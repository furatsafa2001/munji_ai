"""The decision layer's beat rules, fed by the real beat model and quality gate.

    python -m munji_ai.eval.eval_beats_model --part vt          # sustained VT (CUDB, VFDB)
    python -m munji_ai.eval.eval_beats_model --part nsrdb       # false alerts per day
    python -m munji_ai.eval.eval_beats_model --part runs        # the 547 VT runs
    python -m munji_ai.eval.eval_beats_model --part naming      # VF or VT naming (shadow)
    python -m munji_ai.eval.eval_beats_model --part vt --no-gate
    python -m munji_ai.eval.eval_beats_model --part runs --oracle

Needs checkpoints/beat_segmenter.pt and checkpoints/quality_gate.pt (Furat's
models; not in git). --oracle replaces the model with the expert beat labels
(a perfect model, gate always passing): it checks the pipeline end to end
without PyTorch, and is the reference the model is compared with.

Why
---
eval_beats_vt ran the rules on expert labels, and every one of its 547 runs
was under 30 s, so the sustained-VT alert (WCT) was never tested on real
VT. CUDB and VFDB hold sustained VT but no beat labels, so only the beat
model can label their beats. This runs the chain the app will run:

    20 s windows every 10 s (config, "beat"), each normalised as in training
    -> beat model mask -> beats_from_mask -> DecisionLayer.on_beats, with
    usable = P(usable) >= GATE_MIN_P["morphology"], sub-windows combined by
    GATE_COMBINER (gate scoring exactly as eval/gated_beat.py, where the 0.90
    line was measured)

Parts
-----
vt      every VT episode in CUDB and VFDB (rhythm annotations). Episodes of
        30 s or more should raise WCT; shorter ones should not, and may show
        as NSVT. WCT alerts away from VT are counted, split into during VF
        (also an emergency, and ShockableNet's) and elsewhere.
nsrdb   healthy 24 h recordings: WCT alerts, NSVT, bigeminy and trigeminy per
        day, and the peak hourly PVC burden, for the model and for the expert
        labels side by side. Only the annotated span is used.
runs    data/cache/vt_runs.npz: NSVT and WCT per run, by length, expert labels
        and model side by side.
naming  the VF-or-VT guess (config.VFVT_NAMING_ENABLED, off) against the
        rhythm annotation, for every VF and VT episode in CUDB and VFDB. A
        ShockableNet alert is reproduced NAMING_DELAY_SEC into the episode
        (4 positive windows of 2.5 s confirm it); the beat windows up to that
        moment go through on_beats first, in time order, and the guess is read
        from the alert's evidence, the same code path the app runs. Outcomes:
        correct, "VF/VT" (no decision: what the app shows today, safe) and
        wrong (VF called VT or VT called VF: the one that matters). Naming
        should only be turned on if wrong is near zero.

ShockableNet does not run here, so rule 3 (no WCT while VF/VT is open) never
applies: every WCT the beat path would raise is counted.

Test-split patients are skipped in every part.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config as C
from ..data import preprocess as pp
from ..data import registry
from ..data.loader import BEAT_MAP, _patient_id, list_records, load_record
from ..data.splits import assign
from ..data.windows import _episode_bpm, paint_mask, rhythm_intervals, split_vt_by_rate
from ..decision.decision_layer import DecisionLayer, beats_from_mask

WIN = C.window_samples("beat")
STRIDE = C.stride_samples("beat")
AROUND_SEC = 60.0
NAMING_DELAY_SEC = 10.0          # ShockableNet confirmation: 4 windows x 2.5 s
BEAT_BINS = ((3, 4), (5, 9), (10, 29), (30, 10 ** 9))


# ------------------------------------------------------------------- labeller
class Labeller:
    """Beat masks and gate decisions for windows of one record."""

    def __init__(self, beat_ckpt: Path | None, gate_ckpt: Path | None, gate: bool):
        self.oracle = beat_ckpt is None
        self.gate_ckpt = gate_ckpt if gate else None
        if not self.oracle:
            import torch

            from ..models.beat_segmenter import build, clean_mask
            blob = torch.load(beat_ckpt, map_location="cpu", weights_only=False)
            self._model = build()
            self._model.load_state_dict(blob["state_dict"])
            self._model.eval()
            self._clean = clean_mask
            self._torch = torch
            print(f"beat model  {beat_ckpt}  epoch {blob.get('epoch')}  "
                  f"val beat F1 {blob.get('val_beat_f1', float('nan')):.3f}")
        print("gate        " + (f"{gate_ckpt}  P(usable) >= {C.GATE_MIN_P['morphology']} "
                                f"({C.GATE_COMBINER} of 5 s sub-windows)"
                                if self.gate_ckpt else "off: every window counts as usable"))

    def run(self, rec, starts: list[int]):
        """(masks (n, WIN), usable (n,)) for windows starting at `starts`."""
        n = len(starts)
        if n == 0:
            return np.zeros((0, WIN), np.uint8), np.zeros(0, bool)
        if self.oracle:
            cls = np.array([BEAT_MAP.get(s, "?") for s in rec.beat_symbols])
            idx = np.asarray(rec.beat_samples)
            masks = []
            for s in starts:
                m = (idx >= s) & (idx < s + WIN)
                masks.append(paint_mask(WIN, idx[m] - s, cls[m], rec.fs))
            usable = np.ones(n, bool)
            if self.gate_ckpt:
                usable = self._gate(np.stack([pp.normalize(rec.signal[s:s + WIN]) for s in starts]))
            return np.stack(masks), usable
        X = np.stack([pp.normalize(rec.signal[s:s + WIN]) for s in starts]).astype(np.float32)
        out = []
        with self._torch.no_grad():
            for i in range(0, n, 16):
                xb = self._torch.from_numpy(X[i:i + 16])
                out.append(self._clean(self._model(xb).argmax(dim=1).numpy()))
        masks = np.concatenate(out)
        usable = self._gate(X) if self.gate_ckpt else np.ones(n, bool)
        return masks, usable

    def _gate(self, X):
        from .gated_beat import combine, gate_scores
        p_sub = gate_scores(np.asarray(X, np.float32), self.gate_ckpt)
        return combine(p_sub, C.GATE_COMBINER, C.GATE_MIN_P["morphology"])


def through_layer(rec, lab: Labeller, lo: int, hi: int):
    """Windows over samples [lo, hi) -> decision layer, as the app sends them."""
    lo, hi = max(0, lo), min(len(rec.signal), hi)
    starts = list(range(lo, max(lo, hi - WIN) + 1, STRIDE)) if hi - lo >= WIN else []
    masks, usable = lab.run(rec, starts)
    dl, alerts = DecisionLayer(), []
    for s, m, u in zip(starts, masks, usable):
        t0 = s / rec.fs
        alerts += dl.on_beats(t0 + WIN / rec.fs, beats_from_mask(m, t0, rec.fs), usable=bool(u))
    dl.on_beats(hi / rec.fs + WIN / rec.fs, [], usable=False)    # closes an open run
    through_layer.last_masks = (starts, masks)
    return dl, alerts, list(zip(starts, usable))


def beats_in(starts, masks, fs, t0: float, t1: float) -> list:
    """Model beats in [t0, t1], read off every window, duplicates dropped."""
    out = []
    for st, m in zip(starts, masks):
        if st / fs > t1 or (st / fs + WIN / fs) < t0:
            continue
        out += [b for b in beats_from_mask(m, st / fs, fs) if t0 <= b[0] <= t1]
    out.sort()
    kept = []
    for bt, c in out:
        if not kept or bt > kept[-1][0] + C.BEAT_DEDUP_SEC:
            kept.append((bt, c))
    return kept


def eligible(ds_key: str, raw: Path, max_records: int = 0) -> list[str]:
    """Records outside the test split (and not a test-only dataset)."""
    ds = registry.get(ds_key)
    try:
        names = list_records(ds_key, raw)
    except Exception as e:                       # dataset not downloaded
        print(f"  {ds_key}: skipped ({e})")
        return []
    names = [n for n in names if ds.role != "val" and assign(_patient_id(ds, n)) != "test"]
    return names[:max_records] if max_records else names


def _pct(v):
    return f"{np.mean(v):.0%}" if len(v) else "  -"


# ----------------------------------------------------------------------- parts
def part_vt(lab: Labeller, raw: Path, max_records: int) -> None:
    rows, far = [], {"vf": 0, "other": 0}
    hours = 0.0
    for ds_key in ("cudb", "vfdb"):
        for name in eligible(ds_key, raw, max_records):
            rec = load_record(ds_key, name, raw)
            ivs = rhythm_intervals(rec)
            vt = [(s, e) for s, e, lab_ in ivs if lab_ == "VT"]
            vf = [(s, e) for s, e, lab_ in ivs if lab_ == "VF"]
            dl, alerts, used = through_layer(rec, lab, 0, len(rec.signal))
            hours += len(rec.signal) / rec.fs / 3600.0
            fs = rec.fs
            st_all, m_all = through_layer.last_masks
            for s, e in vt:
                ts, te = s / fs, e / fs
                win_ok = [u for st, u in used if st < e and st + WIN > s]
                bts = beats_in(st_all, m_all, fs, ts, te)
                n_b = len(bts)
                model_bpm = 60.0 * (n_b - 1) / (bts[-1][0] - bts[0][0]) if n_b > 2 and bts[-1][0] > bts[0][0] else 0.0
                ref = _episode_bpm(rec.signal[s:e], fs)
                rows.append({
                    "ref_bpm": float(ref) if ref else float("nan"),
                    "model_bpm": model_bpm,
                    "pvc": sum(c == "PVC" for _, c in bts) / n_b if n_b else float("nan"),
                    "n": sum(c == "N" for _, c in bts) / n_b if n_b else float("nan"),
                    "dur": te - ts, "patient": rec.patient,
                    "wct": any(a.kind == "WCT" and ts <= a.t <= te + C.VT_SUSTAINED_SEC for a in alerts),
                    "nsvt": any(a0 <= te and a1 >= ts for a0, a1, _, _ in dl.nsvt + dl.long_runs),
                    "gate": float(np.mean(win_ok)) if win_ok else float("nan")})
            for a in alerts:
                if a.kind != "WCT":
                    continue
                if any(s / fs <= a.t <= e / fs + C.VT_SUSTAINED_SEC for s, e in vt):
                    continue
                in_vf = any(s / fs <= a.t <= e / fs + C.VT_SUSTAINED_SEC for s, e in vf)
                far["vf" if in_vf else "other"] += 1
            print(f"  {ds_key} {name}: {len(vt)} VT, {len(vf)} VF episodes, "
                  f"{sum(a.kind == 'WCT' for a in alerts)} WCT alerts", flush=True)
    print(f"\n{hours:.1f} h of CUDB/VFDB, {len(rows)} VT episodes")
    print(f"\n{'VT episodes':<16}{'n':>5}{'patients':>10}{'WCT alert':>11}{'NSVT/run':>10}{'gate passed':>13}")
    for name, sel in (("under 30 s", lambda r: r["dur"] < C.VT_SUSTAINED_SEC),
                      ("30 s or more", lambda r: r["dur"] >= C.VT_SUSTAINED_SEC)):
        g = [r for r in rows if sel(r)]
        gp = [r["gate"] for r in g if np.isfinite(r["gate"])]
        print(f"{name:<16}{len(g):>5}{len({r['patient'] for r in g}):>10}"
              f"{_pct([r['wct'] for r in g]):>11}{_pct([r['nsvt'] for r in g]):>10}"
              f"{(f'{np.mean(gp):.0%}' if gp else '  -'):>13}")
    print("\nWhat the beat model sees inside VT episodes (medians):")
    print(f"{'VT episodes':<16}{'rate (signal)':>15}{'rate (model)':>14}{'PVC share':>11}{'N share':>9}")
    for name, sel in (("under 30 s", lambda r: r["dur"] < C.VT_SUSTAINED_SEC),
                      ("30 s or more", lambda r: r["dur"] >= C.VT_SUSTAINED_SEC)):
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        med = lambda k: np.nanmedian([r[k] for r in g]) if any(np.isfinite(r[k]) for r in g) else float("nan")
        print(f"{name:<16}{med('ref_bpm'):>15.0f}{med('model_bpm'):>14.0f}"
              f"{med('pvc'):>11.0%}{med('n'):>9.0%}")
    print("Reading: model rate near the signal rate with a low PVC share -> it sees the beats but"
          "\ncalls them normal; model rate far below -> it misses the beats themselves.")
    print(f"\nWCT alerts away from VT: {far['vf']} during VF (ShockableNet's, also an emergency), "
          f"{far['other']} elsewhere ({far['other'] / hours * 24 if hours else 0:.1f} per 24 h)")
    print("Expected: 30 s or more -> WCT alert; under 30 s -> no WCT. 'gate passed' is the share of"
          "\nwindows over the episode the gate let through: VT the gate refuses cannot alert here.")


def part_nsrdb(lab: Labeller, raw: Path, max_records: int) -> None:
    oracle = Labeller(None, None, gate=False) if not lab.oracle else None
    tot = defaultdict(float)
    for name in eligible("nsrdb", raw, max_records):
        rec = load_record("nsrdb", name, raw)
        if len(rec.beat_samples) < 2:
            continue
        lo, hi = int(rec.beat_samples.min()), int(rec.beat_samples.max()) + 1
        days = (hi - lo) / rec.fs / 86400.0
        tot["days"] += days
        for tag, lb in (("model", lab), ("expert", oracle)):
            if lb is None:
                continue
            dl, alerts, used = through_layer(rec, lb, lo, hi)
            rep = dl.report()
            tot[f"{tag}_wct"] += sum(a.kind == "WCT" for a in alerts)
            tot[f"{tag}_nsvt"] += rep["nsvt"]
            tot[f"{tag}_big"] += rep["bigeminy_episodes"]
            tot[f"{tag}_tri"] += rep["trigeminy_episodes"]
            tot[f"{tag}_pvc"] += rep["beats"]["PVC"]
            tot[f"{tag}_beats"] += sum(rep["beats"].values())
            pk = rep["pvc_burden"]["peak_hour"]
            tot[f"{tag}_peak"] = max(tot[f"{tag}_peak"], pk["burden"] if pk else 0.0)
            if tag == "model":
                tot["windows"] += len(used)
                tot["rejected"] += sum(not u for _, u in used)
        print(f"  nsrdb {name}: {days * 24:.1f} h", flush=True)
    d = tot["days"]
    if not d:
        print("no NSRDB records")
        return
    print(f"\nNSRDB, {d * 24:.0f} h (annotated span), gate rejected "
          f"{tot['rejected'] / max(tot['windows'], 1):.1%} of beat windows")
    tags = ["model"] + (["expert"] if not lab.oracle else [])
    shown = {"model": "expert" if lab.oracle else "model", "expert": "expert"}
    print(f"\n{'per 24 h':<22}" + "".join(f"{shown[t]:>10}" for t in tags))
    for key, label in (("wct", "WCT alerts (false)"), ("nsvt", "NSVT"), ("big", "bigeminy episodes"),
                       ("tri", "trigeminy episodes")):
        print(f"{label:<22}" + "".join(f"{tot[f'{t}_{key}'] / d:>10.2f}" for t in tags))
    print(f"{'PVC share of beats':<22}" + "".join(
        f"{tot[f'{t}_pvc'] / max(tot[f'{t}_beats'], 1):>10.2%}" for t in tags))
    print(f"{'peak hourly burden':<22}" + "".join(f"{tot[f'{t}_peak']:>10.1%}" for t in tags))
    print("Healthy subjects: any WCT is false; NSVT, patterns and burden should sit near the expert column.")


def part_runs(lab: Labeller, raw: Path, runs_file: Path) -> None:
    z = np.load(runs_file)
    oracle = Labeller(None, None, gate=False) if not lab.oracle else None
    by_rec = defaultdict(list)
    for i in range(len(z["record"])):
        by_rec[(str(z["dataset"][i]), str(z["record"][i]))].append(i)
    rows = []
    t0 = time.time()
    for (ds_key, name), idx in sorted(by_rec.items()):
        try:
            rec = load_record(ds_key, name, raw)
        except FileNotFoundError:
            continue
        for i in idx:
            scale = rec.fs / float(z["fs"][i])
            s, e = int(z["start"][i] * scale), int(z["end"][i] * scale)
            pad = int(AROUND_SEC * rec.fs)
            row = {"beats": int(z["beats"][i]), "dur": (e - s) / rec.fs}
            for tag, lb in (("model", lab), ("expert", oracle)):
                if lb is None:
                    continue
                dl, alerts, _ = through_layer(rec, lb, s - pad, e + pad)
                ts, te = s / rec.fs, e / rec.fs
                row[f"{tag}_nsvt"] = any(a0 <= te and a1 >= ts for a0, a1, _, _ in dl.nsvt + dl.long_runs)
                row[f"{tag}_wct"] = any(a.kind == "WCT" and ts <= a.t <= te + C.VT_SUSTAINED_SEC
                                        for a in alerts)
            rows.append(row)
        if rows and len(rows) % 50 == 0:
            print(f"  {len(rows)} runs   {time.time() - t0:.0f} s", flush=True)
    tags = ["model"] + (["expert"] if not lab.oracle else [])
    shown = {"model": "expert" if lab.oracle else "model", "expert": "expert"}
    print(f"\n{len(rows)} VT runs")
    print(f"\n{'by length':<14}{'runs':>6}" + "".join(
        f"{shown[t] + ' NSVT':>14}{shown[t] + ' WCT':>12}" for t in tags))
    for lo_b, hi_b in BEAT_BINS:
        g = [r for r in rows if lo_b <= r["beats"] <= hi_b]
        if not g:
            continue
        name = f"{lo_b}-{hi_b} beats" if hi_b < 10 ** 9 else f"{lo_b}+ beats"
        print(f"{name:<14}{len(g):>6}" + "".join(
            f"{_pct([r[t + '_nsvt'] for r in g]):>14}{_pct([r[t + '_wct'] for r in g]):>12}" for t in tags))
    print("Expected (all runs here are under 30 s): NSVT near the expert column, WCT 0%.")


def part_naming(lab: Labeller, raw: Path, max_records: int, delay: float) -> None:
    rows = []
    for ds_key in ("cudb", "vfdb"):
        for name in eligible(ds_key, raw, max_records):
            rec = load_record(ds_key, name, raw)
            fs = rec.fs
            for s, e, label in split_vt_by_rate(rec, rhythm_intervals(rec)):
                if label not in ("VF", "VT_FAST", "VT_SLOW", "VT"):
                    continue
                t_alert = min(s / fs + delay, e / fs)
                lo = max(0, s - int(AROUND_SEC * fs))
                # beat windows that END by the alert, in time order
                starts = list(range(lo, max(lo, int(t_alert * fs) - WIN) + 1, STRIDE))
                masks, usable = lab.run(rec, starts)
                dl = DecisionLayer()
                for st, m, u in zip(starts, masks, usable):
                    t0 = st / fs
                    dl.on_beats(t0 + WIN / fs, beats_from_mask(m, t0, fs), usable=bool(u))
                guess = None
                for k in range(6):                       # 4 of 6 positive: fires at t_alert
                    for a in dl.on_shockable(t_alert - (3 - k) * 2.5 if k <= 3 else t_alert, 0.9):
                        guess = a.evidence.get("shadow_name")
                truth = "VF" if label == "VF" else "VT"
                rows.append({"label": label, "truth": truth, "guess": guess or "VF/VT",
                             "patient": rec.patient, "dur": (e - s) / fs})
            print(f"  {ds_key} {name}: {sum(1 for r in rows if r['patient'] == rec.patient)} episodes", flush=True)
    if not rows:
        print("no VF or VT episodes")
        return
    print(f"\nVF/VT naming, guess read {delay:.0f} s into each episode "
          f"({'expert labels' if lab.oracle else 'beat model'})")
    print(f"\n{'episode':<18}{'n':>5}{'patients':>10}{'called VT':>11}{'called VF':>11}"
          f"{'VF/VT':>8}{'correct':>9}{'WRONG':>8}")
    groups = [("VF", "VF"), ("VT 150+ bpm", "VT_FAST"), ("VT under 150", "VT_SLOW"), ("VT, rate unread", "VT")]
    for title, label in groups:
        g = [r for r in rows if r["label"] == label]
        if not g:
            continue
        n = len(g)
        vt = sum(r["guess"] == "VT" for r in g)
        vf = sum(r["guess"] == "VF" for r in g)
        ab = sum(r["guess"] == "VF/VT" for r in g)
        ok = vf if label == "VF" else vt
        bad = vt if label == "VF" else vf
        print(f"{title:<18}{n:>5}{len({r['patient'] for r in g}):>10}{vt / n:>11.0%}{vf / n:>11.0%}"
              f"{ab / n:>8.0%}{ok / n:>9.0%}{bad / n:>8.0%}")
    shock = [r for r in rows if r["label"] in ("VF", "VT_FAST")]
    if shock:
        wrong = sum((r["truth"] == "VF" and r["guess"] == "VT") or (r["truth"] == "VT" and r["guess"] == "VF")
                    for r in shock)
        print(f"\nShockableNet's territory (VF and VT 150+): {wrong} of {len(shock)} named wrongly "
              f"({wrong / len(shock):.0%}).")
    print("Turn naming on (config.VFVT_NAMING_ENABLED) only if WRONG is near zero on new patients;"
          "\n'VF/VT' is a safe abstention, the alert the app gives today. Slow VT rows are for"
          "\ninformation: ShockableNet does not fire on them, the WCT alert covers them.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("vt", "nsrdb", "runs", "naming"), required=True)
    ap.add_argument("--naming-delay", type=float, default=NAMING_DELAY_SEC,
                    help="seconds into each episode the ShockableNet alert is placed")
    ap.add_argument("--beat-ckpt", type=Path, default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--gate-ckpt", type=Path, default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--no-gate", action="store_true", help="every window counts as usable")
    ap.add_argument("--oracle", action="store_true",
                    help="expert beat labels instead of the model (checks the pipeline)")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--runs", type=Path, default=C.CACHE_DIR / "vt_runs.npz")
    ap.add_argument("--max-records", type=int, default=0, help="per dataset (0 = all)")
    a = ap.parse_args(argv)

    use_gate = not a.no_gate and not a.oracle
    if not a.oracle and not a.beat_ckpt.exists():
        print(f"no beat model at {a.beat_ckpt} (Furat's beat_segmenter.pt); --oracle runs without it")
        return 1
    if use_gate and not a.gate_ckpt.exists():
        print(f"no quality gate at {a.gate_ckpt}; use --no-gate to run without it")
        return 1
    lab = Labeller(None if a.oracle else a.beat_ckpt, a.gate_ckpt, gate=use_gate)
    t0 = time.time()
    if a.part == "vt":
        part_vt(lab, a.raw, a.max_records)
    elif a.part == "nsrdb":
        part_nsrdb(lab, a.raw, a.max_records)
    elif a.part == "naming":
        part_naming(lab, a.raw, a.max_records, a.naming_delay)
    else:
        part_runs(lab, a.raw, a.runs)
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
