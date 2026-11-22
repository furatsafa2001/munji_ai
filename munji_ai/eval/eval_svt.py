"""SVT named by origin, on real recordings, with every model the app runs.

    python -m munji_ai.eval.eval_svt                          # arrhythmia sets, then NSRDB
    python -m munji_ai.eval.eval_svt --day-datasets           # arrhythmia sets only
    python -m munji_ai.eval.eval_svt --max-records 3          # quick trial

The decision layer can name a fast rhythm SVT from the models instead of the
QRS width (config.SVT_NAMING_ENABLED, off; decision_layer.py, "SVT, named by
origin"). It stays off until this shows:

    safety      no rate window named SVT while VT or VF is annotated in it
                (CUDB, VFDB, and any VT in SVDB). VT named SVT is the error
                that matters: it tells a dangerous rhythm as a milder one.
    detection   SVT episodes ("(SVTA") named. Only those holding 150 bpm or
                more over a tachy run of annotated beats can be named by the
                rule; shorter or slower ones are counted apart.
    other       windows named SVT over other rhythms, by annotation. AFL is
                inside the SVT definition (2015 ACC/AHA/HRS); AF is not.
    days        SVT named on healthy 24 h recordings (NSRDB, annotated span),
                per day. A healthy heart at 150 or more is sinus tachycardia,
                which nothing here can tell from SVT.

A record with expert beat labels but no rhythm annotation (SVDB in our copy)
is judged by its beats instead: a named window holding VT_MIN_BEATS PVCs in a
row above 100 bpm is VT (safety), one holding a run of SVT_RUN_MIN_BEATS
supraventricular ectopic beats (A, a, J, S) is SVT, the rest are named over
N beats only (sinus tachycardia, or an SVT the annotators labelled N). SVT and
VT runs are counted from the same labels for detection and safety.

How
---
The rate engine runs on every 40 s window of a record, as in the app. Only
windows where it holds 150 bpm or more can be named, so the models run only
before those: the beat model over BEATS_BACK, ShockableNet over SHOCK_BACK
and RhythmNet over RHYTHM_BACK seconds, enough for what the rule reads
(beats of the rate window, the last 6 ShockableNet windows and its open
episode, the latest rhythm window and an open AF episode). One DecisionLayer
per record, naming on, is fed all of it in time order, as the app would send
it, with lead status unknown (public recordings): ShockableNet every 2.5 s,
RhythmNet every 15 s, the beat model every 10 s behind the gate at the
morphology line, and the rhythm and rate windows behind the gate at the
rhythm line.

Models (none are in git)
    --shock       checkpoints/shock_cv_aha: the cross-validation fold models.
                  A record in the shockable pool is scored by the fold that
                  never saw its patient, any other record by fold 0. A .pt
                  file instead (the final model) scores every record.
    --rhythm      RhythmNet checkpoint (train_rhythm.py output)
    --beat-ckpt   the beat model; --gate-ckpt the quality gate (--no-gate:
                  every window usable)

Test-split records are skipped (eval_beats_model.eligible).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config as C
from ..decision.decision_layer import DecisionConfig, DecisionLayer, beats_from_mask
from ..rules.rate_engine import analyse, held_rates

FS = C.TARGET_FS
RATE_WIN = int(round(C.RULES_WINDOW_SEC * FS))
RATE_STEP = int(round(C.RULES_STRIDE_SEC * FS))
EPISODE_GAP_SEC = DecisionConfig().rate_episode_end * C.RULES_STRIDE_SEC
# How far before a candidate rate window each model runs.
BEATS_BACK = C.RULES_WINDOW_SEC + C.WINDOW_SEC["beat"] + C.WINDOW_STRIDE_SEC["beat"]
SHOCK_BACK = 120.0      # the last 6 windows, and an open episode (4 of 6, ends after 30 s)
RHYTHM_BACK = (DecisionConfig().af_episode_end * C.WINDOW_STRIDE_SEC["rhythm"]
               + C.WINDOW_SEC["rhythm"] + C.WINDOW_STRIDE_SEC["rhythm"])
VENTRICULAR = ("VT", "VF")
# Ambulatory reports count an SVT run from more than 3 beats (Eisenberg et al.,
# J Innov Card Rhythm Manag 2014) or 4 (a Zio patch study protocol,
# NCT03309956); 3 matches the NSVT count. Only runs the rule can name (150 bpm
# held over a tachy run) are scored for detection, so this only sizes the
# list of all runs.
SVT_RUN_MIN_BEATS = 3


# ------------------------------------------------------------ pure helpers
def merge(spans: list, gap: float = 0.0) -> list:
    """Union of (start, end) spans; spans closer than `gap` join."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def regions(ends: list, back: float, total: float) -> list:
    """Spans [end - back, end] around candidate window ends, merged, in seconds."""
    return merge([(max(0.0, e - back), min(total, e)) for e in ends])


def grid(span: tuple, win: int, stride: int, n: int) -> list:
    """Window starts on the record's own grid (multiples of stride) inside span."""
    a, b = int(span[0] * FS), min(n, int(span[1] * FS))
    k0 = -(-a // stride)
    return [s for s in range(k0 * stride, b - win + 1, stride)]


def overlaps(t_end: float, intervals: list) -> dict:
    """label -> seconds it covers in the rate window ending at t_end."""
    lo, out = t_end - C.RULES_WINDOW_SEC, {}
    for s, e, lab in intervals:
        o = min(t_end, e) - max(lo, s)
        if o > 0:
            out[lab] = out.get(lab, 0.0) + o
    return out


def window_kind(t_end: float, intervals: list) -> str:
    """What a named window was: VT/VF if any is in it, then SVT, else the main label."""
    cov = overlaps(t_end, intervals)
    if any(k in cov for k in VENTRICULAR):
        return "VT/VF"
    if "SVT" in cov:
        return "SVT"
    return max(cov, key=cov.get) if cov else "unannotated"


def episode_named(span: tuple, named_ends: list) -> bool:
    """A named rate window overlaps the episode."""
    s, e = span
    return any(t > s and t - C.RULES_WINDOW_SEC < e for t in named_ends)


def can_be_named(span: tuple, beat_times: np.ndarray) -> bool:
    """Annotated beats in the episode hold 150 bpm or more over a tachy run."""
    bt = beat_times[(beat_times >= span[0]) & (beat_times <= span[1])]
    if bt.size < 2:
        return False
    fast = held_rates(np.diff(bt), C.RATE_RUN_BEATS["tachy"])[1]
    return bool(fast >= C.FAST_NARROW_BPM)


def class_runs(times: np.ndarray, classes: np.ndarray, cls: str, min_beats: int) -> list:
    """Runs of min_beats or more consecutive beats of one class, faster than
    VT_MIN_BPM (100) over the run: [(first beat s, last beat s, beats)]."""
    out, i, n = [], 0, len(classes)
    while i < n:
        if classes[i] != cls:
            i += 1
            continue
        j = i
        while j < n and classes[j] == cls:
            j += 1
        k = j - i
        dur = times[j - 1] - times[i]
        if k >= min_beats and dur > 0 and 60.0 * (k - 1) / dur > C.VT_MIN_BPM:
            out.append((float(times[i]), float(times[j - 1]), k))
        i = j
    return out


def beat_kind(t_end: float, times: np.ndarray, classes: np.ndarray) -> str:
    """What a named window was by the expert beat labels; "" without labels."""
    m = (times > t_end - C.RULES_WINDOW_SEC) & (times <= t_end)
    if not m.any():
        return ""
    t, c = times[m], classes[m]
    if class_runs(t, c, "PVC", C.VT_MIN_BEATS):
        return "VT (beats)"
    if class_runs(t, c, "PAC", SVT_RUN_MIN_BEATS):
        return "SVT (beats)"
    return "N only (beats)"


def episodes_of(named_ends: list) -> int:
    """Named windows joined into episodes the way the decision layer joins them."""
    return len(merge([(t, t) for t in named_ends], gap=EPISODE_GAP_SEC))


def _rows(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Each window normalised as false_alarm_rate and af_alarm_rate score them."""
    q25, med, q75 = np.percentile(W, [25, 50, 75], axis=1)
    return (W - med[:, None]) / (((q75 - q25) / 1.349)[:, None] + eps)


# ------------------------------------------------------------------ models
class Models:
    """Every model the decision layer hears from, loaded once."""

    def __init__(self, a):
        import torch

        from ..export.export_tflite import load_model
        from .eval_beats_model import Labeller
        from .eval_rules_holter import Gate
        self.torch, self.device = torch, torch.device(a.device)
        self.folds, self.fold_of, self.single = {}, {}, None
        if a.shock.is_dir():
            from .false_alarm_rate import fold_of_records, load_fold_models
            self.folds = load_fold_models(a.shock, self.device)
            self.fold_of = fold_of_records(a.shock / "shockable_oof.npz", a.cache)
            print(f"shockable   {a.shock}: {len(self.folds)} fold models, out of fold")
        else:
            self.single = load_model("shockable", a.shock)[0].to(self.device)
            print(f"shockable   {a.shock}")
        self.rhythm = load_model("rhythm", a.rhythm)[0].to(self.device)
        print(f"rhythm      {a.rhythm}")
        self.labeller = Labeller(a.beat_ckpt, None if a.no_gate else a.gate_ckpt,
                                 gate=not a.no_gate)
        self.gate = None if a.no_gate else Gate(a.gate_ckpt)

    def _score(self, model, sig, starts, win):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(starts), 512):
                s = np.asarray(starts[i:i + 512])
                x = self.torch.from_numpy(_rows(sig[s[:, None] + np.arange(win)[None, :]])
                                          .astype(np.float32)).to(self.device)
                p = model.predict_proba(x) if hasattr(model, "predict_proba") else model(x)
                out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def shock(self, ds, name, sig, starts):
        model = self.single
        if model is None:
            model = self.folds.get(self.fold_of.get((ds, name), 0), self.folds[0])
        return self._score(model, sig, starts, C.window_samples("shockable"))

    def rhythm_probs(self, sig, starts):
        p = self._score(self.rhythm, sig, starts, C.window_samples("rhythm"))
        idx = [C.RHYTHM_CLASSES.index(k) for k in ("NSR", "AF", "OTHER")]
        return p[:, idx] if len(p) else p

    def gate_blocks(self, sig):
        """P(usable) per 5 s block of the record, or None without a gate."""
        return None if self.gate is None else self.gate.blocks(sig)


def usable(p_block, start: int, win: int) -> bool:
    """A rate or rhythm window passes the gate at the rhythm line, as the
    Holter evaluation reads it (eval_rules_holter, mean of the 5 s blocks)."""
    if p_block is None:
        return True
    blk = C.window_samples("quality")
    seg = p_block[start // blk:(start + win) // blk]
    return bool(seg.size and seg.mean() >= C.GATE_MIN_P["rhythm"])


# ------------------------------------------------------------- one record
def run_record(ds: str, name: str, rec, models: Models, span=None, keep: bool = False) -> dict:
    """Named SVT windows of one record, and what the rate engine saw.

    keep: also return every event fed to the decision layer (for plotting).
    """
    sig, n = rec.signal, len(rec.signal)
    lo, hi = span if span is not None else (0, n)
    starts = [s for s in range(0, n - RATE_WIN + 1, RATE_STEP) if lo <= s and s + RATE_WIN <= hi]
    reports = {s: analyse(sig[s:s + RATE_WIN]) for s in starts}
    cand = [(s + RATE_WIN) / FS for s, r in reports.items()
            if r.fast_run_bpm >= C.FAST_NARROW_BPM]
    out = {"windows": len(starts), "candidates": len(cand), "named": [],
           "hours": (hi - lo) / FS / 3600}
    if not cand:
        return out
    total = n / FS
    p_block = models.gate_blocks(sig)
    ev = []
    win = C.window_samples("shockable")
    sh = sorted({s for r in regions(cand, SHOCK_BACK, total)
                 for s in grid(r, win, C.stride_samples("shockable"), n)})
    for s, p in zip(sh, models.shock(ds, name, sig, sh)):
        ev.append(((s + win) / FS, 0, "shock", float(p)))
    win = C.window_samples("rhythm")
    rh = sorted({s for r in regions(cand, RHYTHM_BACK, total)
                 for s in grid(r, win, C.stride_samples("rhythm"), n)})
    for s, p in zip(rh, models.rhythm_probs(sig, rh)):
        ev.append(((s + win) / FS, 1, "rhythm",
                   (tuple(float(v) for v in p), usable(p_block, s, win))))
    win = C.window_samples("beat")
    breg = regions(cand, BEATS_BACK, total)
    bt = sorted({s for r in breg for s in grid(r, win, C.stride_samples("beat"), n)})
    masks, ok = models.labeller.run(rec, bt)
    for s, m, u in zip(bt, masks, ok):
        ev.append(((s + win) / FS, 2, "beats", (beats_from_mask(m, s / FS, FS), bool(u))))
    for s, r in reports.items():
        t = (s + RATE_WIN) / FS
        if any(a + C.RULES_WINDOW_SEC <= t <= b for a, b in breg):
            ev.append((t, 3, "rate", (r, usable(p_block, s, RATE_WIN))))
    dl = DecisionLayer(DecisionConfig(svt_naming=True))
    ev.sort(key=lambda e: (e[0], e[1]))
    if keep:
        out["events"] = ev
    for t, _, kind, x in ev:
        if kind == "shock":
            dl.on_shockable(t, x)
        elif kind == "rhythm":
            dl.on_rhythm(t, x[0], usable=x[1])
        elif kind == "beats":
            dl.on_beats(t, x[0], usable=x[1])
        else:
            dl.on_rate(t, x[0], usable=x[1])
            if dl.svt_now:
                out["named"].append(t)
    return out


# ------------------------------------------------------------------- parts
def part_arrhythmia(models, raw: Path, datasets: list, max_records: int, show: int) -> dict:
    from ..data.loader import beat_classes, load_record
    from ..data.windows import rhythm_intervals
    from .eval_beats_model import eligible
    tot = defaultdict(lambda: defaultdict(float))
    listed = defaultdict(list)
    for ds in datasets:
        t0 = time.time()
        names = eligible(ds, raw, max_records)
        for j, name in enumerate(names, 1):
            rec = load_record(ds, name, raw)
            iv = [(s / FS, e / FS, lab) for s, e, lab in rhythm_intervals(rec)]
            r = run_record(ds, name, rec, models)
            k = tot[ds]
            k["records"] += 1
            k["windows"] += r["windows"]
            k["candidates"] += r["candidates"]
            beats = np.asarray(rec.beat_samples, dtype=np.float64) / FS
            cls = beat_classes(rec) if beats.size else np.array([], dtype="<U8")
            by_beats = not iv and beats.size > 0
            for t in r["named"]:
                kind = window_kind(t, iv)
                if by_beats:
                    kind = beat_kind(t, beats, cls) or kind
                k[f"named_{kind}"] += 1
                if kind not in ("SVT", "SVT (beats)") and len(listed[kind]) < show:
                    listed[kind].append(f"{ds}/{name} at {t:.0f} s")
            if by_beats:
                for s, e, _ in class_runs(beats, cls, "PVC", C.VT_MIN_BEATS):
                    k["vt_episodes"] += 1
                    k["vt_named"] += episode_named((s, e), r["named"])
                for s, e, _ in class_runs(beats, cls, "PAC", SVT_RUN_MIN_BEATS):
                    able = can_be_named((s, e), beats)
                    named = episode_named((s, e), r["named"])
                    k["svtb_runs"] += 1
                    k["svtb_able"] += able
                    k["svtb_named"] += named
                    k["svtb_able_named"] += able and named
            for s, e, lab in iv:
                if lab in VENTRICULAR:
                    k["vt_episodes"] += 1
                    k["vt_named"] += episode_named((s, e), r["named"])
                elif lab == "SVT":
                    able = can_be_named((s, e), beats) if beats.size else False
                    named = episode_named((s, e), r["named"])
                    k["svt_episodes"] += 1
                    k["svt_able"] += able
                    k["svt_named"] += named
                    k["svt_able_named"] += able and named
            if j % 10 == 0 or j == len(names):
                print(f"  {ds:<8} {j:>3}/{len(names)}  {time.time() - t0:6.0f} s", flush=True)
    return {"tot": tot, "listed": listed}


def part_days(models, raw: Path, datasets: list, max_records: int) -> dict:
    from ..data.loader import load_record
    from .eval_beats_model import eligible
    tot = defaultdict(float)
    for ds in datasets:
        t0 = time.time()
        names = eligible(ds, raw, max_records)
        for j, name in enumerate(names, 1):
            rec = load_record(ds, name, raw)
            span = None
            if len(rec.beat_samples) >= 2:
                span = (int(rec.beat_samples.min()), int(rec.beat_samples.max()) + 1)
            r = run_record(ds, name, rec, models, span)
            tot["hours"] += r["hours"]
            tot["candidates"] += r["candidates"]
            tot["named"] += len(r["named"])
            tot["episodes"] += episodes_of(r["named"])
            if j % 5 == 0 or j == len(names):
                print(f"  {ds:<8} {j:>3}/{len(names)}  {time.time() - t0:6.0f} s", flush=True)
    return tot


def show_arrhythmia(res: dict) -> None:
    tot, listed = res["tot"], res["listed"]
    print("\n" + "=" * 70)
    print("Safety: VT or VF annotated in a window named SVT (must be 0)")
    print(f"  {'dataset':<9}{'records':>8}{'VT/VF episodes':>16}{'named SVT':>11}{'windows':>9}")
    for ds, k in tot.items():
        print(f"  {ds:<9}{k['records']:>8.0f}{k['vt_episodes']:>16.0f}{k['vt_named']:>11.0f}"
              f"{k['named_VT/VF'] + k['named_VT (beats)']:>9.0f}")
    print("  Episodes: rhythm annotation, or 3 PVCs in a row above 100 bpm by beat labels.")
    print("\nDetection: SVT episodes (annotation \"(SVTA\")")
    print(f"  {'dataset':<9}{'episodes':>9}{'named':>7}   {'150+ held 17 beats':>19}{'named':>7}")
    for ds, k in tot.items():
        if k["svt_episodes"]:
            print(f"  {ds:<9}{k['svt_episodes']:>9.0f}{k['svt_named']:>7.0f}   "
                  f"{k['svt_able']:>19.0f}{k['svt_able_named']:>7.0f}")
        if k["svtb_runs"]:
            print(f"  {ds + '*':<9}{k['svtb_runs']:>9.0f}{k['svtb_named']:>7.0f}   "
                  f"{k['svtb_able']:>19.0f}{k['svtb_able_named']:>7.0f}")
    print("  * runs of supraventricular ectopic beats (expert beat labels), no rhythm annotation.")
    print("  Only episodes of 150 bpm or more held over a tachy run can be named.")
    print("\nOther rhythms named SVT (rate windows, by annotation)")
    other = defaultdict(float)
    for k in tot.values():
        for key, v in k.items():
            if key.startswith("named_") and key not in ("named_SVT", "named_VT/VF",
                                                        "named_SVT (beats)", "named_VT (beats)"):
                other[key[6:]] += v
    if other:
        for lab, v in sorted(other.items(), key=lambda kv: -kv[1]):
            print(f"  {lab:<12}{v:>6.0f}")
        print("  AFL is inside the SVT definition (2015 ACC/AHA/HRS); AF and NSR are not.")
    else:
        print("  none")
    sb = sum(k["named_SVT (beats)"] for k in tot.values())
    if sb:
        print(f"\nNamed over a run of supraventricular ectopic beats (beat labels): {sb:.0f} windows")
    for kind, items in listed.items():
        print(f"\n  first {kind} windows named SVT: " + "; ".join(items))


def show_days(tot: dict, datasets: list) -> None:
    h = tot["hours"]
    print("\n" + "=" * 70)
    print(f"Days ({', '.join(datasets)}, annotated span): {h:.0f} h")
    if not h:
        return
    print(f"  rate windows at 150 bpm or more   {tot['candidates']:.0f}")
    print(f"  windows named SVT                 {tot['named']:.0f}")
    print(f"  SVT episodes per 24 h             {tot['episodes'] / h * 24:.2f}")
    print("  Healthy subjects: an SVT here is sinus tachycardia named SVT.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["svdb", "cudb", "vfdb"])
    ap.add_argument("--day-datasets", nargs="*", default=["nsrdb"])
    ap.add_argument("--shock", type=Path, default=Path("checkpoints/shock_cv_aha"))
    ap.add_argument("--rhythm", type=Path, default=Path("checkpoints/rhythm_final/rhythm_net.pt"))
    ap.add_argument("--beat-ckpt", type=Path, default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--gate-ckpt", type=Path, default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--no-gate", action="store_true", help="every window counts as usable")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--max-records", type=int, default=0, help="per dataset (0 = all)")
    ap.add_argument("--show", type=int, default=10, help="list this many wrongly named windows")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    need = [("--shock", a.shock), ("--rhythm", a.rhythm), ("--beat-ckpt", a.beat_ckpt)]
    if not a.no_gate:
        need.append(("--gate-ckpt", a.gate_ckpt))
    missing = [f"{flag} {p}" for flag, p in need if not p.exists()]
    if missing:
        print("missing: " + ", ".join(missing))
        print("Every model the app runs is needed; point the flags at your copies.")
        return 1
    print(f"rule        SVT named by origin (config: 150 bpm, "
          f"{C.SVT_MIN_SUPRA_FRACTION:.0%} N or PAC, no {C.VT_MIN_BEATS} PVCs in a row, "
          f"ShockableNet quiet at {C.SHOCKABLE_THRESHOLD}, not AF)")
    models = Models(a)
    if a.datasets:
        print("\narrhythmia recordings ...", flush=True)
        show_arrhythmia(part_arrhythmia(models, a.raw, a.datasets, a.max_records, a.show))
    if a.day_datasets:
        print("\ndaily life (the slow part) ...", flush=True)
        show_days(part_days(models, a.raw, a.day_datasets, a.max_records), a.day_datasets)
    print("\nTurn SVT naming on (config.SVT_NAMING_ENABLED) only if safety is 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
