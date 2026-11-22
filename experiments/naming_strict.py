"""Can a stricter rule name VT safely? The VF/VT naming rule, swept.

    python experiments/naming_strict.py                    # CUDB + VFDB
    python experiments/naming_strict.py --sddb             # + SDDB VF onsets
    python experiments/naming_strict.py --sddb --delays 10 20 30

Today's rule (decision_layer._vfvt_guess, off in the app): over the 10 s
before the ShockableNet alert, 3 or more PVCs in a row -> "VT"; no beat at
all -> "VF"; anything else -> "VF/VT". With Furat's beat model on CUDB and
VFDB it named about 1 episode in 5 and was wrong 6 times in 115 (5%).

This asks whether a stricter "VT" can be wrong close to never while still
naming some VT:

    run     the longest run of consecutive PVCs is at least L beats
    rate    above 100 bpm inside that run (the definition of VT)
    steady  the RR intervals inside the run vary by at most CV (std / mean);
            monomorphic VT is regular, a "run" read off fibrillation is not
    source  beats from every window, or only from windows the quality gate
            passed

The VF side ("no beat at all" -> VF) is scored separately.

The beat model and the gate run once per episode; every rule is applied to the
stored beats, so the sweep costs nothing extra. The wrong rate among named
episodes is given with a one-sided 95% upper bound (Clopper-Pearson): with this
few episodes "0 wrong" is not "near zero", and the bound says how far from zero
it could still be. The rule that looks best is picked on this same data, so it
is a candidate, not a result: it needs new patients before naming is turned on.

SDDB (Sudden Cardiac Death Holter Database, PhysioNet, ODC-By 1.0): 23 Holter
recordings of up to 25 h; Table 1 on its PhysioNet page gives the VF onset time
of 20 of them. VF only, which measures the dangerous mistake (VF named VT) on
patients that neither the beat model nor ShockableNet has seen. Onset times
are to the second and some VF begins as VT that degenerates, so --delays reads
the guess later into the episode too. Download first:

    aws s3 sync --no-sign-request s3://physionet-open/sddb/1.0.0/ data/raw/sddb/

Reads only. Test-split patients of CUDB and VFDB are skipped; SDDB is in no
split. Needs checkpoints/beat_segmenter.pt and checkpoints/quality_gate.pt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C  # noqa: E402
from munji_ai.data import preprocess as pp  # noqa: E402
from munji_ai.data.loader import Record, load_record  # noqa: E402
from munji_ai.data.windows import rhythm_intervals, split_vt_by_rate  # noqa: E402
from munji_ai.decision.decision_layer import DecisionLayer, beats_from_mask  # noqa: E402
from munji_ai.eval.eval_beats_model import (  # noqa: E402
    AROUND_SEC, NAMING_DELAY_SEC, STRIDE, WIN, Labeller, beats_in, eligible)

LOOKBACK_SEC = C.VFVT_NAMING_LOOKBACK_SEC
RUN_LENGTHS = (3, 4, 5, 6, 8, 10, 12)
CVS = (None, 0.20, 0.15, 0.10)
SOURCES = ("all", "usable")

# PhysioNet SDDB, Table 1, "VF Onset Time (elapsed)", H:MM:SS from the start
# of the record. Records 40 and 49 are paced with no VF, 42 has no VF.
SDDB_VF_ONSET = {
    "30": "07:54:33", "31": "13:42:24", "32": "16:45:18", "33": "04:46:19",
    "34": "06:35:44", "35": "24:34:56", "36": "18:59:01", "37": "01:31:13",
    "38": "08:01:54", "39": "04:37:51", "41": "02:59:24", "43": "15:37:11",
    "44": "19:38:45", "45": "18:09:17", "46": "03:41:47", "47": "06:13:01",
    "48": "02:29:40", "50": "11:45:43", "51": "22:58:23", "52": "02:32:40",
}


# --------------------------------------------------------------------- rules
def longest_pvc_run(beats) -> list:
    """Times of the longest run of consecutive PVCs."""
    best, cur = [], []
    for t, c in beats:
        cur = cur + [t] if c == "PVC" else []
        if len(cur) > len(best):
            best = cur
    return best


def vt_rule(beats, run_len: int | None, cv_max: float | None) -> bool:
    """run_len None: today's rule (3 in a row, nothing else checked)."""
    run = longest_pvc_run(beats)
    if run_len is None:
        return len(run) >= C.VT_MIN_BEATS
    if len(run) < run_len:
        return False
    rr = np.diff(run)
    if rr.size == 0 or rr.mean() <= 0 or 60.0 / rr.mean() <= C.VT_MIN_BPM:
        return False
    if cv_max is None:
        return True
    return rr.size >= 2 and float(rr.std() / rr.mean()) <= cv_max


def guess(beats, run_len, cv_max, name_vf: bool) -> str:
    if not beats:
        return "VF" if name_vf else "VF/VT"
    return "VT" if vt_rule(beats, run_len, cv_max) else "VF/VT"


def upper95(wrong: int, n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound of wrong / n."""
    if n == 0:
        return float("nan")
    if wrong >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(0.95, wrong + 1, n - wrong))


# ------------------------------------------------------------------ episodes
def episode_beats(lab: Labeller, rec: Record, s: int, t_alert: float) -> dict:
    """Beats of the LOOKBACK before t_alert, from windows that end by then.

    Also the decision layer's own guess on the same windows (today's rule,
    the code path the app runs), to check the rules here reproduce it.
    """
    fs = rec.fs
    lo = max(0, s - int(AROUND_SEC * fs))
    starts = list(range(lo, max(lo, int(t_alert * fs) - WIN) + 1, STRIDE))
    masks, usable = lab.run(rec, starts)
    t0, t1 = t_alert - LOOKBACK_SEC, t_alert
    keep = [i for i, u in enumerate(usable) if u]
    out = {"all": beats_in(starts, masks, fs, t0, t1),
           "usable": beats_in([starts[i] for i in keep], [masks[i] for i in keep], fs, t0, t1)}
    dl = DecisionLayer()
    for st, m, u in zip(starts, masks, usable):
        dl.on_beats(st / fs + WIN / fs, beats_from_mask(m, st / fs, fs), usable=bool(u))
    layer = None
    for k in range(6):                                  # 4 of 6 positive: fires at t_alert
        for a in dl.on_shockable(t_alert - (3 - k) * 2.5 if k <= 3 else t_alert, 0.9):
            layer = a.evidence.get("shadow_name")
    out["layer"] = layer or "VF/VT"
    return out


def cudb_vfdb(lab: Labeller, raw: Path, delays, max_records: int) -> list:
    rows = []
    for ds_key in ("cudb", "vfdb"):
        for name in eligible(ds_key, raw, max_records):
            rec = load_record(ds_key, name, raw)
            fs, n = rec.fs, 0
            for s, e, label in split_vt_by_rate(rec, rhythm_intervals(rec)):
                if label not in ("VF", "VT_FAST", "VT_SLOW", "VT"):
                    continue
                for d in delays:
                    t_alert = min(s / fs + d, e / fs)
                    rows.append({"source": ds_key, "label": label, "patient": rec.patient,
                                 "delay": d, "dur": (e - s) / fs,
                                 **episode_beats(lab, rec, s, t_alert)})
                n += 1
            print(f"  {ds_key} {name}: {n} episodes", flush=True)
    return rows


def sddb(lab: Labeller, raw: Path, delays, channel: int) -> list:
    import wfdb

    root = raw / "sddb"
    rows = []
    for name, onset in SDDB_VF_ONSET.items():
        hits = list(root.rglob(f"{name}.hea"))
        if not hits:
            print(f"  sddb {name}: not downloaded, skipped")
            continue
        stem = str(hits[0].with_suffix(""))
        head = wfdb.rdheader(stem)
        fs_in, ch = int(round(head.fs)), min(channel, head.n_sig - 1)
        h, m, s_ = (int(v) for v in onset.split(":"))
        on = h * 3600 + m * 60 + s_
        a = max(0.0, on - AROUND_SEC - 5.0)
        b = min(head.sig_len / fs_in, on + max(delays) + 5.0)
        r = wfdb.rdrecord(stem, sampfrom=int(a * fs_in), sampto=int(b * fs_in), channels=[ch])
        x = np.nan_to_num(np.asarray(r.p_signal[:, 0], np.float64))
        rec = Record("sddb", name, f"sddb:{name}", pp.model_path(x, fs_in), C.TARGET_FS,
                     np.array([], np.int64), np.array([], "<U2"),
                     np.array([], np.int64), np.array([], object))
        fs, on_local = rec.fs, on - a
        for d in delays:
            t_alert = on_local + d
            if t_alert * fs > len(rec.signal):
                continue
            rows.append({"source": "sddb", "label": "VF", "patient": rec.patient,
                         "delay": d, "dur": float("nan"),
                         **episode_beats(lab, rec, int(on_local * fs), t_alert)})
        print(f"  sddb {name}: VF onset {onset}, channel {ch} of {head.n_sig}", flush=True)
    return rows


# ------------------------------------------------------------------- report
def _k(rows, pred) -> str:
    return f"{sum(pred(r) for r in rows)}/{len(rows)}" if rows else "-"


def report(rows: list, delay: float, have_sddb: bool) -> None:
    rows = [r for r in rows if r["delay"] == delay]
    vf_cv = [r for r in rows if r["label"] == "VF" and r["source"] != "sddb"]
    vf_sd = [r for r in rows if r["label"] == "VF" and r["source"] == "sddb"]
    vt_fast = [r for r in rows if r["label"] == "VT_FAST"]
    vt_other = [r for r in rows if r["label"] in ("VT_SLOW", "VT")]
    vf_all = vf_cv + vf_sd

    print(f"\n========== guess read {delay:.0f} s into each episode ==========")
    print(f"episodes: VF {len(vf_cv)} (CUDB/VFDB, {len({r['patient'] for r in vf_cv})} patients)"
          + (f" + {len(vf_sd)} (SDDB)" if have_sddb else "")
          + f", VT 150+ {len(vt_fast)}, VT slow or rate unread {len(vt_other)}")

    same = sum(guess(r["all"], None, None, True) == r["layer"] for r in rows)
    print(f"check: today's rule here matches the decision layer on {same} of {len(rows)} episodes")

    print("\nVT naming. 'VT 150+ named VT' is the gain; 'VF named VT' is the dangerous mistake.")
    hdr = (f"{'source':<8}{'run':>6}{'CV':>6}{'VT 150+ named VT':>18}{'slow VT named VT':>18}"
           f"{'VF named VT':>13}" + (f"{'SDDB VF named VT':>18}" if have_sddb else "")
           + f"{'wrong/named':>13}{'upper 95%':>11}{'patients':>10}")
    print(hdr)
    rules = [("all", None, None)] + [(src, L, cv) for src in SOURCES for L in RUN_LENGTHS for cv in CVS]
    best = None
    for src, L, cv in rules:
        is_vt = lambda r: guess(r[src], L, cv, False) == "VT"  # noqa: E731
        good = sum(is_vt(r) for r in vt_fast)
        bad_rows = [r for r in vf_all if is_vt(r)]
        named, wrong = good + len(bad_rows), len(bad_rows)
        ub = upper95(wrong, named)
        tag = "today" if L is None else str(L)
        print(f"{src:<8}{tag:>6}{('any' if cv is None else f'{cv:.2f}'):>6}"
              f"{_k(vt_fast, is_vt):>18}{_k(vt_other, is_vt):>18}{_k(vf_cv, is_vt):>13}"
              + (f"{_k(vf_sd, is_vt):>18}" if have_sddb else "")
              + f"{f'{wrong}/{named}':>13}{(f'{ub:.0%}' if np.isfinite(ub) else '-'):>11}"
              f"{len({r['patient'] for r in bad_rows}):>10}")
        if L is not None and wrong == 0 and good > 0 and (best is None or good > best[3]):
            best = (src, L, cv, good, ub)

    print("\nVF naming ('no beat at all' -> VF).")
    print(f"{'source':<8}{'VF named VF':>14}" + (f"{'SDDB VF named VF':>18}" if have_sddb else "")
          + f"{'VT 150+ named VF':>18}{'slow VT named VF':>18}{'wrong/named':>13}{'upper 95%':>11}")
    for src in SOURCES:
        is_vf = lambda r: guess(r[src], None, None, True) == "VF"  # noqa: E731
        good = sum(is_vf(r) for r in vf_all)
        wrong = sum(is_vf(r) for r in vt_fast + vt_other)
        ub = upper95(wrong, good + wrong)
        print(f"{src:<8}{_k(vf_cv, is_vf):>14}" + (f"{_k(vf_sd, is_vf):>18}" if have_sddb else "")
              + f"{_k(vt_fast, is_vf):>18}{_k(vt_other, is_vf):>18}"
              f"{f'{wrong}/{good + wrong}':>13}{(f'{ub:.0%}' if np.isfinite(ub) else '-'):>11}")

    print()
    if best is None:
        print("No stricter VT rule named any VT 150+ episode without also naming a VF episode VT.")
    else:
        src, L, cv, good, ub = best
        print(f"Best VT rule with no mistake here: {src} beats, run >= {L}, CV "
              f"{'any' if cv is None else f'<= {cv:.2f}'}: names {good} of {len(vt_fast)} VT 150+ "
              f"episodes; the wrong rate could still be up to {ub:.0%} (95% bound).")
        print("Chosen on this data: a candidate for new patients, not a reason to turn naming on.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sddb", action="store_true", help="add the SDDB VF onsets")
    ap.add_argument("--sddb-channel", type=int, default=0)
    ap.add_argument("--delays", type=float, nargs="+", default=[NAMING_DELAY_SEC],
                    help="seconds into each episode the ShockableNet alert is placed")
    ap.add_argument("--beat-ckpt", type=Path, default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--gate-ckpt", type=Path, default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--max-records", type=int, default=0, help="per dataset (0 = all)")
    a = ap.parse_args(argv)

    for p, what in ((a.beat_ckpt, "beat model"), (a.gate_ckpt, "quality gate")):
        if not p.exists():
            print(f"no {what} at {p}")
            return 1
    if a.sddb and not any((a.raw / "sddb").rglob("*.hea")):
        print(f"no SDDB under {a.raw / 'sddb'}; download it first (see the top of this file)")
        return 1
    lab = Labeller(a.beat_ckpt, a.gate_ckpt, gate=True)
    t0 = time.time()
    rows = cudb_vfdb(lab, a.raw, a.delays, a.max_records)
    if a.sddb:
        rows += sddb(lab, a.raw, a.delays, a.sddb_channel)
    if not rows:
        print("no VF or VT episodes")
        return 1
    for d in a.delays:
        report(rows, d, a.sddb)
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
