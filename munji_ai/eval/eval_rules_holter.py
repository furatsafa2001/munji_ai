"""How accurate the rate rules are on long, beat-annotated recordings.

    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb
    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb icentia11k --max-records 50
    python -m munji_ai.eval.eval_rules_holter --datasets mitdb --allow-test   # final run only

The question
------------
Two different things can go wrong, and this script keeps them apart:

    rule      the threshold or run length is wrong for the patient
    detection the engine found the wrong beats

So the reference is not "this database has no arrhythmia". It is the same
rules applied to the expert-annotated beats: the rate held over the same runs,
the same gaps, the same thresholds. NSRDB subjects are healthy, yet their
rate does fall below 50 at night; that is a correct finding, not a false
alarm. Whatever differs between reference and engine is detection error.

What it reports, per dataset
----------------------------
- Heart rate accuracy, window by window, against ANSI/AAMI EC13: the share of
  windows whose median rate is within 10% or 5 bpm (whichever is greater) of
  the annotated one.
- Window agreement per family at warning level and above.
- Alerts after the decision layer, the figure a caregiver lives with:
  sensitivity (reference alerts the engine also raised within 30 s) and
  false alerts per 24 h (engine alerts with no reference alert within 30 s).

What it does not do
-------------------
- No quality gate: every window counts as usable, so noisy stretches raise
  rate findings the gate would have stopped. False alerts here are an upper
  bound.
- No lead status: lead_off=None, so absence findings follow the (absent) gate.
- Test-split recordings are skipped unless --allow-test. MITDB is a held-out
  benchmark: use it once, at the end.

Reads only. Writes nothing.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config as C
from ..data import splits as S
from ..data.loader import dataset_dir, load_record
from ..data.registry import get
from ..decision.decision_layer import RATE_FAMILIES, DecisionLayer
from ..rules.rate_engine import (SEVERITY, RateReport, analyse, classify_gap,
                                 classify_rate, held_rates)

FS = C.TARGET_FS
# Every WFDB beat code, so the reference never sees a gap where an unusual
# beat (fusion, paced, unclassifiable) was annotated.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
MATCH_SEC = 30.0


def annotated_beats(dataset: str, name: str, root: Path, n_out: int) -> np.ndarray:
    """Every annotated beat, as sample indices at TARGET_FS."""
    import wfdb

    hea = next(dataset_dir(get(dataset), root).rglob(f"{name}.hea"))
    stem = str(hea.with_suffix(""))
    n_in = wfdb.rdheader(stem).sig_len
    for ext in ("atr", "ecg", "qrs"):
        try:
            ann = wfdb.rdann(stem, ext)
        except Exception:                                  # noqa: BLE001
            continue
        sym = np.asarray(ann.symbol)
        keep = np.isin(sym, list(BEAT_SYMBOLS))
        if keep.any():
            return np.round(np.asarray(ann.sample)[keep] * n_out / max(n_in, 1)).astype(np.int64)
    return np.array([], dtype=np.int64)


def reference_report(beats: np.ndarray, start: int, end: int) -> RateReport:
    """The rules applied to the annotated beats of one window."""
    rep = RateReport()
    b = beats[(beats >= start) & (beats < end)] - start
    edges = np.concatenate([[0], b, [end - start - 1]])
    rep.longest_gap_sec = float(np.diff(edges).max() / FS)
    f = classify_gap(rep.longest_gap_sec)
    if f:
        rep.findings.append(f)
    if len(b) >= 2:
        rr = np.diff(b) / FS
        rep.bpm = float(60.0 / np.median(rr))
        slow, _ = held_rates(rr, C.RATE_RUN_BEATS["brady"])
        _, fast = held_rates(rr, C.RATE_RUN_BEATS["tachy"])
        rep.findings.extend(classify_rate(slow, fast))
    return rep


def _worst(rep: RateReport) -> dict:
    out = {}
    for f in rep.findings:
        if f.severity == "info":
            continue
        cur = out.get(f.family)
        if cur is None or SEVERITY.index(f.severity) > SEVERITY.index(cur):
            out[f.family] = f.severity
    return out


def _match(a: list, b: list) -> int:
    """How many alerts in a have an alert of the same family in b within MATCH_SEC."""
    hits = 0
    for t, fam in a:
        if any(fb == fam and abs(t - tb) <= MATCH_SEC for tb, fb in b):
            hits += 1
    return hits


def evaluate_record(sig: np.ndarray, beats: np.ndarray, acc: dict) -> None:
    win = int(C.RULES_WINDOW_SEC * FS)
    step = int(C.RULES_STRIDE_SEC * FS)
    dl_eng, dl_ref = DecisionLayer(), DecisionLayer()
    eng_alerts, ref_alerts = [], []
    for start in range(0, len(sig) - win + 1, step):
        t = (start + win) / FS
        eng = analyse(sig[start:start + win])
        ref = reference_report(beats, start, start + win)

        if np.isfinite(ref.bpm) and np.isfinite(eng.bpm):
            tol = max(0.10 * ref.bpm, 5.0)
            acc["hr_n"] += 1
            acc["hr_ok"] += int(abs(eng.bpm - ref.bpm) <= tol)

        we, wr = _worst(eng), _worst(ref)
        for fam in RATE_FAMILIES:
            e, r = fam in we, fam in wr
            acc[f"win_{fam}"][(r, e)] += 1

        for a in dl_eng.on_rate(t, eng, usable=True, lead_off=None):
            if a.severity != "info":
                eng_alerts.append((t, a.kind))
        for a in dl_ref.on_rate(t, ref, usable=True, lead_off=None):
            if a.severity != "info":
                ref_alerts.append((t, a.kind))

    fam_of = {"bradycardia": "brady", "extreme_bradycardia": "brady",
              "tachycardia": "tachy", "extreme_tachycardia": "tachy",
              "pause": "absence", "asystole": "absence", "fast_narrow": "fast_narrow"}
    eng_f = [(t, fam_of[k]) for t, k in eng_alerts]
    ref_f = [(t, fam_of[k]) for t, k in ref_alerts]
    acc["ref_alerts"] += len(ref_f)
    acc["ref_caught"] += _match(ref_f, eng_f)
    acc["eng_alerts"] += len(eng_f)
    acc["eng_false"] += len(eng_f) - _match(eng_f, ref_f)
    acc["hours"] += len(sig) / FS / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nsrdb"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--allow-test", action="store_true",
                    help="include test-split records (final evaluation only)")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--splits-file", type=Path, default=None)
    a = ap.parse_args()

    use = set(a.splits) | ({"test"} if a.allow_test else set())
    manifest = S.load(a.splits_file)
    for ds in a.datasets:
        names = sorted({n for sp in use for n in S.records_in(manifest, sp, ds)})
        if a.max_records:
            names = names[:a.max_records]
        if not names:
            print(f"\n{ds}: no records in splits {sorted(use)}"
                  + ("" if a.allow_test else " (test split excluded; --allow-test)"))
            continue
        acc = defaultdict(int)
        for fam in RATE_FAMILIES:
            acc[f"win_{fam}"] = defaultdict(int)
        t0 = time.time()
        for i, name in enumerate(names):
            rec = load_record(ds, name, a.raw, with_annotations=False)
            beats = annotated_beats(ds, name, a.raw, len(rec.signal))
            if beats.size < 2:
                print(f"  {ds}/{name}: no beat annotations, skipped")
                continue
            evaluate_record(rec.signal, beats, acc)
            print(f"  {ds} {i + 1}/{len(names)}  {acc['hours']:.1f} h  "
                  f"{time.time() - t0:.0f} s", flush=True)

        h = acc["hours"]
        print(f"\n{ds}: {len(names)} records, {h:.1f} hours")
        if acc["hr_n"]:
            print(f"  heart rate within EC13 tolerance   {acc['hr_ok'] / acc['hr_n']:.3f}"
                  f"  ({acc['hr_n']:,} windows)")
        print(f"  {'family':<12}{'both':>9}{'ref only':>10}{'eng only':>10}{'neither':>11}")
        for fam in RATE_FAMILIES:
            w = acc[f"win_{fam}"]
            print(f"  {fam:<12}{w[(True, True)]:>9,}{w[(True, False)]:>10,}"
                  f"{w[(False, True)]:>10,}{w[(False, False)]:>11,}")
        sens = acc["ref_caught"] / acc["ref_alerts"] if acc["ref_alerts"] else float("nan")
        fpd = acc["eng_false"] * 24.0 / h if h else float("nan")
        print(f"  alerts: reference {acc['ref_alerts']}, engine {acc['eng_alerts']}, "
              f"sensitivity {sens:.2f}, false per 24 h {fpd:.2f}")
    print("\nNo quality gate and no lead status here: false alerts are an upper bound.")


if __name__ == "__main__":
    main()
