"""How accurate the rate rules are on long, beat-annotated recordings.

    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb
    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb icentia11k --max-records 50
    python -m munji_ai.eval.eval_rules_holter --datasets nsrdb --max-records 1 --list-gaps 15
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

Two diagnostics
---------------
--list-gaps N prints the N longest beat-free gaps in the annotations, with
the signal inside each: its peak-to-peak size, the size of the QRS just
before it, and how many beats the engine finds there. A real pause sits on a
live baseline after normal-sized complexes; a detached or failing electrode
shows a collapse in amplitude that nobody annotated.

Only notification findings become alerts (MCOT practice: sustained rate at
or below 30 or at or above 180 for 30 s, pauses, asystole); the graded tiers
are summarised as hours for the doctor's report. Alerts are split by severity, and the time from each alert to the previous
one of the same family is summarised: a rate hovering at a threshold re-opens
an episode every time it crosses, and that shows here as alerts minutes apart.

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
from ..data.loader import _pick_channel, dataset_dir, load_record
from ..data.registry import get
from ..decision.decision_layer import RATE_FAMILIES, DecisionLayer
from ..rules.rate_engine import (SEVERITY, RateReport, analyse, classify_gap,
                                 detect_beats, qrs_amplitudes_mv, rate_findings)

FS = C.TARGET_FS
# Every WFDB beat code, so the reference never sees a gap where an unusual
# beat (fusion, paced, unclassifiable) was annotated.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
MATCH_SEC = 30.0
FROZEN_SEC = 0.5          # a raw value unchanged this long is a dropout, not ECG
FAMILY_OF = {"bradycardia": "brady", "extreme_bradycardia": "brady",
             "sustained_bradycardia": "brady",
             "tachycardia": "tachy", "extreme_tachycardia": "tachy",
             "sustained_tachycardia": "tachy",
             "pause": "absence", "asystole": "absence", "fast_narrow": "fast_narrow"}


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


def missing_mask(dataset: str, name: str, root: Path, n_out: int) -> np.ndarray:
    """True where the raw record has no signal, at TARGET_FS.

    NaN samples, plus any run where the raw value does not change for
    FROZEN_SEC: a live front end always carries some noise, so a frozen value
    is a dropout written as a constant.
    """
    import wfdb

    ds = get(dataset)
    hea = next(dataset_dir(ds, root).rglob(f"{name}.hea"))
    rec = wfdb.rdrecord(str(hea.with_suffix("")))
    raw = np.asarray(rec.p_signal[:, _pick_channel(ds, rec)], dtype=np.float64)
    miss = np.isnan(raw)
    run = max(2, int(FROZEN_SEC * rec.fs))
    same = np.concatenate([[False], np.diff(raw) == 0])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], same.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):     # runs of "equal to previous"
        if b - a + 1 >= run:
            miss[max(a - 1, 0):b] = True
    pos = np.minimum((np.arange(n_out) * len(raw) / max(n_out, 1)).astype(np.int64),
                     len(raw) - 1)
    return miss[pos]


def dropout_gaps(beats: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, list]:
    """Whole beat-to-beat gaps that contain a recording dropout.

    A frozen stretch of a fraction of a second sits inside a quiet stretch of
    several seconds that nobody annotated; skipping only the frozen samples
    leaves the rest to read as asystole on both sides. A gap that contains a
    dropout cannot be scored as a pause or as no pause, so all of it is
    skipped. Returns the mask and the length of each such gap in seconds.
    """
    out = np.zeros(missing.size, dtype=bool)
    lengths = []
    b = np.sort(beats)
    for k in np.flatnonzero(np.diff(b) / FS > C.PAUSE_WARNING_SEC):
        a0, a1 = int(b[k]), int(b[k + 1])
        if missing[a0:a1].any():
            out[a0:a1] = True
            lengths.append((a1 - a0) / FS)
    return out, lengths


def outside_annotation(beats: np.ndarray, n: int) -> np.ndarray:
    """True before the first and after the last annotated beat.

    A reference exists only over the annotated span. A recorder often runs on
    before or after it (set-up, removal) with no QRS on the trace; scored, every
    window there reads as asystole on both sides.
    """
    out = np.zeros(n, dtype=bool)
    if beats.size == 0:
        out[:] = True
        return out
    b = np.sort(beats)
    out[:max(int(b[0]), 0)] = True
    out[min(int(b[-1]) + 1, n):] = True
    return out


def _hms(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s_ = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s_:02d}"


def unannotated_mask(beats: np.ndarray, n: int, longer_than_sec: float) -> np.ndarray:
    """True between consecutive annotated beats further apart than the limit.

    For a corpus whose subjects were screened free of significant arrhythmia
    (NSRDB), a beat-to-beat gap of many seconds is a stretch nobody annotated,
    not a pause: the engine finds full-size complexes inside such gaps. 0 turns
    this off, which is right for corpora that do contain long pauses.
    """
    out = np.zeros(n, dtype=bool)
    if longer_than_sec <= 0 or beats.size < 2:
        return out
    b = np.sort(beats)
    for k in np.flatnonzero(np.diff(b) / FS > longer_than_sec):
        out[int(b[k]):int(b[k + 1])] = True
    return out


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
        findings, held = rate_findings(rr)       # the engine's own rules
        for k, v in held.items():
            setattr(rep, k, v)
        rep.findings.extend(findings)
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


def _episode_spacing(alerts: list, family: str) -> list:
    """Seconds between episode starts of one family.

    An escalation (a more severe alert of the same family within 30 s of the
    previous one) belongs to the same episode and is not counted as a start.
    """
    starts, prev = [], None
    for t, fam, sev in alerts:
        if fam != family:
            continue
        if (prev is not None and t - prev[0] <= 30.0
                and SEVERITY.index(sev) > SEVERITY.index(prev[1])):
            prev = (t, sev)
            continue
        starts.append(t)
        prev = (t, sev)
    return np.diff(starts).tolist()


def gap_report(sig: np.ndarray, beats: np.ndarray, missing: np.ndarray,
               record: str) -> list:
    """Every annotated beat-to-beat gap longer than the pause line."""
    out = []
    b = np.sort(beats)
    gaps = np.diff(b) / FS
    # Trim the T wave of the beat before (about 0.45 s) and the P wave of the
    # beat after (about 0.25 s), so only the stretch between them is measured.
    after, before_next = int(0.45 * FS), int(0.25 * FS)
    for k in np.flatnonzero(gaps > C.PAUSE_WARNING_SEC):
        a0, a1 = int(b[k]), int(b[k + 1])
        if missing[a0:a1].any():
            continue
        seg = sig[a0 + after:max(a1 - before_next, a0 + after + 1)]
        before = b[max(0, k - 9):k + 1]
        qrs = float(np.median(qrs_amplitudes_mv(sig, before, FS))) if len(before) else float("nan")
        found, _ = detect_beats(seg, FS) if seg.size > FS else (np.array([]), 0)
        out.append({"record": record, "t": a0 / FS, "gap": float(gaps[k]),
                    "ptp": float(np.ptp(seg)) if seg.size else float("nan"),
                    "qrs_before": qrs, "engine_beats": int(len(found))})
    return out


def evaluate_record(sig: np.ndarray, beats: np.ndarray, missing: np.ndarray,
                    acc: dict, record: str = "") -> None:
    win = int(C.RULES_WINDOW_SEC * FS)
    step = int(C.RULES_STRIDE_SEC * FS)
    cum = np.concatenate([[0], np.cumsum(missing)])
    dl_eng, dl_ref = DecisionLayer(), DecisionLayer()
    eng_alerts, ref_alerts, eng_alerts_sev = [], [], []
    for start in range(0, len(sig) - win + 1, step):
        t = (start + win) / FS
        complete = cum[start + win] - cum[start] == 0
        if not complete:
            acc["windows_skipped"] += 1
            empty = RateReport()
            dl_eng.on_rate(t, empty, usable=True, lead_off=None, complete=False)
            dl_ref.on_rate(t, empty, usable=True, lead_off=None, complete=False)
            continue
        acc["windows_used"] += 1
        eng = analyse(sig[start:start + win])
        ref = reference_report(beats, start, start + win)

        if np.isfinite(ref.bpm) and np.isfinite(eng.bpm):
            tol = max(0.10 * ref.bpm, 5.0)
            acc["hr_n"] += 1
            acc["hr_ok"] += int(abs(eng.bpm - ref.bpm) <= tol)

        we, wr = _worst(eng), _worst(ref)
        for fam in RATE_FAMILIES:
            acc[f"win_{fam}"][(fam in wr, fam in we)] += 1

        for a in dl_eng.on_rate(t, eng, usable=True, lead_off=None):
            if a.severity != "info":
                eng_alerts.append((t, FAMILY_OF[a.kind]))
                eng_alerts_sev.append((t, FAMILY_OF[a.kind], a.severity))
                acc[f"sev_eng_{FAMILY_OF[a.kind]}_{a.severity}"] += 1
        for a in dl_ref.on_rate(t, ref, usable=True, lead_off=None):
            if a.severity != "info":
                ref_alerts.append((t, FAMILY_OF[a.kind]))
                acc[f"sev_ref_{FAMILY_OF[a.kind]}_{a.severity}"] += 1

    for fam in RATE_FAMILIES:
        ef = [x for x in eng_alerts if x[1] == fam]
        rf = [x for x in ref_alerts if x[1] == fam]
        caught = _match(rf, eng_alerts)
        false = len(ef) - _match(ef, ref_alerts)
        acc[f"ref_{fam}"] += len(rf)
        acc[f"caught_{fam}"] += caught
        acc[f"eng_{fam}"] += len(ef)
        acc[f"false_{fam}"] += false
        acc[f"spacing_{fam}"].extend(_episode_spacing(eng_alerts_sev, fam))
    acc["gaps"].extend(gap_report(sig, beats, missing, record))
    for fam, by_sev in dl_eng.report()["rate_report_windows"].items():
        for sv, n in by_sev.items():
            acc[f"graded_{fam}_{sv}"] += n
    acc["hours_total"] += len(sig) / FS / 3600.0
    acc["hours_missing"] += float(missing.sum()) / FS / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nsrdb"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--allow-test", action="store_true",
                    help="include test-split records (final evaluation only)")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--splits-file", type=Path, default=None)
    ap.add_argument("--unannotated-sec", type=float, default=15.0,
                    help="skip stretches with no annotated beat for longer than this "
                         "(NSRDB subjects are screened free of significant arrhythmia; "
                         "0 = off, for corpora with real long pauses)")
    ap.add_argument("--list-gaps", type=int, default=10,
                    help="print the N longest annotated gaps with their signal")
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
            acc[f"spacing_{fam}"] = []
        acc["gaps"] = []
        acc["dropout_len"] = []
        t0 = time.time()
        for i, name in enumerate(names):
            rec = load_record(ds, name, a.raw, with_annotations=False)
            beats = annotated_beats(ds, name, a.raw, len(rec.signal))
            if beats.size < 2:
                print(f"  {ds}/{name}: no beat annotations, skipped")
                continue
            missing = missing_mask(ds, name, a.raw, len(rec.signal))
            outside = outside_annotation(beats, len(rec.signal))
            acc["hours_outside"] += float((outside & ~missing).sum()) / FS / 3600.0
            span = (f"annotated {_hms(beats.min() / FS)} to {_hms(beats.max() / FS)}"
                    f" of {_hms(len(rec.signal) / FS)}")
            missing = missing | outside
            drop, drop_len = dropout_gaps(beats, missing)
            unann = unannotated_mask(beats, len(rec.signal), a.unannotated_sec)
            acc["dropout_len"].extend(drop_len)
            acc["hours_dropout"] += float((drop & ~missing).sum()) / FS / 3600.0
            acc["hours_unannotated"] += float((unann & ~missing & ~drop).sum()) / FS / 3600.0
            evaluate_record(rec.signal, beats, missing | drop | unann, acc, name)
            print(f"  {ds} {i + 1}/{len(names)}  {name}  {span}  "
                  f"{time.time() - t0:.0f} s", flush=True)

        used_h = acc["windows_used"] * C.RULES_STRIDE_SEC / 3600.0
        print(f"\n{ds}: {len(names)} records, {acc['hours_total']:.1f} h recorded, "
              f"{used_h:.1f} h analysed")
        raw_missing = (acc["hours_missing"] - acc["hours_dropout"] - acc["hours_unannotated"]
                       - acc["hours_outside"])
        dl_ = np.asarray(acc["dropout_len"])
        print(f"  skipped: {acc['hours_outside']:.2f} h outside the annotated span (no reference there)")
        print(f"           {raw_missing:.2f} h missing signal (as the app will)")
        print(f"           {dl_.size} beat gaps containing a dropout ({acc['hours_dropout']:.2f} h)"
              + (f": {int((dl_ <= 6).sum())} of 3-6 s, {int(((dl_ > 6) & (dl_ <= 15)).sum())}"
                 f" of 6-15 s, {int((dl_ > 15).sum())} longer" if dl_.size else ""))
        print(f"           {acc['hours_unannotated']:.2f} h with no annotated beat for over "
              f"{a.unannotated_sec:g} s")
        if acc["hr_n"]:
            print(f"  heart rate within EC13 tolerance   {acc['hr_ok'] / acc['hr_n']:.3f}"
                  f"  ({acc['hr_n']:,} windows)")
        print(f"\n  windows      {'both':>9}{'ref only':>10}{'eng only':>10}{'neither':>11}")
        for fam in RATE_FAMILIES:
            w = acc[f"win_{fam}"]
            print(f"  {fam:<12}{w[(True, True)]:>9,}{w[(True, False)]:>10,}"
                  f"{w[(False, True)]:>10,}{w[(False, False)]:>11,}")
        print(f"\n  alerts       {'reference':>10}{'caught':>8}{'sens':>7}"
              f"{'engine':>8}{'false':>7}{'false/24h':>11}")
        for fam in RATE_FAMILIES:
            r, c = acc[f"ref_{fam}"], acc[f"caught_{fam}"]
            e, f = acc[f"eng_{fam}"], acc[f"false_{fam}"]
            sens = f"{c / r:.2f}" if r else "  -"
            fpd = f"{f * 24.0 / used_h:.2f}" if used_h else "-"
            print(f"  {fam:<12}{r:>10}{c:>8}{sens:>7}{e:>8}{f:>7}{fpd:>11}")

        print(f"\n  by severity  {'ref warn':>9}{'ref crit':>9}{'eng warn':>9}{'eng crit':>9}"
              f"{'eng/24h':>9}   time between episodes (engine)")
        for fam in RATE_FAMILIES:
            rw, rc = acc[f"sev_ref_{fam}_warning"], acc[f"sev_ref_{fam}_critical"]
            ew, ec = acc[f"sev_eng_{fam}_warning"], acc[f"sev_eng_{fam}_critical"]
            per_day = (ew + ec) * 24.0 / used_h if used_h else float("nan")
            sp = np.asarray(acc[f"spacing_{fam}"]) / 60.0
            spacing = (f"median {np.median(sp):.0f} min, {np.mean(sp < 10):.0%} within 10 min"
                       if sp.size else "-")
            print(f"  {fam:<12}{rw:>9}{rc:>9}{ew:>9}{ec:>9}{per_day:>9.1f}   {spacing}")

        print("\n  graded for the doctor's report, never sent (hours of rhythm):")
        for fam in ("brady", "tachy"):
            parts = "   ".join(f"{sv} {acc[f'graded_{fam}_{sv}'] * C.RULES_STRIDE_SEC / 3600:.1f} h"
                               for sv in ("info", "warning", "critical"))
            print(f"    {fam:<8}{parts}")

        g = acc["gaps"]
        if g:
            gap = np.array([x["gap"] for x in g])
            ptp = np.array([x["ptp"] for x in g])
            ratio = ptp / np.array([x["qrs_before"] for x in g])
            beats_in = np.array([x["engine_beats"] for x in g])
            bins = [(C.PAUSE_WARNING_SEC, C.ASYSTOLE_CRITICAL_SEC), (C.ASYSTOLE_CRITICAL_SEC, 15.0),
                    (15.0, 60.0), (60.0, np.inf)]
            counts = "   ".join(f"{lo:g}-{hi:g} s: {int(((gap > lo) & (gap <= hi)).sum())}"
                               if np.isfinite(hi) else f"> {lo:g} s: {int((gap > lo).sum())}"
                               for lo, hi in bins)
            print(f"\n  annotated gaps > {C.PAUSE_WARNING_SEC:g} s: {len(g)}")
            print(f"    by length   {counts}")
            print("    (a healthy heart pauses for seconds; minutes that end in normal"
                  " rhythm are signal loss)")
            print(f"    signal inside smaller than {C.QRS_MIN_AMP_MV} mV peak to peak: "
                  f"{np.mean(ptp < C.QRS_MIN_AMP_MV):.0%}")
            print(f"    signal inside under 20% of the QRS before it:          "
                  f"{np.nanmean(ratio < 0.2):.0%}")
            print(f"    engine finds beats inside (the annotator did not):     "
                  f"{np.mean(beats_in > 0):.0%}")
            if a.list_gaps:
                print(f"\n    {a.list_gaps} longest:")
                print(f"    {'record':<10}{'time':>10}{'gap s':>8}{'ptp mV':>9}"
                      f"{'QRS before':>12}{'engine beats':>14}")
                for x in sorted(g, key=lambda x: -x["gap"])[:a.list_gaps]:
                    hh, rem = divmod(int(x["t"]), 3600)
                    mm, ss = divmod(rem, 60)
                    print(f"    {x['record']:<10}{f'{hh:02d}:{mm:02d}:{ss:02d}':>10}"
                          f"{x['gap']:>8.1f}{x['ptp']:>9.2f}{x['qrs_before']:>12.2f}"
                          f"{x['engine_beats']:>14}")
    print("\nNo quality gate and no lead status here: false alerts are an upper bound.")


if __name__ == "__main__":
    main()
