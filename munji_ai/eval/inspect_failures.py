"""Inspect the patients a model fails on.

    python -m munji_ai.eval.inspect_failures --stage shockable
    python -m munji_ai.eval.inspect_failures --stage shockable --patient cudb:cu21

Reads the out-of-fold predictions written by cross_validate — every patient
predicted once by a model that never saw them — and reports, per failing
patient:

  * which direction it errs: missing danger, or alarming on clean signal
  * how confident it is while wrong, which separates a hard case from a
    confused one
  * whether the signal itself looks degenerate: flat, clipped, or without
    structure
  * how its statistics differ from everyone else's

Why this comes before any architecture change
---------------------------------------------
One patient scoring far below the rest has two possible causes, and they call
for opposite responses. If the recording is genuinely degraded or mislabelled,
the model is fine and the record should be documented or excluded. If the
signal is ordinary and the model still fails, something systematic is wrong
with how a whole class of signal is handled.

Tuning a network before answering that risks spending days fitting an
architecture around a data fault.

Reads only. Writes nothing, trains nothing, and does not touch the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR
from ..data.windows import load_cache

POSITIVE = {"shockable": "shockable", "rhythm": "AF"}


def macro_f1_of(y_true, y_pred, classes) -> float:
    """Macro F1 over the classes this patient actually has.

    Averaging over absent classes is what made the first version of this
    tool misleading. A patient with no shockable windows got F1 0.000 for
    that class, so a flawless record scored 0.500 and was flagged as a
    failure. Worse, cu21 — 53 shockable windows and one negative, sensitivity
    0.962 — came out at 0.486 because the single negative was misread and
    specificity collapsed to zero.

    Scoring only present classes removes both artefacts. Patients whose
    label mix makes F1 meaningless are reported by error rate instead.
    """
    present = [c for c in classes if (y_true == c).any()]
    if not present:
        return float("nan")
    out = []
    for c in present:
        t, p = y_true == c, y_pred == c
        tp = int((t & p).sum())
        prec = tp / max(int(p.sum()), 1)
        rec = tp / max(int(t.sum()), 1)
        out.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(out))


def error_rate(y_true, y_pred) -> float:
    """Share of windows got wrong. Defined for every label mix."""
    return float((y_true != y_pred).mean())


def signal_stats(X: np.ndarray) -> dict:
    """Cheap descriptors that catch a degenerate recording.

    Deliberately not a quality model — these are the checks you would make by
    eye, written down. A flat trace, a clipped one, and one with no beat
    structure all fail in ways a classifier reports as a confident wrong
    answer rather than as bad input.
    """
    X = np.asarray(X, dtype=np.float64)
    per_window_std = X.std(axis=1)
    rng = X.max(axis=1) - X.min(axis=1)

    # a clipped trace spends an unusual share of its samples at the extremes
    hi = np.quantile(X, 0.999)
    lo = np.quantile(X, 0.001)
    at_rail = float(((X >= hi) | (X <= lo)).mean())

    # a trace with beats has sharp excursions; a flat or noise-only one does not
    d = np.abs(np.diff(X, axis=1))
    sharpness = float(np.median(d.max(axis=1) / np.maximum(d.mean(axis=1), 1e-9)))

    return {
        "amplitude": float(np.median(rng)),
        "std": float(np.median(per_window_std)),
        "flat_windows": float((per_window_std < 0.05).mean()),
        "at_rail": at_rail,
        "sharpness": sharpness,
    }


def describe(name, mask, y, pred, score, X, classes, positive, ref, out=print):
    yt, yp, sc = y[mask], pred[mask], score[mask]
    f1 = macro_f1_of(yt, yp, classes)
    err = error_rate(yt, yp)

    t, p = yt == positive, yp == positive
    tp = int((t & p).sum()); fn = int((t & ~p).sum())
    fp = int((~t & p).sum()); tn = int((~t & ~p).sum())

    out(f"\n{'=' * 64}")
    f1_txt = "n/a" if np.isnan(f1) else f"{f1:.3f}"
    out(f"{name}   error {err:.3f}   F1 {f1_txt}   "
        f"({int(mask.sum())} windows)")
    out(f"{'=' * 64}")

    dist = {c: int((yt == c).sum()) for c in classes}
    out(f"  true labels        {dist}")
    absent = [c for c in classes if dist[c] == 0]
    if absent:
        out(f"  note               no {', '.join(absent)} windows here — F1 "
            f"covers the present classes only")

    out(f"\n  direction of error")
    if fn + fp == 0:
        out("    no errors")
    elif fn > fp * 2:
        out(f"    MISSES danger — {fn} missed, {fp} false alarms")
        out("    The dangerous direction. A miss reaches no one.")
    elif fp > fn * 2:
        out(f"    OVER-ALARMS — {fp} false alarms, {fn} missed")
        out("    Costly but not lethal; confirmation rules absorb some of it.")
    else:
        out(f"    both ways — {fn} missed, {fp} false alarms")

    out(f"    sensitivity {tp / max(tp + fn, 1):.3f}   "
        f"specificity {tn / max(tn + fp, 1):.3f}")

    wrong = yt != yp
    if wrong.any():
        conf = np.abs(sc[wrong] - 0.5).mean() * 2
        out(f"\n  confidence while wrong   {conf:.3f}")
        if conf > 0.6:
            out("    Confidently wrong. Not a borderline case — the model has")
            out("    learned something that does not hold for this recording.")
        elif conf < 0.3:
            out("    Uncertain while wrong. A borderline case; a threshold")
            out("    change would move these windows.")

    st = signal_stats(X[mask])
    out(f"\n  signal")
    for k, label, fmt in (("amplitude", "median amplitude", "{:.3f}"),
                          ("std", "median std", "{:.3f}"),
                          ("flat_windows", "flat windows", "{:.1%}"),
                          ("at_rail", "samples at rail", "{:.2%}"),
                          ("sharpness", "peak sharpness", "{:.1f}")):
        v, r = st[k], ref[k]
        ratio = v / r if r else float("inf")
        flag = ""
        if k == "flat_windows" and v > 0.10:
            flag = "   <-- flat stretches"
        elif k == "at_rail" and v > 0.02:
            flag = "   <-- possible clipping"
        elif k == "sharpness" and ratio < 0.6:
            flag = "   <-- weak beat structure"
        elif k in ("amplitude", "std") and (ratio < 0.5 or ratio > 2.0):
            flag = f"   <-- {ratio:.1f}x the cohort"
        out(f"    {label:<18} " + fmt.format(v) +
            f"   (cohort {fmt.format(r)}){flag}")

    verdict = []
    if st["flat_windows"] > 0.10:
        verdict.append("flat stretches — likely lead-off or disconnection")
    if st["at_rail"] > 0.02:
        verdict.append("clipping — the front end saturated")
    if st["sharpness"] < ref["sharpness"] * 0.6:
        verdict.append("weak beat structure — the trace may carry no usable QRS")

    out(f"\n  reading")
    if verdict:
        for v in verdict:
            out(f"    {v}")
        out("    The recording looks degraded. Document or exclude it; this is")
        out("    not evidence against the model.")
    else:
        out("    The signal looks ordinary by these measures. The failure is")
        out("    not explained by degraded input, so it points at the model or")
        out("    at the labels for this record. Worth plotting by hand.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(POSITIVE), default="shockable")
    ap.add_argument("--oof", type=Path, default=None,
                    help="defaults to checkpoints/{stage}_oof.npz")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--patient", default=None, help="inspect one patient")
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="error rate above this counts as failing")
    a = ap.parse_args(argv)

    oof_path = a.oof or Path("checkpoints") / f"{a.stage}_oof.npz"
    if not oof_path.exists():
        print(f"no out-of-fold predictions at {oof_path}\n"
              f"run:  python -m munji_ai.train.cross_validate "
              f"--stage {a.stage}")
        return 1

    d = np.load(oof_path, allow_pickle=False)
    y = np.asarray(d["y"], dtype=str)
    pred = np.asarray(d["pred"], dtype=str)
    score = np.asarray(d["score"], dtype=float)
    patients = np.asarray(d["patient"], dtype=str)

    # Signals come from the caches the folds were built from.
    parts = [load_cache(a.stage, s, a.cache) for s in ("train", "val")]
    X = np.concatenate([p["X"] for p in parts])
    cache_pat = np.concatenate([np.asarray(p["patient"], dtype=str)
                                for p in parts])
    if len(X) != len(y) or not np.array_equal(cache_pat, patients):
        print("the cache no longer matches the saved predictions — rebuild "
              "the cache or re-run cross_validate")
        return 1

    classes = sorted(set(y.tolist()))
    positive = POSITIVE[a.stage]

    scores, errs = {}, {}
    for q in sorted(set(patients.tolist())):
        m = patients == q
        scores[q] = macro_f1_of(y[m], pred[m], classes)
        errs[q] = error_rate(y[m], pred[m])

    ev = np.array(list(errs.values()))
    print(f"stage       {a.stage}")
    print(f"source      {oof_path}")
    print(f"patients    {len(scores)}")
    print(f"error rate  mean {ev.mean():.3f}   median {np.median(ev):.3f}   "
          f"std {ev.std():.3f}")
    print("            ranked by error rate, which is comparable across "
          "patients\n            whatever mix of labels they carry")

    # Cohort reference excludes the failing patients, so a bad record is
    # compared against the normal ones rather than against an average it has
    # itself dragged down.
    ok_mask = np.isin(patients, [q for q, e in errs.items()
                                 if e <= a.threshold])
    ref = signal_stats(X[ok_mask]) if ok_mask.any() else signal_stats(X)

    if a.patient:
        if a.patient not in errs:
            print(f"\nno patient {a.patient!r}. Available, worst first:")
            for q, e in sorted(errs.items(), key=lambda kv: -kv[1])[:10]:
                print(f"  {q:<24} error {e:.3f}")
            return 1
        targets = [a.patient]
    else:
        targets = [q for q, e in sorted(errs.items(), key=lambda kv: -kv[1])
                   if e > a.threshold]
        if not targets:
            print(f"\nno patient above error rate {a.threshold:.2f}. "
                  f"Worst three:")
            for q, e in sorted(errs.items(), key=lambda kv: -kv[1])[:3]:
                print(f"  {q:<24} error {e:.3f}")
            return 0
        print(f"\n{len(targets)} patient(s) above error rate "
              f"{a.threshold:.2f}")

    for q in targets:
        describe(q, patients == q, y, pred, score, X, classes, positive, ref)

    print(f"\n{'=' * 64}")
    print("  A degraded recording and a model failure need opposite responses.")
    print("  Settle which one this is before changing the architecture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
