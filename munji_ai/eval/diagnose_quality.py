"""Test whether contaminated carriers explain the quality gate's failure.

    python -m munji_ai.eval.diagnose_quality --cache data/cache

The gate's labels come from injected SNR: clean signal plus a known amount of
noise gives an exact label. That reasoning holds only if the carrier really is
clean. These are 1980s Holter recordings, so many carriers already contain
artifact before anything is added — meaning a dirty carrier with light
injection gets labelled 'good' while a clean carrier with moderate injection
gets labelled 'qrs_only'. Two similar-looking windows, opposite labels.

This script measures whether that is actually what is happening, rather than
assuming it. Three tests:

  1. Do carriers vary in quality at all, and by how much?
  2. Are the windows the model gets wrong the ones with dirty carriers?
  3. Would filtering dirty carriers separate the classes better?

If test 2 shows no relationship, the diagnosis is wrong and the fix would be
wasted work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, QUALITY_CLASSES
from ..data.windows import load_cache
from ..features.sqi import features, FEATURE_NAMES

QSQI = FEATURE_NAMES.index("qSQI")
KSQI = FEATURE_NAMES.index("kSQI")


def bar(frac: float, width: int = 28) -> str:
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--sample", type=int, default=4000,
                    help="windows to analyse; feature extraction is the slow part")
    a = ap.parse_args(argv)

    d = load_cache("quality", "train", a.cache)
    X, y, ds = d["X"], d["y"].astype(str), d["dataset"].astype(str)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))[: min(a.sample, len(X))]
    X, y, ds = X[idx], y[idx], ds[idx]
    print(f"analysing {len(X):,} windows\n")

    print("extracting features")
    F = np.zeros((len(X), len(FEATURE_NAMES)))
    for i, w in enumerate(X):
        F[i] = features(np.asarray(w, dtype=np.float64))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{len(X):,}")
    q = np.nan_to_num(F[:, QSQI])

    # ---------------------------------------------------------------- test 1
    # If every carrier were clean, qSQI within the 'good' class would be tight.
    # A wide spread means the carriers themselves differ in quality.
    print("\n" + "=" * 64)
    print("TEST 1  carrier quality, measured on windows labelled 'good'")
    print("=" * 64)
    g = q[y == "good"]
    print(f"  n={len(g):,}   median {np.median(g):.3f}   "
          f"p10 {np.percentile(g, 10):.3f}   p90 {np.percentile(g, 90):.3f}")
    print(f"  spread p90-p10: {np.percentile(g, 90) - np.percentile(g, 10):.3f}")
    print("\n  by dataset:")
    for k in sorted(set(ds.tolist())):
        m = (ds == k) & (y == "good")
        if m.sum() < 20:
            continue
        print(f"    {k:<12} n={int(m.sum()):>5}  median qSQI {np.median(q[m]):.3f}  "
              f"{bar(float(np.median(q[m])))}")

    # Spread, not an absolute threshold. What matters is whether windows
    # sharing the label 'good' differ from each other — an absolute cut-off
    # would need calibrating against data we do not have yet.
    spread = float(np.percentile(g, 90) - np.percentile(g, 10))
    below = float((g < np.median(g) - 0.15).mean())
    print(f"\n  'good' windows well below the class median: {below:.1%}")
    if spread > 0.15:
        print(f"  -> carriers are NOT uniform. Windows sharing the label 'good' "
              f"differ by {spread:.2f} in qSQI,")
        print("     which is the inconsistency the model has to learn through.")
    else:
        print("  -> carriers look uniform; contamination is unlikely to be "
              "the main issue")

    # ---------------------------------------------------------------- test 2
    # The decisive one. Train on the cache as it stands, then check whether the
    # errors concentrate in low-qSQI windows.
    print("\n" + "=" * 64)
    print("TEST 2  do errors concentrate in dirty-carrier windows?")
    print("=" * 64)
    try:
        from ..models.sqi_baseline import SQIBaseline
    except ImportError:
        print("  scikit-learn not installed — skipping")
        return 1

    n_tr = int(len(X) * 0.7)
    clf = SQIBaseline(n_estimators=150)
    clf.model.fit(F[:n_tr], y[:n_tr], sample_weight=clf._weights(y[:n_tr]))
    pred = clf.predict_from_features(F[n_tr:])
    yt, qt = y[n_tr:], q[n_tr:]
    wrong = pred != yt

    print(f"  overall accuracy: {1 - wrong.mean():.3f}\n")
    edges = [0.0, 0.4, 0.6, 0.8, 0.95, 1.01]
    print(f"  {'carrier qSQI':<16}{'n':>7}{'error rate':>13}")
    rates = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (qt >= lo) & (qt < hi)
        if m.sum() < 15:
            continue
        r = float(wrong[m].mean())
        rates.append((lo, hi, r, int(m.sum())))
        print(f"  {lo:.2f}-{hi:.2f}      {int(m.sum()):>7}{r:>12.3f}  {bar(r)}")

    if len(rates) >= 2:
        worst, best = max(rates, key=lambda t: t[2]), min(rates, key=lambda t: t[2])
        ratio = worst[2] / max(best[2], 1e-6)
        print(f"\n  dirtiest band errs {ratio:.1f}x more often than the cleanest")
        confirmed = ratio > 2.0
        print("  -> DIAGNOSIS CONFIRMED: errors track carrier quality"
              if confirmed else
              "  -> NOT CONFIRMED: errors are spread evenly, look elsewhere")
    else:
        confirmed = False
        print("  not enough spread to judge")

    # ---------------------------------------------------------------- test 3
    print("\n" + "=" * 64)
    print("TEST 3  would dropping dirty carriers help, and at what cost?")
    print("=" * 64)
    print(f"  {'threshold':<12}{'kept':>9}{'accuracy':>11}{'vs baseline':>14}")
    base = 1 - wrong.mean()
    for thr in (0.0, 0.4, 0.6, 0.7, 0.8):
        keep = q >= thr
        ntr = int(keep[:n_tr].sum())
        if ntr < 200 or keep[n_tr:].sum() < 100:
            continue
        c = SQIBaseline(n_estimators=150)
        ytr = y[:n_tr][keep[:n_tr]]
        c.model.fit(F[:n_tr][keep[:n_tr]], ytr, sample_weight=c._weights(ytr))
        p = c.predict_from_features(F[n_tr:][keep[n_tr:]])
        acc = float((p == y[n_tr:][keep[n_tr:]]).mean())
        delta = acc - base
        print(f"  qSQI >= {thr:.1f}  {keep.mean():>8.1%}{acc:>11.3f}"
              f"{delta:>+13.3f}")

    print("\n" + "=" * 64)
    print("READING THIS")
    print("=" * 64)
    print("""
Test 2 is the one that matters. If the dirtiest band errs several times more
often than the cleanest, carrier contamination is real and filtering is the
fix. If the error rate is flat across bands, the labels are not the problem
and the three-class boundary itself needs rethinking.

Test 3 shows the trade. Accuracy that climbs as the threshold rises means
filtering works; the 'kept' column is what it costs in data.
""".strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())