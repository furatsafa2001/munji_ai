"""Inspect the patients the quality gate fails on.

    python -m munji_ai.eval.inspect_patient --cache data/cache
    python -m munji_ai.eval.inspect_patient --cache data/cache --patient cudb:cu21

One patient scored 0.367 across two separate runs while the rest sat above
0.76. Either that recording is genuinely different, or something systematic is
wrong with how a whole class of signal is handled — and the two need very
different responses. Building a network on top of an unanswered version of
this question risks spending days tuning an architecture around a data fault.

Reports, per failing patient:

  * how its features differ from everyone else's
  * which direction it errs — over-rejecting or over-accepting
  * whether the signal itself looks degenerate (flat, clipped, no structure)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, QUALITY_CLASSES
from ..data.windows import load_cache
from ..features.sqi import FEATURE_NAMES, feature_matrix
from .metrics import macro_f1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--patient", default=None, help="inspect one specific patient")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="macro F1 below this counts as failing")
    a = ap.parse_args(argv)

    from ..models.sqi_baseline import SQIBaseline

    tr = load_cache("quality", "train", a.cache)
    ev = load_cache("quality", a.split, a.cache)
    labels = list(QUALITY_CLASSES)

    print(f"training baseline on {len(tr['y']):,} windows")
    Ftr = feature_matrix(tr["X"])
    clf = SQIBaseline()
    clf.model.fit(Ftr, tr["y"].astype(str),
                  sample_weight=clf._weights(tr["y"].astype(str)))

    print(f"scoring {len(ev['y']):,} windows\n")
    Fev = feature_matrix(ev["X"])
    pred = clf.predict_from_features(Fev)
    yt = ev["y"].astype(str)
    pats = ev["patient"].astype(str)

    scores = {}
    for p in sorted(set(pats.tolist())):
        m = pats == p
        if m.sum() >= 20:
            scores[p] = macro_f1(yt[m], pred[m], labels)

    failing = ([a.patient] if a.patient
               else [p for p, s in scores.items() if s < a.threshold])
    vals = np.array(list(scores.values()))

    print("=" * 66)
    print(f"{len(scores)} patients scored | median {np.median(vals):.3f} "
          f"| {len(failing)} below {a.threshold}")
    print("=" * 66)

    if not failing:
        print("\nno failing patients — nothing to inspect")
        return 0

    for p in failing:
        m = pats == p
        if not m.any():
            print(f"\n{p}: not in this split")
            continue

        yp, yy, F, X = pred[m], yt[m], Fev[m], ev["X"][m]
        print(f"\n{'-' * 66}\n{p}   macro F1 {scores.get(p, float('nan')):.3f}   "
              f"n={int(m.sum())}\n{'-' * 66}")

        # Which direction does it fail? Over-rejecting hides real signal;
        # over-accepting lets noise through. They call for opposite fixes.
        print("  confusion")
        for t in labels:
            row = [int(((yy == t) & (yp == q)).sum()) for q in labels]
            print(f"    true {t:<10} -> " +
                  "  ".join(f"{q}={n}" for q, n in zip(labels, row)))

        fr = int(((yy == "usable") & (yp == "unusable")).sum())
        fa = int(((yy == "unusable") & (yp == "usable")).sum())
        print(f"\n  wrongly rejected {fr}   wrongly accepted {fa}   "
              f"-> {'over-rejecting' if fr > fa else 'over-accepting'}")

        # Where this patient sits relative to everyone else, in standard
        # deviations. A feature far out on its own explains a lot.
        others = Fev[~m]
        print(f"\n  {'feature':>10}  {'this':>9}  {'others':>9}  {'z':>7}")
        for i, name in enumerate(FEATURE_NAMES):
            mu, sd = others[:, i].mean(), others[:, i].std() + 1e-9
            here = F[:, i].mean()
            z = (here - mu) / sd
            flag = "  <<<" if abs(z) > 2 else ""
            print(f"  {name:>10}  {here:>9.3f}  {mu:>9.3f}  {z:>+7.2f}{flag}")

        # Degenerate signal checks: these are data faults, not model faults.
        Xf = np.asarray(X, dtype=np.float64)
        rng_amp = float(np.mean(np.ptp(Xf, axis=1)))
        flat = float(np.mean(np.std(Xf, axis=1) < 0.05))
        sat = float(np.mean(np.mean(np.abs(Xf) > 0.99 * np.max(np.abs(Xf)),
                                    axis=1) > 0.05))
        print(f"\n  signal: mean p2p {rng_amp:.2f} | flat windows {flat:.1%} "
              f"| clipped {sat:.1%}")

        verdict = ("recording looks degenerate — a data fault, not a model fault"
                   if flat > 0.2 or rng_amp < 0.5 else
                   "signal looks normal — the failure is the model's, not the data's")
        print(f"  -> {verdict}")

    print(f"\n{'=' * 66}")
    print("""
Two failure shapes, two responses.

Degenerate signal, or features several sigma from everyone else: the recording
is unlike the training distribution and no architecture fixes that. Exclude it
or accept it as a known limitation.

Normal signal with ordinary features: the model genuinely cannot handle this
case, and a network with more capacity may. That is a reason to build one.

The direction matters too. Over-rejecting is the dangerous side — a gate that
discards this patient's signal blinds every downstream model for them.
""".strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())