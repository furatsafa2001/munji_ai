"""Train and evaluate the SQI baseline on the real quality cache.

    python -m munji_ai.eval.eval_quality
    python -m munji_ai.eval.eval_quality --cache data/cache --save models/sqi.pkl

Reports three things, in order of importance:

  1. the target check       specificity >= 0.95, sensitivity >= 0.90
  2. rhythm rejection bias  does the gate discard arrhythmias as noise
  3. per-patient spread     does it collapse on a subset of people

The third matters as much as the first. A gate at 0.95 pooled that fails
completely on 10% of patients is not a gate at 0.95.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, QUALITY_CLASSES
from ..data.windows import load_cache
from .metrics import format_bias, report, rhythm_rejection_bias

# Datasets whose underlying rhythm is normal. Everything else is arrhythmic and
# must not be rejected at a higher rate — see rhythm_rejection_bias.
NORMAL_SOURCES = ("nsrdb",)


def composition(d: dict, name: str) -> None:
    ds, y = d["dataset"], d["y"]
    print(f"\n{name}: {len(y):,} windows, {len(set(d['patient'].tolist()))} patients")
    for k in sorted(set(ds.tolist())):
        m = ds == k
        dist = {c: int((y[m] == c).sum()) for c in QUALITY_CLASSES}
        print(f"  {k:<12} n={int(m.sum()):>6,}  {dist}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--split", default="test", choices=("val", "test"),
                    help="evaluate on val while iterating; test once, at the end")
    a = ap.parse_args(argv)

    from ..models.sqi_baseline import SQIBaseline

    tr = load_cache("quality", "train", a.cache)
    ev = load_cache("quality", a.split, a.cache)

    composition(tr, "train")
    composition(ev, a.split)

    overlap = set(tr["patient"].tolist()) & set(ev["patient"].tolist())
    if overlap:
        raise AssertionError(f"patient leak between train and {a.split}: {overlap}")
    print(f"\nno patient overlap between train and {a.split}")

    print(f"\ntraining SQI baseline on {len(tr['y']):,} windows")
    model = SQIBaseline().fit(tr["X"], tr["y"])

    print(f"predicting on {len(ev['y']):,} windows")
    pred = model.predict(ev["X"])

    print()
    print(report(ev["y"], pred, list(QUALITY_CLASSES),
                 patients=ev["patient"], title=f"SQI baseline — {a.split}"))

    print()
    print(format_bias(rhythm_rejection_bias(pred, ev["dataset"], NORMAL_SOURCES)))

    if a.save:
        print(f"\nsaved -> {model.save(a.save)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())