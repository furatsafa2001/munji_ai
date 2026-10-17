"""Measure the detector on the windows a quality gate would actually pass.

    python -m munji_ai.eval.gated_eval
    python -m munji_ai.eval.gated_eval --reject 0.20

Why the ungated number is the wrong one
---------------------------------------
Sinus specificity is 0.952 against an AHA target of 0.99, measured over every
window in the cache. But the device does not work that way. The quality gate
runs first, and the shockable detector only ever sees what the gate passes. A
window too noisy to judge never reaches it, so a false alarm on that window is
a number we are charging to the wrong model.

The false-alarm literature for wearables says the same thing from the other
direction: a two-phase framework using signal quality and activity type is the
standard way to cut false arrhythmia alarms in continuous monitoring. The gate
is not an excuse, it is the architecture.

Model 1 has not been trained here, so this uses the four classic SQIs in
features/sqi.py as a stand-in gate. They are deterministic, need no
checkpoint, and are the published baseline the CNN gate was built to beat — so
a gate built on them is, if anything, weaker than the real one. Any
improvement measured here is a floor, not a ceiling.

    qSQI    agreement between two independent R-peak detectors
    pSQI    share of spectral power in the QRS band
    kSQI    kurtosis, which impulsive artifact raises sharply
    basSQI  share of power above 1 Hz, so baseline wander lowers it

What this cannot tell you
-------------------------
Gating raises specificity by removing windows, so it always looks like an
improvement. The question is whether the windows removed are the ones a device
would genuinely refuse to judge, or merely the ones the model got wrong. Two
guards against reading it the easy way:

  the rejected rate is printed every time, next to every other number

  sensitivity on shockable windows is printed for the rejected group too. If
  the gate is throwing away real episodes, that shows up there, and a gate
  that improves specificity by discarding fibrillation is worse than no gate

Reads only. Writes nothing, trains nothing, does not touch the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, RAW_DIR, window_samples
from ..data.loader import load_record
from ..data.windows import load_cache, rhythm_intervals
from ..features import sqi

POSITIVE = "shockable"


def scores(y_true, y_pred, positive=POSITIVE) -> dict:
    t, p = y_true == positive, y_pred == positive
    tp = int((t & p).sum()); fn = int((t & ~p).sum())
    fp = int((~t & p).sum()); tn = int((~t & ~p).sum())
    return {"sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "n_pos": tp + fn, "n_neg": tn + fp, "fp": fp, "fn": fn}


def rhythm_at(record_key, dataset, offset, n, cache, root) -> str:
    key = (dataset, record_key)
    if key not in cache:
        try:
            cache[key] = rhythm_intervals(load_record(dataset, record_key, root))
        except Exception:
            cache[key] = []
    best, best_cover = "UNLABELLED", 0
    for s, e, lab in cache[key]:
        cover = max(0, min(e, offset + n) - max(s, offset))
        if cover > best_cover:
            best, best_cover = lab, cover
    return best if best_cover > 0 else "UNLABELLED"


def quality_score(X: np.ndarray) -> np.ndarray:
    """One usability score per window from the four classic SQIs.

    Deliberately a plain mean of normalised indices rather than a fitted
    combination. A fitted gate would be tuned on the same windows it is
    judging, which would make the result circular — the point here is to find
    out whether low-quality windows carry the false alarms, not to build the
    best possible gate.
    """
    out = np.zeros(len(X))
    for i, x in enumerate(X):
        x = np.asarray(x, dtype=np.float64)
        try:
            q = sqi.q_sqi(x)                       # 0..1, higher is better
            p = sqi.p_sqi(x)                       # 0..1
            b = sqi.bas_sqi(x)                     # 0..1
            k = sqi.k_sqi(x)
            # kurtosis has no natural ceiling; a clean ECG sits around 5, and
            # impulsive artifact pushes it far higher. Map it to 0..1 with
            # 5 as the ideal and penalise distance in either direction.
            k_n = float(np.clip(1.0 - abs(k - 5.0) / 20.0, 0.0, 1.0))
            out[i] = float(np.mean([q, p, b, k_n]))
        except Exception:
            out[i] = 0.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path,
                    default=Path("checkpoints") / "shockable_oof.npz")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--reject", type=float, default=None,
                    help="share of lowest-quality windows to gate out")
    a = ap.parse_args(argv)

    if not a.oof.exists():
        print(f"no out-of-fold predictions at {a.oof}\n"
              f"run:  python -m munji_ai.train.cross_validate --stage shockable")
        return 1

    d = np.load(a.oof, allow_pickle=False)
    y = np.asarray(d["y"], dtype=str)
    pred = np.asarray(d["pred"], dtype=str)
    records = np.asarray(d["record"], dtype=str)
    offsets = np.asarray(d["offset"], dtype=np.int64)

    parts = [load_cache("shockable", s, a.cache) for s in ("train", "val")]
    X = np.concatenate([p["X"] for p in parts])
    datasets = np.concatenate([np.asarray(p["dataset"], dtype=str)
                               for p in parts])
    if len(X) != len(y):
        print("cache no longer matches the saved predictions — re-run "
              "cross_validate")
        return 1

    print(f"source      {a.oof}")
    print(f"windows     {len(y):,}")
    print("computing signal quality indices — this takes a minute\n")
    q = quality_score(X)

    n = window_samples("shockable")
    iv: dict = {}
    labels = np.array([rhythm_at(r, ds, o, n, iv, a.raw)
                       for r, ds, o in zip(records, datasets, offsets)])
    sinus = np.isin(labels, ("N", "NSR"))

    base = scores(y, pred)
    base_sinus = scores(y[sinus], pred[sinus]) if sinus.any() else None
    print("ungated — every window judged (what we have been reporting)")
    print(f"  sensitivity {base['sensitivity']:.3f}   "
          f"specificity {base['specificity']:.3f}")
    if base_sinus:
        print(f"  sinus specificity {base_sinus['specificity']:.3f}   "
              f"({base_sinus['n_neg']:,} windows)")

    print("\nquality distribution")
    for name, mask in (("correct", y == pred), ("wrong", y != pred)):
        if mask.any():
            print(f"  {name:<10} median {np.median(q[mask]):.3f}   "
                  f"p10 {np.quantile(q[mask], 0.10):.3f}   ({int(mask.sum()):,})")
    fp_mask = (y != POSITIVE) & (pred == POSITIVE)
    if fp_mask.any() and (y != POSITIVE).any():
        tn_mask = (y != POSITIVE) & (pred != POSITIVE)
        print(f"  false alarms median {np.median(q[fp_mask]):.3f}   "
              f"vs correct rejections {np.median(q[tn_mask]):.3f}")
        if np.median(q[fp_mask]) >= np.median(q[tn_mask]) - 0.01:
            print("\n  The false alarms are NOT on worse signal. A quality")
            print("  gate will not remove them, and the ungated number is")
            print("  already the honest one. Stop here.")
            return 0

    rejects = [a.reject] if a.reject is not None else \
        [0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"\n{'gated out':>10}{'sens':>8}{'spec':>8}{'sinus sp':>10}"
          f"{'kept':>8}{'episodes lost':>15}")
    for r in rejects:
        thr = np.quantile(q, r)
        keep = q > thr
        s = scores(y[keep], pred[keep])
        ss = scores(y[keep & sinus], pred[keep & sinus]) if (keep & sinus).any() \
            else {"specificity": float("nan")}
        lost = int(((~keep) & (y == POSITIVE)).sum())
        print(f"{r:>10.0%}{s['sensitivity']:>8.3f}{s['specificity']:>8.3f}"
              f"{ss['specificity']:>10.3f}{keep.mean():>8.1%}"
              f"{lost:>15,}")

    print("\n" + "=" * 62)
    print("  'episodes lost' is the count of genuinely shockable windows the")
    print("  gate discarded. A gate that buys specificity with those is not a")
    print("  gate, it is a blind spot. Quote it beside any gated figure.")
    print("\n  This stands in for Model 1 using the classic indices. The real")
    print("  gate is a trained CNN and should do better, so treat these as a")
    print("  floor — and re-measure once that model exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
