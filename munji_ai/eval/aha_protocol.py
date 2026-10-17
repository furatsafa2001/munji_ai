"""Evaluate under the AHA protocol, next to the all-windows figure.

    python -m munji_ai.eval.aha_protocol

Why two numbers
---------------
Sinus specificity on every window in the cache is 0.963. Published AED work
reports 0.993 to 1.000 for the same class. Those figures are not measured the
way ours is.

The protocol those papers follow, and the AHA statement they cite, evaluates
on segments that are artifact-free and contain a single continuous rhythm.
Papers say so directly: episodes are cut into samples "containing a single
continuous rhythm type", segments that "did not conform to the specifications
of the AHA framework" are excluded, and one group reports that of 75 patients
across CUDB, VFDB and SDDB, 40 "remained following the culling". Test sets are
often the manufacturer's own curated strips rather than raw PhysioNet.

We evaluate on everything. A window that is half fibrillation and half noise
counts as fibrillation here, because SHOCKABLE_PURITY is 0.5. A window on the
boundary of an episode counts. A window during motion artifact counts. Those
are the hard cases, and they are where most of the errors sit — so the two
figures are measuring different things.

This script applies the published selection to our out-of-fold predictions
and reports both. Nothing is retrained and nothing in the prediction file
changes; only the subset being scored.

The two conditions
------------------
single rhythm   the window's dominant rhythm label covers at least 95% of
                it. Below that it straddles a boundary and is excluded, as
                the papers exclude it.

artifact-free   signal quality indices (features/sqi.py) above a threshold.
                The papers do this by hand; we do it with the same four
                indices the quality-gate baseline uses.

How to read the result
----------------------
Report both figures, always together, with the share of windows each one
covers. The all-windows number describes what the device does in the field.
The AHA-protocol number is the one that can be set beside 0.99. Quoting only
the second would be the dishonest version of what this script does.

Reads only. Writes nothing, trains nothing, does not touch the test split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, RAW_DIR, window_samples
from ..data.loader import load_record
from ..data.windows import load_cache, rhythm_intervals
from ..features import sqi

POSITIVE = "shockable"
TARGETS = {"VF": ("sensitivity", 0.90), "VFL": ("sensitivity", 0.90),
           "VT": ("sensitivity", 0.75), "N": ("specificity", 0.99),
           "NSR": ("specificity", 0.99), "ASYS": ("specificity", 0.95)}


def rhythm_and_purity(record_key, dataset, offset, n, cache, root):
    """Dominant rhythm over the window and the fraction of it that rhythm
    covers. Purity is what the papers' 'single continuous rhythm' condition
    turns into once written down."""
    key = (dataset, record_key)
    if key not in cache:
        try:
            cache[key] = rhythm_intervals(load_record(dataset, record_key, root))
        except Exception:
            cache[key] = []
    cover = defaultdict(int)
    for s, e, lab in cache[key]:
        c = max(0, min(e, offset + n) - max(s, offset))
        if c > 0:
            cover[lab] += c
    if not cover:
        return "UNLABELLED", 0.0
    lab, c = max(cover.items(), key=lambda kv: kv[1])
    return lab, c / n


def quality(X: np.ndarray) -> np.ndarray:
    """Mean of the four classic SQIs per window, 0 to 1. Same construction
    as gated_eval.py, so the two scripts agree on what 'clean' means."""
    out = np.zeros(len(X))
    for i, x in enumerate(X):
        x = np.asarray(x, dtype=np.float64)
        try:
            k = sqi.k_sqi(x)
            k_n = float(np.clip(1.0 - abs(k - 5.0) / 20.0, 0.0, 1.0))
            out[i] = float(np.mean([sqi.q_sqi(x), sqi.p_sqi(x),
                                    sqi.bas_sqi(x), k_n]))
        except Exception:
            out[i] = 0.0
    return out


def scores(y_true, y_pred, positive=POSITIVE) -> dict:
    t, p = y_true == positive, y_pred == positive
    tp = int((t & p).sum()); fn = int((t & ~p).sum())
    fp = int((~t & p).sum()); tn = int((~t & ~p).sum())
    return {"sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "n_pos": tp + fn, "n_neg": tn + fp}


def per_rhythm(y, pred, labels, mask, out=print):
    groups = defaultdict(list)
    for i in np.flatnonzero(mask):
        groups[(labels[i], y[i] == POSITIVE)].append(i)
    for (lab, is_pos), idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 10:
            continue
        idxs = np.array(idxs)
        s = scores(y[idxs], pred[idxs])
        metric = "sensitivity" if is_pos else "specificity"
        name = lab if lab not in ("VT", "VF", "VFL") else \
            f"{lab} ({'shockable' if is_pos else 'slow/excl'})"
        tgt = TARGETS.get(lab)
        flag = ""
        if tgt and tgt[0] == metric:
            flag = "  ok" if s[metric] >= tgt[1] else "  BELOW"
        out(f"    {name:<18}{len(idxs):>8,}   {metric} {s[metric]:.3f}{flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path,
                    default=Path("checkpoints") / "shockable_oof.npz")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--purity", type=float, default=0.95,
                    help="minimum share of the window one rhythm must cover")
    ap.add_argument("--quality", type=float, default=None,
                    help="minimum SQI score; default is the 20th percentile")
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
    print("recovering rhythm purity and signal quality — a minute or two\n")

    n = window_samples("shockable")
    iv: dict = {}
    lab_pur = [rhythm_and_purity(r, ds, o, n, iv, a.raw)
               for r, ds, o in zip(records, datasets, offsets)]
    labels = np.array([lp[0] for lp in lab_pur])
    purity = np.array([lp[1] for lp in lab_pur])
    q = quality(X)
    q_thr = a.quality if a.quality is not None else float(np.quantile(q, 0.20))

    pure = purity >= a.purity
    clean = q >= q_thr
    aha = pure & clean
    sinus = np.isin(labels, ("N", "NSR"))
    is_pos = y == POSITIVE

    print("selection")
    print(f"  single rhythm (purity >= {a.purity:.2f})   "
          f"{pure.mean():>6.1%} of windows kept")
    print(f"  artifact-free (SQI >= {q_thr:.3f})       "
          f"{clean.mean():>6.1%} of windows kept")
    print(f"  both                                 "
          f"{aha.mean():>6.1%} of windows kept   ({int(aha.sum()):,})")
    print(f"  shockable windows kept               "
          f"{(aha & is_pos).sum() / max(is_pos.sum(), 1):>6.1%}")
    print(f"  sinus windows kept                   "
          f"{(aha & sinus).sum() / max(sinus.sum(), 1):>6.1%}")

    if aha.sum() < 200 or (aha & is_pos).sum() < 20 or (aha & sinus).sum() < 20:
        print("\n  Too few windows survive the selection to score. Either the")
        print("  rhythm intervals could not be recovered from the raw records")
        print("  (every window reads UNLABELLED), or the thresholds are too")
        print("  strict for this cache. Try --purity 0.9 or --quality 0.5.")
        return 1

    print("\n" + "=" * 66)
    print(f"{'':<22}{'all windows':>14}{'AHA protocol':>14}{'change':>10}")
    print("=" * 66)
    a_all, a_aha = scores(y, pred), scores(y[aha], pred[aha])
    s_all = scores(y[sinus], pred[sinus])
    s_aha = scores(y[aha & sinus], pred[aha & sinus])
    rows = (("sensitivity", a_all["sensitivity"], a_aha["sensitivity"]),
            ("specificity", a_all["specificity"], a_aha["specificity"]),
            ("sinus specificity", s_all["specificity"], s_aha["specificity"]))
    for name, v0, v1 in rows:
        print(f"  {name:<20}{v0:>14.3f}{v1:>14.3f}{v1 - v0:>+10.3f}")
    print(f"  {'windows scored':<20}{len(y):>14,}{int(aha.sum()):>14,}")
    print(f"  {'sinus windows':<20}{s_all['n_neg']:>14,}{s_aha['n_neg']:>14,}")

    print("\n  AHA target for sinus specificity is 0.99. Published AED work")
    print("  under this protocol reports 0.993 to 1.000.")

    print("\nper rhythm — all windows")
    per_rhythm(y, pred, labels, np.ones(len(y), bool))
    print("\nper rhythm — AHA protocol")
    per_rhythm(y, pred, labels, aha)

    excluded = ~aha
    if excluded.any():
        e = scores(y[excluded], pred[excluded])
        print(f"\nthe {int(excluded.sum()):,} windows the protocol excludes")
        print(f"  sensitivity {e['sensitivity']:.3f}   "
              f"specificity {e['specificity']:.3f}")
        print("  These are boundary and artifact windows. The device will")
        print("  meet them in the field, so this is not a number to discard —")
        print("  it is the part of the all-windows figure the protocol hides.")

    print("\n" + "=" * 66)
    print("  Quote both columns together, with the share of windows each")
    print("  covers. One figure describes the device; the other is the one")
    print("  that compares to the literature. Neither stands alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
