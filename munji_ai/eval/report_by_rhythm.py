"""Report shockable-detector performance per rhythm type, not pooled.

    python -m munji_ai.eval.report_by_rhythm
    python -m munji_ai.eval.report_by_rhythm --oof checkpoints/shockable_oof.npz

Why pooled specificity is the wrong number
------------------------------------------
Cross-validation gave specificity 0.864 against an AHA target of 0.99, which
looks like a large shortfall. It is not comparable.

The AHA figure is specificity *for normal sinus rhythm*. Published AED work
reports one number per rhythm class — separate figures for NSR, asystole and
everything else — because the classes are not equally hard and a single
average hides which one is failing.

Our negatives are mostly not sinus rhythm. CUDB and VFDB are recordings of
people who arrested, so "not shockable" here means the quiet stretches of an
arrest: bradycardia, asystole, idioventricular rhythm, noise. Comparing a
pooled number over that mixture against a target defined for healthy sinus
rhythm compares two different things.

This script splits the result the way the literature does, by reading the
rhythm label each window actually carried before it was collapsed into the
binary shockable / not_shockable decision.

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

# AHA / AAMI targets, per class. A single pooled number has no target to
# compare against — these do.
TARGETS = {
    "VF": ("sensitivity", 0.90),
    "VFL": ("sensitivity", 0.90),
    "VT_FAST": ("sensitivity", 0.75),
    "N": ("specificity", 0.99),
    "NSR": ("specificity", 0.99),
    "ASYS": ("specificity", 0.95),
}


def rhythm_at(record_key: str, dataset: str, offset: int, n: int,
              cache: dict, root: Path) -> str:
    """The rhythm label covering a window, from the loader's own intervals."""
    key = (dataset, record_key)
    if key not in cache:
        try:
            rec = load_record(dataset, record_key, root)
            cache[key] = rhythm_intervals(rec)
        except Exception:
            cache[key] = []
    best, best_cover = "UNLABELLED", 0
    for s, e, lab in cache[key]:
        cover = max(0, min(e, offset + n) - max(s, offset))
        if cover > best_cover:
            best, best_cover = lab, cover
    return best if best_cover > 0 else "UNLABELLED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path,
                    default=Path("checkpoints") / "shockable_oof.npz")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
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
    datasets = np.concatenate([np.asarray(p["dataset"], dtype=str)
                               for p in parts])
    if len(datasets) != len(y):
        print("cache no longer matches the saved predictions — re-run "
              "cross_validate")
        return 1

    n = window_samples("shockable")
    iv_cache: dict = {}
    labels = np.array([rhythm_at(r, ds, o, n, iv_cache, a.raw)
                       for r, ds, o in zip(records, datasets, offsets)])

    print(f"source      {a.oof}")
    print(f"windows     {len(y):,}")
    print(f"convention  one figure per rhythm class, as AED studies report\n")

    # A rhythm label is not a class. "VT" in these databases means anything
    # ventricular above 100 bpm, which is the clinical definition — so it spans
    # both sides of our 150 bpm split: the fast half is deliberately shockable,
    # the slow half deliberately not. Scoring the two together produces a
    # number that is neither a sensitivity nor a specificity, and the first
    # version of this script reported exactly that: "VT sensitivity 0.586",
    # mostly counting windows the model was correct to call safe.
    #
    # So each rhythm is split by the binary label its windows actually carry,
    # and each half gets the metric that applies to it.
    groups = defaultdict(lambda: [0, 0])        # (label, is_pos) -> [ok, total]
    for lab, t, p in zip(labels, y, pred):
        key = (lab, t == "shockable")
        groups[key][1] += 1
        groups[key][0] += int(t == p)

    rows = sorted(((lab, is_pos, ok, tot)
                   for (lab, is_pos), (ok, tot) in groups.items()),
                  key=lambda r: -r[3])

    print(f"{'rhythm':<18}{'windows':>9}{'metric':>14}{'value':>8}"
          f"{'target':>9}{'':>4}")
    for lab, is_pos, ok, tot in rows:
        metric = "sensitivity" if is_pos else "specificity"
        rate = ok / max(tot, 1)
        name = lab if lab not in ("VT", "VF", "VFL") else \
            f"{lab} ({'shockable' if is_pos else 'slow/excl'})"
        tgt = TARGETS.get("VT_FAST" if (lab == "VT" and is_pos) else lab)
        tgt_txt, flag = "—", ""
        if tgt and tgt[0] == metric:
            tgt_txt = f"{tgt[1]:.2f}"
            flag = "  ok" if rate >= tgt[1] else "  BELOW"
        print(f"{name:<18}{tot:>9,}{metric:>14}{rate:>8.3f}"
              f"{tgt_txt:>9}{flag}")

    print("\n  A rhythm label spanning both sides of the 150 bpm split is")
    print("  shown twice. The slow half was excluded from the shockable class")
    print("  by design, so the model calling it safe is correct, not a miss.")

    sinus = [k for k in groups if k[0] in ("N", "NSR") and not k[1]]
    if sinus:
        ok = sum(groups[k][0] for k in sinus)
        tot = sum(groups[k][1] for k in sinus)
        print(f"\n  sinus rhythm specificity   {ok / max(tot, 1):.3f}   "
              f"({tot:,} windows)")
        print("  This is the figure the AHA 0.99 target is defined against.")

    non_sinus = sum(tot for lab, is_pos, _, tot in rows
                    if lab not in ("N", "NSR") and not is_pos)
    total_neg = int((y == "not_shockable").sum())
    if total_neg:
        print(f"\n  of {total_neg:,} negative windows, {non_sinus:,} "
              f"({non_sinus / total_neg:.0%}) are not sinus rhythm")
        print("  They come from the quiet stretches of arrest recordings, so a")
        print("  pooled specificity is mostly measuring those, not healthy")
        print("  hearts. That is the gap a normal-rhythm corpus would close.")

    unl = sum(tot for (lab, _), (_, tot) in groups.items()
              if lab == "UNLABELLED")
    if unl:
        print(f"\n  {unl:,} windows carried no rhythm label and are excluded "
              f"from\n  the per-class reading above.")

    print("\n" + "=" * 62)
    print("  Quote these per class. A single pooled specificity has no")
    print("  published target to compare against and hides which class is")
    print("  actually failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
