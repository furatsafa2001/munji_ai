"""Test whether mask fragmentation explains the beat classifier's precision gap.

    python -m munji_ai.eval.diagnose_beat --cache data/cache
    python -m munji_ai.eval.diagnose_beat --cache data/cache --split test

Runs on an existing checkpoint. No retraining.

The trained model reaches QRS sensitivity 0.995 with precision 0.878, and the
same pattern holds per class: PVC 0.899/0.798, PAC 0.677/0.557. Published work
on the same task reports the opposite balance (PVC 0.977/0.969, PAC
0.714/0.928), so the gap is in precision, not in detection.

Over-detection has two plausible causes, and they need different fixes:

  fragmentation   one beat painted as several regions, because mask_to_events
                  splits on any class change or background gap. Each extra
                  fragment is a false positive even though the beat was found.
  spurious        regions appearing where no beat exists at all.

This measures which it is, then sweeps two post-processing rules against the
existing predictions:

  merge gap    adjacent same-class regions closer than N ms become one
  min width    regions narrower than N ms are dropped

If fragmentation dominates, these recover precision without retraining and
without touching sensitivity. If spurious regions dominate, they will not help
and the fix belongs in training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import BEAT_CLASSES, CACHE_DIR, TARGET_FS
from ..data.windows import load_cache
from ..train.train_beat import beat_scores, mask_to_events


def runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous non-background regions as (start, end, class)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        c = int(mask[i])
        if c == 0:
            i += 1
            continue
        j = i
        while j < n and int(mask[j]) == c:
            j += 1
        out.append((i, j, c))
        i = j
    return out


def clean_mask(mask: np.ndarray, merge_gap: int, min_width: int) -> np.ndarray:
    """Close small same-class gaps, then drop regions that are too narrow.

    Order matters: merging first lets two fragments that are each below the
    width threshold survive as one region of adequate width. Dropping first
    would delete both and turn a detected beat into a miss.
    """
    out = mask.copy()
    rs = runs(out)

    if merge_gap > 0:
        for (s1, e1, c1), (s2, e2, c2) in zip(rs, rs[1:]):
            if c1 == c2 and (s2 - e1) <= merge_gap:
                out[e1:s2] = c1

    if min_width > 0:
        for s, e, _ in runs(out):
            if (e - s) < min_width:
                out[s:e] = 0
    return out


def fragmentation_report(true_masks, pred_masks, fs=TARGET_FS) -> dict:
    """How many predicted regions land inside each true beat's tolerance."""
    tol = int(0.150 * fs)
    counts, widths_t, widths_p, gaps = [], [], [], []

    for tm, pm in zip(true_masks, pred_masks):
        tr, pr = runs(tm), runs(pm)
        widths_t += [e - s for s, e, _ in tr]
        widths_p += [e - s for s, e, _ in pr]

        pc = [(s + e) // 2 for s, e, _ in pr]
        for s, e, _ in tr:
            centre = (s + e) // 2
            counts.append(sum(1 for c in pc if abs(c - centre) <= tol))

        for (s1, e1, c1), (s2, e2, c2) in zip(pr, pr[1:]):
            if c1 == c2:
                gaps.append(s2 - e1)

    counts = np.array(counts)
    gaps = np.array(gaps) if gaps else np.array([0])
    return {
        "beats": len(counts),
        "missed": float((counts == 0).mean()),
        "exact": float((counts == 1).mean()),
        "fragmented": float((counts > 1).mean()),
        "extra_from_fragmentation": int(np.clip(counts - 1, 0, None).sum()),
        "true_width_median_ms": float(np.median(widths_t) / fs * 1000),
        "pred_width_median_ms": float(np.median(widths_p) / fs * 1000),
        "pred_width_p10_ms": float(np.percentile(widths_p, 10) / fs * 1000),
        "same_class_gap_p25_ms": float(np.percentile(gaps, 25) / fs * 1000),
        "same_class_gap_median_ms": float(np.median(gaps) / fs * 1000),
        "n_pred_regions": len(widths_p),
        "n_true_regions": len(widths_t),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N windows, for a quick run")
    a = ap.parse_args(argv)

    from ..models.beat_segmenter import build

    if not a.ckpt.exists():
        print(f"no checkpoint at {a.ckpt} — train the beat segmenter first")
        return 1

    d = load_cache("beat", a.split, a.cache)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["y"])
    if a.limit:
        X, Y = X[: a.limit], Y[: a.limit]

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"checkpoint epoch {blob.get('epoch')} | "
          f"val beat F1 {blob.get('val_beat_f1', float('nan')):.3f}")
    print(f"scoring {len(X):,} {a.split} windows\n")

    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=16)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).argmax(dim=1).numpy())
    P = np.concatenate(preds)

    # ------------------------------------------------------------ diagnosis
    fr = fragmentation_report(Y, P)
    print("=" * 66)
    print("IS THE MODEL FRAGMENTING BEATS?")
    print("=" * 66)
    print(f"  true beats                {fr['beats']:,}")
    print(f"  predicted regions         {fr['n_pred_regions']:,}  "
          f"(true regions {fr['n_true_regions']:,})")
    print()
    print(f"  found exactly once        {fr['exact']:.1%}")
    print(f"  found as several pieces   {fr['fragmented']:.1%}")
    print(f"  not found                 {fr['missed']:.1%}")
    print()
    print(f"  extra regions from splitting one beat: "
          f"{fr['extra_from_fragmentation']:,}")

    excess = fr["n_pred_regions"] - fr["n_true_regions"]
    share = (fr["extra_from_fragmentation"] / excess) if excess > 0 else 0.0
    print(f"  excess predictions overall:            {excess:,}")
    if excess > 0:
        print(f"  -> {share:.0%} of the excess is fragmentation, "
              f"{1 - share:.0%} is spurious regions")

    print()
    print(f"  region width  true {fr['true_width_median_ms']:.0f} ms  "
          f"predicted {fr['pred_width_median_ms']:.0f} ms  "
          f"(10th pct {fr['pred_width_p10_ms']:.0f} ms)")
    print(f"  gap between same-class regions: "
          f"median {fr['same_class_gap_median_ms']:.0f} ms, "
          f"25th pct {fr['same_class_gap_p25_ms']:.0f} ms")

    # --------------------------------------------------------------- sweep
    print("\n" + "=" * 66)
    print("DOES POST-PROCESSING RECOVER PRECISION?")
    print("=" * 66)

    base = beat_scores(Y, P)
    print(f"\n  {'merge':>7}{'min w':>7}  {'QRS sens':>9}{'QRS prec':>9}"
          f"{'PVC sens':>9}{'PVC prec':>9}{'PAC sens':>9}{'PAC prec':>9}"
          f"{'beat F1':>9}")

    def row(tag_a, tag_b, s):
        q, v, c = (s["per_class"]["QRS (any beat)"], s["per_class"]["PVC"],
                   s["per_class"]["PAC"])
        print(f"  {tag_a:>7}{tag_b:>7}  {q['sensitivity']:>9.3f}"
              f"{q['precision']:>9.3f}{v['sensitivity']:>9.3f}"
              f"{v['precision']:>9.3f}{c['sensitivity']:>9.3f}"
              f"{c['precision']:>9.3f}{s['macro_f1']:>9.3f}")

    row("—", "—", base)

    best, best_cfg = base["macro_f1"], (0, 0)
    for gap_ms in (0, 30, 50, 80):
        for width_ms in (0, 40, 60, 80):
            if gap_ms == 0 and width_ms == 0:
                continue
            g = int(gap_ms * TARGET_FS / 1000)
            w = int(width_ms * TARGET_FS / 1000)
            C = np.stack([clean_mask(m, g, w) for m in P])
            s = beat_scores(Y, C)
            row(f"{gap_ms}ms", f"{width_ms}ms", s)
            if s["macro_f1"] > best:
                best, best_cfg = s["macro_f1"], (gap_ms, width_ms)

    print("\n" + "=" * 66)
    gain = best - base["macro_f1"]
    if gain > 0.005:
        print(f"  best: merge {best_cfg[0]} ms, min width {best_cfg[1]} ms")
        print(f"  beat F1 {base['macro_f1']:.3f} -> {best:.3f}  ({gain:+.3f})")
        print("  -> fragmentation was real; apply this in post-processing")
    else:
        print(f"  no configuration beats the raw output by more than 0.005")
        print("  -> the excess predictions are spurious regions, not split "
              "beats.\n     Precision has to be fixed in training: lower the "
              "class-weight cap\n     (currently 20.0) so the model stops "
              "over-calling ectopics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())