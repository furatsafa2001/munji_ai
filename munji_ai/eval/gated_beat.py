"""Measure the beat segmenter on the windows the quality gate would pass.

    python -m munji_ai.eval.gated_beat --cache data/cache
    python -m munji_ai.eval.gated_beat --cache data/cache --split test

Reads only. Trains nothing, writes nothing.


WHY THE UNGATED NUMBER IS THE WRONG ONE
----------------------------------------
The reported beat F1 of 0.846 was measured over every window in the cache. The
device does not work that way. The quality gate runs first and the beat
classifier only ever sees what it passes, so a mistake on a window the gate
would have refused is a number charged to the wrong model.

The existing gated_eval.py does this for the shockable detector, but it uses
the four classic SQIs as a stand-in gate because the real one was untrained at
the time. It is trained now, so this uses the actual checkpoint.


THE SUB-WINDOW PROBLEM
----------------------
The gate judges 5-second windows; beat windows are 20 seconds. Each beat
window therefore carries four gate decisions, and they can disagree.

Three ways to combine them are reported, because the choice is a real one and
the decision layer has to make it:

  mean     average P(usable) across the four. Tolerant of one bad second.
  min      the worst sub-window decides. A single burst of artifact
           disqualifies the window.
  all      every sub-window must clear the threshold. Equivalent to min at the
           same threshold, kept separate because it is the rule people
           usually mean when they say "the window passed".

mean is the default. A 20-second window with one noisy second still carries
nineteen usable ones, and the segmenter works per sample rather than needing
the whole window clean.


WHAT THIS CANNOT TELL YOU
-------------------------
Gating always looks like an improvement, because it removes windows. The
question is whether the windows removed are ones a device would genuinely
refuse to judge, or merely ones the model got wrong. Two guards:

  the rejected share is printed next to every number

  scores on the REJECTED group are printed too. If the gate is throwing away
  windows the model handled correctly, that shows up there, and a gate that
  improves the headline by discarding beats it could have classified is not
  an improvement at all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import (BEAT_CLASSES, CACHE_DIR, GATE_MIN_P, QUALITY_CLASSES,
                      window_samples)
from ..data.windows import load_cache
from ..train.train_beat import beat_scores, sample_macro_f1

COMBINERS = ("mean", "min", "all")

# config.GATE_COMBINER names the one the device uses. The others stay in the
# report so the choice can be re-checked on new data rather than taken on
# trust.


def gate_scores(X: np.ndarray, ckpt: Path, device: str = "cpu") -> np.ndarray:
    """P(usable) for every 5 s sub-window of every beat window.

    Returns (n_windows, n_sub). Beat windows are an exact multiple of the
    quality window, so no padding or overlap is needed.
    """
    from ..models.quality_gate import build

    qlen = window_samples("quality")
    n, L = X.shape
    if L % qlen:
        raise ValueError(
            f"beat window {L} is not a multiple of quality window {qlen}; "
            f"the sub-window split assumes it is")
    n_sub = L // qlen

    blob = torch.load(ckpt, map_location=device, weights_only=False)
    gate = build().to(device)
    gate.load_state_dict(blob["state_dict"])
    gate.eval()

    flat = torch.from_numpy(
        X.reshape(n * n_sub, qlen).astype(np.float32))
    usable_idx = QUALITY_CLASSES.index("usable")

    out = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(flat), batch_size=256):
            p = torch.softmax(gate(xb.to(device)), dim=-1)[:, usable_idx]
            out.append(p.cpu().numpy())
    return np.concatenate(out).reshape(n, n_sub)


def combine(p_sub: np.ndarray, how: str, thr: float) -> np.ndarray:
    """Collapse per-sub-window probabilities into one pass/fail per window."""
    if how == "mean":
        return p_sub.mean(axis=1) >= thr
    if how in ("min", "all"):
        return p_sub.min(axis=1) >= thr
    raise ValueError(f"unknown combiner {how!r}")


def row(tag: str, y, pred, n_total: int) -> None:
    if len(y) == 0:
        print(f"  {tag:<22}{'—':>8}{'—':>10}{'—':>10}{'—':>10}{'—':>10}"
              f"{0:>9.1%}")
        return
    s = beat_scores(y, pred)
    pc = s["per_class"]
    print(f"  {tag:<22}{len(y):>8,}{pc['QRS (any beat)']['sensitivity']:>10.3f}"
          f"{pc['PVC']['f1']:>10.3f}{pc['PAC']['f1']:>10.3f}"
          f"{s['macro_f1']:>10.3f}{len(y) / n_total:>9.1%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--beat-ckpt", type=Path,
                    default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--gate-ckpt", type=Path,
                    default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    for p, what in ((a.beat_ckpt, "beat segmenter"), (a.gate_ckpt, "quality gate")):
        if not p.exists():
            print(f"no {what} checkpoint at {p}")
            return 1

    from ..models.beat_segmenter import build as build_beat
    from ..models.beat_segmenter import clean_mask

    d = load_cache("beat", a.split, a.cache)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["y"])
    if a.limit:
        X, Y = X[: a.limit], Y[: a.limit]
    n_total = len(X)

    print(f"split       {a.split}")
    print(f"windows     {n_total:,}")
    print(f"patients    {len(set(d['patient'][:n_total].astype(str)))}")

    # ------------------------------------------------------------ beat model
    blob = torch.load(a.beat_ckpt, map_location="cpu", weights_only=False)
    beat = build_beat()
    beat.load_state_dict(blob["state_dict"])
    beat.eval()
    print(f"beat ckpt   epoch {blob.get('epoch')} | "
          f"val beat F1 {blob.get('val_beat_f1', float('nan')):.3f}")

    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),
                                batch_size=16):
            preds.append(clean_mask(beat(xb).argmax(dim=1).numpy()))
    P = np.concatenate(preds)

    # ------------------------------------------------------------- the gate
    print(f"\nscoring the gate on {n_total * (X.shape[1] // window_samples('quality')):,}"
          f" sub-windows")
    p_sub = gate_scores(X, a.gate_ckpt)
    print(f"  P(usable)   median {np.median(p_sub):.3f}  "
          f"p10 {np.percentile(p_sub, 10):.3f}  "
          f"p90 {np.percentile(p_sub, 90):.3f}")

    # --------------------------------------------------------------- report
    hdr = (f"\n  {'group':<22}{'windows':>8}{'QRS sens':>10}{'PVC F1':>10}"
           f"{'PAC F1':>10}{'beat F1':>10}{'share':>9}")

    print("\n" + "=" * 80)
    print(f"UNGATED — every window, which is not how the device runs")
    print("=" * 80)
    print(hdr)
    row("all windows", Y, P, n_total)

    print("\n" + "=" * 80)
    print("GATED — at the configured operating point")
    print("=" * 80)
    thr = GATE_MIN_P["morphology"]
    print(f"\n  GATE_MIN_P['morphology'] = {thr}")
    for how in COMBINERS:
        keep = combine(p_sub, how, thr)
        print(hdr if how == COMBINERS[0] else "")
        row(f"passed ({how})", Y[keep], P[keep], n_total)
        row(f"  rejected ({how})", Y[~keep], P[~keep], n_total)

    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP — mean combiner")
    print("=" * 80)
    print(hdr)
    base = beat_scores(Y, P)["macro_f1"]
    for t in (0.0, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        keep = combine(p_sub, "mean", t)
        row(f"thr {t:.2f}", Y[keep], P[keep], n_total)

    print("\n" + "=" * 80)
    print("READING THIS")
    print("=" * 80)
    print(f"""
The ungated beat F1 is {base:.3f}. Every gated number above it is bought by
discarding windows, and the share column is the price.

The row that matters is the rejected group. If its scores are close to the
passed group's, the gate is discarding windows the model handled fine and the
threshold is too strict. If they are much worse, the gate is doing its job:
removing signal the model could not have judged.

Pick the operating point where the rejected group is clearly worse AND the
share kept is high enough to keep monitoring continuous. A gate that passes
60% of windows is not a gate, it is a 40% blind spot.
""".strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
