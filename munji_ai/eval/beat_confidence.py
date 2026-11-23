"""Does the beat model know when it is guessing?

    python -m munji_ai.eval.beat_confidence --part val
    python -m munji_ai.eval.beat_confidence --part val --limit 500

Why
---
beat_guard.py showed that inside short ventricular runs the beat model does
not lose beats: its beat count matches the expert's (median 1.00). What it gets
wrong there is the label. A true PVC inside a run is called N, and no
comparison of beat counts can see that, because the counts agree.

What could see it is the model's own probability. The segmenter outputs a
probability for every class at every sample; the label is only the largest of
them. If a PVC called N still carries a raised P(PVC), the information is in
the model and the argmax throws it away. Then two things become possible
without retraining:

    flag      beats called N with P(PVC) above a level are marked uncertain
    relabel   a lower decision level for PVC, the prior adjustment the rhythm
              stage already uses (CinC 2017 entries, commit 04cbff3)

If instead the model is confidently wrong, the information is not there and
only retraining can put it there.

What it measures
----------------
Per expert beat: the model's mean P(PVC) and mean confidence (largest class
probability) over the beat's samples, split by what the beat really is, what
the model called it, and whether it sits in a run of 3 or more consecutive
PVCs. Then a sweep over a PVC decision level, printing for each level what it
recovers and what it costs. The cost is counted as true N and PAC beats turned
PVC, per beat: the false-PVC rate per beat that the decision layer's VT rule
depends on, and which had not been measured before.

How beats are judged
--------------------
On the expert's beat regions: the model's output is read over the samples the
expert marked and the label taken by majority. Model beats the expert never
marked (false beats on background) are not counted, so the cost column is a
floor, not the whole false-PVC rate. clean_mask is not applied; it removes
fragments under 60 ms and rarely changes a region's majority.

Reads the cache only. Trains nothing, and never opens the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import config as C

CLASSES = tuple(C.BEAT_CLASSES)
I_BG, I_N, I_PAC, I_PVC = (CLASSES.index(c) for c in ("background", "N", "PAC", "PVC"))
RUN_LEN = 3
# A level of 0.50 or above changes nothing: P(PVC) >= 0.5 is already the
# largest class. Only levels below it can add PVC labels.
SWEEP = (0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10)
NAME = {I_BG: "missed", I_N: "N", I_PAC: "PAC", I_PVC: "PVC"}


def regions(mask) -> list:
    """[(start, end, class index)] for each contiguous non-background run."""
    m = np.asarray(mask).astype(int).ravel()
    out, i, n = [], 0, m.size
    while i < n:
        c = m[i]
        if c == I_BG:
            i += 1
            continue
        j = i
        while j < n and m[j] == c:
            j += 1
        out.append((i, j, int(c)))
        i = j
    return out


def pvc_stretch(classes: list) -> list:
    """For each beat, the length of the consecutive-PVC stretch it is in (0 if not a PVC)."""
    out = [0] * len(classes)
    i = 0
    while i < len(classes):
        if classes[i] != I_PVC:
            i += 1
            continue
        j = i
        while j < len(classes) and classes[j] == I_PVC:
            j += 1
        for k in range(i, j):
            out[k] = j - i
        i = j
    return out


def majority(labels) -> int:
    return int(np.bincount(labels, minlength=len(CLASSES)).argmax())


def collect(ckpt: Path, X, Y, batch: int) -> list:
    """One record per expert beat, read off the model's softmax output."""
    import torch

    from ..models.beat_segmenter import build

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"beat model  {ckpt}  epoch {blob.get('epoch')}  "
          f"val beat F1 {blob.get('val_beat_f1', float('nan')):.3f}")

    beats = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch], np.float32))
            P = torch.softmax(model(xb), dim=1).numpy()          # (B, classes, L)
            for p, y in zip(P, Y[i:i + batch]):
                regs = regions(y)
                stretch = pvc_stretch([c for _, _, c in regs])
                for (s, e, c), k in zip(regs, stretch):
                    pr = p[:, s:e]
                    lab = pr.argmax(axis=0)
                    at = []
                    for t in SWEEP:
                        adj = lab.copy()
                        adj[pr[I_PVC] >= t] = I_PVC
                        at.append(majority(adj))
                    beats.append({"true": c, "run": k,
                                  "p_pvc": float(pr[I_PVC].mean()),
                                  "conf": float(pr.max(axis=0).mean()),
                                  "pred": majority(lab), "at": at})
            if i and i % (batch * 50) == 0:
                print(f"  {i:,}/{len(X):,} windows", flush=True)
    return beats


def row(label: str, group: list) -> None:
    if not group:
        print(f"{label:<36}{0:>8}{'-':>10}{'-':>10}{'-':>10}")
        return
    pp = np.array([b["p_pvc"] for b in group])
    cf = np.array([b["conf"] for b in group])
    print(f"{label:<36}{len(group):>8,}{np.median(pp):>10.2f}"
          f"{np.percentile(pp, 90):>10.2f}{np.median(cf):>10.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="val", choices=("val",),
                    help="cached split to read (the test split is never opened)")
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--beat-ckpt", type=Path,
                    default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--limit", type=int, default=0,
                    help="use at most this many windows (0 = all)")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)

    path = a.cache / f"beat_{a.part}.npz"
    z = np.load(path)
    X, Y = z["X"], z["y"]
    if a.limit:
        X, Y = X[:a.limit], Y[:a.limit]
    print(f"{len(X):,} windows from {path}\n")

    beats = collect(a.beat_ckpt, X, Y, a.batch)

    true_n = [b for b in beats if b["true"] == I_N]
    true_pac = [b for b in beats if b["true"] == I_PAC]
    single = [b for b in beats if b["true"] == I_PVC and b["run"] < RUN_LEN]
    in_run = [b for b in beats if b["true"] == I_PVC and b["run"] >= RUN_LEN]
    print(f"\n{len(beats):,} expert beats   N {len(true_n):,}   PAC {len(true_pac):,}   "
          f"PVC single or pair {len(single):,}   PVC in a run of {RUN_LEN}+ {len(in_run):,}")
    if beats:
        pvc_all = len(single) + len(in_run)
        print(f"PVC prevalence {pvc_all / len(beats):.2%} of beats")

    print("\nWHAT THE MODEL BELIEVED (P(PVC) averaged over each beat's samples)")
    print(f"{'beat':<36}{'beats':>8}{'P(PVC) med':>10}{'p90':>10}{'conf med':>10}")
    row("true N, called N", [b for b in true_n if b["pred"] == I_N])
    row("true N, called PVC", [b for b in true_n if b["pred"] == I_PVC])
    row("true PVC single/pair, called PVC", [b for b in single if b["pred"] == I_PVC])
    row("true PVC single/pair, called N", [b for b in single if b["pred"] == I_N])
    row(f"true PVC in run {RUN_LEN}+, called PVC", [b for b in in_run if b["pred"] == I_PVC])
    row(f"true PVC in run {RUN_LEN}+, called N", [b for b in in_run if b["pred"] == I_N])
    missed = [b for b in single + in_run if b["pred"] == I_BG]
    print(f"{'true PVC, not found (background)':<36}{len(missed):>8,}")
    print("Reading: if 'in run, called N' sits clearly above 'true N, called N' (compare its"
          "\nmedian with the N row's p90), the model holds the right answer and throws it"
          "\naway at the argmax. If it sits near zero, the model is confidently wrong and"
          "\nonly retraining can help.")

    def sens(g, key):
        return f"{np.mean([key(b) == I_PVC for b in g]):.1%}" if g else "-"

    nonpvc = true_n + true_pac
    print("\nPVC DECISION LEVEL (label a sample PVC when P(PVC) is at least this)")
    print(f"{'level':<14}{'false PVC':>11}{'per beat':>10}{'sens single':>13}"
          f"{'sens in run':>13}{'sens all':>10}")

    def line(label, key):
        fp = sum(key(b) == I_PVC for b in nonpvc)
        per = fp / len(beats) if beats else float("nan")
        print(f"{label:<14}{fp:>11,}{per:>10.3%}{sens(single, key):>13}"
              f"{sens(in_run, key):>13}{sens(single + in_run, key):>10}")

    line("argmax today", lambda b: b["pred"])
    for j, t in enumerate(SWEEP):
        line(f"{t:.2f}", lambda b, j=j: b["at"][j])
    print("'false PVC' is true N and PAC beats called PVC; 'per beat' divides it by every"
          "\nexpert beat. That is the rate the VT rule's simulation assumed under 0.5%, and"
          "\nit is a floor: false beats on background are not counted here."
          "\nA level is worth taking only if 'sens in run' climbs while 'per beat' stays low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
