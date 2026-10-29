"""What would it cost to make OTHER better?

    python -m munji_ai.eval.other_tradeoff

OTHER recall is 0.67 and half of all errors are OTHER windows called NSR. The
obvious move is to weight OTHER harder in training. Before spending three
hours on a CPU to find out, this measures the shape of that trade-off in
seconds, on the model already trained.

How, without retraining
-----------------------
Re-weighting a class in the loss and shifting its decision boundary afterwards
are not the same operation, but they move the same axis: both buy recall on
one class by taking precision from it, and both take that precision out of the
classes it gets confused with. So the slope measured here is a good estimate
of the slope a re-weighted run would land on, and the sign is certain.

The script runs the saved checkpoint over the validation split, adds a bias to
the OTHER logit, and sweeps it. At bias 0 the numbers reproduce the trained
model exactly, which is the control.

What to read
------------
Two columns decide it: OTHER recall, and AF sensitivity. AF is the reason this
model exists, and the class the client named. If pulling OTHER up costs AF more
than a point or so, the trade is not worth making, whatever it does to overall
accuracy.

Accuracy itself is in the table but should not drive the decision. Nearly half
of this data is NSR, so accuracy mostly measures how well the model handles the
easy majority.

Reads only. Trains nothing, writes nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..config import CACHE_DIR, RHYTHM_CLASSES
from ..data.windows import load_cache
from ..models.rhythm_net import build

OTHER_IDX = RHYTHM_CLASSES.index("OTHER")
AF_IDX = RHYTHM_CLASSES.index("AF")


def metrics(y_true, y_pred) -> dict:
    out = {"accuracy": float((y_true == y_pred).mean())}
    f1s = []
    for c in RHYTHM_CLASSES:
        t, p = y_true == c, y_pred == c
        tp = int((t & p).sum())
        rec = tp / max(int(t.sum()), 1)
        prec = tp / max(int(p.sum()), 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        out[f"{c}_recall"] = rec
        out[f"{c}_prec"] = prec
        out[f"{c}_f1"] = f1
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    # AF specificity: of everything that is not AF, how much was not called AF
    t, p = y_true == "AF", y_pred == "AF"
    out["AF_spec"] = int((~t & ~p).sum()) / max(int((~t).sum()), 1)
    return out


@torch.no_grad()
def logits_for(model, X, device, batch=128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints") / "rhythm_net.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--split", default="val", choices=("val",))
    a = ap.parse_args(argv)

    if not a.ckpt.exists():
        print(f"no checkpoint at {a.ckpt} — train the rhythm model first")
        return 1

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(ckpt["state_dict"])

    d = load_cache("rhythm", a.split, a.cache)
    y = np.asarray(d["y"], dtype=str)
    L = logits_for(model, d["X"], torch.device("cpu"))

    base = metrics(y, np.array([RHYTHM_CLASSES[i] for i in L.argmax(1)]))
    counts = {c: int((y == c).sum()) for c in RHYTHM_CLASSES}

    print(f"checkpoint  {a.ckpt}")
    print(f"split       {a.split}  ({len(y):,} windows)  {counts}\n")
    print("sweeping a bias added to the OTHER logit; 0.0 is the trained model\n")

    print(f"{'bias':>6}{'OTHER rec':>11}{'OTHER prec':>12}{'OTHER F1':>10}"
          f"{'AF sens':>9}{'AF spec':>9}{'NSR rec':>9}{'macro F1':>10}"
          f"{'acc':>7}")

    best = None
    for b in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        adj = L.copy()
        adj[:, OTHER_IDX] += b
        m = metrics(y, np.array([RHYTHM_CLASSES[i] for i in adj.argmax(1)]))
        drop = base["AF_recall"] - m["AF_recall"]
        flag = ""
        if drop <= 0.010 and m["macro_f1"] > base["macro_f1"]:
            if best is None or m["macro_f1"] > best[1]:
                best = (b, m["macro_f1"])
                flag = "  <--"
        elif drop > 0.010:
            flag = "  AF cost"
        print(f"{b:>6.2f}{m['OTHER_recall']:>11.3f}{m['OTHER_prec']:>12.3f}"
              f"{m['OTHER_f1']:>10.3f}{m['AF_recall']:>9.3f}{m['AF_spec']:>9.3f}"
              f"{m['NSR_recall']:>9.3f}{m['macro_f1']:>10.3f}"
              f"{m['accuracy']:>7.3f}{flag}")

    print("\n" + "=" * 72)
    if best is None:
        print("  Every setting that lifts OTHER costs AF more than a point.")
        print("  The two classes are entangled: OTHER windows look like sinus")
        print("  rhythm to this model, and pushing the boundary toward OTHER")
        print("  drags AF across it too. Re-weighting in training would land on")
        print("  the same curve. Leave it, and report OTHER as a review flag")
        print("  rather than a diagnosis.")
    else:
        b, f1 = best
        print(f"  A bias of {b:.2f} raises macro F1 to {f1:.3f} without costing")
        print("  AF more than a point. That is worth a re-weighted training run:")
        print(f"    class_weights(..., max_weight=4.0) with OTHER scaled by")
        print(f"    roughly exp({b:.2f}) ≈ {np.exp(b):.1f}x")
        print("  Re-measure on cross-validation afterwards; this is one split.")

    print("\n  Caveat: these are single-split validation numbers, not")
    print("  out-of-fold. They show the shape of the trade, not a result to")
    print("  quote. Accuracy is included for completeness and should not drive")
    print("  the decision — NSR is nearly half this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
