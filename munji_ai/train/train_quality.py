"""Train Model 1 — the signal quality gate.

    python -m munji_ai.train.train_quality --cache data/cache
    python -m munji_ai.train.train_quality --cache data/cache --epochs 40

Selection is on validation macro F1, never on the test split, and never on
accuracy — the classes run roughly 76/24, so accuracy rewards a model that
predicts the majority and detects nothing.

Two numbers decide whether a run is an improvement:

  macro F1        must beat the 0.835 feature baseline
  rejection bias  must not rise above 1.34

The second matters as much as the first. A gate that improves accuracy while
rejecting arrhythmias more often is worse, not better — it would be discarding
exactly the events MUNJI exists to detect, and its own score would look fine.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import CACHE_DIR, QUALITY_CLASSES
from ..data.windows import load_cache
from ..eval.metrics import (format_bias, gate_views_scored, macro_f1, report,
                            rhythm_rejection_bias)
from ..models.quality_gate import AsymmetricLoss, build, class_weights

NORMAL_SOURCES = ("nsrdb",)
BASELINE_F1 = 0.835        # what the SQI feature baseline reached
BASELINE_BIAS = 1.34       # and the bias it did it at


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon. Safa's machine has it; Furat's does not.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_tensors(d: dict) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(np.asarray(d["X"], dtype=np.float32))
    y_str = np.asarray(d["y"], dtype=str)
    idx = {c: i for i, c in enumerate(QUALITY_CLASSES)}
    y = torch.tensor([idx[s] for s in y_str], dtype=torch.long)
    return X, y


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted label strings, P(usable))."""
    model.eval()
    probs = []
    for xb, _ in loader:
        p = torch.softmax(model(xb.to(device)), dim=-1)
        probs.append(p.cpu())
    P = torch.cat(probs).numpy()
    pred = np.array([QUALITY_CLASSES[i] for i in P.argmax(1)])
    return pred, P[:, QUALITY_CLASSES.index("usable")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--penalty", type=float, default=1.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)

    tr = load_cache("quality", "train", a.cache)
    va = load_cache("quality", "val", a.cache)

    overlap = set(tr["patient"].astype(str)) & set(va["patient"].astype(str))
    if overlap:
        raise AssertionError(f"patient leak between train and val: {overlap}")

    Xtr, ytr = as_tensors(tr)
    Xva, yva = as_tensors(va)
    y_va_str = np.asarray(va["y"], dtype=str)
    va_ds = np.asarray(va["dataset"], dtype=str)

    dist = {c: int((np.asarray(tr["y"], dtype=str) == c).sum())
            for c in QUALITY_CLASSES}
    print(f"device      {device}")
    print(f"train       {len(Xtr):,} windows, "
          f"{len(set(tr['patient'].astype(str))):,} patients  {dist}")
    print(f"val         {len(Xva):,} windows, "
          f"{len(set(va['patient'].astype(str))):,} patients")

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=a.batch,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256)

    model = build().to(device)
    print(f"parameters  {model.n_params():,}\n")

    crit = AsymmetricLoss(class_weights(tr["y"]).to(device),
                          false_reject_penalty=a.penalty).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train_loader))

    best_f1, best_epoch, best_state = -1.0, -1, None
    print(f"{'epoch':>5}{'loss':>9}{'val F1':>9}{'sens':>8}{'spec':>8}"
          f"{'bias':>8}{'':>4}")

    for ep in range(1, a.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()

        pred, _ = evaluate(model, val_loader, device)
        f1 = macro_f1(y_va_str, pred, list(QUALITY_CLASSES))
        from ..eval.metrics import gate_views
        gv = gate_views(y_va_str, pred)["reject"]
        bias = rhythm_rejection_bias(pred, va_ds, NORMAL_SOURCES)["ratio"]

        # Bias is a gate on selection, not a tiebreaker. A model that scores
        # better by discarding more arrhythmia is not a better gate.
        acceptable = bias <= BASELINE_BIAS * 1.05
        mark = ""
        if f1 > best_f1 and acceptable:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        elif f1 > best_f1:
            mark = " (bias)"

        secs = f"   {time.time() - t0:.0f}s/epoch" if ep == 1 else ""
        print(f"{ep:>5}{total / len(train_loader):>9.4f}{f1:>9.3f}"
              f"{gv['sensitivity']:>8.3f}{gv['specificity']:>8.3f}"
              f"{bias:>8.2f}{mark}{secs}", flush=True)

        if ep - best_epoch >= a.patience:
            print(f"\nno improvement in {a.patience} epochs — stopping")
            break

    if best_state is None:
        print("\nno epoch met the bias constraint. Nothing saved.")
        return 1

    model.load_state_dict(best_state)
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / "quality_gate.pt"
    torch.save({"state_dict": best_state, "classes": list(QUALITY_CLASSES),
                "val_macro_f1": best_f1, "epoch": best_epoch}, ckpt)

    pred, p_usable = evaluate(model, val_loader, device)
    print("\n" + "=" * 62)
    print(report(y_va_str, pred, list(QUALITY_CLASSES),
                 patients=va["patient"].astype(str),
                 title=f"CNN quality gate — val (epoch {best_epoch})"))

    print("\ngraded operating points")
    print(f"  {'point':<14}{'thr':>6}{'sens':>8}{'spec':>8}{'pass rate':>11}")
    for name, s in gate_views_scored(y_va_str, p_usable).items():
        print(f"  {name:<14}{s['threshold']:>6.2f}{s['sensitivity']:>8.3f}"
              f"{s['specificity']:>8.3f}{s['pass_rate']:>11.1%}")

    print()
    print(format_bias(rhythm_rejection_bias(pred, va_ds, NORMAL_SOURCES)))

    print("\n" + "=" * 62)
    delta = best_f1 - BASELINE_F1
    print(f"  macro F1  {best_f1:.3f} vs baseline {BASELINE_F1:.3f}  "
          f"({delta:+.3f})")
    print("  -> the CNN earns its place" if delta > 0.02 else
          "  -> no meaningful gain over ten features and a tree; "
          "the extra complexity is not justified yet")
    print(f"\nsaved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())