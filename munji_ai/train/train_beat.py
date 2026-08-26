"""Train Model 2 — the beat classifier, as segmentation.

    python -m munji_ai.train.train_beat --cache data/cache
    python -m munji_ai.train.train_beat --cache data/cache --epochs 40

Selection is on validation macro F1 over the three beat classes, excluding
background. Background is 87% of samples, so including it would let a model
that finds no beats at all still score well.

Two evaluations are reported, and they answer different questions.

  Per-sample     how precisely the mask traces each complex. Useful for
                 diagnosing the model, not for judging clinical usefulness.

  Per-beat       whether each annotated beat was found and given the right
                 class, matching within a tolerance. This is what a clinician
                 means by "did it detect the PVC", and it is the number that
                 belongs in any report. The reference paper's 0.999 QRS
                 sensitivity is a per-beat figure.

A model can look strong per-sample while missing beats outright, because
missing a 140 ms complex costs 35 samples out of 5000.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import BEAT_CLASSES, CACHE_DIR, TARGET_FS
from ..data.windows import load_cache
from ..models.beat_segmenter import (SegmentationLoss, build,
                                     class_weights_from_masks)

# A detection counts as matching an annotation if their centres fall within
# this distance. 150 ms is the usual tolerance in the beat-detection
# literature and is well inside one RR interval at any plausible rate.
MATCH_TOLERANCE_MS = 150.0
BEAT_IDS = {c: i for i, c in enumerate(BEAT_CLASSES)}
EVENT_CLASSES = [c for c in BEAT_CLASSES if c != "background"]


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_to_events(mask: np.ndarray) -> list[tuple[int, str]]:
    """Contiguous non-background runs -> (centre sample, class).

    The mask marks a region around each beat, so a beat is a run of identical
    non-zero labels. Taking the centre gives a position comparable to an
    annotated R-peak.
    """
    out = []
    i, n = 0, len(mask)
    while i < n:
        c = int(mask[i])
        if c == 0:
            i += 1
            continue
        j = i
        while j < n and int(mask[j]) == c:
            j += 1
        out.append(((i + j) // 2, BEAT_CLASSES[c]))
        i = j
    return out


def beat_scores(true_masks: np.ndarray, pred_masks: np.ndarray,
                fs: int = TARGET_FS) -> dict:
    """Per-beat sensitivity and precision, greedily matched one-to-one.

    Greedy matching matters: without it, several predicted fragments of one
    beat all match the same annotation and precision is overstated.
    """
    tol = MATCH_TOLERANCE_MS * fs / 1000.0
    tp = {c: 0 for c in EVENT_CLASSES}
    fn = {c: 0 for c in EVENT_CLASSES}
    fp = {c: 0 for c in EVENT_CLASSES}
    qrs_tp = qrs_fn = qrs_fp = 0

    for tm, pm in zip(true_masks, pred_masks):
        te, pe = mask_to_events(tm), mask_to_events(pm)
        used = [False] * len(pe)

        for pos, cls in te:
            best, best_d = -1, tol + 1
            for k, (ppos, _) in enumerate(pe):
                if used[k]:
                    continue
                d = abs(ppos - pos)
                if d <= tol and d < best_d:
                    best, best_d = k, d
            if best < 0:
                fn[cls] += 1
                qrs_fn += 1
                continue
            used[best] = True
            qrs_tp += 1
            # Located correctly, but was it the right kind of beat?
            if pe[best][1] == cls:
                tp[cls] += 1
            else:
                fn[cls] += 1
                fp[pe[best][1]] += 1

        for k, (_, cls) in enumerate(pe):
            if not used[k]:
                fp[cls] += 1
                qrs_fp += 1

    def prf(t, f_n, f_p):
        sens = t / (t + f_n) if (t + f_n) else float("nan")
        prec = t / (t + f_p) if (t + f_p) else float("nan")
        f1 = (2 * sens * prec / (sens + prec)
              if sens and prec and not np.isnan(sens + prec) else 0.0)
        return {"sensitivity": sens, "precision": prec, "f1": f1,
                "support": t + f_n}

    per = {c: prf(tp[c], fn[c], fp[c]) for c in EVENT_CLASSES}
    per["QRS (any beat)"] = prf(qrs_tp, qrs_fn, qrs_fp)
    scored = [v["f1"] for c, v in per.items()
              if c in EVENT_CLASSES and v["support"] > 0]
    return {"per_class": per, "macro_f1": float(np.mean(scored)) if scored else 0.0}


def sample_macro_f1(true_masks: np.ndarray, pred_masks: np.ndarray) -> float:
    """Macro F1 over beat classes only — background excluded on purpose."""
    t, p = true_masks.ravel(), pred_masks.ravel()
    f1s = []
    for c in EVENT_CLASSES:
        i = BEAT_IDS[c]
        tp = int(((t == i) & (p == i)).sum())
        fn = int(((t == i) & (p != i)).sum())
        fp = int(((t != i) & (p == i)).sum())
        if tp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dice", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)

    tr = load_cache("beat", "train", a.cache)
    va = load_cache("beat", "val", a.cache)

    overlap = set(tr["patient"].astype(str)) & set(va["patient"].astype(str))
    if overlap:
        raise AssertionError(f"patient leak between train and val: {overlap}")

    Xtr = torch.from_numpy(np.asarray(tr["X"], dtype=np.float32))
    Ytr = torch.from_numpy(np.asarray(tr["y"], dtype=np.int64))
    Xva = torch.from_numpy(np.asarray(va["X"], dtype=np.float32))
    Yva = torch.from_numpy(np.asarray(va["y"], dtype=np.int64))
    yva_np = np.asarray(va["y"])

    counts = {c: int((np.asarray(tr["y"]) == i).sum())
              for i, c in enumerate(BEAT_CLASSES)}
    total = sum(counts.values())
    print(f"device      {device}")
    print(f"train       {len(Xtr):,} windows, "
          f"{len(set(tr['patient'].astype(str))):,} patients")
    print(f"val         {len(Xva):,} windows, "
          f"{len(set(va['patient'].astype(str))):,} patients")
    print("mask mix    " + "  ".join(f"{c} {counts[c] / total:.1%}"
                                     for c in BEAT_CLASSES))

    model = build().to(device)
    w = class_weights_from_masks(tr["y"]).to(device)
    print(f"parameters  {model.n_params():,}")
    print("weights     " + "  ".join(f"{c} {v:.1f}"
                                     for c, v in zip(BEAT_CLASSES, w.tolist())))
    print()

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=a.batch,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=32)

    crit = SegmentationLoss(w, dice_weight=a.dice).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train_loader))

    best, best_ep, best_state = -1.0, -1, None
    print(f"{'epoch':>5}{'loss':>9}{'sample F1':>11}{'beat F1':>10}"
          f"{'QRS sens':>10}{'':>3}")

    for ep in range(1, a.epochs + 1):
        model.train()
        tot, t0 = 0.0, time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()

        pred = predict(model, val_loader, device)
        s_f1 = sample_macro_f1(yva_np, pred)
        bs = beat_scores(yva_np, pred)
        b_f1 = bs["macro_f1"]
        qrs = bs["per_class"]["QRS (any beat)"]["sensitivity"]

        # Selection is on the per-beat number: it is what the clinical claim
        # rests on, and a model can trace masks well while missing beats.
        mark = ""
        if b_f1 > best:
            best, best_ep = b_f1, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"

        secs = f"   {time.time() - t0:.0f}s/epoch" if ep == 1 else ""
        print(f"{ep:>5}{tot / len(train_loader):>9.4f}{s_f1:>11.3f}"
              f"{b_f1:>10.3f}{qrs:>10.3f}{mark}{secs}", flush=True)

        if ep - best_ep >= a.patience:
            print(f"\nno improvement in {a.patience} epochs — stopping")
            break

    if best_state is None:
        print("\nnothing to save")
        return 1

    model.load_state_dict(best_state)
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / "beat_segmenter.pt"
    torch.save({"state_dict": best_state, "classes": list(BEAT_CLASSES),
                "val_beat_f1": best, "epoch": best_ep}, ckpt)

    pred = predict(model, val_loader, device)
    bs = beat_scores(yva_np, pred)

    print("\n" + "=" * 62)
    print(f"beat segmenter — val (epoch {best_ep})")
    print("=" * 62)
    print(f"\n{'class':<18}{'sens':>9}{'prec':>9}{'f1':>9}{'beats':>10}")
    for c, v in bs["per_class"].items():
        print(f"{c:<18}{v['sensitivity']:>9.3f}{v['precision']:>9.3f}"
              f"{v['f1']:>9.3f}{v['support']:>10,}")
    print(f"\n{'macro F1 (beats)':<18}{bs['macro_f1']:>9.3f}")
    print(f"{'macro F1 (samples)':<18}{sample_macro_f1(yva_np, pred):>9.3f}")

    ref = bs["per_class"]["QRS (any beat)"]["sensitivity"]
    print(f"\nreference: the paper reports QRS sensitivity and precision up to "
          f"0.999\nand PVC sensitivity 0.820-0.986 across datasets.")
    print(f"this run:  QRS sensitivity {ref:.3f}")
    print("\nBoth are validation numbers on different data — treat the "
          "comparison as\na sanity check on the order of magnitude, not a "
          "like-for-like result.")
    print(f"\nsaved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())