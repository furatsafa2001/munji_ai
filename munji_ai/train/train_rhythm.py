"""Train Model 2 — the rhythm classifier.

    python -m munji_ai.train.train_rhythm --cache data/cache
    python -m munji_ai.train.train_rhythm --cache data/cache --epochs 40

Selection is on validation macro F1 over NSR and AF only, never on the test
split and never on accuracy.

Leaving OTHER out of selection needs saying plainly. It has tens of training
windows against thousands, so its F1 swings on a handful of predictions and
would add noise to every comparison between epochs without measuring anything
real. It is still scored, still printed, and still counted in the confusion
matrix — it just does not choose the checkpoint.

That is a reporting decision, not a way to make the numbers look better. The
OTHER row will be poor. It should be, and it should be visible.

Also reported is the effect of config.CONFIRMATION["af"], which wants 2 of 3
consecutive positive windows before an alert. At a 15-second stride that is
about a minute of sustained AF before the patient hears anything — deliberate,
since paroxysmal AF matters over months and a single noisy window should never
raise an alarm.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import CACHE_DIR, CONFIRMATION, RHYTHM_CLASSES
from ..data.windows import load_cache
from ..eval.metrics import macro_f1, report
from ..models.rhythm_net import (SCORED_CLASSES, RhythmLoss, build,
                                 class_weights)

POSITIVE = "AF"


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_tensors(d: dict) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(np.asarray(d["X"], dtype=np.float32))
    y_str = np.asarray(d["y"], dtype=str)
    idx = {c: i for i, c in enumerate(RHYTHM_CLASSES)}
    y = torch.tensor([idx[s] for s in y_str], dtype=torch.long)
    return X, y


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray,
                  positive: str = POSITIVE) -> dict:
    """One-vs-rest scores for a single class."""
    pos_t, pos_p = y_true == positive, y_pred == positive
    tp = int((pos_t & pos_p).sum())
    fn = int((pos_t & ~pos_p).sum())
    fp = int((~pos_t & pos_p).sum())
    tn = int((~pos_t & ~pos_p).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def confirmed_alarms(y_true, y_pred, records, offsets, rule=(2, 3),
                     positive: str = POSITIVE) -> dict:
    """Apply the N-of-M confirmation rule and recount at episode level.

    A run of confirmed windows is one alarm, not several: an episode that
    raises the phone once is the unit the patient experiences.
    """
    n, m = rule
    by_rec = defaultdict(list)
    for i, r in enumerate(records):
        by_rec[r].append(i)

    caught = missed = false_alarms = true_alarms = 0
    for idxs in by_rec.values():
        order = sorted(idxs, key=lambda i: offsets[i])
        pred = [y_pred[i] == positive for i in order]
        true = [y_true[i] == positive for i in order]

        fired = [sum(pred[max(0, i - m + 1):i + 1]) >= n
                 for i in range(len(pred))]

        prev = False
        for i, f in enumerate(fired):
            if f and not prev:
                if any(true[max(0, i - m + 1):i + 1]):
                    true_alarms += 1
                else:
                    false_alarms += 1
            prev = f

        prev = False
        for i, t in enumerate(true):
            if t and not prev:
                end = i
                while end < len(true) and true[end]:
                    end += 1
                if any(fired[i:min(end + m, len(fired))]):
                    caught += 1
                else:
                    missed += 1
            prev = t

    return {"episodes_caught": caught, "episodes_missed": missed,
            "false_alarms": false_alarms, "true_alarms": true_alarms,
            "episode_sensitivity": caught / max(caught + missed, 1)}


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    for xb, _ in loader:
        probs.append(torch.softmax(model(xb.to(device)), dim=-1).cpu())
    P = torch.cat(probs).numpy()
    pred = np.array([RHYTHM_CLASSES[i] for i in P.argmax(1)])
    return pred, P[:, RHYTHM_CLASSES.index(POSITIVE)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--penalty", type=float, default=1.3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)

    tr = load_cache("rhythm", "train", a.cache)
    va = load_cache("rhythm", "val", a.cache)

    overlap = set(tr["patient"].astype(str)) & set(va["patient"].astype(str))
    if overlap:
        raise AssertionError(f"patient leak between train and val: {overlap}")

    Xtr, ytr = as_tensors(tr)
    Xva, yva = as_tensors(va)
    y_va_str = np.asarray(va["y"], dtype=str)
    va_rec = np.asarray(va["record"], dtype=str)
    va_off = np.asarray(va["offset"], dtype=np.int64)

    y_tr_str = np.asarray(tr["y"], dtype=str)
    dist = {c: int((y_tr_str == c).sum()) for c in RHYTHM_CLASSES}
    print(f"device      {device}")
    print(f"train       {len(Xtr):,} windows, "
          f"{len(set(tr['patient'].astype(str))):,} patients  {dist}")
    print(f"val         {len(Xva):,} windows, "
          f"{len(set(va['patient'].astype(str))):,} patients")

    n_other = dist.get("OTHER", 0)
    if n_other < 200:
        print(f"\n  OTHER has {n_other} training windows. It is reported but "
              f"excluded from\n  selection, and its score should not be "
              f"quoted as a capability.")

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=a.batch,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128)

    model = build().to(device)
    w = class_weights(tr["y"])
    print(f"\nparameters  {model.n_params():,}")
    print(f"weights     {dict(zip(RHYTHM_CLASSES, [round(v, 2) for v in w.tolist()]))}"
          f"  (capped)")
    print(f"selected on macro F1 over {SCORED_CLASSES}\n")

    crit = RhythmLoss(w.to(device), af_miss_penalty=a.penalty).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train_loader))

    best_f1, best_epoch, best_state = -1.0, -1, None
    print(f"{'epoch':>5}{'loss':>9}{'F1':>8}{'AF sens':>10}{'AF spec':>10}"
          f"{'OTHER F1':>10}{'':>3}")

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
        f1 = macro_f1(y_va_str, pred, list(SCORED_CLASSES))
        other_f1 = macro_f1(y_va_str, pred, ["OTHER"])
        s = binary_scores(y_va_str, pred)

        mark = ""
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"

        secs = f"   {time.time() - t0:.0f}s/epoch" if ep == 1 else ""
        print(f"{ep:>5}{total / len(train_loader):>9.4f}{f1:>8.3f}"
              f"{s['sensitivity']:>10.3f}{s['specificity']:>10.3f}"
              f"{other_f1:>10.3f}{mark}{secs}", flush=True)

        if ep - best_epoch >= a.patience:
            print(f"\nno improvement in {a.patience} epochs — stopping")
            break

    if best_state is None:
        print("\nnothing saved.")
        return 1

    model.load_state_dict(best_state)
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / "rhythm_net.pt"
    torch.save({"state_dict": best_state, "classes": list(RHYTHM_CLASSES),
                "val_macro_f1_scored": best_f1, "epoch": best_epoch}, ckpt)

    pred, p_af = evaluate(model, val_loader, device)
    s = binary_scores(y_va_str, pred)

    print("\n" + "=" * 62)
    print(report(y_va_str, pred, list(RHYTHM_CLASSES),
                 patients=va["patient"].astype(str),
                 title=f"rhythm classifier — val (epoch {best_epoch})"))

    print("\nAF operating points")
    print(f"  {'threshold':<12}{'sens':>8}{'spec':>8}{'ppv':>8}{'alarm rate':>12}")
    for thr in (0.30, 0.50, 0.70, 0.90):
        p = np.where(p_af >= thr, POSITIVE, "NSR")
        t = binary_scores(y_va_str, p)
        print(f"  {thr:<12.2f}{t['sensitivity']:>8.3f}{t['specificity']:>8.3f}"
              f"{t['ppv']:>8.3f}{(p == POSITIVE).mean():>12.1%}")

    rule = CONFIRMATION["af"]
    c = confirmed_alarms(y_va_str, pred, va_rec, va_off, rule)
    print(f"\nafter {rule[0]}-of-{rule[1]} confirmation")
    print(f"  episodes caught      {c['episodes_caught']}")
    print(f"  episodes missed      {c['episodes_missed']}")
    print(f"  episode sensitivity  {c['episode_sensitivity']:.3f}")
    print(f"  false alarms         {c['false_alarms']}  "
          f"(from {s['fp']} isolated false positives)")

    print("\n" + "=" * 62)
    print(f"  macro F1 (NSR, AF)  {best_f1:.3f}")
    print(f"  AF sensitivity      {s['sensitivity']:.3f}")
    print(f"  AF specificity      {s['specificity']:.3f}")
    print("\n  Training AF comes from long-term AF recordings, so its share "
          "here is\n  far above what a home monitor sees. Expect the false "
          "alarm rate in the\n  field to be higher than this validation "
          "figure suggests.")
    print(f"\nsaved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
