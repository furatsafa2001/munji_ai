"""Train Model 3 — the shockable rhythm detector.

    python -m munji_ai.train.train_shockable --cache data/cache
    python -m munji_ai.train.train_shockable --cache data/cache --epochs 40

Selection is on validation specificity subject to a sensitivity floor, never
on the test split and never on accuracy. The floor is the AHA/AAMI figure for
VF, not a tuning choice: below 0.90 sensitivity this is not a shock-advisory
algorithm, whatever its average score.

    sensitivity  >= 0.90     hard gate on selection
    specificity              maximised among epochs that clear the gate

Reported alongside is the effect of config.CONFIRMATION["vf"], which requires
2 of 3 consecutive positive windows before an alarm fires. Window-level
specificity overstates how noisy the product actually is, because isolated
false positives never reach the patient. The number that matters for a home
monitor is how many alarms survive confirmation.

One caveat travels with every number this prints. CUDB and VFDB are curated
recordings of patients who arrested, so shockable rhythm is 28% of these
windows; in a home device it is closer to zero. Detection is also known to be
easier on public repository data than on real out-of-hospital arrest, where VF
several minutes in looks nothing like the clean high-amplitude onset these
databases captured. Treat what follows as a comparison figure, not as expected
field performance.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import CACHE_DIR, CONFIRMATION, SHOCKABLE_CLASSES
from ..data.windows import load_cache
from ..eval.metrics import macro_f1, report
from ..models.shockable_net import (SENSITIVITY_FLOOR, MissPenaltyLoss, build,
                                    class_weights)

POSITIVE = "shockable"


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
    idx = {c: i for i, c in enumerate(SHOCKABLE_CLASSES)}
    y = torch.tensor([idx[s] for s in y_str], dtype=torch.long)
    return X, y


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sensitivity, specificity and PPV for the shockable class."""
    pos_t, pos_p = y_true == POSITIVE, y_pred == POSITIVE
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


def confirmed_alarms(y_true, y_pred, records, offsets, rule=(2, 3)) -> dict:
    """Apply the N-of-M confirmation rule and recount at alarm level.

    Windows are grouped per record and ordered in time. An alarm fires when at
    least N of the last M windows are positive. A run of confirmed windows is
    one alarm, not several — an episode raising the phone once is the unit the
    patient experiences.
    """
    n, m = rule
    by_rec = defaultdict(list)
    for i, r in enumerate(records):
        by_rec[r].append(i)

    caught = missed = false_alarms = true_alarms = 0
    for idxs in by_rec.values():
        order = sorted(idxs, key=lambda i: offsets[i])
        pred = [y_pred[i] == POSITIVE for i in order]
        true = [y_true[i] == POSITIVE for i in order]

        fired = [sum(pred[max(0, i - m + 1):i + 1]) >= n
                 for i in range(len(pred))]

        # Count runs rather than windows: consecutive alarms are one event.
        prev = False
        for i, f in enumerate(fired):
            if f and not prev:
                true_alarms += 1 if any(true[max(0, i - m + 1):i + 1]) else 0
                false_alarms += 0 if any(true[max(0, i - m + 1):i + 1]) else 1
            prev = f

        # Did each true episode get confirmed anywhere inside it?
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
    pred = np.array([SHOCKABLE_CLASSES[i] for i in P.argmax(1)])
    return pred, P[:, SHOCKABLE_CLASSES.index(POSITIVE)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--penalty", type=float, default=2.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)

    tr = load_cache("shockable", "train", a.cache)
    va = load_cache("shockable", "val", a.cache)

    overlap = set(tr["patient"].astype(str)) & set(va["patient"].astype(str))
    if overlap:
        raise AssertionError(f"patient leak between train and val: {overlap}")

    Xtr, ytr = as_tensors(tr)
    Xva, yva = as_tensors(va)
    y_va_str = np.asarray(va["y"], dtype=str)
    va_rec = np.asarray(va["record"], dtype=str)
    va_off = np.asarray(va["offset"], dtype=np.int64)

    dist = {c: int((np.asarray(tr["y"], dtype=str) == c).sum())
            for c in SHOCKABLE_CLASSES}
    print(f"device      {device}")
    print(f"train       {len(Xtr):,} windows, "
          f"{len(set(tr['patient'].astype(str))):,} patients  {dist}")
    print(f"val         {len(Xva):,} windows, "
          f"{len(set(va['patient'].astype(str))):,} patients")

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=a.batch,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256)

    model = build().to(device)
    print(f"parameters  {model.n_params():,}")
    print(f"floor       sensitivity >= {SENSITIVITY_FLOOR:.2f} "
          f"(AHA/AAMI for VF)\n")

    crit = MissPenaltyLoss(class_weights(tr["y"]).to(device),
                           miss_penalty=a.penalty).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train_loader))

    best_spec, best_epoch, best_state = -1.0, -1, None
    print(f"{'epoch':>5}{'loss':>9}{'sens':>8}{'spec':>8}{'ppv':>8}"
          f"{'F1':>8}{'':>4}")

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
        s = binary_scores(y_va_str, pred)
        f1 = macro_f1(y_va_str, pred, list(SHOCKABLE_CLASSES))

        # Sensitivity is a gate on selection, not a tiebreaker. A model that
        # scores better by missing arrests is not a better detector.
        mark = ""
        if s["sensitivity"] >= SENSITIVITY_FLOOR and s["specificity"] > best_spec:
            best_spec, best_epoch = s["specificity"], ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        elif s["specificity"] > best_spec:
            mark = " (sens)"

        secs = f"   {time.time() - t0:.0f}s/epoch" if ep == 1 else ""
        print(f"{ep:>5}{total / len(train_loader):>9.4f}"
              f"{s['sensitivity']:>8.3f}{s['specificity']:>8.3f}"
              f"{s['ppv']:>8.3f}{f1:>8.3f}{mark}{secs}", flush=True)

        if best_epoch > 0 and ep - best_epoch >= a.patience:
            print(f"\nno improvement in {a.patience} epochs — stopping")
            break

    if best_state is None:
        print(f"\nno epoch reached {SENSITIVITY_FLOOR:.0%} sensitivity. "
              f"Nothing saved.\nRaise --penalty or train longer; do not lower "
              f"the floor to make a run pass.")
        return 1

    model.load_state_dict(best_state)
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / "shockable_net.pt"
    torch.save({"state_dict": best_state, "classes": list(SHOCKABLE_CLASSES),
                "val_specificity": best_spec, "epoch": best_epoch}, ckpt)

    pred, p_shock = evaluate(model, val_loader, device)
    s = binary_scores(y_va_str, pred)

    print("\n" + "=" * 62)
    print(report(y_va_str, pred, list(SHOCKABLE_CLASSES),
                 patients=va["patient"].astype(str),
                 title=f"shockable detector — val (epoch {best_epoch})"))

    print("\ngraded operating points")
    print(f"  {'threshold':<12}{'sens':>8}{'spec':>8}{'ppv':>8}{'alarm rate':>12}")
    for thr in (0.30, 0.50, 0.70, 0.90):
        p = np.where(p_shock >= thr, POSITIVE, "not_shockable")
        t = binary_scores(y_va_str, p)
        rate = (p == POSITIVE).mean()
        flag = "  <-- floor" if t["sensitivity"] < SENSITIVITY_FLOOR else ""
        print(f"  {thr:<12.2f}{t['sensitivity']:>8.3f}{t['specificity']:>8.3f}"
              f"{t['ppv']:>8.3f}{rate:>12.1%}{flag}")

    rule = CONFIRMATION["vf"]
    c = confirmed_alarms(y_va_str, pred, va_rec, va_off, rule)
    print(f"\nafter {rule[0]}-of-{rule[1]} confirmation")
    print(f"  episodes caught      {c['episodes_caught']}")
    print(f"  episodes missed      {c['episodes_missed']}")
    print(f"  episode sensitivity  {c['episode_sensitivity']:.3f}")
    print(f"  false alarms         {c['false_alarms']}  "
          f"(from {s['fp']} isolated false positives)")

    print("\n" + "=" * 62)
    print(f"  sensitivity  {s['sensitivity']:.3f}  "
          f"(floor {SENSITIVITY_FLOOR:.2f})")
    print(f"  specificity  {s['specificity']:.3f}")
    print("\n  Numbers are on curated arrest recordings, where shockable "
          "rhythm\n  is 28% of windows. Field prevalence is near zero and "
          "field VF looks\n  different. This is a comparison figure, not "
          "expected performance.")
    print(f"\nsaved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
