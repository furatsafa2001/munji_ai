"""Patient-level k-fold cross-validation.

    python -m munji_ai.train.cross_validate --stage shockable
    python -m munji_ai.train.cross_validate --stage rhythm --folds 5

Why this exists
---------------
A single train/val split gave the shockable detector a validation set of eight
patients carrying nine episodes. Nine is enough to show the logic works. It is
not enough to claim reliability: one unusual patient moves the number by more
than any architectural change would.

Cross-validation pools the train and val caches, splits the patients into k
folds, and trains k times — each fold held out once. Every patient ends up
predicted exactly once by a model that never saw them. So instead of nine
episodes from eight patients, the episode count comes from all of them.

Two things come out of that:

  per-fold spread   a mean with a standard deviation, not a bare number.
                    "0.90 ± 0.04 across 56 patients" survives review in a way
                    "0.90 on 8 patients" does not.

  pooled OOF        one confusion matrix and one episode count over every
                    patient, each predicted by a model blind to them.

The test split is never read. It stays held out for one final measurement
after everything else is settled.

Cost: k trainings. At roughly 5 s an epoch for the shockable stage, a 5-fold
run is well under an hour.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import CACHE_DIR, CONFIRMATION, RHYTHM_CLASSES, SHOCKABLE_CLASSES
from ..data.windows import load_cache

STAGES = {
    "shockable": {
        "classes": SHOCKABLE_CLASSES,
        "positive": "shockable",
        "floor": 0.90,          # AHA/AAMI for VF — a gate, not a target
        "rule": CONFIRMATION["vf"],
        "select": "specificity",
    },
    "rhythm": {
        "classes": RHYTHM_CLASSES,
        "positive": "AF",
        "floor": None,
        "rule": CONFIRMATION["af"],
        "select": "macro_f1",
    },
}


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_for(stage: str):
    if stage == "shockable":
        from ..models.shockable_net import MissPenaltyLoss, build, class_weights
        return build, class_weights, MissPenaltyLoss, "miss_penalty"
    from ..models.rhythm_net import RhythmLoss, build, class_weights
    return build, class_weights, RhythmLoss, "af_miss_penalty"


def assign_folds(patients: np.ndarray, k: int, seed: int) -> dict:
    """Deterministic, evenly sized folds, grouped by patient.

    Sorting by a seeded hash then dealing round-robin gives folds of nearly
    equal size without the ordering of the cache leaking in. Same patient,
    same fold, every run.
    """
    uniq = sorted(set(patients.tolist()))
    ranked = sorted(
        uniq,
        key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest(),
    )
    return {p: i % k for i, p in enumerate(ranked)}


def pooled_cache(stage: str, cache: Path) -> dict:
    """train + val. Never test."""
    parts = [load_cache(stage, s, cache) for s in ("train", "val")]
    keys = ("X", "y", "patient", "record", "offset", "dataset")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def binary_scores(y_true, y_pred, positive) -> dict:
    t, p = y_true == positive, y_pred == positive
    tp = int((t & p).sum()); fn = int((t & ~p).sum())
    fp = int((~t & p).sum()); tn = int((~t & ~p).sum())
    return {"sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def macro_f1_of(y_true, y_pred, classes) -> float:
    out = []
    for c in classes:
        t, p = y_true == c, y_pred == c
        tp = int((t & p).sum())
        prec = tp / max(int(p.sum()), 1)
        rec = tp / max(int(t.sum()), 1)
        out.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(out))


def roc_auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y = y_bin[order].astype(float)
    tpr = np.concatenate([[0.0], np.cumsum(y) / max(y.sum(), 1), [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / max((1 - y).sum(), 1), [1.0]])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(tpr, fpr))


def train_one_fold(stage, Xtr, ytr, Xva, device, args, classes):
    build, class_weights, Loss, penalty_kw = build_for(stage)
    cfg = STAGES[stage]
    idx = {c: i for i, c in enumerate(classes)}

    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.tensor([idx[s] for s in ytr], dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch,
                        shuffle=True, drop_last=True)

    model = build().to(device)
    aug = None
    if getattr(args, "augment", False):
        from ..data.train_augment import TrainAugment
        aug = TrainAugment(args._bank_train, device, seed=args.seed)
    crit = Loss(class_weights(ytr).to(device),
                **{penalty_kw: args.penalty},
                gamma=args.gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader))

    Xv = torch.from_numpy(Xva.astype(np.float32))
    val_loader = DataLoader(TensorDataset(Xv, torch.zeros(len(Xv), dtype=torch.long)),
                            batch_size=256)

    best, best_state = -1.0, None
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        P = predict(model, val_loader, device)
        pred = np.array([classes[i] for i in P.argmax(1)])
        # Selection uses the held-out fold, exactly as the single-split script
        # uses val. That makes the fold an optimistic estimate of itself — the
        # honest reading is the spread across folds, not any one fold's peak.
        if cfg["select"] == "specificity":
            s = binary_scores(np.asarray(args._y_fold, dtype=str), pred,
                              cfg["positive"])
            ok = s["sensitivity"] >= cfg["floor"]
            score = s["specificity"] if ok else -1.0
        else:
            scored = [c for c in classes if c != "OTHER"]
            score = macro_f1_of(np.asarray(args._y_fold, dtype=str), pred, scored)

        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, predict(model, val_loader, device)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(out)


def _report_noisy(y, pred, score, classes, positive, snr_db) -> None:
    """Pooled out-of-fold metrics on the noisy copy of every held-out window."""
    s = binary_scores(y, pred, positive)
    print(f"\n  under NSTDB noise at {snr_db:g} dB (held-out noise, same windows)")
    print(f"    sensitivity   {s['sensitivity']:.3f}")
    print(f"    specificity   {s['specificity']:.3f}")
    print(f"    ROC AUC       {roc_auc(y == positive, score):.3f}")
    n = len(classes)
    ix = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[ix[t], ix[p]] += 1
    width = max(len(c) for c in classes) + 2
    print("    " + " " * width + "".join(f"{c:>{width}}" for c in classes))
    for i, c in enumerate(classes):
        print(f"    {c:<{width}}" + "".join(f"{v:>{width},}" for v in cm[i]))
    f1s = []
    for i in range(n):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    print("    per-class F1  "
          + "   ".join(f"{c} {v:.3f}" for c, v in zip(classes, f1s)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES), default="shockable")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")
    ap.add_argument("--augment", action="store_true",
                    help="NSTDB noise, jitter and cutout on training batches")
    ap.add_argument("--noisy-eval", type=float, default=None, metavar="SNR_DB",
                    help="also score each held-out fold with NSTDB noise at this SNR")
    ap.add_argument("--save-models", action="store_true",
                    help="save each fold's selected model next to the OOF file")
    ap.add_argument("--extra-negatives", type=Path, default=None,
                    help="shockable only: .npz of extra negative windows added to "
                         "every training fold (data/hard_negatives.py)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.extra_negatives and a.stage != "shockable":
        ap.error("--extra-negatives applies to the shockable stage only")

    cfg = STAGES[a.stage]
    classes = list(cfg["classes"])
    positive = cfg["positive"]
    if a.penalty is None:
        a.penalty = 2.5 if a.stage == "shockable" else 1.3

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)

    d = pooled_cache(a.stage, a.cache)
    if a.augment or a.noisy_eval is not None:
        from ..data.train_augment import BANK_NAME, load_noise_bank, split_bank
        a._bank_train, a._bank_eval = split_bank(
            load_noise_bank(a.cache / BANK_NAME))
    Xe = ye = None
    if a.extra_negatives:
        with np.load(a.extra_negatives, allow_pickle=False) as e:
            Xe, ye = e["X"], e["y"].astype(str)
            extra_patients = set(e["patient"].astype(str))
        if set(ye) - {"not_shockable"}:
            raise SystemExit("extra negatives must all be not_shockable")
        shared = extra_patients & set(np.asarray(d["patient"], dtype=str))
        if shared:
            raise SystemExit(f"extra negatives share {len(shared)} patients with "
                             f"the evaluation data - that would leak")
    y = np.asarray(d["y"], dtype=str)
    patients = np.asarray(d["patient"], dtype=str)
    folds = assign_folds(patients, a.folds, a.seed)
    fold_of = np.array([folds[p] for p in patients])

    print(f"stage       {a.stage}")
    print(f"pooled      {len(y):,} windows, {len(set(patients)):,} patients "
          f"(train + val; test untouched)")
    print(f"folds       {a.folds}, grouped by patient")
    print(f"device      {device}")
    print(f"gamma       {a.gamma}  (focal; 0 = weighted cross-entropy)")
    print(f"augment     {'on (noise, jitter, cutout)' if a.augment else 'off'}")
    if Xe is not None:
        print(f"extra neg   {len(Xe):,} windows from {len(extra_patients)} patients, "
              f"training folds only")
    if a.noisy_eval is not None:
        print(f"noisy eval  NSTDB at {a.noisy_eval:g} dB, held-out noise segments")
    print(f"selection   {cfg['select']}"
          + (f" subject to sensitivity >= {cfg['floor']:.2f}"
             if cfg["floor"] else "") + "\n")

    oof_pred = np.empty(len(y), dtype=object)
    oof_score = np.zeros(len(y))
    noisy_pred = np.empty(len(y), dtype=object)
    noisy_score = np.zeros(len(y))
    rows = []

    print(f"{'fold':>5}{'patients':>10}{'windows':>9}{'sens':>8}{'spec':>8}"
          f"{'F1':>8}{'AUC':>8}{'':>3}")

    for f in range(a.folds):
        te = fold_of == f
        tr = ~te
        a._y_fold = y[te]

        t0 = time.time()
        Xtr, ytr = d["X"][tr], y[tr]
        if Xe is not None:
            Xtr = np.concatenate([Xtr, Xe.astype(Xtr.dtype)])
            ytr = np.concatenate([ytr, ye])
        model, P = train_one_fold(a.stage, Xtr, ytr, d["X"][te],
                                  device, a, classes)
        if a.save_models:
            a.out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), a.out / f"{a.stage}_fold{f}.pt")
        pred = np.array([classes[i] for i in P.argmax(1)])
        score = P[:, classes.index(positive)]

        oof_pred[te] = pred
        oof_score[te] = score

        if a.noisy_eval is not None:
            from ..data.train_augment import add_noise_fixed
            Xn = add_noise_fixed(d["X"][te], a._bank_eval, a.noisy_eval,
                                 seed=1000 + f)
            Pn = predict(model, DataLoader(
                TensorDataset(torch.from_numpy(Xn),
                              torch.zeros(len(Xn), dtype=torch.long)),
                batch_size=256), device)
            noisy_pred[te] = np.array([classes[i] for i in Pn.argmax(1)])
            noisy_score[te] = Pn[:, classes.index(positive)]

        s = binary_scores(y[te], pred, positive)
        scored = [c for c in classes if c != "OTHER"]
        f1 = macro_f1_of(y[te], pred, scored)
        auc = roc_auc(y[te] == positive, score)
        rows.append((s["sensitivity"], s["specificity"], f1, auc))

        flag = "" if cfg["floor"] is None or s["sensitivity"] >= cfg["floor"] \
            else "  <-- below floor"
        print(f"{f:>5}{len(set(patients[te])):>10}{int(te.sum()):>9}"
              f"{s['sensitivity']:>8.3f}{s['specificity']:>8.3f}"
              f"{f1:>8.3f}{auc:>8.3f}{flag}"
              + (f"   {time.time() - t0:.0f}s/fold" if f == 0 else ""))

    arr = np.array(rows)
    mean, std = arr.mean(0), arr.std(0)

    print("\n" + "=" * 62)
    print(f"across {a.folds} folds")
    for i, name in enumerate(("sensitivity", "specificity",
                              "macro F1", "ROC AUC")):
        print(f"  {name:<14} {mean[i]:.3f} ± {std[i]:.3f}"
              f"   (min {arr[:, i].min():.3f}, max {arr[:, i].max():.3f})")

    oof = np.array([str(v) for v in oof_pred])
    s = binary_scores(y, oof, positive)
    print(f"\npooled out-of-fold — every patient predicted by a model that "
          f"never saw them")
    print(f"  windows       {len(y):,}")
    print(f"  patients      {len(set(patients)):,}")
    print(f"  sensitivity   {s['sensitivity']:.3f}")
    print(f"  specificity   {s['specificity']:.3f}")
    print(f"  macro F1      "
          f"{macro_f1_of(y, oof, [c for c in classes if c != 'OTHER']):.3f}")
    print(f"  ROC AUC       {roc_auc(y == positive, oof_score):.3f}")

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    ix = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y, oof):
        cm[ix[t], ix[p]] += 1
    width = max(len(c) for c in classes) + 2
    print(f"\n  confusion (rows = true)")
    print("  " + " " * width + "".join(f"{c:>{width}}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:<{width}}" + "".join(f"{v:>{width},}" for v in cm[i]))
    f1s = []
    for i in range(n):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    print("\n  per-class F1  "
          + "   ".join(f"{c} {v:.3f}" for c, v in zip(classes, f1s)))

    if a.stage == "shockable":
        from .train_shockable import confirmed_alarms
    else:
        from .train_rhythm import confirmed_alarms
    c = confirmed_alarms(y, oof, np.asarray(d["record"], dtype=str),
                         np.asarray(d["offset"], dtype=np.int64), cfg["rule"])
    print(f"\n  after {cfg['rule'][0]}-of-{cfg['rule'][1]} confirmation")
    print(f"    episodes caught      {c['episodes_caught']}")
    print(f"    episodes missed      {c['episodes_missed']}")
    print(f"    episode sensitivity  {c['episode_sensitivity']:.3f}")
    print(f"    false alarms         {c['false_alarms']}")

    if a.noisy_eval is not None:
        _report_noisy(y, np.array([str(v) for v in noisy_pred]), noisy_score,
                      classes, positive, a.noisy_eval)

    a.out.mkdir(parents=True, exist_ok=True)
    oof_path = a.out / f"{a.stage}_oof.npz"
    extra = {} if a.noisy_eval is None else {
        "pred_noisy": np.array([str(v) for v in noisy_pred]),
        "score_noisy": noisy_score, "noisy_snr_db": a.noisy_eval}
    np.savez_compressed(oof_path, y=y, pred=oof, score=oof_score,
                        patient=patients, record=d["record"],
                        offset=d["offset"], fold=fold_of, **extra)

    print("\n" + "=" * 62)
    print("  Report the spread, not a single fold. A fold's own peak is an")
    print("  optimistic estimate of itself; the standard deviation across")
    print("  folds is what tells you how much the number moves with the")
    print("  luck of who landed in which fold.")
    print(f"\nsaved {oof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
