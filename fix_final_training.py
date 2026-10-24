"""Bring the final training scripts in line with what cross-validation adopted.

Why
---
Every improvement so far was measured with cross_validate.py, but the models
that ship come from train_rhythm.py and train_shockable.py, which still train
the old way. A final model has to be trained exactly as it was validated:

    RhythmNet      focal loss (already) + training augmentation
    ShockableNet   training augmentation + Icentia11k hard negatives
                   + operating threshold 0.65 with the 2-of-3 rule

What changes
------------
    train_rhythm.py      --augment
    train_shockable.py   --augment, --extra-negatives, --threshold (0.65)

Both checkpoints now record how they were trained (augment, gamma, extra
negatives, threshold, confirmation rule), so a model file says which
operating point it was validated for. Without the new flags both scripts
behave exactly as before. config.py is not touched: the threshold travels
with the checkpoint until the decision layer reads it.

Selection is unchanged and still never looks at the test split.

How to run
----------
From the repo root:

    python fix_final_training.py
    python tests/test_final_training.py

Revert with:

    git checkout munji_ai/train/train_rhythm.py munji_ai/train/train_shockable.py
"""

from pathlib import Path

RH = Path("munji_ai/train/train_rhythm.py")
SH = Path("munji_ai/train/train_shockable.py")
NEEDS = Path("munji_ai/data/train_augment.py")

AUG_BLOCK = r'''    aug = None
    if a.augment:
        from ..data.train_augment import (BANK_NAME, TrainAugment,
                                          load_noise_bank, split_bank)
        bank_train, _ = split_bank(load_noise_bank(a.cache / BANK_NAME))
        aug = TrainAugment(bank_train, device, seed=a.seed)
        print("augment     on (noise, jitter, cutout), training batches only")
'''

LOOP_OLD = r'''        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()'''
LOOP_NEW = r'''        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad()'''

# ------------------------------------------------------------------ rhythm
RH_ARG_OLD = r'''    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")'''
RH_ARG_NEW = RH_ARG_OLD + r'''
    ap.add_argument("--augment", action="store_true",
                    help="NSTDB noise, jitter and cutout on training batches")'''

RH_MODEL_OLD = r'''    model = build().to(device)
    w = class_weights(tr["y"])
'''
RH_MODEL_NEW = RH_MODEL_OLD + AUG_BLOCK

RH_SAVE_OLD = r'''    torch.save({"state_dict": best_state, "classes": list(RHYTHM_CLASSES),
                "val_macro_f1_scored": best_f1, "epoch": best_epoch}, ckpt)'''
RH_SAVE_NEW = r'''    torch.save({"state_dict": best_state, "classes": list(RHYTHM_CLASSES),
                "val_macro_f1_scored": best_f1, "epoch": best_epoch,
                "gamma": a.gamma, "augment": a.augment,
                "confirmation": tuple(CONFIRMATION["af"])}, ckpt)'''

# --------------------------------------------------------------- shockable
SH_ARG_OLD = r'''    ap.add_argument("--penalty", type=float, default=2.5)'''
SH_ARG_NEW = SH_ARG_OLD + r'''
    ap.add_argument("--augment", action="store_true",
                    help="NSTDB noise, jitter and cutout on training batches")
    ap.add_argument("--extra-negatives", type=Path, default=None,
                    help=".npz of extra negative windows for training only "
                         "(data/hard_negatives.py)")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="operating threshold on P(shockable), validated on full "
                         "days with eval/false_alarm_rate.py")'''

SH_EXTRA_OLD = r'''    overlap = set(tr["patient"].astype(str)) & set(va["patient"].astype(str))
    if overlap:
        raise AssertionError(f"patient leak between train and val: {overlap}")
'''
SH_EXTRA_NEW = SH_EXTRA_OLD + r'''
    if a.extra_negatives:
        with np.load(a.extra_negatives, allow_pickle=False) as e:
            ex = {k: e[k] for k in ("X", "y", "patient")}
        if set(ex["y"].astype(str)) - {"not_shockable"}:
            raise SystemExit("extra negatives must all be not_shockable")
        shared = set(ex["patient"].astype(str)) & (
            set(tr["patient"].astype(str)) | set(va["patient"].astype(str)))
        if shared:
            raise SystemExit(f"extra negatives share {len(shared)} patients with "
                             f"train or val - that would leak")
        tr = {"X": np.concatenate([tr["X"], ex["X"].astype(tr["X"].dtype)]),
              "y": np.concatenate([np.asarray(tr["y"], dtype=str),
                                   ex["y"].astype(str)]),
              "patient": np.concatenate([np.asarray(tr["patient"], dtype=str),
                                         ex["patient"].astype(str)])}
        print(f"extra neg   {len(ex['X']):,} windows from "
              f"{len(set(ex['patient'].astype(str)))} patients, train only")
'''

SH_MODEL_OLD = r'''    model = build().to(device)
    print(f"parameters  {model.n_params():,}")
'''
SH_MODEL_NEW = SH_MODEL_OLD + AUG_BLOCK

SH_SAVE_OLD = r'''    torch.save({"state_dict": best_state, "classes": list(SHOCKABLE_CLASSES),
                "val_specificity": best_spec, "epoch": best_epoch}, ckpt)'''
SH_SAVE_NEW = r'''    torch.save({"state_dict": best_state, "classes": list(SHOCKABLE_CLASSES),
                "val_specificity": best_spec, "epoch": best_epoch,
                "threshold": a.threshold, "augment": a.augment,
                "extra_negatives": str(a.extra_negatives) if a.extra_negatives else None,
                "confirmation": tuple(CONFIRMATION["vf"])}, ckpt)'''

SH_THR_OLD = r'''    for thr in (0.30, 0.50, 0.70, 0.90):
        p = np.where(p_shock >= thr, POSITIVE, "not_shockable")'''
SH_THR_NEW = r'''    for thr in sorted({0.30, 0.50, a.threshold, 0.70, 0.90}):
        p = np.where(p_shock >= thr, POSITIVE, "not_shockable")'''

SH_OP_OLD = r'''    print("\n" + "=" * 62)
    print(f"  sensitivity  {s['sensitivity']:.3f}  "'''
SH_OP_NEW = r'''    p_op = np.where(p_shock >= a.threshold, POSITIVE, "not_shockable")
    c_op = confirmed_alarms(y_va_str, p_op, va_rec, va_off, rule)
    s_op = binary_scores(y_va_str, p_op)
    print(f"\nat the operating threshold {a.threshold:.2f}")
    print(f"  sensitivity          {s_op['sensitivity']:.3f}  "
          f"(floor {SENSITIVITY_FLOOR:.2f})")
    print(f"  specificity          {s_op['specificity']:.3f}")
    print(f"  episodes caught      {c_op['episodes_caught']}/"
          f"{c_op['episodes_caught'] + c_op['episodes_missed']}")
    print(f"  false alarms         {c_op['false_alarms']}")

    print("\n" + "=" * 62)
    print(f"  sensitivity  {s['sensitivity']:.3f}  "'''

PLAN = {
    RH: [(RH_ARG_OLD, RH_ARG_NEW, "--augment flag"),
         (RH_MODEL_OLD, RH_MODEL_NEW, "augmentation set up"),
         (LOOP_OLD, LOOP_NEW, "augmentation in the training loop"),
         (RH_SAVE_OLD, RH_SAVE_NEW, "checkpoint records how it was trained")],
    SH: [(SH_ARG_OLD, SH_ARG_NEW, "--augment, --extra-negatives, --threshold"),
         (SH_EXTRA_OLD, SH_EXTRA_NEW, "extra negatives, leakage checked"),
         (SH_MODEL_OLD, SH_MODEL_NEW, "augmentation set up"),
         (LOOP_OLD, LOOP_NEW, "augmentation in the training loop"),
         (SH_SAVE_OLD, SH_SAVE_NEW, "checkpoint records threshold and rule"),
         (SH_THR_OLD, SH_THR_NEW, "operating threshold in the table"),
         (SH_OP_OLD, SH_OP_NEW, "report at the operating threshold")],
}


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    for p in (RH, SH):
        if not p.exists():
            print(f"MISSING  {p}\nrun this from the repo root")
            return 1
    if not NEEDS.exists():
        print("STOP  the augmentation change (PR #2) is not on this branch.")
        return 1
    files = {p: read(p) for p in (RH, SH)}
    text = {p: t for p, (t, _) in files.items()}
    if "--augment" in text[RH] and "--extra-negatives" in text[SH]:
        print("already applied - nothing to do")
        return 0
    print("applying\n")
    ok = True
    for path, steps in PLAN.items():
        for old, new, label in steps:
            tag = f"{path.stem:<16}"
            if new in text[path]:
                print(f"  present  {tag}{label}")
            elif text[path].count(old) == 1:
                text[path] = text[path].replace(old, new, 1)
                print(f"  fixed    {tag}{label}")
            else:
                print(f"  FAILED   {tag}{label} - expected code not found")
                ok = False
    if not ok:
        print("\nnothing written - both files are untouched")
        return 1
    for path, (_, eol) in files.items():
        write(path, text[path], eol)
    print("\napplied. config, caches and models are untouched.\n")
    print("now run:")
    print("  python tests/test_final_training.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
