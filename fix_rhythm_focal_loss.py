"""Give the rhythm classifier the focal loss its model card already describes.

Why
---
The RhythmNet model card lists the loss as

    RhythmLoss(class_weights, af_miss_penalty=1.3, label_smoothing=0.05,
               gamma=2.0)

but RhythmLoss in rhythm_net.py has no gamma and no focal term. fix_focal_loss.py
patched shockable_net.py only. So every rhythm number reported so far came from
plain weighted cross-entropy, and the card describes a loss that did not
produce them.

This brings the code in line with the card, the same way the shockable
detector was changed: identical focal term, gamma exposed, and gamma = 0
reproducing the old loss exactly so the comparison is controlled.

Focal loss down-weights windows the model already gets right, so the gradient
stops being owned by thousands of easy sinus windows and reaches the boundary
cases. For this model that is the OTHER class and the OTHER -> NSR confusion
(740 windows in the pooled matrix). It also regularises confidence, which
matters here because CONFIRMATION["af"] acts on the probability, not the label.

What changes
------------
    munji_ai/models/rhythm_net.py      RhythmLoss gains gamma (default 2.0)
    munji_ai/train/cross_validate.py   --gamma flag, passed to either loss
    munji_ai/train/train_rhythm.py     --gamma flag

Nothing else. Same architecture, data, splits, selection rule and penalty.
The config hash does not move and no cache is invalidated.

The shockable detector is not affected: MissPenaltyLoss already defaults to
gamma 2.0, and --gamma defaults to 2.0, so its runs behave exactly as before.

How to run
----------
From the repo root:

    python fix_rhythm_focal_loss.py
    python tests/test_rhythm_loss.py

Then the controlled comparison, both with the same flags you used for the
reported rhythm run:

    python -m munji_ai.train.cross_validate --stage rhythm --gamma 0
    python -m munji_ai.train.cross_validate --stage rhythm --gamma 2

gamma 0 must reproduce the current numbers (macro F1 NSR+AF 0.925, AF sens
0.920, AF spec 0.973). If it does not, stop: something other than the loss
moved. Then compare gamma 2 against it on AF sensitivity, AF specificity,
OTHER F1, and the OTHER -> NSR cell of the confusion matrix.

Revert with:

    git checkout munji_ai/models/rhythm_net.py munji_ai/train/cross_validate.py munji_ai/train/train_rhythm.py
"""

from pathlib import Path

MODEL = Path("munji_ai/models/rhythm_net.py")
CV = Path("munji_ai/train/cross_validate.py")
TRAIN = Path("munji_ai/train/train_rhythm.py")


# ---------------------------------------------------------------- rhythm_net.py
INIT_OLD = '''    def __init__(self, class_weights: torch.Tensor | None = None,
                 af_miss_penalty: float = 1.3,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(RHYTHM_CLASSES)))
        self.penalty = af_miss_penalty
        self.smoothing = label_smoothing'''

INIT_NEW = '''    def __init__(self, class_weights: torch.Tensor | None = None,
                 af_miss_penalty: float = 1.3,
                 label_smoothing: float = 0.05,
                 gamma: float = 2.0):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(RHYTHM_CLASSES)))
        self.penalty = af_miss_penalty
        self.smoothing = label_smoothing
        # Focusing parameter, as in MissPenaltyLoss. 0 recovers the weighted
        # cross-entropy used before this change exactly.
        self.gamma = gamma'''

FWD_OLD = '''    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")
        pred = logits.argmax(dim=-1)
        missed_af = (target == AF_IDX) & (pred != AF_IDX)
        return (losses * torch.where(missed_af, self.penalty, 1.0)).mean()'''

FWD_NEW = '''    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")

        # Focal modulation, identical to MissPenaltyLoss. (1 - p_true)^gamma
        # shrinks the loss on windows already classified correctly, so the
        # thousands of easy sinus windows stop owning the gradient and the
        # boundary cases (OTHER, and OTHER called NSR) get the capacity.
        if self.gamma > 0:
            p_true = F.softmax(logits, dim=-1).gather(
                1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            losses = losses * (1.0 - p_true) ** self.gamma

        pred = logits.argmax(dim=-1)
        missed_af = (target == AF_IDX) & (pred != AF_IDX)
        return (losses * torch.where(missed_af, self.penalty, 1.0)).mean()'''

DOC_OLD = '''    """Weighted cross-entropy with a bounded weight and a small AF miss cost.'''

DOC_NEW = '''    """Focal-weighted cross-entropy with a bounded weight and a small AF miss cost.

    The focal term decides which windows the model attends to; the miss
    penalty decides what a missed AF window costs. Same split of concerns as
    MissPenaltyLoss in the shockable detector.'''


# ------------------------------------------------------------ cross_validate.py
CV_ARG_OLD = '''    ap.add_argument("--penalty", type=float, default=None)'''
CV_ARG_NEW = '''    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")'''

CV_LOSS_OLD = '''    crit = Loss(class_weights(ytr).to(device),
                **{penalty_kw: args.penalty}).to(device)'''
CV_LOSS_NEW = '''    crit = Loss(class_weights(ytr).to(device),
                **{penalty_kw: args.penalty},
                gamma=args.gamma).to(device)'''

CV_LOG_OLD = '''    print(f"device      {device}")'''
CV_LOG_NEW = '''    print(f"device      {device}")
    print(f"gamma       {a.gamma}  (focal; 0 = weighted cross-entropy)")'''


# -------------------------------------------------------------- train_rhythm.py
TR_ARG_OLD = '''    ap.add_argument("--penalty", type=float, default=1.3)'''
TR_ARG_NEW = '''    ap.add_argument("--penalty", type=float, default=1.3)
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal loss gamma; 0 = plain weighted cross-entropy")'''

TR_LOSS_OLD = '''    crit = RhythmLoss(w.to(device), af_miss_penalty=a.penalty).to(device)'''
TR_LOSS_NEW = '''    crit = RhythmLoss(w.to(device), af_miss_penalty=a.penalty,
                      gamma=a.gamma).to(device)'''


PLAN = [
    # (file, old, new, label, required)
    (MODEL, INIT_OLD, INIT_NEW, "rhythm_net    gamma parameter", True),
    (MODEL, FWD_OLD, FWD_NEW, "rhythm_net    focal modulation", True),
    (MODEL, DOC_OLD, DOC_NEW, "rhythm_net    docstring", False),
    (CV, CV_ARG_OLD, CV_ARG_NEW, "cross_validate --gamma flag", True),
    (CV, CV_LOSS_OLD, CV_LOSS_NEW, "cross_validate gamma into loss", True),
    (CV, CV_LOG_OLD, CV_LOG_NEW, "cross_validate gamma in header", False),
    (TRAIN, TR_ARG_OLD, TR_ARG_NEW, "train_rhythm  --gamma flag", True),
    (TRAIN, TR_LOSS_OLD, TR_LOSS_NEW, "train_rhythm  gamma into loss", True),
]


def read(path: Path) -> tuple[str, str]:
    """Return (text with LF endings, original line ending).

    Read without newline translation so a CRLF file (Windows checkout) is
    detected and written back as CRLF, instead of every line showing as
    changed in git.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write(path: Path, text: str, eol: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


def main() -> int:
    missing = [p for p in (MODEL, CV, TRAIN) if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING  {p}")
        print("run this from the repo root")
        return 1

    files = {p: read(p) for p in (MODEL, CV, TRAIN)}
    text = {p: t for p, (t, _) in files.items()}

    if "Focal modulation" in text[MODEL] and "--gamma" in text[CV]:
        print("already applied — nothing to do")
        return 0

    print("applying\n")
    ok = True
    for path, old, new, label, required in PLAN:
        if new in text[path]:
            print(f"  present  {label}")
            continue
        if text[path].count(old) == 1:
            text[path] = text[path].replace(old, new, 1)
            print(f"  fixed    {label}")
        else:
            print(f"  FAILED   {label} — expected code not found"
                  + ("" if required else " (optional, skipped)"))
            ok = ok and not required

    if not ok:
        print("\nnothing written — all three files are untouched")
        return 1

    for path, (_, eol) in files.items():
        write(path, text[path], eol)

    print("\napplied. config hash unchanged, no cache invalidated.\n")
    print("now run:")
    print("  python tests/test_rhythm_loss.py")
    print("  python -m munji_ai.train.cross_validate --stage rhythm --gamma 0")
    print("  python -m munji_ai.train.cross_validate --stage rhythm --gamma 2")
    print("\ngamma 0 must reproduce macro F1 0.925 / AF sens 0.920 / AF spec 0.973.")
    print("Revert with:")
    print("  git checkout munji_ai/models/rhythm_net.py "
          "munji_ai/train/cross_validate.py munji_ai/train/train_rhythm.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
