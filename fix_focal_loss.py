"""Train the shockable detector with focal loss instead of cross-entropy.

Why
---
Sinus specificity is 0.952 against an AHA target of 0.99. Eight attempts at
closing that gap by changing the data have been measured; two helped. What the
failure inspection showed is not a model that cannot tell the classes apart —
it is a model that is sure of itself while wrong. Patient after patient came
back marked "Confidently wrong", with mean confidence above 0.6 on the windows
it got wrong.

That is a calibration problem, not a capacity problem, and cross-entropy is
known to cause it: a high-capacity network overfits the negative log-likelihood
and ends up overconfident.

Focal loss addresses it directly. It down-weights examples the model already
gets right, so the gradient stops being dominated by the thousands of obvious
sinus windows and goes to the boundary cases instead. And it improves
calibration as a side effect, acting as an implicit entropy regulariser that
avoids the low-entropy, overconfident predictions cross-entropy produces.

It is also the standard choice in this field rather than an invention here: a
2026 comparative evaluation of ECG architectures trains every model with focal
loss, AdamW and ReduceLROnPlateau, at alpha 1 and gamma 2.

What changes
------------
Only the loss. Same architecture, same data, same splits, same selection rule,
same sensitivity floor. The config hash does not move, no cache is
invalidated, and nothing of Furat's is touched — this edits shockable_net.py
and nothing else.

The miss penalty stays. Focal loss handles which examples the model attends
to; the penalty encodes that missing a fibrillating heart and raising a false
alarm are not equally bad. They are separate concerns and both are needed.

gamma is exposed rather than fixed:

    gamma = 0    identical to the weighted cross-entropy used until now
    gamma = 2    the published default
    gamma > 3    sharper focus on hard cases, less stable

Starting at 2 and comparing against gamma 0 is a controlled test, since gamma 0
reproduces the current behaviour exactly.

How to read the result
----------------------
Specificity should rise. Watch sensitivity in the same breath: focal loss can
quietly de-emphasise the positive class once the model finds it easy, and a
detector that gains specificity by missing arrests is worse than the one it
replaced. The 0.90 floor in the selection rule guards against that, but check
the per-fold minimum too.

Run from the repo root:

    python fix_focal_loss.py
    python -m munji_ai.train.cross_validate --stage shockable
    python -m munji_ai.eval.report_by_rhythm

Compare sinus specificity against 0.952. Revert with:

    git checkout munji_ai/models/shockable_net.py
"""

from pathlib import Path
import sys

MODEL = Path("munji_ai/models/shockable_net.py")

OLD = '''    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")
        pred = logits.argmax(dim=-1)
        missed = (target == SHOCKABLE_IDX) & (pred != SHOCKABLE_IDX)
        return (losses * torch.where(missed, self.penalty, 1.0)).mean()'''

NEW = '''    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")

        # Focal modulation. Cross-entropy keeps pushing on windows the model
        # already classifies correctly, and with thousands of obvious sinus
        # windows that is where most of the gradient goes. The (1 - p)^gamma
        # term shrinks their contribution so the boundary cases get the
        # capacity instead.
        #
        # It also regularises confidence. The failure inspection found the
        # model was "confidently wrong" on most of its weak patients — mean
        # confidence above 0.6 while mistaken — which is the overconfidence
        # cross-entropy is known to produce in high-capacity networks.
        #
        # gamma = 0 reduces this to exactly the weighted cross-entropy used
        # before, which makes it a controlled comparison rather than a
        # rewrite.
        if self.gamma > 0:
            p_true = F.softmax(logits, dim=-1).gather(
                1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            losses = losses * (1.0 - p_true) ** self.gamma

        pred = logits.argmax(dim=-1)
        missed = (target == SHOCKABLE_IDX) & (pred != SHOCKABLE_IDX)
        return (losses * torch.where(missed, self.penalty, 1.0)).mean()'''

OLD_INIT = '''    def __init__(self, class_weights: torch.Tensor | None = None,
                 miss_penalty: float = 2.5,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(SHOCKABLE_CLASSES)))
        self.penalty = miss_penalty
        self.smoothing = label_smoothing'''

NEW_INIT = '''    def __init__(self, class_weights: torch.Tensor | None = None,
                 miss_penalty: float = 2.5,
                 label_smoothing: float = 0.05,
                 gamma: float = 2.0):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(SHOCKABLE_CLASSES)))
        self.penalty = miss_penalty
        self.smoothing = label_smoothing
        # Focusing parameter. 2.0 is the published default and what the 2026
        # comparative evaluation of ECG architectures uses throughout. Set it
        # to 0 to recover the previous weighted cross-entropy exactly.
        self.gamma = gamma'''

DOC_OLD = '''    The penalty is larger than the gate's 1.5 because the consequences are not
    comparable'''

DOC_NEW = '''    Since the detector moved to focal loss, two mechanisms act together and
    should not be confused. The focal term decides which examples the model
    attends to — it stops the easy majority from owning the gradient. The miss
    penalty decides what an error costs once made. One is about attention, the
    other about consequence.

    The penalty is larger than the gate's 1.5 because the consequences are not
    comparable'''


def patch(text: str, old: str, new: str, label: str):
    for o, n in ((old, new),
                 (old.replace("\\n", "\\r\\n"), new.replace("\\n", "\\r\\n"))):
        if o in text:
            print(f"  fixed    {label}")
            return text.replace(o, n, 1), True
    print(f"  FAILED   {label} — expected code not found")
    return text, False


def main() -> int:
    if not MODEL.exists():
        print(f"MISSING  {MODEL} — run this from the repo root")
        return 1

    text = MODEL.read_text()
    if "Focal modulation" in text:
        print("already applied — nothing to do")
        return 0

    print("applying\n")
    ok = True
    text, h = patch(text, OLD_INIT, NEW_INIT, "1  gamma parameter")
    ok = ok and h
    text, h = patch(text, OLD, NEW, "2  focal modulation in the loss")
    ok = ok and h
    text, _ = patch(text, DOC_OLD, DOC_NEW, "3  docstring note")

    if not ok:
        print("\nnothing written — the model file is untouched")
        return 1

    MODEL.write_text(text)
    print("\napplied. config hash unchanged, no cache invalidated.\n")
    print("now run:")
    print("  python -m munji_ai.train.cross_validate --stage shockable")
    print("  python -m munji_ai.eval.report_by_rhythm")
    print("\nCompare sinus specificity against 0.952, and check the per-fold")
    print("minimum sensitivity has not fallen below 0.90. Revert with:")
    print("  git checkout munji_ai/models/shockable_net.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
