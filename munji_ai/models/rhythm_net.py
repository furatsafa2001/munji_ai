"""Model 2 — rhythm classifier.

A 1D CNN that labels a 30-second window as normal sinus rhythm, atrial
fibrillation, or other. AF is the reason this model exists: it is the largest
single modifiable risk factor for stroke, it is often paroxysmal and therefore
missed by a clinic ECG, and continuous monitoring is the only way to catch it.

Why 30 seconds when the other models use 5
------------------------------------------
AF is not a shape you can see in one beat. It is an absence — no P wave — and
an irregularity: the interval between beats varies with no pattern. Judging
"no pattern" needs enough beats to establish that no pattern exists. At a
resting rate a 5-second window holds five or six beats, which is not enough to
distinguish irregular from merely variable. Thirty seconds holds thirty-odd,
and that is the shortest window on which the judgement is reliable.

The consequence is a six-times longer input, so the stack is one block deeper
and the dilations run further. The receptive field has to span many beats, not
one complex.

What the network has to learn implicitly
-----------------------------------------
Classical AF detectors compute RR intervals first, then measure their
irregularity. This model gets the raw trace, so it has to find the beats and
judge their spacing on its own. That is a real cost — it makes the model
harder to interrogate than an RR-statistics detector — and a real benefit: no
R-peak detector to fail under motion artifact, which is the single most common
way an ambulatory pipeline breaks.

The OTHER class
---------------
Not a rhythm. A bucket holding atrial flutter, supraventricular tachycardia,
bradycardia and block — conditions with nothing in common except that they are
neither sinus nor fibrillation. Training data for it is scarce (tens of
windows against thousands), and it cannot be synthesised: interpolating
between an atrial flutter and a bradycardia produces a trace that does not
exist in nature.

So the class is kept, but it does not drive selection and its score is
reported separately. Keeping it means the model has somewhere to put a rhythm
it does not recognise. Dropping it would force every flutter and every SVT
into NSR or AF, and calling a real arrhythmia "normal" is the worse failure.

Shape: (batch, 1, 7500) -> (batch, 3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RHYTHM_CLASSES, window_samples

AF_IDX = RHYTHM_CLASSES.index("AF")
OTHER_IDX = RHYTHM_CLASSES.index("OTHER")

# Classes the model is actually held to. OTHER is scored and reported but does
# not decide which epoch is kept — see the note above.
SCORED_CLASSES = tuple(c for c in RHYTHM_CLASSES if c != "OTHER")


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> GELU -> pool.

    A local copy rather than an import from the other models, for the same
    reason they keep their own: these three are tuned against different
    failures and should not move together by accident.
    """

    def __init__(self, c_in: int, c_out: int, k: int = 7, dilation: int = 1,
                 pool: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=(k // 2) * dilation,
                              dilation=dilation, bias=False)
        self.norm = nn.BatchNorm1d(c_out)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.pool(F.gelu(self.norm(self.conv(x)))))


class RhythmNet(nn.Module):
    """1D CNN over a raw normalised 30-second window.

    Six blocks with dilations doubling to 32. After pooling, one activation at
    the deepest layer sees several seconds of trace — long enough to compare
    one beat interval against the next, which is the whole question for AF.

    Mean, max and standard deviation are pooled together. Standard deviation
    carries most of the signal here: regular rhythm produces evenly spaced
    activations and low variance, fibrillation produces uneven ones and high
    variance. That is irregularity measured directly, without ever detecting a
    beat.
    """

    def __init__(self, n_classes: int = len(RHYTHM_CLASSES),
                 widths=(16, 32, 64, 96, 128, 160), dropout: float = 0.1):
        super().__init__()
        dilations = (1, 2, 4, 8, 16, 32)
        chans = (1,) + tuple(widths)
        self.blocks = nn.Sequential(*[
            ConvBlock(chans[i], chans[i + 1], dilation=dilations[i],
                      dropout=dropout)
            for i in range(len(widths))
        ])
        self.head = nn.Sequential(
            nn.Linear(widths[-1] * 3, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, n_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.blocks(x)
        pooled = torch.cat(
            [h.mean(dim=-1), h.amax(dim=-1), h.std(dim=-1)], dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def predict_proba(self, x) -> torch.Tensor:
        """P(AF) per window.

        Graded, not thresholded: config.CONFIRMATION["af"] wants 2 of 3
        consecutive positives before an alert, so the caller needs the score
        rather than a decision.
        """
        self.eval()
        return F.softmax(self(x), dim=-1)[:, AF_IDX]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RhythmLoss(nn.Module):
    """Weighted cross-entropy with a bounded weight and a small AF miss cost.

    Two guards, both about not letting a rare class wreck training.

    Inverse-frequency weighting would give OTHER a weight near eighty, since it
    is well under one percent of the data. The model would then spend its
    capacity chasing a few dozen windows and lose the classes that matter, so
    weights are capped.

    The AF penalty is small — 1.3, against 2.5 in the shockable detector. The
    asymmetry is real but not comparable: a missed fibrillation is a stroke
    risk left unflagged over months, not a death in minutes, and there will be
    thousands more windows from the same patient. Over-weighting it would trade
    a modest gain in recall for a flood of false alarms that gets the monitor
    ignored.
    """

    def __init__(self, class_weights: torch.Tensor | None = None,
                 af_miss_penalty: float = 1.3,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(RHYTHM_CLASSES)))
        self.penalty = af_miss_penalty
        self.smoothing = label_smoothing

    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")
        pred = logits.argmax(dim=-1)
        missed_af = (target == AF_IDX) & (pred != AF_IDX)
        return (losses * torch.where(missed_af, self.penalty, 1.0)).mean()


def class_weights(labels, max_weight: float = 4.0) -> torch.Tensor:
    """Inverse-frequency weights over RHYTHM_CLASSES, capped.

    The cap is what makes this usable. Uncapped, OTHER's weight is set by how
    few examples of it exist rather than by how much it matters, and a class
    the model cannot learn ends up dominating the gradient.
    """
    import numpy as np

    labels = np.asarray(labels, dtype=str)
    n = len(labels)
    w = [min(max_weight,
             n / (len(RHYTHM_CLASSES) * max((labels == c).sum(), 1)))
         for c in RHYTHM_CLASSES]
    return torch.tensor(w, dtype=torch.float32)


def build(**kw) -> RhythmNet:
    return RhythmNet(**kw)


if __name__ == "__main__":
    m = build()
    x = torch.randn(4, window_samples("rhythm"))
    print(f"params    {m.n_params():,}")
    print(f"logits    {tuple(m(x).shape)}")
    print(f"P(AF)     {[round(v, 3) for v in m.predict_proba(x).tolist()]}")
    print(f"scored    {SCORED_CLASSES}")
