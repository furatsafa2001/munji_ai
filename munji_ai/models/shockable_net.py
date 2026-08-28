"""Model 3 — shockable rhythm detector.

A small 1D CNN that decides whether a 5-second window carries a rhythm that
needs a shock: ventricular fibrillation, ventricular flutter, or rapid VT.

The error asymmetry is the opposite of the quality gate's, and steeper. Missing
a fibrillating heart costs a life; raising an alarm on a clean trace costs a
phone call. So the loss penalises misses, not false alarms — the reverse of
AsymmetricLoss in quality_gate.py, and deliberately so.

Targets come from the AHA/AAMI convention for shock-advisory algorithms rather
than from anything this project invented:

    VF          sensitivity  > 0.90
    rapid VT    sensitivity  > 0.75
    NSR         specificity  > 0.99

Selection therefore maximises specificity subject to a sensitivity floor. A
model that scores well on macro F1 while missing one VF window in five is not
a better detector, however good the average looks.

Shape: (batch, 1, 1250) -> (batch, 2)
Budget: well under 1M parameters; four models share one phone.

What the network has to see
---------------------------
Organised rhythm has a repeating narrow spike. Fibrillation has neither — it
is a broad, wandering, quasi-sinusoidal wave with no reproducible complex.
Distinguishing the two is a question about structure across the whole window,
not about any single feature of one beat, which is why the deeper blocks are
dilated: they need a receptive field spanning several seconds to judge whether
anything repeats at all.

The hard case is not VF against normal — it is VF against muscle artifact,
which is also broad, also chaotic, and far more common in a home device. That
is what the quality gate upstream exists to remove, and why this model's
reported numbers only hold for windows the gate passed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SHOCKABLE_CLASSES, window_samples

SHOCKABLE_IDX = SHOCKABLE_CLASSES.index("shockable")

# AHA/AAMI floor for VF detection. Not a tuning knob — a model below this is
# not a shock-advisory algorithm, whatever else it scores.
SENSITIVITY_FLOOR = 0.90


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> GELU, with optional downsampling.

    Deliberately a local copy of the block in quality_gate.py rather than an
    import. The two models are tuned against different failures, and a change
    made for the gate should not silently alter a detector whose sensitivity
    is a safety claim.
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


class ShockableNet(nn.Module):
    """1D CNN over a raw normalised 5-second window.

    Three pooled statistics are concatenated rather than two. Mean and max are
    the same pair the gate uses. Standard deviation is added because it is the
    single most direct summary of the thing that separates these classes: an
    organised rhythm is mostly flat with brief spikes, so its activations vary
    sharply, while fibrillation is continuously active and varies little. That
    contrast is exactly what a variance statistic captures and what an average
    hides.
    """

    def __init__(self, n_classes: int = len(SHOCKABLE_CLASSES),
                 widths=(16, 32, 64, 96, 128), dropout: float = 0.1):
        super().__init__()
        dilations = (1, 1, 2, 4, 8)
        chans = (1,) + tuple(widths)
        self.blocks = nn.Sequential(*[
            ConvBlock(chans[i], chans[i + 1], dilation=dilations[i],
                      dropout=dropout)
            for i in range(len(widths))
        ])
        self.head = nn.Sequential(
            nn.Linear(widths[-1] * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
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
        """P(shockable) per window.

        Returned graded rather than thresholded because the alerting rule is
        not per-window: config.CONFIRMATION requires 2 of 3 consecutive
        positives before an alarm fires, and a single window must never
        trigger a phone call.
        """
        self.eval()
        return F.softmax(self(x), dim=-1)[:, SHOCKABLE_IDX]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MissPenaltyLoss(nn.Module):
    """Cross-entropy weighted against missing a shockable rhythm.

    Mirror image of the gate's AsymmetricLoss. There the risk was discarding
    good signal; here it is calling a fibrillating heart normal.

    The penalty is larger than the gate's 1.5 because the consequences are not
    comparable — but it is still bounded, and for the same reason the gate
    bounds its own. Push it far enough and the model calls everything
    shockable: sensitivity reaches 1.0, specificity collapses, and a monitor
    that alarms constantly gets ignored or switched off. That failure is not
    visible in a sensitivity column, so watch specificity while tuning it.
    """

    def __init__(self, class_weights: torch.Tensor | None = None,
                 miss_penalty: float = 2.5,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(SHOCKABLE_CLASSES)))
        self.penalty = miss_penalty
        self.smoothing = label_smoothing

    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")
        pred = logits.argmax(dim=-1)
        missed = (target == SHOCKABLE_IDX) & (pred != SHOCKABLE_IDX)
        return (losses * torch.where(missed, self.penalty, 1.0)).mean()


def class_weights(labels) -> torch.Tensor:
    """Inverse-frequency weights over SHOCKABLE_CLASSES."""
    import numpy as np

    labels = np.asarray(labels, dtype=str)
    n = len(labels)
    w = [n / (len(SHOCKABLE_CLASSES) * max((labels == c).sum(), 1))
         for c in SHOCKABLE_CLASSES]
    return torch.tensor(w, dtype=torch.float32)


def build(**kw) -> ShockableNet:
    return ShockableNet(**kw)


if __name__ == "__main__":
    m = build()
    x = torch.randn(4, window_samples("shockable"))
    print(f"params        {m.n_params():,}")
    print(f"logits        {tuple(m(x).shape)}")
    print(f"P(shockable)  {[round(v, 3) for v in m.predict_proba(x).tolist()]}")
