"""Model 1 — signal quality gate.

A small 1D CNN that decides whether a 5-second window carries recoverable QRS
structure. Everything downstream sees only what this passes, so its errors are
not symmetric: rejecting good signal blinds the whole pipeline for that patient,
while passing a slightly noisy window costs a little accuracy later.

The design targets one specific failure the feature baseline could not handle.
Ten hand-crafted features are aggregate numbers over the whole window, so a low
kurtosis reads the same whether the QRS is buried in noise or genuinely broad
and rounded — which is what ventricular tachycardia looks like. The baseline
therefore rejected cudb patients whose signal was clean but whose complexes
were wide, calling a dangerous rhythm "bad signal". A convolutional stack sees
the waveform in time and can separate chaos from a broad regular pattern.

Shape: (batch, 1, 1250) -> (batch, 2)
Budget: well under 1M parameters; this runs on a phone alongside three others.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import QUALITY_CLASSES, window_samples


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> GELU, with optional downsampling.

    Dilation widens the receptive field without extra parameters or another
    pooling step. At 250 Hz a single beat spans roughly 200 samples, so the
    deeper blocks need to see across several beats to judge whether a pattern
    repeats — that periodicity is the difference between a broad regular rhythm
    and noise.
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


class QualityGate(nn.Module):
    """1D CNN over a raw normalised window.

    Global pooling rather than flattening keeps the parameter count low and
    makes the model independent of input length, so changing the window from
    5 s to 10 s needs no architectural change.

    Both average and max pooling are concatenated. Average captures the general
    character of the window; max captures whether any strong transient exists
    at all. A window can be quiet on average and still contain one decisive
    artifact, and vice versa — a flat trace with no transient anywhere is
    lead-off, not signal.
    """

    def __init__(self, n_classes: int = len(QUALITY_CLASSES),
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
            nn.Linear(widths[-1] * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.blocks(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def predict_proba(self, x) -> torch.Tensor:
        """P(usable) per window — the graded score downstream stages threshold."""
        self.eval()
        return F.softmax(self(x), dim=-1)[:, QUALITY_CLASSES.index("usable")]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AsymmetricLoss(nn.Module):
    """Cross-entropy weighted against wrongly rejecting usable signal.

    Two adjustments, and they do different jobs. Class weights correct for the
    roughly 76/24 split so the minority class is not ignored. The extra penalty
    on false rejection encodes that the two mistakes are not equally costly.

    The weight is deliberately modest. At 3.0 during baseline work the model
    stopped predicting 'unusable' at all: specificity hit target while
    sensitivity went to zero, which is a gate that gates nothing.
    """

    def __init__(self, class_weights: torch.Tensor | None = None,
                 false_reject_penalty: float = 1.5,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(QUALITY_CLASSES)))
        self.penalty = false_reject_penalty
        self.smoothing = label_smoothing
        self.usable = QUALITY_CLASSES.index("usable")

    def forward(self, logits, target):
        losses = F.cross_entropy(logits, target, weight=self.w,
                                 label_smoothing=self.smoothing,
                                 reduction="none")
        pred = logits.argmax(dim=-1)
        false_reject = (target == self.usable) & (pred != self.usable)
        return (losses * torch.where(false_reject, self.penalty, 1.0)).mean()


def class_weights(labels) -> torch.Tensor:
    """Inverse-frequency weights over QUALITY_CLASSES."""
    import numpy as np

    labels = np.asarray(labels, dtype=str)
    n = len(labels)
    w = [n / (len(QUALITY_CLASSES) * max((labels == c).sum(), 1))
         for c in QUALITY_CLASSES]
    return torch.tensor(w, dtype=torch.float32)


def build(**kw) -> QualityGate:
    return QualityGate(**kw)


if __name__ == "__main__":
    m = build()
    x = torch.randn(4, window_samples("quality"))
    print(f"params      {m.n_params():,}")
    print(f"logits      {tuple(m(x).shape)}")
    print(f"P(usable)   {m.predict_proba(x).tolist()}")