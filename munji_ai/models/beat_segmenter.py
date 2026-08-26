"""Model 2 — beat classifier as 1D segmentation.

Predicts a class for every sample: background, normal beat, PAC, or PVC. That
framing, rather than one label per window, is deliberate and follows Sensors
2026 (MDPI 26/2/513), which reports near-perfect QRS detection (sensitivity and
precision up to 0.999) with PVC sensitivity from 0.820 on AHA to 0.986 on
MIT 11.

The property that matters for MUNJI is what the paper's title makes explicit:
no explicit R-peak detection, no handcrafted features, no fixed-length windows.
A pipeline that detects R-peaks first has a single point of failure — under
motion artifact the detector fails and everything downstream fails with it. A
segmentation model degrades gradually instead: it may blur a beat boundary, but
it does not lose the whole window.

Architecture, following the paper:

    stem        Conv1d(k=4, s=4) + LayerNorm
    encoder     4 stages of ConvNeXt V2 blocks, 3 downsamples of Conv1d(k=2, s=2)
    block       DWConv(k=7) -> LN -> PWConv(4x) -> GELU -> GRN -> PWConv -> residual
    decoder     simple upsample + skip concatenation
    head        per-sample logits at input resolution

Global Response Normalization is the V2 contribution: it increases competition
between channels, which the original paper found necessary once the masked
autoencoder pretraining was added. We keep it because the ECG paper does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import BEAT_CLASSES, window_samples


class LayerNorm1d(nn.Module):
    """LayerNorm over channels for (N, C, L) tensors.

    ConvNeXt normalises over the channel dimension, not the spatial one, which
    for 1D data means transposing around the operation. Doing it inline keeps
    the block readable and matches the reference implementation's semantics.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GRN(nn.Module):
    """Global Response Normalization, ConvNeXt V2.

    Normalises each channel by its own magnitude relative to the average across
    channels, so a channel that fires everywhere is damped and a rarely-active
    one is amplified. On this task the rare signal is the ectopic beat, which is
    exactly the kind of feature a magnitude-dominated network tends to lose.

    Operates on (N, C, L).
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)          # per-channel energy
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)   # relative to others
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock1d(nn.Module):
    """Depthwise 7-wide conv, then an inverted bottleneck with GRN.

    Kernel 7 at 250 Hz spans 28 ms, comfortably inside a QRS complex, so the
    first layers see complex shape rather than whole beats. Width comes from
    depth and downsampling instead, which is cheaper than wide kernels.
    """

    def __init__(self, channels: int, expansion: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm1d(channels)
        self.pw1 = nn.Conv1d(channels, channels * expansion, 1)
        self.grn = GRN(channels * expansion)
        self.pw2 = nn.Conv1d(channels * expansion, channels, 1)
        self.drop_path = drop_path

    def forward(self, x):
        h = self.pw2(self.grn(F.gelu(self.pw1(self.norm(self.dwconv(x))))))
        if self.drop_path > 0.0 and self.training:
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
            h = h * mask / keep
        return x + h


class Down(nn.Module):
    """LayerNorm then strided conv, as in the paper's downsampling layers."""

    def __init__(self, c_in: int, c_out: int, k: int = 2, s: int = 2):
        super().__init__()
        self.norm = LayerNorm1d(c_in)
        self.conv = nn.Conv1d(c_in, c_out, k, stride=s)

    def forward(self, x):
        return self.conv(self.norm(x))


class Up(nn.Module):
    """Simple decoder block: resize to the skip's length, concatenate, mix.

    Interpolating to the skip's exact size rather than doubling avoids an
    off-by-one whenever a length is odd — 5000 samples reduce to 1250, 625, 312,
    156, and 312 is not 625/2. Fixed upsampling factors would silently misalign
    the skip connection, and a mask shifted by one sample is a mask that no
    longer marks the beat it was drawn for.
    """

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv1d(c_in + c_skip, c_out, 3, padding=1),
            LayerNorm1d(c_out),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.mix(torch.cat([x, skip], dim=1))


class BeatSegmenter(nn.Module):
    """1D U-Net with a ConvNeXt V2 encoder.

    Input  (N, L) or (N, 1, L)
    Output (N, n_classes, L) — logits per sample, at input resolution
    """

    def __init__(self, n_classes: int = len(BEAT_CLASSES),
                 widths=(24, 48, 96, 128), depths=(2, 2, 3, 2),
                 drop_path: float = 0.05):
        super().__init__()
        assert len(widths) == len(depths) == 4

        # Stem reduces by 4 immediately: at 250 Hz sample-level resolution
        # carries no information a 4-sample window does not, and the saving
        # applies to every layer after it.
        self.stem = nn.Sequential(
            nn.Conv1d(1, widths[0], 4, stride=4),
            LayerNorm1d(widths[0]),
        )

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (w, d) in enumerate(zip(widths, depths)):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock1d(w, drop_path=drop_path * i / max(len(widths) - 1, 1))
                for _ in range(d)
            ]))
            if i < len(widths) - 1:
                self.downs.append(Down(w, widths[i + 1]))

        rev = list(reversed(widths))
        self.ups = nn.ModuleList([
            Up(rev[i], rev[i + 1], rev[i + 1]) for i in range(len(rev) - 1)
        ])

        # Back to input resolution, undoing the stem's stride of 4.
        self.head = nn.Sequential(
            nn.Conv1d(widths[0], widths[0], 3, padding=1),
            LayerNorm1d(widths[0]),
            nn.GELU(),
            nn.Conv1d(widths[0], n_classes, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        length = x.shape[-1]

        h = self.stem(x)
        skips = []
        for i, stage in enumerate(self.stages):
            h = stage(h)
            if i < len(self.downs):
                skips.append(h)
                h = self.downs[i](h)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)

        h = F.interpolate(h, size=length, mode="linear", align_corners=False)
        return self.head(h)

    @torch.no_grad()
    def predict_mask(self, x) -> torch.Tensor:
        self.eval()
        return self(x).argmax(dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SegmentationLoss(nn.Module):
    """Weighted cross-entropy plus soft Dice, per class.

    Cross-entropy alone is dominated by background: roughly 87% of samples carry
    no beat, so a model that predicts background everywhere already scores well
    on it. Dice is computed per class over the whole window and is insensitive
    to how large the background is, which keeps the rare classes visible in the
    gradient. Using both means neither failure mode goes unpunished.
    """

    def __init__(self, class_weights: torch.Tensor | None = None,
                 dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(len(BEAT_CLASSES)))
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.w)
        probs = F.softmax(logits, dim=1)
        oh = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 2, 1).float()
        inter = (probs * oh).sum(dim=(0, 2))
        denom = probs.sum(dim=(0, 2)) + oh.sum(dim=(0, 2))
        dice = 1.0 - ((2 * inter + self.eps) / (denom + self.eps))
        # Classes absent from the batch would otherwise contribute a spurious
        # perfect score and dilute the signal from the ones that are present.
        present = oh.sum(dim=(0, 2)) > 0
        dice = dice[present].mean() if present.any() else logits.new_zeros(())
        return ce + self.dice_weight * dice


def class_weights_from_masks(masks, cap: float = 20.0) -> torch.Tensor:
    """Inverse-frequency weights, capped.

    Ectopics occupy roughly 1.7% of samples each against 87% background, so raw
    inverse frequency gives them weights above 50 and training becomes unstable.
    The cap keeps the correction useful without letting a handful of samples
    dominate every gradient step.
    """
    import numpy as np

    m = np.asarray(masks).ravel()
    total = len(m)
    w = []
    for i in range(len(BEAT_CLASSES)):
        n = max(int((m == i).sum()), 1)
        w.append(min(total / (len(BEAT_CLASSES) * n), cap))
    return torch.tensor(w, dtype=torch.float32)


def build(**kw) -> BeatSegmenter:
    return BeatSegmenter(**kw)


if __name__ == "__main__":
    m = build()
    L = window_samples("beat")
    x = torch.randn(2, L)
    out = m(x)
    print(f"classes     {BEAT_CLASSES}")
    print(f"input       {tuple(x.shape)}")
    print(f"output      {tuple(out.shape)}")
    print(f"params      {m.n_params():,}")
    print(f"mask        {tuple(m.predict_mask(x).shape)}")
    for alt in (2000, 3751, 7500):
        print(f"length {alt:>5} -> {tuple(m(torch.randn(1, alt)).shape)}")