"""Training-time augmentation for the rhythm and shockable models.

    python -m munji_ai.data.train_augment --build     # once, needs NSTDB in data/raw

Why
---
Both models train on clean hospital and Holter recordings and are deployed on
an elderly patient walking, sleeping and praying with a chest patch. In
wearable defibrillators, artifact is the main cause of false alarms, so a model
that has never seen motion noise during training is being asked to generalise
to its most common input. augment.py already loads NSTDB for the quality gate;
nothing used it for these two models.

Three label-preserving augmentations, applied on the fly to training batches
only:

    noise    real electrode-motion (em) and muscle (ma) artifact from NSTDB,
             at 3 to 20 dB SNR. 3 dB is the quality gate's usable boundary, so
             the models practise on everything the gate can let through.
    jitter   small white noise, 20 to 30 dB SNR.
    cutout   a short stretch (up to 5% of the window) set to baseline,
             like a brief contact loss.

After mixing, each window is renormalised with the same median / IQR rule as
preprocess.normalize, because in deployment the noise is present before
normalisation, not after.

What is deliberately left out
-----------------------------
    time warping     changes the heart rate, and rate defines labels here
                     (VT is shockable only at 150 bpm or more)
    baseline wander  the 0.5 Hz high-pass already removes it; scaling the
                     residual up to a target SNR would invent a wander that
                     the device never sees
    synthetic beats  averaging rhythms makes traces that do not exist

Held-out noise for evaluation
-----------------------------
Each NSTDB record is split in time: the first 70% feeds training, the last 30%
is reserved for the noisy evaluation in cross_validate (--noisy-eval). A model
can therefore never be scored on an artifact waveform it trained on.

This module does not change augment.py or anything the quality gate uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..config import CACHE_DIR, RAW_DIR, TARGET_FS

NOISE_KINDS = ("em", "ma")
BANK_NAME = "noise_bank.npz"
TRAIN_SPLIT = 0.7

# Training-only knobs. Kept here rather than in config.py because nothing in
# deployment reads them and no cache depends on them.
P_NOISE = 0.5
NOISE_SNR_DB = (3.0, 20.0)
P_JITTER = 0.3
JITTER_SNR_DB = (20.0, 30.0)
P_CUTOUT = 0.2
CUTOUT_MAX_FRAC = 0.05


# ------------------------------------------------------------------ noise bank

def build_noise_bank(path: Path | None = None, root: Path = RAW_DIR) -> Path:
    """NSTDB em and ma, resampled to TARGET_FS and run through the model path."""
    from . import preprocess as pp
    from .augment import load_noise

    path = Path(path or Path(CACHE_DIR) / BANK_NAME)
    raw = load_noise(root, NOISE_KINDS)
    bank = {k: pp.model_path(v, TARGET_FS).astype(np.float32) for k, v in raw.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fs=TARGET_FS, **bank)
    for k, v in bank.items():
        print(f"  {k}  {len(v) / TARGET_FS / 60:.1f} min")
    print(f"saved {path}")
    return path


def load_noise_bank(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it once with\n"
            f"    python -m munji_ai.data.train_augment --build\n"
            f"(needs NSTDB: python -m munji_ai.data.download nstdb)")
    # Context manager so the file handle is released at once: Windows will
    # not delete or overwrite a file that is still open.
    with np.load(path, allow_pickle=False) as d:
        if int(d["fs"]) != TARGET_FS:
            raise ValueError(f"{path.name} was built at {int(d['fs'])} Hz, not {TARGET_FS}")
        return {k: np.array(d[k]) for k in NOISE_KINDS if k in d.files}


def split_bank(bank: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """(training part, evaluation part): first 70% and last 30% of each record."""
    train, held = {}, {}
    for k, v in bank.items():
        cut = int(len(v) * TRAIN_SPLIT)
        train[k], held[k] = v[:cut], v[cut:]
    return train, held


# ---------------------------------------------------------------- torch pieces

def renormalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise preprocess.normalize: (x - median) / (IQR / 1.349)."""
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device,
                                       dtype=x.dtype), dim=1)
    scale = ((q[2] - q[0]) / 1.349).unsqueeze(1)
    return (x - q[1].unsqueeze(1)) / (scale + eps)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2, dim=1) + 1e-12)


class NoiseMixer:
    """Adds NSTDB segments at a requested SNR, from a random record and offset."""

    def __init__(self, bank: dict[str, np.ndarray], device, seed: int = 0):
        self.device = torch.device(device)
        self.kinds = sorted(bank)
        self.bank = [torch.from_numpy(np.asarray(bank[k], dtype=np.float32)).to(self.device)
                     for k in self.kinds]
        self.gen = torch.Generator().manual_seed(seed)

    def rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def segments(self, n: int, length: int) -> torch.Tensor:
        kind = torch.randint(0, len(self.bank), (n,), generator=self.gen)
        out = torch.empty(n, length, device=self.device)
        ar = torch.arange(length, device=self.device)
        for i, b in enumerate(self.bank):
            m = kind == i
            k = int(m.sum())
            if k == 0:
                continue
            if b.numel() <= length:
                raise ValueError(f"noise record '{self.kinds[i]}' is shorter than a window")
            off = torch.randint(0, b.numel() - length, (k,), generator=self.gen).to(self.device)
            out[m.to(self.device)] = b[off.unsqueeze(1) + ar.unsqueeze(0)]
        return out - out.mean(dim=1, keepdim=True)

    def mix(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        seg = self.segments(x.shape[0], x.shape[1])
        target = _rms(x) / torch.pow(10.0, snr_db / 20.0)
        return x + seg * (target / _rms(seg)).unsqueeze(1)


class TrainAugment:
    """Callable on a (batch, length) tensor of normalised windows."""

    def __init__(self, bank_train: dict[str, np.ndarray], device, seed: int = 0):
        self.mixer = NoiseMixer(bank_train, device, seed)

    def _uniform(self, n: int, lo_hi: tuple[float, float]) -> torch.Tensor:
        lo, hi = lo_hi
        return lo + (hi - lo) * self.mixer.rand(n)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                     # (batch, 1, length)
            return self(x.squeeze(1)).unsqueeze(1)
        b, length = x.shape
        x = x.clone()

        m = self.mixer.rand(b) < P_NOISE
        if m.any():
            x[m] = self.mixer.mix(x[m], self._uniform(int(m.sum()), NOISE_SNR_DB))

        j = self.mixer.rand(b) < P_JITTER
        if j.any():
            w = torch.randn(int(j.sum()), length, generator=self.mixer.gen).to(x.device)
            snr = self._uniform(int(j.sum()), JITTER_SNR_DB)
            target = _rms(x[j]) / torch.pow(10.0, snr / 20.0)
            x[j] = x[j] + w * (target / _rms(w)).unsqueeze(1)

        c = self.mixer.rand(b) < P_CUTOUT
        if c.any():
            span = (self.mixer.rand(b) * CUTOUT_MAX_FRAC * length).long()
            start = (self.mixer.rand(b) * (length - span)).long()
            pos = torch.arange(length, device=x.device).unsqueeze(0)
            mask = (pos >= start.unsqueeze(1)) & (pos < (start + span).unsqueeze(1))
            x = x.masked_fill(mask & c.unsqueeze(1), 0.0)

        return renormalize(x)


@torch.no_grad()
def add_noise_fixed(X: np.ndarray, bank_eval: dict[str, np.ndarray], snr_db: float,
                    seed: int, chunk: int = 256) -> np.ndarray:
    """Deterministic noisy copy of X for evaluation, from held-out noise only."""
    mixer = NoiseMixer(bank_eval, "cpu", seed)
    out = []
    for i in range(0, len(X), chunk):
        x = torch.from_numpy(np.asarray(X[i:i + chunk], dtype=np.float32))
        y = mixer.mix(x, torch.full((len(x),), float(snr_db)))
        out.append(renormalize(y).numpy())
    return np.concatenate(out) if out else np.empty_like(X, dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build data/cache/noise_bank.npz")
    ap.add_argument("--root", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.build:
        ap.print_help()
        return 0
    build_noise_bank(a.out, a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
