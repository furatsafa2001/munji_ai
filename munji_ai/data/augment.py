"""Noise augmentation.

Public ECG corpora were recorded on resting or lightly-active subjects under
supervision. MUNJI runs on an ambulatory elderly patient walking, sleeping, and
praying. Injecting real recorded artifact at controlled SNR is the cheapest
available mitigation for that gap, and it doubles as the label source for the
signal quality gate: SNR is known exactly, so quality labels are exact.

NSTDB supplies three artifact types recorded from real electrodes:
  bw  baseline wander
  em  electrode motion
  ma  muscle artifact
"""

from __future__ import annotations

import numpy as np

from ..config import NOISE_SNR_DB_RANGE, RAW_DIR, TARGET_FS
from . import preprocess as pp
from .registry import NOISE_RECORDS

# SNR thresholds mapping to the three-class quality scheme. Chosen so the
# middle band corresponds to "R peaks still findable, morphology unreliable".
QUALITY_BANDS = {"good": 12.0, "qrs_only": 3.0}  # dB, lower bound of each class


def load_noise(root=RAW_DIR, kinds=NOISE_RECORDS) -> dict[str, np.ndarray]:
    """Load NSTDB artifact records, resampled to TARGET_FS."""
    import wfdb
    from .loader import dataset_dir
    from .registry import get

    d = dataset_dir(get("nstdb"), root)
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        hits = list(d.rglob(f"{kind}.hea"))
        if not hits:
            continue
        rec = wfdb.rdrecord(str(hits[0].with_suffix("")))
        x = np.nan_to_num(np.asarray(rec.p_signal[:, 0], dtype=np.float64))
        out[kind] = pp.resample_to(x, int(round(rec.fs)))
    if not out:
        raise FileNotFoundError(f"no NSTDB noise records under {d}")
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float,
        rng: np.random.Generator | None = None) -> np.ndarray:
    """Add `noise` to `clean` scaled to the requested SNR.

    A random offset into the noise record is used so the model cannot memorise
    one artifact waveform.
    """
    rng = rng or np.random.default_rng()
    clean = np.asarray(clean, dtype=np.float64)
    n = len(clean)

    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    off = int(rng.integers(0, max(len(noise) - n, 1)))
    seg = noise[off : off + n].copy()
    seg -= np.mean(seg)

    target = _rms(clean) / (10.0 ** (snr_db / 20.0))
    return clean + seg * (target / _rms(seg))


def quality_from_snr(snr_db: float) -> str:
    if snr_db >= QUALITY_BANDS["good"]:
        return "good"
    if snr_db >= QUALITY_BANDS["qrs_only"]:
        return "qrs_only"
    return "unusable"


def augment_window(
    x: np.ndarray,
    noises: dict[str, np.ndarray],
    rng: np.random.Generator | None = None,
    snr_range: tuple[float, float] = NOISE_SNR_DB_RANGE,
) -> tuple[np.ndarray, float, str]:
    """Return (noisy signal, snr_db, quality label)."""
    rng = rng or np.random.default_rng()
    kind = str(rng.choice(sorted(noises)))
    snr = float(rng.uniform(*snr_range))
    return mix(x, noises[kind], snr, rng), snr, quality_from_snr(snr)


def make_quality_set(
    windows: np.ndarray,
    noises: dict[str, np.ndarray],
    seed: int = 0,
    clean_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a labelled quality-gate training set from clean windows.

    A share of windows is left untouched so the 'good' class contains genuinely
    clean signal and not only high-SNR mixtures.
    """
    rng = np.random.default_rng(seed)
    X, y, snrs = [], [], []
    for w in windows:
        if rng.random() < clean_fraction:
            X.append(np.asarray(w, dtype=np.float64))
            y.append("good")
            snrs.append(np.inf)
        else:
            noisy, snr, label = augment_window(w, noises, rng)
            X.append(noisy)
            y.append(label)
            snrs.append(snr)
    return np.asarray(X), np.asarray(y, dtype="<U9"), np.asarray(snrs)


def simulate_lead_off(x: np.ndarray, fs: int = TARGET_FS,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Flat, low-amplitude segment mimicking a detached electrode.

    The asystole rule must reject these. Training and testing without them
    guarantees false cardiac-arrest alerts in the field.
    """
    rng = rng or np.random.default_rng()
    out = np.asarray(x, dtype=np.float64).copy()
    dur = int(rng.uniform(2.0, min(6.0, len(x) / fs)) * fs)
    start = int(rng.integers(0, max(len(out) - dur, 1)))
    out[start : start + dur] = rng.normal(0, 0.01, size=len(out[start : start + dur]))
    return out
