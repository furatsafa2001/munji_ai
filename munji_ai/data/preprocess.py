"""Signal preprocessing.

Two filter paths exist on purpose:

  diagnostic  0.05-40 Hz  preserves the ST segment, used for storage
  model       0.5-40 Hz   standard monitoring band, used for model inputs

ST detection is out of scope for user-facing output, but the raw signal is
still stored at diagnostic bandwidth. A 0.5 Hz high-pass discards ST
information permanently, so that decision is kept out of the acquisition path.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig

from ..config import (
    DIAGNOSTIC_BAND,
    FILTER_ORDER,
    MAINS_HZ,
    MODEL_BAND,
    NOTCH_Q,
    TARGET_FS,
)


def resample_to(x: np.ndarray, fs_in: int, fs_out: int = TARGET_FS) -> np.ndarray:
    """Rational resampling. Anti-aliased, and exact for integer ratios."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    from math import gcd

    g = gcd(int(fs_in), int(fs_out))
    return sig.resample_poly(np.asarray(x, dtype=np.float64), fs_out // g, fs_in // g)


def _sos_bandpass(low: float, high: float, fs: int) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    return sig.butter(FILTER_ORDER, [low / nyq, high / nyq], btype="band", output="sos")


def bandpass(x: np.ndarray, band: tuple[float, float], fs: int = TARGET_FS) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    filtfilt is used so the filter introduces no group delay — R-peak timing
    stays aligned with the annotations. This is offline-only; the on-device
    path must use a causal filter and will shift peaks slightly.
    """
    return sig.sosfiltfilt(_sos_bandpass(*band, fs), np.asarray(x, dtype=np.float64))


def notch(x: np.ndarray, freq: float = MAINS_HZ, fs: int = TARGET_FS) -> np.ndarray:
    if freq >= fs / 2.0:
        return np.asarray(x, dtype=np.float64)
    b, a = sig.iirnotch(freq, NOTCH_Q, fs)
    return sig.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def diagnostic_path(x: np.ndarray, fs_in: int) -> np.ndarray:
    """Storage path — ST segment preserved."""
    x = resample_to(x, fs_in)
    return notch(bandpass(x, DIAGNOSTIC_BAND))


def model_path(x: np.ndarray, fs_in: int, apply_notch: bool = True) -> np.ndarray:
    """Model input path — standard monitoring band."""
    x = resample_to(x, fs_in)
    x = bandpass(x, MODEL_BAND)
    return notch(x) if apply_notch else x


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Robust per-window normalisation.

    Median and IQR rather than mean and standard deviation: a single motion
    spike inflates the standard deviation enough to flatten the whole window,
    which is exactly the situation the quality gate needs to see clearly.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.349  # IQR -> sigma for a normal distribution
    return (x - med) / (scale + eps)


def sliding_windows(
    x: np.ndarray, win: int, stride: int, drop_partial: bool = True
) -> np.ndarray:
    """Frame a 1-D signal into overlapping windows. Returns (n_windows, win)."""
    x = np.asarray(x)
    if len(x) < win:
        return np.empty((0, win), dtype=x.dtype)
    n = 1 + (len(x) - win) // stride
    out = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * stride, x.strides[0]), writeable=False
    )
    out = np.ascontiguousarray(out)
    if not drop_partial and (len(x) - win) % stride:
        tail = np.zeros((1, win), dtype=x.dtype)
        rest = x[n * stride :][:win]
        tail[0, : len(rest)] = rest
        out = np.vstack([out, tail])
    return out
