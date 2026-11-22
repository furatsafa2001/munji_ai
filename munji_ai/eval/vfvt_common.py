"""Shared by the VF-or-VT experiments, and free of PyTorch.

vf_vt_feasibility.py, vt_detect_feasibility.py and vf_vt_cardiologist.py read
the same published measures off a window and score them the same way; they
live here so the measures cannot drift apart between experiments.

    features(x)      the published measures of one 5 s window (FEATURE_NAMES):
                     Lempel-Ziv complexity, permutation and spectral entropy,
                     dominant frequency and its concentration, periodicity,
                     kurtosis, skewness, zero-crossing rate, rate from the signal
    make_clf(kind)   "logistic" (scaled, C 0.1) or "forest", balanced classes
    auc(y, p)        rank AUC, NaN scores ignored
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .. import config as C
from ..data.windows import _episode_bpm

FS = C.TARGET_FS

FEATURE_NAMES = ["lz_complexity", "perm_entropy", "spectral_entropy", "dominant_hz",
                 "concentration", "periodicity", "kurtosis", "skewness",
                 "zero_crossings", "bpm_signal"]


def lempel_ziv(x: np.ndarray) -> float:
    """Normalised Lempel-Ziv complexity of the median-binarised signal."""
    s = (x > np.median(x)).astype(np.uint8).tobytes()
    n, i, c, seen = len(s), 0, 0, set()
    while i < n:
        k = 1
        while i + k <= n and s[i:i + k] in seen:
            k += 1
        seen.add(s[i:i + k])
        c += 1
        i += k
    return c * np.log2(n) / n


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 2) -> float:
    n = len(x) - (order - 1) * delay
    pats = np.stack([x[i * delay:i * delay + n] for i in range(order)], axis=1)
    codes = np.argsort(pats, axis=1) @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(math.factorial(order)))


def features(x: np.ndarray) -> np.ndarray:
    z = (x - x.mean()) / (x.std() + 1e-8)
    f, p = welch(z, FS, nperseg=512)
    m = (f >= 1) & (f <= 15)
    pm = p[m] / p[m].sum()
    k = int(np.argmax(p[m]))
    band = np.abs(f[m] - f[m][k]) <= 0.5
    ac = np.correlate(z, z, "full")[len(z) - 1:] / len(z)
    bpm = _episode_bpm(x, FS)
    return np.array([
        lempel_ziv(z),
        permutation_entropy(z),
        float(-(pm * np.log2(pm + 1e-12)).sum() / np.log2(len(pm))),
        float(f[m][k]),
        float(pm[band].sum()),
        float(ac[int(0.2 * FS):int(0.6 * FS)].max()),
        float(kurtosis(z)),
        float(skew(z)),
        float(np.mean(np.abs(np.diff(np.sign(z))) > 0)),
        float(bpm) if bpm else np.nan,
    ])


def make_clf(kind: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "forest":
        return make_pipeline(SimpleImputer(), RandomForestClassifier(
            300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1))
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=5000))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
