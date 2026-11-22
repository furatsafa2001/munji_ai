"""Ten published measures that tell VF from VT on one window.

    from munji_ai.features.vfvt import FEATURE_NAMES, features
    f = features(window)        # FEATURE_NAMES order; NaN where not measurable

VF is disorganised: complex, high in entropy, its power spread over a band,
little repetition. VT repeats one complex: periodic, its power at one
frequency, a rate that can be counted. These measures read that difference
(Lempel-Ziv complexity: Zhang et al., IEEE TBME 1999; permutation entropy:
Bandt and Pompe, PRL 2002; spectral and autocorrelation measures as compared
by Amann et al., BioMed Eng OnLine 2005):

    lz_complexity     Lempel-Ziv complexity of the signal cut at its median
    perm_entropy      permutation entropy, order 3, delay 2 samples
    spectral_entropy  entropy of the Welch spectrum between 1 and 15 Hz
    dominant_hz       frequency of that spectrum's peak
    concentration     share of the 1-15 Hz power within 0.5 Hz of the peak
    periodicity       highest autocorrelation at lags of 0.2 to 0.6 s
    kurtosis          excess kurtosis
    skewness          skewness
    zero_crossings    share of steps where the signal changes sign
    bpm_signal        median rate from the peaks of the rectified signal
                      (data.windows._episode_bpm); NaN under 3 peaks

All but the rate are read off the window standardised to mean 0 and standard
deviation 1, and the rate's peak threshold scales with the window, so gain
and offset change nothing: the raw window and ShockableNet's normalised one
give the same values (tests/test_vfvt_namer.py checks it).

The VF-or-VT namer (models/vfvt_namer.py, exports/vfvt_namer.json) was
fitted on exactly these definitions, and the app computes them again in
Dart. A change here changes what the shipped namer reads: retrain and
re-export it; tests/test_vfvt_namer.py fails on the golden windows until then.
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
    """Normalised permutation entropy. Ties keep sample order (a stable sort),
    so every platform codes a pattern the same way."""
    n = len(x) - (order - 1) * delay
    pats = np.stack([x[i * delay:i * delay + n] for i in range(order)], axis=1)
    codes = np.argsort(pats, axis=1, kind="stable") @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(math.factorial(order)))


def features(x: np.ndarray) -> np.ndarray:
    """The ten measures of one window, in FEATURE_NAMES order."""
    x = np.asarray(x, np.float64)
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
