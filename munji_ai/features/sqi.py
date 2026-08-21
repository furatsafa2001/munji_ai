"""Classic signal quality indices.

These four features are the established baseline in the ECG quality literature
(Zhao & Zhang, Front. Physiol. 2018), where fused SQIs reached roughly 94-97%
accuracy on PhysioNet data. Deep models in the same literature report about
94% accuracy with 91% sensitivity and 95% specificity — a narrow margin.

Building this first is deliberate. If a four-feature tree hits the target, a
CNN is not worth the parameters, the latency, or the opacity. And if the CNN
does win, the margin is measured against a real baseline rather than asserted.

    qSQI    agreement between two independent R-peak detectors
    pSQI    share of spectral power in the QRS band
    kSQI    kurtosis — impulsive artifact raises it sharply
    basSQI  share of power below 1 Hz — baseline wander

Everything here is deterministic and cheap enough to run on the device.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis, skew

from ..config import TARGET_FS

QRS_BAND = (5.0, 15.0)      # QRS energy concentrates here
FULL_BAND = (1.0, 40.0)
BASELINE_BAND = (0.0, 1.0)
MATCH_TOLERANCE_MS = 60.0   # two detectors agree if peaks fall within this
MIN_DYNAMIC_RANGE = 1e-3    # below this the window is flat — lead-off, not signal
PLAUSIBLE_BPM = (20.0, 300.0)


# --------------------------------------------------------------- R detection


def _integrate(x: np.ndarray, fs: int, win_ms: float) -> np.ndarray:
    w = max(int(win_ms * fs / 1000.0), 1)
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")


def detect_r_pantompkins(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Pan-Tompkins style: band-pass, differentiate, square, integrate."""
    nyq = fs / 2.0
    sos = sig.butter(2, [QRS_BAND[0] / nyq, min(QRS_BAND[1], nyq * 0.99) / nyq],
                     btype="band", output="sos")
    y = sig.sosfiltfilt(sos, x)
    y = np.diff(y, prepend=y[0]) ** 2
    y = _integrate(y, fs, 120.0)
    thr = np.mean(y) + 0.5 * np.std(y)
    peaks, _ = sig.find_peaks(y, height=thr, distance=int(0.25 * fs))
    return peaks


def detect_r_envelope(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Wide-band amplitude detector — deliberately fragile.

    qSQI only works if the two detectors fail *differently*. Pan-Tompkins
    band-passes to 5-15 Hz, which suppresses most artifact and makes it robust.
    This one runs on a near-full band, so noise produces spurious peaks here
    that the robust detector never sees. The disagreement is the signal.
    """
    nyq = fs / 2.0
    sos = sig.butter(2, [2.0 / nyq, min(40.0, nyq * 0.99) / nyq],
                     btype="band", output="sos")
    y = np.abs(sig.sosfiltfilt(sos, x))
    y = _integrate(y, fs, 40.0)
    thr = np.percentile(y, 70)
    # 0.30 s minimum spacing keeps T waves from being counted as beats. Tuned so
    # the two detectors agree exactly on clean signal and diverge sharply under
    # artifact — a detector that is unreliable on clean signal makes qSQI noise.
    peaks, _ = sig.find_peaks(y, height=thr, distance=int(0.30 * fs))
    return peaks


def q_sqi(x: np.ndarray, fs: int = TARGET_FS) -> float:
    """Fraction of peaks the two detectors agree on, one-to-one.

    Clean signal → both find the same beats. Noise → the fragile detector
    latches onto artifact the robust one ignores, and agreement collapses.

    Two guards matter here. A flat window (lead-off) makes both detectors chase
    the same numerical dither and agree perfectly on nothing, so dynamic range
    is checked first. And matching is greedy one-to-one: without it a dense
    artifact train scores high because many spurious peaks all match the same
    real beat.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(x) < MIN_DYNAMIC_RANGE or np.std(x) < MIN_DYNAMIC_RANGE:
        return 0.0

    a, b = detect_r_pantompkins(x, fs), detect_r_envelope(x, fs)
    if len(a) == 0 or len(b) == 0:
        return 0.0

    dur_min = len(x) / fs / 60.0
    for peaks in (a, b):
        bpm = len(peaks) / dur_min
        if not (PLAUSIBLE_BPM[0] <= bpm <= PLAUSIBLE_BPM[1]):
            return 0.0

    tol = MATCH_TOLERANCE_MS * fs / 1000.0
    b = np.asarray(b, dtype=np.float64)
    used = np.zeros(len(b), dtype=bool)
    matched = 0
    for p in a:
        d = np.abs(b - float(p))
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            matched += 1
    return float(matched / max(len(a), len(b)))


# ------------------------------------------------------------------ spectral


def _band_power(x: np.ndarray, fs: int, lo: float, hi: float) -> float:
    nper = min(len(x), int(fs * 2))
    f, p = sig.welch(x, fs=fs, nperseg=max(nper, 32))
    return float(np.trapezoid(p[(f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)]))


def p_sqi(x: np.ndarray, fs: int = TARGET_FS) -> float:
    """QRS-band power over full-band power. Falls as broadband noise rises."""
    full = _band_power(x, fs, *FULL_BAND)
    return float(_band_power(x, fs, *QRS_BAND) / full) if full > 0 else 0.0


def bas_sqi(x: np.ndarray, fs: int = TARGET_FS) -> float:
    """1 minus the sub-1 Hz share. Falls with baseline wander."""
    total = _band_power(x, fs, 0.0, FULL_BAND[1])
    if total <= 0:
        return 0.0
    return float(1.0 - _band_power(x, fs, *BASELINE_BAND) / total)


def k_sqi(x: np.ndarray) -> float:
    """Kurtosis. A clean ECG is spiky (QRS); flat noise is not.

    Returned raw — impulsive motion artifact pushes it very high, so the
    relationship with quality is not monotonic and the classifier should learn
    the band rather than a threshold.
    """
    return float(kurtosis(x, fisher=True, bias=False))


def s_sqi(x: np.ndarray) -> float:
    """Skewness. Asymmetry from the dominant R deflection."""
    return float(skew(x, bias=False))


# ------------------------------------------------------------------ assembly

FEATURE_NAMES = ("qSQI", "pSQI", "kSQI", "basSQI", "sSQI",
                 "rate_bpm", "rr_cv", "amp_iqr", "flat_frac", "sat_frac")


def features(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Full feature vector for one window. Order matches FEATURE_NAMES."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < fs or not np.isfinite(x).all():
        return np.zeros(len(FEATURE_NAMES))

    peaks = detect_r_pantompkins(x, fs)
    dur = len(x) / fs
    rate = len(peaks) / dur * 60.0
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        rr_cv = float(np.std(rr) / (np.mean(rr) + 1e-9))
    else:
        rr_cv = 0.0

    q75, q25 = np.percentile(x, [75, 25])
    # Lead-off shows up as a near-flat stretch; clipping shows up as saturation.
    d = np.abs(np.diff(x))
    flat = float(np.mean(d < (np.median(d) * 0.05 + 1e-12)))
    lim = np.max(np.abs(x))
    sat = float(np.mean(np.abs(x) > 0.99 * lim)) if lim > 0 else 0.0

    return np.array([
        q_sqi(x, fs), p_sqi(x, fs), k_sqi(x), bas_sqi(x, fs), s_sqi(x),
        rate, rr_cv, float(q75 - q25), flat, sat,
    ], dtype=np.float64)


def feature_matrix(X: np.ndarray, fs: int = TARGET_FS,
                   verbose: bool = False) -> np.ndarray:
    """Feature vectors for a stack of windows. Shape (n, len(FEATURE_NAMES))."""
    X = np.asarray(X)
    out = np.zeros((len(X), len(FEATURE_NAMES)))
    for i, w in enumerate(X):
        out[i] = features(np.asarray(w, dtype=np.float64), fs)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  features {i + 1:,}/{len(X):,}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
