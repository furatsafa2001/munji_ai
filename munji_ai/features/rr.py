"""RR-interval features for the rhythm classifier.

    from munji_ai.features.rr import RR_FEATURES, rr_features
    f = rr_features(window)          # one 30-second window -> (12,)

Why
---
RhythmNet reads the raw trace and has to infer beat spacing on its own. That
works well for AF (irregular, no pattern) but leaves the OTHER class weak: in
the pooled confusion matrix about 740 OTHER windows are called NSR. OTHER is
mostly rhythms whose signature is in the *timing* of beats rather than their
shape: ectopic beats and bigeminy (alternating short-long), pauses and
dropped beats (block), and rates outside the normal band.

These are exactly what classical RR statistics measure, and the best entries
of the PhysioNet/CinC 2017 challenge (the source of the OTHER class here)
combined such features with a learned representation. This module supplies
them; RhythmNet concatenates them with its pooled CNN embedding.

The CNN still sees the raw trace, so if the beat detector fails under motion
the network is not blind: `qsqi` and `valid` tell the head how far to trust
the RR numbers in each window.

Features, all computed on one window
------------------------------------
    hr         median heart rate / 100
    cv         SD of RR / mean RR                 overall irregularity
    rmssd_n    RMSSD / mean RR                    beat-to-beat irregularity
    pnn50      share of successive changes > 50 ms
    premature  share of RR < 0.85 x median        early (ectopic) beats
    long       share of RR > 1.5 x median         pauses, dropped beats
    tpr        turning-point ratio                ~0.67 random, ~1 bigeminy
    entropy    Shannon entropy of RR histogram    AF high, NSR low
               (bins fixed at 0.5x to 1.5x the median RR)
    sd_ratio   Poincare SD1 / SD2                 AF near 1, NSR small
    range      log(max RR / min RR)
    qsqi       agreement of two beat detectors    trust in the numbers above
    valid      1 if enough beats were found, else 0 and everything else is 0

RMSSD/mean, TPR and Shannon entropy are the three statistics of the Dash et al.
AF detector; the rest cover the non-AF rhythms that fill OTHER.

No feature uses anything outside its own window, so computing them before the
patient-level split cannot leak information across folds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config import TARGET_FS
from .sqi import MIN_DYNAMIC_RANGE, detect_r_pantompkins, q_sqi

RR_FEATURES = ("hr", "cv", "rmssd_n", "pnn50", "premature", "long", "tpr",
               "entropy", "sd_ratio", "range", "qsqi", "valid")
RR_VERSION = 1            # bump when any feature definition changes

MIN_BEATS = 4             # fewer and the statistics are noise
RR_PLAUSIBLE = (0.2, 3.0) # seconds; 20 to 300 bpm. Outside is a detector error
HIST_BINS = 16
CLIP = 5.0                # no feature is allowed to blow up the head


def rr_features(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """All RR features for one window, as float32 in RR_FEATURES order."""
    out = np.zeros(len(RR_FEATURES), dtype=np.float32)
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(x) < MIN_DYNAMIC_RANGE or np.std(x) < MIN_DYNAMIC_RANGE:
        return out

    peaks = detect_r_pantompkins(x, fs)
    if len(peaks) < MIN_BEATS:
        return out

    rr = np.diff(peaks) / fs
    rr = rr[(rr >= RR_PLAUSIBLE[0]) & (rr <= RR_PLAUSIBLE[1])]
    if len(rr) < MIN_BEATS - 1:
        return out

    med, mean = float(np.median(rr)), float(np.mean(rr))
    d = np.diff(rr)

    f = dict.fromkeys(RR_FEATURES, 0.0)
    f["hr"] = 60.0 / med / 100.0
    f["cv"] = float(np.std(rr)) / mean
    f["rmssd_n"] = float(np.sqrt(np.mean(d ** 2))) / mean if d.size else 0.0
    f["pnn50"] = float(np.mean(np.abs(d) > 0.05)) if d.size else 0.0
    f["premature"] = float(np.mean(rr < 0.85 * med))
    f["long"] = float(np.mean(rr > 1.5 * med))

    if d.size >= 2:
        turns = (d[:-1] * d[1:]) < 0
        f["tpr"] = float(np.mean(turns))

    # Bins span a fixed band around the median (0.5x to 1.5x), not min..max.
    # Scaling the bins to the data would make the tiny jitter of a sinus
    # rhythm look as spread out as fibrillation.
    edges = (0.5 * med, 1.5 * med)
    counts, _ = np.histogram(np.clip(rr, *edges), bins=HIST_BINS, range=edges)
    p = counts[counts > 0] / counts.sum()
    f["entropy"] = float(-(p * np.log(p)).sum() / np.log(HIST_BINS))

    lo, hi = float(rr.min()), float(rr.max())
    f["range"] = float(np.log(hi / lo))

    if d.size >= 2:
        sd1 = float(np.std(d)) / np.sqrt(2.0)
        sd2_sq = 2.0 * float(np.var(rr)) - sd1 ** 2
        if sd2_sq > 1e-12:
            f["sd_ratio"] = sd1 / np.sqrt(sd2_sq)

    f["qsqi"] = float(q_sqi(x, fs))
    f["valid"] = 1.0

    out[:] = [f[k] for k in RR_FEATURES]
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=CLIP, neginf=0.0),
                   0.0, CLIP).astype(np.float32)


def rr_features_batch(X: np.ndarray, fs: int = TARGET_FS,
                      progress: bool = True) -> np.ndarray:
    """rr_features over every row of X."""
    F = np.zeros((len(X), len(RR_FEATURES)), dtype=np.float32)
    step = max(1, len(X) // 10)
    for i, x in enumerate(X):
        F[i] = rr_features(x, fs)
        if progress and (i + 1) % step == 0:
            print(f"  rr features  {i + 1:,}/{len(X):,}", flush=True)
    return F


def _fingerprint(record: np.ndarray, offset: np.ndarray) -> str:
    """Identity of the window set, so a stale feature file is never reused."""
    h = hashlib.sha256()
    h.update("\n".join(map(str, record)).encode())
    h.update(np.asarray(offset, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def cached_rr_features(X: np.ndarray, record: np.ndarray, offset: np.ndarray,
                       path: Path, fs: int = TARGET_FS) -> np.ndarray:
    """Load RR features from `path`, or compute and save them.

    The file is reused only if it was built from exactly these windows (same
    records, same offsets, same order) under the same RR_VERSION. Anything
    else is recomputed, never silently trusted.
    """
    path = Path(path)
    key = _fingerprint(record, offset)
    if path.exists():
        d = np.load(path, allow_pickle=False)
        if (str(d["key"]) == key and int(d["version"]) == RR_VERSION
                and d["F"].shape == (len(X), len(RR_FEATURES))):
            print(f"  rr features  loaded {path.name}")
            return d["F"]
        print(f"  rr features  {path.name} is stale, rebuilding")

    F = rr_features_batch(X, fs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, F=F, key=key, version=RR_VERSION,
                        names=np.array(RR_FEATURES))
    print(f"  rr features  saved {path.name}")
    return F
