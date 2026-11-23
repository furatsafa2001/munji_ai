"""Shockable labels, one per window: the AHA framework as published work applies it.

    from munji_ai.data.shockable_labels import label_windows, window_class
    labels = label_windows(rec.signal, rec.fs, rhythm_intervals(rec), starts, win)
    classes = [window_class(lab) for lab in labels]  # shockable / not_shockable / None

Why per window
--------------
VT used to be split by one rate per episode (windows.split_vt_by_rate, from
windows._episode_bpm: peaks above twice the median deviation of the whole
episode). A smooth, sine-like VT has no peak that prominent, and the quieter
stretches of an episode fall under it, so the rate read far too slow or not at
all. In CUDB and VFDB, 47% of the windows called VT_SLOW (trained as NOT
shockable) were 150 bpm or more by autocorrelation, and none of the 13
VT_SLOW episodes was mostly 100 to 150 (experiments/vt_rate_check.py). An
episode also changes rate as it runs, so one rate per episode is wrong even
when it is measured well.

The rules (config, "Shockable labels, one per window")
------------------------------------------------------
    more than one rhythm in the window        None          excluded
    NOISE, UNKNOWN                            kept as is    excluded
    VF or flutter, 200 uV peak to peak or more    VF        shockable
    VF or flutter under 200 uV                VF_FINE       intermediate, excluded
    VT at 150 bpm or more                     VT_FAST       shockable
    VT under 150 bpm                          VT_SLOW       intermediate, excluded
    VT with no regular rate (polymorphic)     VT_IRREGULAR  shockable
    any other rhythm                          its class     not shockable

Intermediate is AHA's category: no performance goal, reported, trained
neither way. Excluded windows are not put in the cache. Irregular VT is
polymorphic VT, which AHA counts as shockable VT and ACLS shocks like VF;
in the ECG Fragment Database every irregular VT window was VF or torsade de
pointes by the cardiologists' labels (config, SHOCKABLE_WINDOW_RHYTHMS).

Training and scoring
--------------------
The cache takes only single-rhythm windows (SHOCKABLE_WINDOW_PURITY, 1.0), as
Figuera et al. 2016 do. Scoring on whole recordings (false_alarm_rate,
shockable_report) keeps the dominant-rhythm rule its episode timing was
measured with (SHOCKABLE_PURITY, 0.5), with the same per-window rate and
amplitude rules: an alarm over an intermediate window is counted apart,
neither true nor false, and hard-negative mining never takes one.

The rate
--------
How far apart the window repeats itself: the autocorrelation peak at the
shortest lag that is within VT_RATE_NEAR_TOP of the highest peak, searched
over VT_RATE_LAG_SEC (300 to 50 bpm). Taking the shortest such lag keeps a
regular rhythm from being read at half its rate off a multiple of its cycle.
The height of the highest peak is the window's regularity, 1 when it repeats
exactly; under VT_RATE_MIN_REGULARITY the window does not repeat (fibrillation,
polymorphic VT, artifact) and gets no rate. There is no beat detection and no
amplitude threshold, so a small VT reads the same rate as a large one: the
property ANSI/AAMI EC13 tests in heart-rate meters with VT at half, normal
and double size (experiments/shockable_labels_check.py runs those waveforms).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .. import config as C


def acf_rates(X: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """(rate in bpm, regularity) for every row of X; rate NaN when irregular."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    X = X - X.mean(axis=1, keepdims=True)
    n = X.shape[1]
    F = np.fft.rfft(X, 2 * n, axis=1)
    ac = np.fft.irfft(F * np.conj(F), 2 * n, axis=1)[:, :n]
    a0 = ac[:, :1]
    ac = np.where(a0 > 0, ac / np.where(a0 > 0, a0, 1.0), 0.0)
    ac = ac * n / (n - np.arange(n))[None, :]          # unbiased: no fall with lag
    k0 = int(round(C.VT_RATE_LAG_SEC[0] * fs))
    k1 = min(int(round(C.VT_RATE_LAG_SEC[1] * fs)), n - 1)
    rate, reg = np.full(len(X), np.nan), np.zeros(len(X))
    for i in range(len(X)):
        seg = ac[i, k0:k1 + 1]
        p, _ = find_peaks(seg)
        if p.size == 0:
            continue
        h = seg[p]
        reg[i] = float(h.max())
        if reg[i] < C.VT_RATE_MIN_REGULARITY:
            continue
        lag = p[int(np.argmax(h >= C.VT_RATE_NEAR_TOP * reg[i]))] + k0
        rate[i] = 60.0 * fs / lag
    return rate, reg


def _single_rhythm(intervals, lo: int, hi: int, purity: float):
    """The rhythm covering at least `purity` of [lo, hi), else None."""
    cover: dict = {}
    for s, e, lab in intervals:
        overlap = min(hi, e) - max(lo, s)
        if overlap > 0:
            cover[lab] = cover.get(lab, 0) + overlap
    if not cover:
        return None
    lab, amount = max(cover.items(), key=lambda kv: kv[1])
    return lab if amount >= purity * (hi - lo) - 1e-9 else None


def label_windows(signal: np.ndarray, fs: int, intervals, starts, win: int,
                  purity: float | None = None) -> list:
    """One label per window [s, s + win): see the module docstring.

    intervals are windows.rhythm_intervals(rec): (start, end, rhythm class),
    VT not yet split by rate. purity: share of the window one rhythm must
    cover (default SHOCKABLE_WINDOW_PURITY, the cache's).
    """
    purity = C.SHOCKABLE_WINDOW_PURITY if purity is None else purity
    out, vt = [], []
    for k, lo in enumerate(starts):
        lab = _single_rhythm(intervals, lo, lo + win, purity)
        if lab == "VF":
            lab = "VF" if np.ptp(signal[lo:lo + win]) >= C.FINE_VF_MAX_PTP_MV else "VF_FINE"
        elif lab == "VT":
            vt.append(k)
        out.append(lab)
    if vt:
        rate, _ = acf_rates(np.stack([signal[starts[k]:starts[k] + win] for k in vt]), fs)
        for k, r in zip(vt, rate):
            out[k] = ("VT_IRREGULAR" if not np.isfinite(r)
                      else "VT_FAST" if r >= C.VT_FAST_BPM else "VT_SLOW")
    return out


# AHA's intermediate rhythms: no goal, reported, trained neither way.
INTERMEDIATE = ("VT_SLOW", "VF_FINE")


def window_class(label) -> str | None:
    """shockable, not_shockable, or None for a window the cache leaves out."""
    if label is None or label in C.SHOCKABLE_WINDOW_EXCLUDED:
        return None
    return "shockable" if label in C.SHOCKABLE_WINDOW_RHYTHMS else "not_shockable"


def record_window_labels(rec, n: int | None = None, purity: float | None = None) -> list:
    """Labels for a record's shockable windows at the deployment stride, in order.

    n: only the first n windows (an evaluation that scored n). A record with
    no rhythm annotation at all gets "UNLABELLED" throughout: not shockable,
    as the evaluation has always counted unannotated recordings (healthy
    days); the cache takes nothing from them (windows.extract_shockable).
    """
    from .windows import rhythm_intervals
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    total = 1 + (len(rec.signal) - win) // stride if len(rec.signal) >= win else 0
    count = total if n is None else min(n, total)
    iv = rhythm_intervals(rec)
    if not iv:
        return ["UNLABELLED"] * count
    return label_windows(rec.signal, rec.fs, iv, [i * stride for i in range(count)], win, purity)


def scoring_flags(rec, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(shockable, intermediate) for the first n windows, as scoring reads them.

    Dominant rhythm over the window (SHOCKABLE_PURITY), so episode onsets and
    delays are measured as before; per-window rate and amplitude rules, so a
    fast or polymorphic VT is shockable whatever its episode, and a slow VT or
    fine VF is intermediate. Windows past the record's end are neither.
    """
    labs = record_window_labels(rec, n, purity=C.SHOCKABLE_PURITY)
    shock, inter = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i, lab in enumerate(labs):
        shock[i] = lab in C.SHOCKABLE_WINDOW_RHYTHMS
        inter[i] = lab in INTERMEDIATE
    return shock, inter
