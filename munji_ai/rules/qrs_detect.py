"""QRS detection with the decision rules of Pan and Tompkins (1985).

    from munji_ai.rules.qrs_detect import detect_qrs
    r_peaks = detect_qrs(window_mv, fs)

Why not features.sqi.detect_r_pantompkins
-----------------------------------------
That function keeps Pan-Tompkins' filtering but replaces its decision rules
with one fixed threshold per window (mean + 0.5 SD of the integrated signal).
It is fine for the signal-quality baseline, which only needs it to agree or
disagree with a second detector, and it stays as it is so that baseline does
not change. For counting beats it fails in a specific, documented way: one
burst of artifact raises the threshold for the whole window and every real
beat under it is lost. On NSRDB record 16272 that turned an hour of motion
artifact into false asystole and false sustained bradycardia, with the missed
beats normal-sized (median 1.9 mV). Pan-Tompkins++ (arXiv 2211.03171)
documents the same failure after exceptionally large signal components.

The decision rules are those of Pan-Tompkins++ (Imtiaz and Khan, arXiv
2211.03171), which keeps the 1985 structure and fixes its documented failures,
including "no R-peak after an exceptionally high R-peak":

    adaptive levels   signal and noise peak levels update with every peak
                      (1/8 new); threshold 1 = noise + 1/4 (signal - noise)
    threshold 2       0.4 x threshold 1 (PT: 0.5, which skipped low beats)
    search-back       no QRS for 1 s or 166% of the mean of the last 8 RR:
                      the largest peak since 360 ms after the last R is a QRS
                      if above threshold 3 = 1/2 threshold 2 + 1/2 the mean of
                      the last 3 QRS and the next 3 peaks; levels then update
                      fast (3/4 new)
    stall rule        no QRS for 1.4 s: the largest such peak above 20% of
                      threshold 2 is a QRS (fast update). This is what brings
                      the threshold back down after a burst of artifact.
    refractory        231 ms (up to 260 bpm)
    T-wave check      a peak within 360 ms, or within half the mean RR, whose
                      mean slope over 70 ms is under 60% of the last QRS's
    learning phase    the first 2 s: threshold 1 = max / 3, threshold 2 =
                      mean / 2

Added here, not in either paper, and labelled as ours:
  - a peak is accepted only if its QRS exceeds the EC13 floor (0.15 mV peak
    to valley);
  - re-learning: when nothing has been accepted for the stall time plus 2 s,
    the original learning phase is re-run on the last 2 s. Measured need: an
    artifact burst 840x a QRS in the integrated signal leaves even the stall
    rule's threshold 17x too high, and the levels never come down otherwise;  is what keeps the stall rule and re-learning, both permissive by
design, from turning noise or P waves into beats during a real pause.

Processing is offline and zero-phase (sosfiltfilt) like the rest of the
model path; the decisions run in time order. The paper's 60 ms flattop
smoothing before integration is not used: the 0.5-40 Hz model path already
removes most of what it targets.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig

BAND_HZ = (5.0, 18.0)          # Pan-Tompkins++ (PT: 5-15 Hz)
INTEGRATION_MS = 150.0         # moving-window integration
REFRACTORY_MS = 231.0          # Pan-Tompkins++: up to 260 bpm (PT: 200 ms)
T_WAVE_MS = 360.0
T_SLOPE = 0.6                  # Pan-Tompkins++ (PT: 0.5)
SEARCHBACK_RR = 1.66           # x mean of the last 8 RR intervals
SEARCHBACK_SEC = 1.0           # Pan-Tompkins++: or 1 s, whichever comes first
STALL_SEC = 1.4                # Pan-Tompkins++: no QRS for 1.4 s after a huge peak
LEARN_SEC = 2.0
LOCATE_MS = 75.0               # R located at the band-passed maximum within this
AMP_HALF_MS = 60.0             # amplitude read over +/- this (as the rate engine)
AMP_RECOVER_FRACTION = 0.5     # ours: recovered beats >= half the usual QRS
AMP_MEMORY_RANGE = (0.5, 1.5)  # ours: only beats this close to the usual size are remembered
RR_MEMORY_RANGE = (0.6, 1.6)   # ours: ... and arriving at this fraction of the mean RR


def _preprocess(x: np.ndarray, fs: int):
    nyq = fs / 2.0
    sos = sig.butter(2, [BAND_HZ[0] / nyq, min(BAND_HZ[1], nyq * 0.99) / nyq],
                     btype="band", output="sos")
    bp = sig.sosfiltfilt(sos, x)
    d = np.diff(bp, prepend=bp[0])
    w = max(int(INTEGRATION_MS / 1000.0 * fs), 1)
    integ = np.convolve(d * d, np.ones(w) / w, mode="same")
    return bp, d, integ


def detect_qrs(x: np.ndarray, fs: int, min_amp_mv: float | None = None,
               count_rejected: bool = False):
    """R-peak sample indices.

    min_amp_mv: a peak is accepted as a QRS only if the signal around it is
    larger than this, peak to valley (ANSI/AAMI EC13, 0.15 mV). Applied inside
    the decisions, so a sub-floor peak counts as noise and never resets the
    signal level. With count_rejected=True, also returns how many otherwise
    accepted peaks the floor turned down.
    """
    x = np.asarray(x, dtype=np.float64)
    empty = np.array([], dtype=np.int64)
    if x.size < int(0.5 * fs):
        return (empty, 0) if count_rejected else empty
    bp, d, integ = _preprocess(x, fs)
    refractory = int(REFRACTORY_MS / 1000.0 * fs)
    cand, _ = sig.find_peaks(integ, distance=max(refractory, 1))
    if cand.size == 0:
        return (empty, 0) if count_rejected else empty

    learn = integ[:max(int(LEARN_SEC * fs), 1)]
    thr1 = float(learn.max()) / 3.0
    thr2 = 0.5 * float(learn.mean())
    spk, npk = thr1, thr2
    locate = int(LOCATE_MS / 1000.0 * fs)
    amp_half = int(AMP_HALF_MS / 1000.0 * fs)
    slope_win = int(0.070 * fs)
    rejected = 0

    def r_at(c):
        a, b = max(0, c - locate), min(x.size, c + locate + 1)
        return a + int(np.argmax(np.abs(bp[a:b])))

    def amp(c):
        r = r_at(c)
        seg = x[max(0, r - amp_half):min(x.size, r + amp_half + 1)]
        return float(np.ptp(seg)) if seg.size else 0.0

    def big_enough(c):
        return min_amp_mv is None or amp(c) > min_amp_mv

    # The patient's own QRS size: median of the last 8 beats accepted the
    # normal way, leaving out anything over 3x that median so a burst of
    # artifact cannot enter it. Ours, not in the papers; see module docstring.
    amp_mem: list[float] = []

    def remember(c):
        """Add a beat to the size memory only if it looks like a sinus beat:
        its size near the usual one, and arriving at a plausible interval.
        Artifact peaks are neither, so they cannot teach the memory a false
        size."""
        a = amp(c)
        if amp_mem:
            med = float(np.median(amp_mem))
            if not (AMP_MEMORY_RANGE[0] * med <= a <= AMP_MEMORY_RANGE[1] * med):
                return
            if len(qrs) >= 3:
                rr = qrs[-1] - qrs[-2]
                mrr = float(np.mean(np.diff(qrs[-9:-1]))) if len(qrs) >= 3 else rr
                if not (RR_MEMORY_RANGE[0] * mrr <= rr <= RR_MEMORY_RANGE[1] * mrr):
                    return
        amp_mem.append(a)
        del amp_mem[:-8]

    def like_this_patient(c):
        """A recovered beat must be at least half the patient's usual QRS."""
        if not amp_mem:
            return True
        return amp(c) >= AMP_RECOVER_FRACTION * float(np.median(amp_mem))

    def slope(c):
        return float(np.mean(np.abs(d[max(0, c - slope_win):c + 1])))

    qrs: list[int] = []
    last_slope = None

    def mean_rr():
        return float(np.mean(np.diff(qrs[-9:]))) if len(qrs) >= 2 else None

    def accept(c, fast: bool):
        nonlocal spk, last_slope
        qrs.append(c)
        v = float(integ[c])
        spk = (0.75 * v + 0.25 * spk) if fast else (0.125 * v + 0.875 * spk)
        last_slope = slope(c)

    learn_n = int(LEARN_SEC * fs)
    stall_n = int(STALL_SEC * fs)

    def relearn(c):
        """Re-run the learning phase on the 2 s before c (our extension).

        After a burst of artifact hundreds of times a QRS, even the stall rule's
        20% of threshold 2 stays far above every real beat, and the levels only
        fall when something is accepted. Re-applying the paper's own start-up
        rule to the last 2 s brings them back once the signal is clean, and
        candidates in those 2 s are reconsidered with the new threshold.
        """
        nonlocal spk, npk, thr1, thr2
        saved = (spk, npk, thr1, thr2)
        seg = integ[max(0, c - learn_n):c + 1]
        thr1 = float(seg.max()) / 3.0
        thr2 = 0.5 * float(seg.mean())
        spk, npk = thr1, thr2
        start = (qrs[-1] + refractory) if qrs else 0
        found = 0
        for r in cand[(cand >= max(start, c - learn_n)) & (cand < c)]:
            r = int(r)
            if qrs and r - qrs[-1] < refractory:
                continue
            if integ[r] > thr1 and big_enough(r) and like_this_patient(r):
                accept(r, fast=False)
                found += 1
        if not found:
            # Nothing of the patient's usual size: this was not lost beats
            # under a raised threshold, it is a real absence (or P waves only).
            # Keep the old levels, so the threshold never drops to P-wave size.
            spk, npk, thr1, thr2 = saved

    for idx, c in enumerate(cand):
        c = int(c)
        if qrs and c - qrs[-1] < refractory:
            continue
        since = c - (qrs[-1] if qrs else 0)
        if since > stall_n + learn_n and integ[c] <= thr1:
            relearn(c)
            thr1 = npk + 0.25 * (spk - npk)
            thr2 = 0.4 * thr1
            if qrs and c - qrs[-1] < refractory:
                continue
        v = float(integ[c])
        rr = (c - qrs[-1]) if qrs else None
        mrr = mean_rr()
        if v > thr1:
            is_t = (rr is not None and last_slope is not None
                    and (rr < T_WAVE_MS / 1000.0 * fs or (mrr and rr < 0.5 * mrr))
                    and slope(c) < T_SLOPE * last_slope)
            if not is_t and big_enough(c):
                accept(c, fast=False)
                remember(c)
            else:
                rejected += int(not is_t)
                npk = 0.125 * v + 0.875 * npk
        else:
            found = False
            long_gap = rr is not None and (rr > SEARCHBACK_SEC * fs
                                           or (mrr and rr > SEARCHBACK_RR * mrr))
            if long_gap:
                lo = qrs[-1] + int(T_WAVE_MS / 1000.0 * fs)
                win = cand[(cand > lo) & (cand <= c)]
                if win.size:
                    best = int(win[np.argmax(integ[win])])
                    nxt = cand[idx + 1:idx + 4]
                    ref = np.concatenate([integ[np.asarray(qrs[-3:])], integ[nxt]])
                    thr3 = 0.5 * thr2 + 0.5 * float(ref.mean())
                    stalled = rr > STALL_SEC * fs
                    if (integ[best] > thr3 or (stalled and integ[best] > 0.2 * thr2)) \
                            and big_enough(best) and like_this_patient(best):
                        accept(best, fast=True)
                        found = True
            if not found:
                npk = 0.125 * v + 0.875 * npk
        thr1 = npk + 0.25 * (spk - npk)
        thr2 = 0.4 * thr1

    peaks = np.unique(np.asarray([r_at(c) for c in qrs], dtype=np.int64))
    return (peaks, rejected) if count_rejected else peaks
