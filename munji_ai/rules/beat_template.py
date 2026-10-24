"""Patient-specific beat template: the baseline a VF alarm is checked against.

    tmpl = build_template(signal, fs, is_clear)        # once, at setup
    if beats_present(window, tmpl, fs):                 # on each VF candidate
        ...  the patient's own heartbeats are still there: not fibrillation

Why
---
ShockableNet reads motion artifact as fibrillation, because on one lead both
look like disorganised activity. On a full ambulatory day that costs dozens of
false alarms. Wearable defibrillators solve the same problem the same way:

    LifeVest   compares the current QRS morphology with the patient's normal
               rhythm baseline template, recorded at device setup; if they
               match, the device keeps monitoring instead of alarming.
    ASSURE     an adaptive matched filter learns the patient's specific QRS
               morphology, to tell real QRS complexes from noise.

The physiology is what makes it work. Artifact is added on top of a heart that
is still beating normally, so the patient's own complexes are still in the
trace underneath. In fibrillation they are gone: there are no organised QRS
complexes at all. Finding several beats that match the patient's own template,
at a plausible rate, is therefore evidence against VF that a window classifier
cannot see by itself.

How
---
build_template   finds the first clear minute (no annotated arrhythmia, a
                 plausible regular rate, beats that agree with each other) and
                 keeps the median beat, 120 ms before the R peak to 200 ms
                 after it. That stands in for the supervised baseline a
                 clinician records at setup.
match_beats      slides the template over a window as a normalised cross-
                 correlation, the matched filter, and keeps peaks above a
                 threshold at least 300 ms apart.
beats_present    true when enough matched beats are found at a plausible
                 rate (about 37 to 180 bpm).

The template also answers the wide-QRS question raised for WCT: a patient
whose normal QRS is already wide carries that width in their own template.

This module only decides. Thresholds are proposals to be measured with
eval/false_alarm_rate.py and reviewed by a cardiologist before any use.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..config import TARGET_FS
from ..features.sqi import detect_r_pantompkins

PRE_S, POST_S = 0.12, 0.20       # template span around the R peak
SEGMENT_S = 60.0                 # length of the baseline stretch
STEP_S = 30.0                    # how far to move when a stretch is rejected
SEARCH_S = 30 * 60.0             # give up after this much signal
MIN_BEATS_BASELINE = 40          # a minute of plausible rhythm
MIN_QUALITY = 0.85               # median agreement of the beats with the template
RR_OK = (0.33, 1.5)              # 40 to 180 bpm for the baseline stretch
REFRACTORY_S = 0.30              # two matches closer than this are one beat
RR_PLAUSIBLE = (0.33, 1.6)       # about 37 to 180 bpm for a veto


def _znorm_rows(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    M = M - M.mean(axis=-1, keepdims=True)
    return M / (M.std(axis=-1, keepdims=True) + eps)


def template_length(fs: int = TARGET_FS) -> tuple[int, int]:
    return int(round(PRE_S * fs)), int(round(POST_S * fs))


def build_template(signal: np.ndarray, fs: int = TARGET_FS, is_clear=None):
    """(template, start_sample, quality) for the first clear minute, or None.

    `is_clear(lo, hi)` says whether a stretch is free of annotated arrhythmia;
    without it every stretch is a candidate.
    """
    pre, post = template_length(fs)
    seg = int(SEGMENT_S * fs)
    stop = min(len(signal), int(SEARCH_S * fs)) - seg
    for lo in range(0, max(stop, 0) + 1, int(STEP_S * fs)):
        hi = lo + seg
        if is_clear is not None and not is_clear(lo, hi):
            continue
        x = np.asarray(signal[lo:hi], dtype=np.float64)
        peaks = detect_r_pantompkins(x, fs)
        if len(peaks) < MIN_BEATS_BASELINE:
            continue
        rr = np.diff(peaks) / fs
        if np.mean((rr >= RR_OK[0]) & (rr <= RR_OK[1])) < 0.9:
            continue
        peaks = peaks[(peaks - pre >= 0) & (peaks + post <= len(x))]
        if len(peaks) < MIN_BEATS_BASELINE:
            continue
        B = _znorm_rows(np.stack([x[p - pre:p + post] for p in peaks]))
        tmpl = _znorm_rows(np.median(B, axis=0)[None, :])[0]
        quality = float(np.median(B @ tmpl / len(tmpl)))
        if quality >= MIN_QUALITY:
            return tmpl, lo, quality
    return None


def ncc_trace(x: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of the template at every offset."""
    L = len(tmpl)
    if len(x) < L:
        return np.zeros(0)
    W = _znorm_rows(sliding_window_view(np.asarray(x, dtype=np.float64), L))
    return W @ tmpl / L


def match_beats(x: np.ndarray, tmpl: np.ndarray, thr: float,
                fs: int = TARGET_FS, ncc: np.ndarray | None = None) -> np.ndarray:
    """Positions where the template matches at `thr` or above, one per beat."""
    c = ncc_trace(x, tmpl) if ncc is None else ncc
    cand = np.flatnonzero(c >= thr)
    if cand.size == 0:
        return cand
    gap = int(REFRACTORY_S * fs)
    kept: list[int] = []
    for i in cand[np.argsort(-c[cand], kind="stable")]:
        if all(abs(i - k) >= gap for k in kept):
            kept.append(int(i))
    return np.sort(np.asarray(kept))


def beats_present(x: np.ndarray, tmpl: np.ndarray, thr: float, min_beats: int,
                  fs: int = TARGET_FS, ncc: np.ndarray | None = None) -> bool:
    """Enough of the patient's own beats, at a plausible rate, to rule out VF."""
    m = match_beats(x, tmpl, thr, fs, ncc)
    if len(m) < min_beats:
        return False
    rr = np.diff(m) / fs
    return bool(RR_PLAUSIBLE[0] <= np.median(rr) <= RR_PLAUSIBLE[1])
