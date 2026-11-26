"""The patient's own beat (the template) and how much every beat looks like it.

    from munji_ai.rules import beat_match as bm
    peaks, ext = bm.beat_windows(x, peaks, fs)       # samples around every R peak
    tpl = bm.first_template(peaks / fs, ext, fs)     # patch-on: the first qualifying 10 s
    shape, size = bm.match(ext, tpl, fs)             # every beat against it
    bm.alike(shape, size), bm.unlike(shape, size)
    stored = tpl.to_dict(); tpl = bm.Template.from_dict(stored)   # kept on the phone

Used by the fast narrow rhythm (SVT) rules, docs/fast_narrow_rhythm.md.

Where it comes from
-------------------
Moved unchanged from experiments/template_wct.py: the "dominant" calibration
and the shape and size match that produced every SVT result. Nothing was
retuned in the move; tests/test_beat_match.py checks that both give the same
numbers. The wide (T-wave) match of that file is not moved: the SVT rules do
not use it.

Not the same as rules/beat_template.py. That one is the VF veto's baseline
(the first clear minute, 120 ms before to 200 ms after R) and is left as it is.

The template
------------
Learnt once at patch-on, from the first 10 s stretch (counted from the start
of the wear) with:

- a rate of 40 to 100 bpm, from 6 beats or more;
- one main shape: the beat that most of the stretch's beats match at a
  correlation of 0.90 or more, holding 60% of them. When 80% of the beats
  already match the stretch's median beat, that median is taken as it is.

The template is the median of those beats, so ectopic beats in the stretch are
left out instead of spoiling it.

It stays fixed for the whole wear and is stored on the phone (to_dict /
from_dict), so a restarted app does not learn a new one, perhaps during an
arrhythmia. A new patch learns a new one. Re-learning during a wear was tried
and dropped: it learnt a VT's own shape (VFDB 420).

The match
---------
Every beat is the signal 100 ms either side of its R peak. Its shape is the
best Pearson correlation with the template over shifts of up to 40 ms either
way; its size is its peak-to-peak range, at that shift, over the template's.

    alike    shape 0.90 or more, and size from half to twice the template's
    unlike   shape under 0.60, or size under half or over twice

A beat with no template is neither (NaN).

These are measurement constants, chosen in experiments/template_wct.py, not
clinical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRE_SEC = POST_SEC = 0.10      # beat window either side of R
LAG_SEC = 0.04                 # alignment search either side
CAL_SEC = 10.0                 # calibration stretch
CAL_BPM = (40.0, 100.0)        # its rate
CAL_MIN_BEATS = 6
CAL_MATCH = 0.90               # "one shape": a correlation of 0.90 or more ...
CAL_SHARE = 0.80               # ... for 80% of the beats with the median beat (taken at once)
CAL_DOMINANT = 0.60            # ... or for 60% of them with the main shape
ALIKE_C = 0.90                 # alike: keeps the template's shape
UNLIKE_C = 0.60                # unlike: has lost it
SIZE_FACTOR = 2.0              # size outside half to twice: not the same beat
TEMPLATE_VERSION = 1           # bump when the stored form changes


def sizes(fs: int) -> tuple:
    """(samples before R, beat length L, alignment lag) at fs."""
    pre, post, lag = (int(round(v * fs)) for v in (PRE_SEC, POST_SEC, LAG_SEC))
    return pre, pre + post + 1, lag


def beat_windows(x: np.ndarray, peaks: np.ndarray, fs: int) -> tuple:
    """(peaks kept, (n, L + 2 lag) samples around each): the beat window plus
    room to shift it. Beats too close to either end of x are dropped."""
    pre, L, lag = sizes(fs)
    peaks = np.asarray(peaks, np.int64)
    ok = (peaks - pre - lag >= 0) & (peaks - pre + L + lag <= len(x))
    p = peaks[ok]
    if p.size == 0:
        return p, np.zeros((0, L + 2 * lag))
    return p, np.asarray(x)[(p - pre - lag)[:, None] + np.arange(L + 2 * lag)[None, :]]


def _z(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    n = np.sqrt((a * a).sum(axis=-1, keepdims=True))
    return a / np.where(n > 0, n, np.inf)


def best_corr(ext: np.ndarray, tpl: np.ndarray, with_shift: bool = False):
    """Best Pearson correlation of each beat with tpl over the shifts; with
    with_shift, also the shift (0 to 2 lag) where it was found."""
    L = tpl.size
    t = _z(np.asarray(tpl, np.float64)[None, :])[0]
    r, k = np.empty(len(ext)), np.empty(len(ext), np.int64)
    for i in range(0, len(ext), 4096):
        w = _z(np.lib.stride_tricks.sliding_window_view(ext[i:i + 4096], L, axis=1))
        c = w @ t
        r[i:i + 4096], k[i:i + 4096] = c.max(axis=1), c.argmax(axis=1)
    return (r, k) if with_shift else r


def stretch_template(ext: np.ndarray, L: int, lag: int):
    """The stretch's median beat if 80% of its beats match it, else None."""
    if len(ext) < CAL_MIN_BEATS:
        return None
    med = np.median(ext[:, lag:lag + L], axis=0)
    r, k = best_corr(ext, med, with_shift=True)
    good = r >= CAL_MATCH
    if good.mean() < CAL_SHARE:
        return None
    w = np.lib.stride_tricks.sliding_window_view(ext[good], L, axis=1)
    return np.median(w[np.arange(int(good.sum())), k[good]], axis=0)


def dominant_template(ext: np.ndarray, L: int, lag: int):
    """The median of the stretch's main shape if it holds 60% of the beats,
    else None. The main shape is the beat that most beats match at 0.90 or
    more (ties: the one they match best). A stretch that stretch_template
    accepts is taken as it would be."""
    if len(ext) < CAL_MIN_BEATS:
        return None
    quick = stretch_template(ext, L, lag)
    if quick is not None:
        return quick
    R = np.stack([best_corr(ext, ext[i, lag:lag + L]) for i in range(len(ext))])
    m = int(np.argmax((R >= CAL_MATCH).sum(axis=1) + 1e-3 * R.mean(axis=1)))
    good = R[m] >= CAL_MATCH
    if good.sum() < CAL_MIN_BEATS or good.mean() < CAL_DOMINANT:
        return None
    _, k = best_corr(ext[good], ext[m, lag:lag + L], with_shift=True)
    w = np.lib.stride_tricks.sliding_window_view(ext[good], L, axis=1)
    return np.median(w[np.arange(int(good.sum())), k], axis=0)


def calibrations(t: np.ndarray, ext: np.ndarray, fs: int) -> tuple:
    """(stretches, counts): (start s, end s, template) of every CAL_SEC
    stretch, counted from t = 0, that qualifies by rate and shape; and how
    many stretches there were, how many had the rate, how many also one shape.
    t: each beat's time in seconds; ext: beat_windows' samples for those beats."""
    _, L, lag = sizes(fs)
    out, n = [], {"stretches": 0, "rate": 0, "shape": 0}
    if len(t) < CAL_MIN_BEATS:
        return out, n
    edges = np.arange(0.0, t[-1] + CAL_SEC, CAL_SEC)
    idx = np.searchsorted(t, edges)
    n["stretches"] = len(edges) - 1
    for k in range(len(edges) - 1):
        i, j = int(idx[k]), int(idx[k + 1])
        if j - i < CAL_MIN_BEATS or t[j - 1] <= t[i]:
            continue
        bpm = 60.0 * (j - i - 1) / (t[j - 1] - t[i])
        if not CAL_BPM[0] <= bpm <= CAL_BPM[1]:
            continue
        n["rate"] += 1
        tpl = dominant_template(ext[i:j], L, lag)
        if tpl is not None:
            n["shape"] += 1
            out.append((float(edges[k]), float(edges[k + 1]), tpl))
    return out, n


@dataclass
class Template:
    """The patient's own beat: L samples at fs, learnt from start to end
    seconds after the patch went on."""
    waveform: np.ndarray
    fs: int
    start: float
    end: float

    def to_dict(self) -> dict:
        """A JSON-ready form to keep on the phone."""
        return {"version": TEMPLATE_VERSION, "fs": int(self.fs), "start": float(self.start),
                "end": float(self.end), "waveform": [float(v) for v in self.waveform]}

    @classmethod
    def from_dict(cls, d: dict) -> "Template":
        """Back from to_dict. Refuses another version, or a length that does
        not fit its fs: a stored template the code cannot read is not used."""
        if d.get("version") != TEMPLATE_VERSION:
            raise ValueError(f"template version {d.get('version')}, expected {TEMPLATE_VERSION}")
        fs = int(d["fs"])
        w = np.asarray(d["waveform"], np.float64)
        if w.ndim != 1 or w.size != sizes(fs)[1] or not np.all(np.isfinite(w)):
            raise ValueError(f"template of {w.size} samples does not fit fs {fs}")
        return cls(waveform=w, fs=fs, start=float(d["start"]), end=float(d["end"]))


def first_template(t: np.ndarray, ext: np.ndarray, fs: int, blocked=()) -> Template | None:
    """The patch-on template: the first stretch that qualifies and does not
    overlap any (start s, end s) in blocked. In the app nothing is blocked;
    on annotated recordings, stretches touching a ventricular, paced, noisy
    or unknown rhythm are (as the experiments did)."""
    cal, _ = calibrations(t, ext, fs)
    for s, e, w in cal:
        if not any(bs < e and s < be for bs, be in blocked):
            return Template(waveform=np.asarray(w, np.float64), fs=int(fs), start=s, end=e)
    return None


def match(ext: np.ndarray, tpl: Template | None, fs: int) -> tuple:
    """(shape, size) for every beat against the template: the best
    correlation over the shifts, and the peak-to-peak size at that shift over
    the template's. NaN for every beat when there is no template."""
    n = len(ext)
    if tpl is None or n == 0:
        return np.full(n, np.nan), np.full(n, np.nan)
    if int(tpl.fs) != int(fs):
        raise ValueError(f"template at {tpl.fs} Hz, beats at {fs} Hz")
    _, L, _ = sizes(fs)
    r, k = best_corr(ext, tpl.waveform, with_shift=True)
    aligned = np.lib.stride_tricks.sliding_window_view(ext, L, axis=1)[np.arange(n), k]
    size = np.ptp(aligned, axis=1) / max(float(np.ptp(tpl.waveform)), 1e-9)
    return r, size


def alike(shape: np.ndarray, size: np.ndarray, c: float = ALIKE_C) -> np.ndarray:
    """Keeps the template's shape (c or more) and size (half to twice). No
    template (NaN): never alike."""
    with np.errstate(invalid="ignore"):
        return (shape >= c) & (size >= 1.0 / SIZE_FACTOR) & (size <= SIZE_FACTOR)


def unlike(shape: np.ndarray, size: np.ndarray, c: float = UNLIKE_C) -> np.ndarray:
    """Has lost the template's shape (under c) or size (outside half to
    twice). No template (NaN): never unlike."""
    with np.errstate(invalid="ignore"):
        return (shape < c) | (size < 1.0 / SIZE_FACTOR) | (size > SIZE_FACTOR)
