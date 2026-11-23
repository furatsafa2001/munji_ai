"""Sustained VT without the beat model: every beat against the patient's own.

    python experiments/template_wct.py                        # all sets
    python experiments/template_wct.py --negatives nsrdb      # quicker first look
    python experiments/template_wct.py --max-records 2        # smoke test
    python experiments/template_wct.py --no-gate --workers 6
    python experiments/template_wct.py --cal strict           # the first run's calibration
    python experiments/template_vtac.py                       # the beat test on VTaC

Why
---
The WCT alert counts PVCs from the beat model. On the 11 VT episodes of 30 s
or more in VFDB it fired on none: the model missed most VT beats (rate 48
against 114 on the signal) and called 77% of the rest normal. ShockableNet
covers VT only at 150 bpm or more, so VT from 100 to 150 has no working
detector today.

The idea
--------
The one the subcutaneous ICD uses to tell VT from a fast rhythm that comes
from above (S-ICD, Boston Scientific): the patient's own beat is stored at
rest as a template, and during a fast rhythm every beat is compared with it.
Beats that keep their usual shape mean a conducted rhythm (sinus, AF, SVT);
beats that lost it mean a ventricular origin. In the START study (Gold et
al., J Cardiovasc Electrophysiol 2012) this reached 100% sensitivity for
VT/VF and 98% specificity for SVT. QRS template matching on Holter recordings
separated ventricular beats at 98.4% sensitivity and 98.9% specificity
(PubMed 17805974). One lead is enough: the S-ICD senses one vector. Nothing
here is trained.

What runs, as the app would run it
----------------------------------
beats     the rate engine's own detector (rules.rate_engine.detect_beats:
          Pan-Tompkins++ with the EC13 floor), not the beat model, on the
          rate engine's 40 s windows, each run afresh. Fresh windows matter:
          the detector keeps the patient's usual QRS size and will not take a
          beat under half of it, so VT smaller than the sinus QRS is only
          found by a window that starts inside the VT and learns it there.
template  calibration, as at patch-on: the first 10 s stretch with a rate of
          40 to 100 and one main shape. --cal dominant (the default): the
          main shape is the beat that most of the stretch's beats match at
          0.90 or more, and it must hold 60% of them; the template is the
          median of those beats, so ectopic beats in the stretch are left
          out instead of spoiling it. Where rhythm labels exist, the stretch
          must not touch a ventricular, paced, noisy or unknown rhythm
          (CAL_EXCLUDED, the records' own tokens): sinus rhythm with frequent
          ventricular ectopy (VFDB's HGEA), first-degree block or a nodal
          rhythm qualify, as the patient's usual rhythm at patch-on would.
          --cal strict, the first run: 80% of the beats at 0.90 or more with
          their median, and only a stretch labelled NSR, AF, AFL or sinus
          brady. It found no template in 14 of 69 records, among them the
          two VFDB records whose pictures show VT from 100 to 150 bpm (607,
          611), so both were missed.
          The template stays fixed for the whole recording, as at patch-on.
          A template re-learnt during the recording was tried and dropped:
          it learnt the VT's own shape (VFDB 420: 98% of the VT's beats
          unlike the fixed template, 21% unlike the re-learnt one).
          Beats for calibration are the newest stride of every window.
match     every beat against the template, three ways:
            shape  Pearson correlation over 100 ms either side of R, the best
                   of shifts up to 40 ms (the first runs' only measure)
            wide   the same from 100 ms before to 250 ms after R, so the T
                   wave counts too; its template is the median of the same
                   calibration beats, aligned as the shape aligned them
            size   the beat's peak-to-peak over the shape span, aligned the
                   same way, as a share of the template's
          A beat is unlike the template when its correlation is under C, or,
          with the size check on, its size is under half or over twice the
          template's. VFDB 607 is why: its VT beats are a third of the
          patient's own beat but keep the shape over 200 ms (correlation
          0.83). The wide span and the factor 2 are ours, tested here.
rule      at the end of every window, over its last 30 s of beats (the WCT
          rule's span and its 80% share): rate above R (monitor-style trimmed
          mean, as the rate engine) and at least 80% of the beats unlike the
          template; then optionally:
            M          the beats alike each other (monomorphic: median
                       correlation with their own median beat at least M)
            onset      the fast rhythm began suddenly: where the running rate
                       (median of 4 RR intervals) first rose above R, the cycle
                       length fell by 9% or more against the 8 intervals
                       before. Sinus tachycardia speeds up gradually; VT starts
                       at once. The newest run with a beat in the 30 s is
                       judged, so the tail of a gradual run stays held back.
                       A run that never started suddenly still alerts once it
                       has lasted 60 s (the ICDs' sustained-rate safety; the
                       60 s is ours).
            stability  the rhythm is regular: the median change between
                       successive RR intervals over the 30 s is 40 ms or less.
                       VT is regular; AF is not.
          Onset 9% and stability 40 ms are the ICD settings Brugada et al.
          (J Cardiovasc Electrophysiol 1998) programmed and found best; there
          a sustained-rate safety took VT sensitivity from 90% to 100% while
          specificity for supraventricular rhythms fell from 96% to 83%.
          One alert per episode, re-armed after 30 s without the condition,
          as the decision layer does.
stride    2.5 s on the VT sets (the rate engine's), --neg-stride (5 s) on the
          long negative sets to keep them quick.
gate      with checkpoints/quality_gate.pt and PyTorch: the 30 s must average
          P(usable) >= 0.90 (GATE_MIN_P morphology), scored on 20 s windows
          normalised and split into 5 s as the beat path does.

Configurations
--------------
The first two runs swept R, C and M. Here each new check is added alone to
the run-2 rule (R 100, C 0.6, M 0.7), then all of them together, then the
whole rule with one setting moved at a time (CONFIGS).

Scored
------
VT episodes from the rhythm labels of every set that has them: 30 s or more
should alert, under 30 s should not. Alerts in VF are counted apart (an
emergency anyway, and ShockableNet's). Every other alert is false, per 24 h:
CUDB/VFDB outside VT and VF, NSRDB (healthy), LTAFDB (atrial fibrillation,
the commonest fast rhythm from above) and SVDB (supraventricular
arrhythmias). Negative sets use their annotated span only, as
eval_beats_model does for NSRDB. CUDB labels VT in few records (cu02 in the
first run), so an alert there may be real VT: false alerts are listed with
record, time and the rhythm label, to be looked at. VFDB labels much more
than slow VT as "VT": in the first run's pictures only 611 and part of 607
were VT from 100 to 150 bpm. Pictures of every VT episode:
experiments/template_wct_plot.py.

For every VT episode of 30 s or more: the rate on the signal and from the
detector, the share of beats unlike the template by each measure, their
median size, the rhythm's regularity, the cycle-length drop at its start,
and whether the template came from before or after the episode. That shows
why an episode is missed: the detector, the template or the rule.

Reads only. Test-split patients are skipped. Every figure is picked on this
data: the result says whether the direction works, not the final numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C  # noqa: E402
from munji_ai.data import preprocess as pp  # noqa: E402
from munji_ai.data.loader import load_record  # noqa: E402
from munji_ai.data.windows import _episode_bpm, rhythm_intervals  # noqa: E402
from munji_ai.eval.eval_beats_model import eligible  # noqa: E402
from munji_ai.rules.rate_engine import _trimmed_mean, detect_beats  # noqa: E402

WIN_SEC = C.RULES_WINDOW_SEC           # 40 s, the rate engine's window
POS_STEP = C.RULES_STRIDE_SEC          # 2.5 s, the rate engine's stride
PRE_SEC = POST_SEC = 0.10              # beat window either side of R
WIDE_POST_SEC = 0.25                   # "wide": to 250 ms after R, T wave included
LAG_SEC = 0.04                         # alignment search either side
CAL_SEC = 10.0                         # calibration stretch
CAL_BPM = (40.0, 100.0)
CAL_MIN_BEATS = 6
CAL_MATCH, CAL_SHARE = 0.90, 0.80      # "one shape": 80% of beats at 0.90 or more
CAL_RHYTHMS = {"NSR", "AF", "AFL", "BRADY"}   # --cal strict: conducted rhythms
CAL_DOMINANT = 0.60                    # --cal dominant: the main shape holds 60% of the beats
CAL_EXCLUDED = {"(VT", "(VF", "(VFL", "(VFIB", "(VER", "(IVR", "(PM", "(P", "(ASYS",
                "(NOISE", "(UNKNOWN"}  # --cal dominant: raw rhythm tokens no template comes from
CALS = ("dominant", "strict")
SUSTAIN = C.VT_SUSTAINED_SEC           # 30 s
FRACTION = C.VT_SUSTAINED_PVC_FRACTION  # 0.80
GATE_BLOCK_SEC = C.WINDOW_SEC["quality"]   # 5 s
SIZE_FACTOR = 2.0                      # unlike when under 1/2 or over 2 times the template's size
RUN_RR = 4                             # RR intervals in the running rate that starts a run
ONSET_BEFORE = 8                       # intervals before the start the drop is measured against
ONSET_DROP = 0.09                      # sudden: cycle length down 9% or more (Brugada 1998)
ONSET_OVERRIDE_SEC = 60.0              # a gradual run still alerts after this long (ours)
STABILITY_MS = 40.0                    # regular: median successive RR change (Brugada 1998)

BASE = {"match": "shape", "size": False, "C": 0.6, "R": 100.0, "M": 0.7,
        "onset": False, "stab": False, "gate": False}
ALL = {"size": True, "onset": True, "stab": True}


def _cfg(name: str, **kw):
    return name, {**BASE, **kw}


CONFIGS = [
    _cfg("run 2 rule"),
    _cfg("+ size", size=True),
    _cfg("+ wide window", match="wide"),
    _cfg("+ onset", onset=True),
    _cfg("+ stability", stab=True),
    _cfg("all: size, onset, stability", **ALL),
    _cfg("all, wide window", **ALL, match="wide"),
    _cfg("all, R 120", **ALL, R=120.0),
    _cfg("all, C 0.5", **ALL, C=0.5),
    _cfg("all, C 0.7", **ALL, C=0.7),
    _cfg("all, no M", **ALL, M=None),
    _cfg("all, gate on", **ALL, gate=True),
]
SHOW = ("run 2 rule", "all: size, onset, stability")   # per episode and false-alert lists
SHOW_C = BASE["C"]


def configs(gated: bool):
    return [(n, c) for n, c in CONFIGS if gated or not c["gate"]]


def _unlike_key(c: dict):
    return c["match"], c["size"], c["C"]


# ------------------------------------------------------------------- beats
def windowed_beats(x: np.ndarray, fs: int, step_sec: float):
    """The app's view: detect_beats on every 40 s window, each run afresh.

    Returns the window ends (samples) and each window's R peaks (samples)."""
    W, S = int(WIN_SEC * fs), int(round(step_sec * fs))
    ends = np.arange(W, len(x) + 1, S)
    return ends, [np.asarray(detect_beats(x[e - W:e], fs)[0], np.int64) + (e - W) for e in ends]


def stream(ends: np.ndarray, beats: list, step_sec: float, fs: int) -> np.ndarray:
    """One beat list for the recording: the newest stride of every window
    (all of the first window), duplicates dropped."""
    if not beats:
        return np.zeros(0, np.int64)
    S = int(round(step_sec * fs))
    p = np.sort(np.concatenate([beats[0]] + [b[b >= e - S] for e, b in zip(ends[1:], beats[1:])]))
    if p.size > 1:
        p = p[np.concatenate([[True], np.diff(p) > int(C.BEAT_DEDUP_SEC * fs)])]
    return p


def index_of(pos: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Indices in sorted pos of the values in want that are present."""
    if pos.size == 0 or want.size == 0:
        return np.zeros(0, np.int64)
    k = np.clip(np.searchsorted(pos, want), 0, pos.size - 1)
    return k[pos[k] == want]


# -------------------------------------------------------------- morphology
def _sizes(fs: int):
    pre, post, lag = (int(round(v * fs)) for v in (PRE_SEC, POST_SEC, LAG_SEC))
    return pre, pre + post + 1, lag


def extended(x: np.ndarray, peaks: np.ndarray, fs: int):
    """(peaks kept, (n, L + 2 lag) samples around each): the beat window plus
    room to shift it. Beats too close to either end are dropped."""
    pre, L, lag = _sizes(fs)
    ok = (peaks - pre - lag >= 0) & (peaks - pre + L + lag <= len(x))
    p = peaks[ok]
    if p.size == 0:
        return p, np.zeros((0, L + 2 * lag))
    return p, x[(p - pre - lag)[:, None] + np.arange(L + 2 * lag)[None, :]]


def wide_windows(x: np.ndarray, peaks: np.ndarray, fs: int):
    """(Lw, (n, Lw + 2 lag) float32): the wide span around every peak, the
    recording's edges padded so the rows stay aligned with peaks. Each row
    starts where extended()'s does, so the same shift aligns both."""
    pre, _, lag = _sizes(fs)
    Lw = pre + int(round(WIDE_POST_SEC * fs)) + 1
    pad = Lw + 2 * lag
    xp = np.pad(np.asarray(x, np.float32), (pad, pad), mode="edge")
    start = np.asarray(peaks, np.int64) - pre - lag + pad
    return Lw, xp[start[:, None] + np.arange(Lw + 2 * lag)[None, :]]


def _z(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    n = np.sqrt((a * a).sum(axis=-1, keepdims=True))
    return a / np.where(n > 0, n, np.inf)


def best_corr(ext: np.ndarray, tpl: np.ndarray, with_shift: bool = False):
    """Best Pearson correlation of each beat with tpl over the shifts."""
    L = tpl.size
    t = _z(np.asarray(tpl, np.float64)[None, :])[0]
    r, k = np.empty(len(ext)), np.empty(len(ext), np.int64)
    for i in range(0, len(ext), 4096):
        w = _z(np.lib.stride_tricks.sliding_window_view(ext[i:i + 4096], L, axis=1))
        c = w @ t
        r[i:i + 4096], k[i:i + 4096] = c.max(axis=1), c.argmax(axis=1)
    return (r, k) if with_shift else r


def stretch_template(ext: np.ndarray, L: int, lag: int):
    """Median beat of a stretch if the stretch is one shape, else None."""
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
    """Median of the stretch's main shape if it holds CAL_DOMINANT of the beats.

    The main shape is the beat that most beats match at CAL_MATCH (ties: the
    one they match best). A stretch the strict rule accepts is taken as it
    would be, so clean recordings calibrate exactly as before.
    """
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


def wide_template(ext: np.ndarray, wext: np.ndarray, tpl: np.ndarray, Lw: int):
    """The wide template: the median wide span of the calibration beats that
    match the shape template, each shifted as the shape match shifted it."""
    if len(ext) == 0:
        return None
    r, k = best_corr(ext, tpl, with_shift=True)
    good = r >= CAL_MATCH
    if good.sum() < 3:
        return None
    w = np.lib.stride_tricks.sliding_window_view(wext[good], Lw, axis=1)
    return np.median(w[np.arange(int(good.sum())), k[good]], axis=0)


def match(ext: np.ndarray, wext: np.ndarray, tpl, tplw, L: int, Lw: int) -> dict:
    """Every beat against the template: shape and wide correlation, size
    ratio. NaN where there is no template."""
    n = len(ext)
    out = {k: np.full(n, np.nan) for k in ("shape", "wide", "size")}
    if tpl is None or n == 0:
        return out
    r, k = best_corr(ext, tpl, with_shift=True)
    out["shape"] = r
    aligned = np.lib.stride_tricks.sliding_window_view(ext, L, axis=1)[np.arange(n), k]
    out["size"] = np.ptp(aligned, axis=1) / max(float(np.ptp(tpl)), 1e-9)
    if tplw is not None:
        out["wide"] = best_corr(wext, tplw)
    return out


def unlike(feat: dict, key) -> np.ndarray:
    """Per beat: unlike the template by (match, size check, C). No template
    (NaN) is never unlike."""
    m, size, c = key
    with np.errstate(invalid="ignore"):
        u = feat["wide" if m == "wide" else "shape"] < c
        if size:
            u |= (feat["size"] < 1.0 / SIZE_FACTOR) | (feat["size"] > SIZE_FACTOR)
    return u


def self_alike(ext: np.ndarray, L: int, lag: int) -> float:
    """Median correlation of beats with their own median beat (monomorphic)."""
    if len(ext) < 3:
        return float("nan")
    return float(np.median(best_corr(ext, np.median(ext[:, lag:lag + L], axis=0))))


def calibrations(t: np.ndarray, ext: np.ndarray, L: int, lag: int, mode: str = "dominant"):
    """(stretches, counts): (start, end, template) of every CAL_SEC stretch
    that qualifies by rate and shape, and how many stretches there were, how
    many had the rate (CAL_BPM, CAL_MIN_BEATS) and how many also one shape."""
    out, n = [], {"stretches": 0, "rate": 0, "shape": 0}
    if len(t) < CAL_MIN_BEATS:
        return out, n
    shape = dominant_template if mode == "dominant" else stretch_template
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
        tpl = shape(ext[i:j], L, lag)
        if tpl is not None:
            n["shape"] += 1
            out.append((float(edges[k]), float(edges[k + 1]), tpl))
    return out, n


def raw_intervals(rec, lo: int, hi: int) -> list:
    """(start s, end s, token) from the record's own rhythm markers, in
    seconds from lo: the tokens as written ("(HGEA", "(VER"), before they are
    folded into rhythm classes."""
    if len(rec.rhythm_starts) == 0:
        return []
    s = np.asarray(rec.rhythm_starts, np.int64)
    o = np.argsort(s, kind="stable")
    s = s[o]
    tok = [str(rec.rhythm_labels[i]).split("\x00")[0].strip() for i in o]
    e = np.append(s[1:], len(rec.signal))
    return [((max(a, lo) - lo) / rec.fs, (min(b, hi) - lo) / rec.fs, t)
            for a, b, t in zip(s, e, tok) if b > a and b > lo and a < hi]


def why_no_template(n: dict) -> str:
    """The calibration counts of a record, read out."""
    return (f"{n['stretches']} stretches of {CAL_SEC:.0f} s, {n['rate']} at {CAL_BPM[0]:.0f}-"
            f"{CAL_BPM[1]:.0f} bpm, {n['shape']} of one main shape, {n['allowed']} in an allowed rhythm")


# ------------------------------------------------------------ onset, rhythm
def onset_runs(ts: np.ndarray, R: float):
    """Per stream beat, for the rate R: the index of the first beat of the
    fast run it is in (-1 if the running rate is not above R), and per run
    start the drop in cycle length there (NaN: not enough beats before)."""
    n = len(ts)
    start = np.full(n, -1, np.int64)
    drop = {}
    if n <= RUN_RR:
        return start, drop
    rr = np.diff(ts)
    run = np.full(n, np.nan)
    run[RUN_RR:] = np.median(np.lib.stride_tricks.sliding_window_view(rr, RUN_RR), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fast = 60.0 / run > R
    s0 = -1
    for i in range(n):
        if fast[i]:
            if s0 < 0:
                s0 = i
                a = s0 - RUN_RR - ONSET_BEFORE
                drop[s0] = (1.0 - run[s0] / float(np.median(rr[a:s0 - RUN_RR]))) if a >= 0 else np.nan
            start[i] = s0
        else:
            s0 = -1
    return start, drop


def onset_ok(t_end: np.ndarray, ts: np.ndarray, start: np.ndarray, drop: dict) -> np.ndarray:
    """Per window end: does the onset check let an alert through? It judges
    the newest fast run with a beat in the window's last 30 s, so the tail of
    a gradual run, where the 30 s rate is still above R after the run ended,
    stays held back. Yes when there is no such run (the 30 s rule judges the
    rate), when the run began suddenly or cannot be judged, or when it has
    lasted ONSET_OVERRIDE_SEC."""
    out = np.ones(len(t_end), bool)
    if len(ts) == 0:
        return out
    newest = np.maximum.accumulate(np.where(start >= 0, np.arange(len(ts)), -1))
    i_last = np.searchsorted(ts, t_end, side="right") - 1
    for w, i in enumerate(i_last):
        j = int(newest[i]) if i >= 0 else -1
        if j < 0 or ts[j] <= t_end[w] - SUSTAIN:
            continue
        s0 = int(start[j])
        d = drop.get(s0, np.nan)
        if np.isfinite(d) and d < ONSET_DROP and ts[j] - ts[s0] < ONSET_OVERRIDE_SEC:
            out[w] = False
    return out


def instability_ms(beat_samples: np.ndarray, fs: int) -> float:
    """Median change between successive RR intervals, ms."""
    if beat_samples.size < 4:
        return float("nan")
    rr = np.diff(beat_samples) / fs
    return float(np.median(np.abs(np.diff(rr))) * 1000.0)


# -------------------------------------------------------------------- gate
def gate_blocks(x: np.ndarray, ckpt: Path) -> np.ndarray:
    """P(usable) of every 5 s block, from 20 s windows as the beat path."""
    from munji_ai.eval.gated_beat import gate_scores
    W = C.window_samples("beat")
    n = len(x) // W
    if n == 0:
        return np.zeros(0)
    X = np.stack([pp.normalize(x[k * W:(k + 1) * W]) for k in range(n)]).astype(np.float32)
    return gate_scores(X, ckpt).reshape(-1)


# ----------------------------------------------------------------- decisions
def decision_table(ends, wbeats, pos, u: dict, ext, L, lag, fs, gate_p) -> dict:
    """What the rules need, at the end of every window, over its last 30 s.

    The share of unlike beats is computed for every window and every unlike
    definition; the rate, the monomorphic check and the regularity only where
    some rule could fire, which is rare away from VT and keeps 24 h records
    quick.
    """
    n = len(ends)
    t_end = ends / fs
    ok = np.zeros(n, bool)
    frac = {k: np.full(n, np.nan) for k in u}
    bpm, mono, stab = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    S30 = int(SUSTAIN * fs)
    rmin = min(c["R"] for _, c in CONFIGS)
    for w in range(n):
        b = wbeats[w]
        b = b[b >= ends[w] - S30]
        if b.size < 3 or (b[-1] - b[0]) / fs < SUSTAIN - 60.0 / C.VT_MIN_BPM:
            continue                     # 30 s of beats, as the decision layer
        idx = index_of(pos, b)
        if idx.size < 3:
            continue
        ok[w] = True
        top = -1.0
        for k, uk in u.items():
            frac[k][w] = float(np.mean(uk[idx]))
            top = max(top, frac[k][w])
        if top >= FRACTION:
            m = _trimmed_mean(np.sort(np.diff(b) / fs), C.RATE_RUN_TRIM)
            bpm[w] = 60.0 / m if m > 0 else np.nan
            if bpm[w] > rmin:
                mono[w] = self_alike(ext[idx], L, lag)
                stab[w] = instability_ms(b, fs)
    usable = None
    if gate_p is not None:
        cs = np.concatenate([[0.0], np.cumsum(gate_p)])
        b0 = np.floor((t_end - SUSTAIN) / GATE_BLOCK_SEC + 1e-9).astype(int)
        b1 = np.ceil(t_end / GATE_BLOCK_SEC - 1e-9).astype(int)
        usable = np.zeros(n, bool)
        v = (b1 <= len(gate_p)) & (b1 > b0)
        usable[v] = (cs[b1[v]] - cs[b0[v]]) / (b1[v] - b0[v]) >= C.GATE_MIN_P["morphology"]
    return {"t": t_end, "ok": ok, "bpm": bpm, "mono": mono, "stab": stab, "frac": frac,
            "usable": usable}


def alert_times(t: np.ndarray, cond: np.ndarray, step: float) -> np.ndarray:
    """First time of every episode of cond. Re-armed once cond has been false
    for more than SUSTAIN (decision_layer._vt_step); on a grid of `step` that
    is a gap of more than SUSTAIN + step between two true decisions."""
    tt = t[cond]
    if tt.size == 0:
        return tt
    return tt[np.concatenate([[True], np.diff(tt) > SUSTAIN + step + 1e-6])]


def config_alerts(dec: dict, onset: dict, c: dict, step: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        cond = dec["ok"] & (dec["frac"][_unlike_key(c)] >= FRACTION) & (dec["bpm"] > c["R"])
        if c["M"] is not None:
            cond &= dec["mono"] >= c["M"]
        if c["stab"]:
            cond &= dec["stab"] <= STABILITY_MS
    if c["onset"]:
        cond &= onset[c["R"]]
    if c["gate"]:
        cond &= dec["usable"]
    return alert_times(dec["t"], cond, step)


# ------------------------------------------------------------------ record
def beats_and_template(rec, step: float, annotated_span: bool = False, mode: str = "dominant") -> dict:
    """Beats, templates and the match of every beat, as the app would get
    them. Shared by process() and template_wct_plot.py. mode: the
    calibration, "dominant" or "strict" (CALS).

    x, fs, ivs   the signal analysed (the annotated span or all) and its rhythm
                 intervals in seconds from its start
    ends, wbeats every window's end and R peaks (samples)
    pos, ext     every distinct R peak (sorted) and the samples around it
    si, ts       the stream (the newest stride of every window) as indices
                 into pos, and its times in seconds
    cal, first   every calibration stretch that qualifies (start, end,
                 template) and the one used as patch-on (None: no template)
    cal_counts   stretches, then with the rate, one shape, an allowed rhythm
    wide         the wide template (None without one)
    corr         per position in pos: "shape" and "wide" correlation and the
                 "size" ratio against the fixed template (NaN without one)
    """
    fs = rec.fs
    lo, hi = 0, len(rec.signal)
    if annotated_span and len(rec.beat_samples) >= 2:
        lo, hi = int(rec.beat_samples.min()), int(rec.beat_samples.max()) + 1
    x = rec.signal[lo:hi]
    ivs = [((max(s, lo) - lo) / fs, (min(e, hi) - lo) / fs, lab)
           for s, e, lab in rhythm_intervals(rec) if e > lo and s < hi]
    pre, L, lag = _sizes(fs)

    ends, wbeats = windowed_beats(x, fs, step)
    sb = stream(ends, wbeats, step, fs)
    every = np.unique(np.concatenate([sb] + wbeats)) if wbeats else sb
    pos, ext = extended(x, every, fs)
    Lw, wext = wide_windows(x, pos, fs)
    tpos = pos / fs
    si = index_of(pos, sb)                     # the stream, as indices into pos
    ts = tpos[si]

    cal, counts = calibrations(ts, ext[si], L, lag, mode)
    if mode == "strict":
        conducted = [(s, e) for s, e, lab in ivs if lab in CAL_RHYTHMS]
        allowed = [c for c in cal if not ivs or any(s <= c[0] and c[1] <= e for s, e in conducted)]
    else:
        blocked = [(s, e) for s, e, tok in raw_intervals(rec, lo, hi) if tok in CAL_EXCLUDED]
        allowed = [c for c in cal if not any(s < c[1] and c[0] < e for s, e in blocked)]
    counts["allowed"] = len(allowed)
    first = allowed[0] if allowed else None

    tplw = None
    if first is not None:
        cal_idx = si[(ts >= first[0]) & (ts < first[1])]
        tplw = wide_template(ext[cal_idx], wext[cal_idx], first[2], Lw)
    corr = match(ext, wext, first[2] if first else None, tplw, L, Lw)
    return {"x": x, "fs": fs, "ivs": ivs, "L": L, "Lw": Lw, "lag": lag, "ends": ends,
            "wbeats": wbeats, "pos": pos, "ext": ext, "si": si, "ts": ts, "cal": cal,
            "first": first, "wide": tplw, "corr": corr, "cal_counts": counts}


def process(rec, gate_ckpt, annotated_span: bool, step: float, mode: str = "dominant") -> dict:
    b = beats_and_template(rec, step, annotated_span, mode)
    x, fs, ivs, L, lag = b["x"], b["fs"], b["ivs"], b["L"], b["lag"]
    ends, wbeats, pos, ext = b["ends"], b["wbeats"], b["pos"], b["ext"]
    si, ts, first, corr = b["si"], b["ts"], b["first"], b["corr"]
    gated = gate_ckpt is not None
    cfgs = configs(gated)

    u = {k: unlike(corr, k) for k in {_unlike_key(c) for _, c in cfgs}}
    gate_p = gate_blocks(x, gate_ckpt) if gated else None
    dec = decision_table(ends, wbeats, pos, u, ext, L, lag, fs, gate_p)
    runs = {R: onset_runs(ts, R) for R in {c["R"] for _, c in cfgs}}
    onset = {R: onset_ok(dec["t"], ts, *runs[R]) for R in runs}
    alerts = {name: config_alerts(dec, onset, c, step) for name, c in cfgs}

    start100, drop100 = runs[BASE["R"]]
    episodes = []
    for s, e, lab in ivs:
        if lab != "VT":
            continue
        i, j = np.searchsorted(ts, [s, e])
        k = si[i:j]
        ref = _episode_bpm(x[int(s * fs):int(e * fs)], fs)
        has = first is not None and k.size > 0
        near = [d for s0, d in drop100.items() if s - 10.0 <= ts[s0] <= s + SUSTAIN]

        def share(key, k=k, has=has):
            return float(np.mean(unlike(corr, key)[k])) if has else float("nan")

        episodes.append({
            "s": s, "e": e, "dur": e - s,
            "rate_sig": float(ref) if ref else float("nan"),
            "rate_det": float(60.0 / np.median(np.diff(ts[i:j]))) if j - i >= 3 else float("nan"),
            "beats": int(j - i),
            "unlike": share(("shape", False, SHOW_C)),
            "unlike_size": share(("shape", True, SHOW_C)),
            "unlike_wide": share(("wide", False, SHOW_C)),
            "size": float(np.nanmedian(corr["size"][k])) if has else float("nan"),
            "corr": float(np.median(corr["shape"][k])) if has else float("nan"),
            "alike": self_alike(ext[k], L, lag),
            "stab": instability_ms(pos[k], fs) if k.size else float("nan"),
            "onset": near[0] if near else float("nan"),
            "template": "none" if first is None else ("before" if first[1] <= s else "after")})
    usable_share = float("nan")
    if gated and dec["ok"].any():
        usable_share = float(np.mean(dec["usable"][dec["ok"]]))
    return {"dataset": rec.dataset, "name": rec.name, "patient": rec.patient,
            "hours": len(x) / fs / 3600.0, "n_beats": int(len(ts)), "ivs": ivs,
            "first": first[1] if first else None, "n_cal": len(b["cal"]), "cal_counts": b["cal_counts"],
            "alerts": alerts, "episodes": episodes, "usable_share": usable_share}


def _work(job) -> dict:
    """One record, in a worker process."""
    ds, name, raw, gate, negative, step, one_thread, mode = job
    t1 = time.time()
    try:
        if gate is not None and one_thread:
            import torch
            torch.set_num_threads(1)
        r = process(load_record(ds, name, raw), gate, annotated_span=negative, step=step, mode=mode)
    except Exception as e:                       # one bad record must not end the run
        return {"dataset": ds, "name": name, "error": f"{type(e).__name__}: {e}"}
    r["secs"] = time.time() - t1
    return r


# ------------------------------------------------------------------ scoring
def classify(a: float, ivs) -> str:
    """'VT', 'VF', or 'other:<label at the alert>' for one alert time."""
    for kind in ("VT", "VF"):
        if any(lab == kind and s <= a <= e + SUSTAIN for s, e, lab in ivs):
            return kind
    lab = next((lab for s, e, lab in ivs if s <= a < e), "unlabelled")
    return "other:" + lab


def score(results: list, name: str) -> dict:
    out = {"long": [0, 0], "short": [0, 0], "vf": 0, "false": defaultdict(int),
           "false_list": [], "caught": {}}
    for r in results:
        al = r["alerts"][name]
        for ep in r["episodes"]:
            hit = al[(al >= ep["s"]) & (al <= ep["e"] + SUSTAIN)]
            key = "long" if ep["dur"] >= SUSTAIN else "short"
            out[key][0] += int(hit.size > 0)
            out[key][1] += 1
            out["caught"][(r["dataset"], r["name"], ep["s"])] = float(hit[0] - ep["s"]) if hit.size else None
        for a in al:
            k = classify(float(a), r["ivs"])
            if k == "VF":
                out["vf"] += 1
            elif k != "VT":
                out["false"][r["dataset"]] += 1
                out["false_list"].append((r["dataset"], r["name"], float(a), k[6:]))
    return out


def _mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def _f(v, fmt="{:.0f}"):
    return fmt.format(v) if v is not None and np.isfinite(v) else "-"


def report(results: list, gated: bool) -> None:
    hours = defaultdict(float)
    for r in results:
        hours[r["dataset"]] += r["hours"]
    eps = [(r, ep) for r in results for ep in r["episodes"]]
    long_ = [(r, ep) for r, ep in eps if ep["dur"] >= SUSTAIN]
    short = [(r, ep) for r, ep in eps if ep["dur"] < SUSTAIN]
    print("\n" + ", ".join(f"{k} {v:.1f} h" for k, v in hours.items()))
    no_tpl = [r for r in results if r["first"] is None]
    print(f"records without a template: {len(no_tpl)} of {len(results)}")
    for r in no_tpl:
        print(f"  {r['dataset']} {r['name']}: {why_no_template(r['cal_counts'])}")
    if gated:
        us = [r["usable_share"] for r in results if np.isfinite(r["usable_share"])]
        print(f"gate: {np.mean(us):.0%} of 30 s decisions usable (mean over records)")

    cfgs = configs(gated)
    sc = {name: score(results, name) for name, _ in cfgs}

    print("\nVT episodes of 30 s or more, one by one (template fixed; times from the start of the record)")
    print(f"{'set':<6}{'record':<8}{'start':>7}{'dur':>5}{'rate':>6}{'det':>5}{'beats':>6}{'shape':>7}"
          f"{'+size':>7}{'wide':>6}{'size':>6}{'stab':>6}{'onset':>7}{'template':>9}{'run 2':>7}{'all':>6}")
    for r, ep in long_:
        k = (r["dataset"], r["name"], ep["s"])
        a, b = (sc[n]["caught"][k] for n in SHOW)
        print(f"{r['dataset']:<6}{r['name']:<8}{_mmss(ep['s']):>7}{ep['dur']:>5.0f}{_f(ep['rate_sig']):>6}"
              f"{_f(ep['rate_det']):>5}{ep['beats']:>6}{_f(ep['unlike'], '{:.0%}'):>7}"
              f"{_f(ep['unlike_size'], '{:.0%}'):>7}{_f(ep['unlike_wide'], '{:.0%}'):>6}"
              f"{_f(ep['size'], '{:.2f}'):>6}{_f(ep['stab']):>6}{_f(ep['onset'], '{:.0%}'):>7}"
              f"{ep['template']:>9}{_f(a):>7}{_f(b):>6}")
    print(f"(rate / det: rate on the signal and from the detector; shape / +size / wide: share of beats unlike"
          f"\n the template, C {SHOW_C}; size: median beat size against the template's; stab: median successive"
          f"\n RR change, ms; onset: cycle-length drop where the rate first rose above 100; run 2 / all:"
          f"\n seconds from the episode start to the alert of '{SHOW[0]}' and '{SHOW[1]}', '-' none)")
    for name, g in (("30 s or more", long_), ("under 30 s", short)):
        if not g:
            continue

        def med(key, g=g):
            v = [ep[key] for _, ep in g if np.isfinite(ep[key])]
            return float(np.median(v)) if v else float("nan")

        exp = sum(ep["rate_sig"] * ep["dur"] / 60.0 for _, ep in g if np.isfinite(ep["rate_sig"]))
        got = sum(ep["beats"] for _, ep in g if np.isfinite(ep["rate_sig"]))
        print(f"{name}: {len(g)} episodes, {len({r['patient'] for r, _ in g})} patients. Medians: rate "
              f"signal {_f(med('rate_sig'))}, detector {_f(med('rate_det'))}; unlike by shape "
              f"{_f(med('unlike'), '{:.0%}')}, with size {_f(med('unlike_size'), '{:.0%}')}, wide "
              f"{_f(med('unlike_wide'), '{:.0%}')}; stab {_f(med('stab'))} ms. Beats found / expected "
              f"from the signal rate: " + (f"{got / exp:.0%}" if exp else "-"))

    print(f"\nEach check added to the run-2 rule, then all together. VT columns: episodes alerted / episodes "
          f"({len(long_)} of 30 s or more, {len(short)} shorter). False alerts per 24 h.")
    sets = list(hours)
    print(f"{'configuration':<30}{'VT>=30s':>9}{'VT<30s':>8}{'in VF':>7}" + "".join(f"{s:>9}" for s in sets))
    for name, _ in cfgs:
        s = sc[name]
        print(f"{name:<30}{s['long'][0]:>5}/{s['long'][1]:<3}{s['short'][0]:>4}/{s['short'][1]:<3}"
              f"{s['vf']:>7}" + "".join(f"{s['false'][d] / (hours[d] / 24.0):>9.2f}" for d in sets))
    print(f"(R {BASE['R']:.0f}, C {BASE['C']}, M {BASE['M']} unless the name says; size: unlike under 1/"
          f"{SIZE_FACTOR:.0f} or over {SIZE_FACTOR:.0f} times the template's size;"
          f"\n onset: cycle length down {ONSET_DROP:.0%} at the run's start, or the run has lasted "
          f"{ONSET_OVERRIDE_SEC:.0f} s; stability: successive RR change {STABILITY_MS:.0f} ms or less)")

    for name in SHOW:
        fl = sc[name]["false_list"]
        print(f"\nFalse alerts, '{name}': {len(fl)}"
              + (" (most first; record: count, then time and rhythm label at the alert)" if fl else ""))
        by_rec = defaultdict(list)
        for d, n, a, lab in fl:
            by_rec[(d, n)].append((a, lab))
        for (d, n), v in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))[:25]:
            print(f"  {d} {n}: {len(v)}  " + ", ".join(f"{_mmss(a)} {lab}" for a, lab in v[:6])
                  + (" ..." if len(v) > 6 else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positives", nargs="*", default=["cudb", "vfdb"],
                    help="sets read whole, at the rate engine's stride (VT and VF episodes)")
    ap.add_argument("--negatives", nargs="*", default=["nsrdb", "ltafdb", "svdb"],
                    help="sets read over their annotated span (false alerts)")
    ap.add_argument("--neg-stride", type=float, default=5.0, help="seconds between windows on negative sets")
    ap.add_argument("--gate-ckpt", type=Path, default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--max-records", type=int, default=0, help="per set (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="records processed in parallel")
    ap.add_argument("--cal", choices=CALS, default="dominant",
                    help="template calibration: dominant shape (default) or the first run's strict rule")
    a = ap.parse_args(argv)

    gate = None if a.no_gate else a.gate_ckpt
    if gate is not None and not gate.exists():
        print(f"gate off: no {gate}")
        gate = None
    elif gate is not None and importlib.util.find_spec("torch") is None:
        print("gate off: PyTorch is not installed")
        gate = None
    print("gate " + (f"on: {gate}, P(usable) >= {C.GATE_MIN_P['morphology']}" if gate else "off"))
    print(f"calibration {a.cal}")

    order = list(a.positives) + list(a.negatives)
    jobs = [(ds, name, a.raw, gate, ds in a.negatives, a.neg_stride if ds in a.negatives else POS_STEP,
             a.workers > 1, a.cal)
            for ds in order for name in eligible(ds, a.raw, a.max_records)]
    if not jobs:
        print("no records")
        return 1
    print(f"{len(jobs)} records, {a.workers} at a time")

    t0, results, failed = time.time(), [], []

    def done(r):
        if "error" in r:
            failed.append(r)
            print(f"  {r['dataset']} {r['name']}: FAILED {r['error']}", flush=True)
            return
        results.append(r)
        print(f"  {r['dataset']} {r['name']}: {r['hours']:.1f} h, {r['n_beats']} beats, "
              + (f"template at {_mmss(r['first'])}" if r["first"] is not None else "no template")
              + f", {len(r['episodes'])} VT episodes, {r['secs']:.0f} s", flush=True)

    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for f in as_completed([pool.submit(_work, j) for j in jobs]):
                done(f.result())
    else:
        for j in jobs:
            done(_work(j))
    if not results:
        print("no record could be read")
        return 1
    results.sort(key=lambda r: (order.index(r["dataset"]), r["name"]))
    report(results, gate is not None)
    if failed:
        print(f"\n{len(failed)} records failed: " + ", ".join(f"{r['dataset']} {r['name']}" for r in failed))
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
