"""Sustained VT without the beat model: every beat against the patient's own.

    python experiments/template_wct.py                        # all sets
    python experiments/template_wct.py --negatives nsrdb      # quicker first look
    python experiments/template_wct.py --max-records 2        # smoke test
    python experiments/template_wct.py --no-gate --workers 6
    python experiments/template_wct.py --cal strict          # the first run's calibration

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
VT/VF and 98% specificity for SVT. QRS template matching on Holter recordings separated ventricular
beats at 98.4% sensitivity and 98.9% specificity (PubMed 17805974). One lead
is enough: the S-ICD senses one vector. Nothing here is trained.

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
          "fixed" keeps that template for the whole recording. "adaptive"
          moves to every later stretch that qualifies, labels not used
          (posture and electrode changes); no stretch above 100 bpm
          qualifies, so VT above 100 cannot become the template. Beats for
          calibration are the newest stride of every window.
match     each beat: Pearson correlation with the template over 100 ms either
          side of R, the best of shifts up to 40 ms.
rule      at the end of every window, over its last 30 s of beats (the WCT
          rule's span and its 80% share): rate above R (monitor-style trimmed
          mean, as the rate engine), at least 80% of the beats under
          correlation C, and optionally the beats alike each other
          (monomorphic: median correlation with their own median beat at
          least M). One alert per episode, re-armed after 30 s without the
          condition, as the decision layer does.
stride    2.5 s on the VT sets (the rate engine's), --neg-stride (5 s) on the
          long negative sets to keep them quick.
gate      with checkpoints/quality_gate.pt and PyTorch: the 30 s must average
          P(usable) >= 0.90 (GATE_MIN_P morphology), scored on 20 s windows
          normalised and split into 5 s as the beat path does. Results come
          with and without it.

R is swept at 100 (the definition of VT, the client's figure) and 120 (the
notification threshold of ambulatory patch monitoring: ARIC, Zio AT); C at
0.5 (S-ICD's figure), 0.6 and 0.7; M off and 0.7.

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
record, time and the rhythm label, to be looked at. Pictures of every VT
episode: experiments/template_wct_plot.py.

For every VT episode of 30 s or more: the rate on the signal (the peak count
used to split VT by rate) and from the detector, the share of beats unlike
the template, and whether the template came from before or after the
episode. That shows why an episode is missed: the detector, the template or
the rule.

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
RATES = (100.0, 120.0)
CORRS = (0.5, 0.6, 0.7)
MONOS = (None, 0.7)
TEMPLATES = ("fixed", "adaptive")
SHOW_C, SHOW_M = 0.6, 0.7              # detailed per episode, template "fixed"


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
def decision_table(ends, wbeats, pos, corr: dict, ext, L, lag, fs, gate_p) -> dict:
    """What the rules need, at the end of every window, over its last 30 s.

    The share of unlike beats is computed for every window; the rate and the
    monomorphic check only where some rule could fire, which is rare away
    from VT and keeps 24 h records quick.
    """
    n = len(ends)
    t_end = ends / fs
    ok = np.zeros(n, bool)
    frac = {k: np.full((n, len(CORRS)), np.nan) for k in corr}
    bpm, mono = np.full(n, np.nan), np.full(n, np.nan)
    cuts = np.asarray(CORRS)[None, :]
    S30 = int(SUSTAIN * fs)
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
        for k, cr in corr.items():
            f = (cr[idx][:, None] < cuts).mean(axis=0)      # NaN (no template) is never unlike
            frac[k][w] = f
            top = max(top, float(f[-1]))
        if top >= FRACTION:
            m = _trimmed_mean(np.sort(np.diff(b) / fs), C.RATE_RUN_TRIM)
            bpm[w] = 60.0 / m if m > 0 else np.nan
            if bpm[w] > min(RATES):
                mono[w] = self_alike(ext[idx], L, lag)
    usable = None
    if gate_p is not None:
        cs = np.concatenate([[0.0], np.cumsum(gate_p)])
        b0 = np.floor((t_end - SUSTAIN) / GATE_BLOCK_SEC + 1e-9).astype(int)
        b1 = np.ceil(t_end / GATE_BLOCK_SEC - 1e-9).astype(int)
        usable = np.zeros(n, bool)
        v = (b1 <= len(gate_p)) & (b1 > b0)
        usable[v] = (cs[b1[v]] - cs[b0[v]]) / (b1[v] - b0[v]) >= C.GATE_MIN_P["morphology"]
    return {"t": t_end, "ok": ok, "bpm": bpm, "mono": mono, "frac": frac, "usable": usable}


def alert_times(t: np.ndarray, cond: np.ndarray, step: float) -> np.ndarray:
    """First time of every episode of cond. Re-armed once cond has been false
    for more than SUSTAIN (decision_layer._vt_step); on a grid of `step` that
    is a gap of more than SUSTAIN + step between two true decisions."""
    tt = t[cond]
    if tt.size == 0:
        return tt
    return tt[np.concatenate([[True], np.diff(tt) > SUSTAIN + step + 1e-6])]


def combos(gated: bool):
    for g in ((False, True) if gated else (False,)):
        for tpl in TEMPLATES:
            for R in RATES:
                for ci, c in enumerate(CORRS):
                    for M in MONOS:
                        yield g, tpl, R, ci, c, M


def combo_alerts(dec: dict, tpl: str, R: float, ci: int, M, gated: bool, step: float) -> np.ndarray:
    cond = dec["ok"] & (dec["frac"][tpl][:, ci] >= FRACTION) & (dec["bpm"] > R)
    if M is not None:
        cond &= dec["mono"] >= M
    if gated:
        cond &= dec["usable"]
    return alert_times(dec["t"], cond, step)


# ------------------------------------------------------------------ record
def beats_and_template(rec, step: float, annotated_span: bool = False, mode: str = "dominant") -> dict:
    """Beats, templates and correlations of one record, as the app would get
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
    corr         per position in pos: correlation with the fixed and the
                 adaptive template (NaN without a template)
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

    corr = {k: np.full(len(pos), np.nan) for k in TEMPLATES}
    if first is not None:
        corr["fixed"] = best_corr(ext, first[2])
        tpls = [first] + [c for c in cal if c[0] >= first[1]]
        which = np.clip(np.searchsorted(np.array([c[1] for c in tpls]), tpos, side="right") - 1, 0, None)
        bounds = np.searchsorted(which, np.arange(len(tpls) + 1))   # tpos is sorted, so is which
        for k in range(len(tpls)):
            a, b = bounds[k], bounds[k + 1]
            if b > a:
                corr["adaptive"][a:b] = best_corr(ext[a:b], tpls[k][2])
    return {"x": x, "fs": fs, "ivs": ivs, "L": L, "lag": lag, "ends": ends, "wbeats": wbeats,
            "pos": pos, "ext": ext, "si": si, "ts": ts, "cal": cal, "first": first, "corr": corr,
            "cal_counts": counts}


def process(rec, gate_ckpt, annotated_span: bool, step: float, mode: str = "dominant") -> dict:
    b = beats_and_template(rec, step, annotated_span, mode)
    x, fs, ivs, L, lag = b["x"], b["fs"], b["ivs"], b["L"], b["lag"]
    ends, wbeats, pos, ext = b["ends"], b["wbeats"], b["pos"], b["ext"]
    si, ts, cal, first, corr = b["si"], b["ts"], b["cal"], b["first"], b["corr"]

    gate_p = gate_blocks(x, gate_ckpt) if gate_ckpt is not None else None
    dec = decision_table(ends, wbeats, pos, corr, ext, L, lag, fs, gate_p)
    alerts = {(g, tpl, R, ci, M): combo_alerts(dec, tpl, R, ci, M, g, step)
              for g, tpl, R, ci, _, M in combos(gate_p is not None)}

    episodes = []
    for s, e, lab in ivs:
        if lab != "VT":
            continue
        i, j = np.searchsorted(ts, [s, e])
        k = si[i:j]
        ref = _episode_bpm(x[int(s * fs):int(e * fs)], fs)
        has = first is not None and k.size > 0
        episodes.append({
            "s": s, "e": e, "dur": e - s,
            "rate_sig": float(ref) if ref else float("nan"),
            "rate_det": float(60.0 / np.median(np.diff(ts[i:j]))) if j - i >= 3 else float("nan"),
            "beats": int(j - i),
            "unlike": float(np.mean(corr["fixed"][k] < SHOW_C)) if has else float("nan"),
            "unlike_adapt": float(np.mean(corr["adaptive"][k] < SHOW_C)) if has else float("nan"),
            "corr": float(np.median(corr["fixed"][k])) if has else float("nan"),
            "alike": self_alike(ext[k], L, lag),
            "template": "none" if first is None else ("before" if first[1] <= s else "after")})
    usable_share = float("nan")
    if gate_p is not None and dec["ok"].any():
        usable_share = float(np.mean(dec["usable"][dec["ok"]]))
    return {"dataset": rec.dataset, "name": rec.name, "patient": rec.patient,
            "hours": len(x) / fs / 3600.0, "n_beats": int(len(ts)), "ivs": ivs,
            "first": first[1] if first else None, "n_cal": len(cal), "cal_counts": b["cal_counts"],
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


def score(results: list, g, tpl, R, ci, M) -> dict:
    out = {"long": [0, 0], "short": [0, 0], "vf": 0, "false": defaultdict(int),
           "false_list": [], "caught": {}}
    for r in results:
        al = r["alerts"][(g, tpl, R, ci, M)]
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

    sc = {(g, tpl, R, ci, M): score(results, g, tpl, R, ci, M) for g, tpl, R, ci, c, M in combos(gated)}
    ci_show = CORRS.index(SHOW_C)

    print(f"\nVT episodes of 30 s or more, one by one (template fixed, alerts at C {SHOW_C} M {SHOW_M}, "
          f"no gate; times from the start of the record)")
    print(f"{'set':<7}{'record':<9}{'start':>8}{'dur':>6}{'rate sig':>9}{'rate det':>9}{'beats':>7}"
          f"{'unlike':>8}{'adapt':>7}{'corr':>6}{'alike':>7}{'template':>10}{'R100':>6}{'R120':>6}")
    for r, ep in long_:
        k = (r["dataset"], r["name"], ep["s"])
        c100 = sc[(False, "fixed", 100.0, ci_show, SHOW_M)]["caught"][k]
        c120 = sc[(False, "fixed", 120.0, ci_show, SHOW_M)]["caught"][k]
        print(f"{r['dataset']:<7}{r['name']:<9}{_mmss(ep['s']):>8}{ep['dur']:>6.0f}{_f(ep['rate_sig']):>9}"
              f"{_f(ep['rate_det']):>9}{ep['beats']:>7}{_f(ep['unlike'], '{:.0%}'):>8}"
              f"{_f(ep['unlike_adapt'], '{:.0%}'):>7}{_f(ep['corr'], '{:.2f}'):>6}"
              f"{_f(ep['alike'], '{:.2f}'):>7}{ep['template']:>10}{_f(c100):>6}{_f(c120):>6}")
    print(f"(unlike / adapt: share of beats under correlation {SHOW_C} with the fixed / adaptive template;"
          "\n corr: their median correlation with the fixed one; alike: with their own median beat;"
          "\n R100 / R120: seconds from the episode start to the alert, '-' none)")
    for name, g in (("30 s or more", long_), ("under 30 s", short)):
        if not g:
            continue

        def med(key, g=g):
            v = [ep[key] for _, ep in g if np.isfinite(ep[key])]
            return float(np.median(v)) if v else float("nan")

        exp = sum(ep["rate_sig"] * ep["dur"] / 60.0 for _, ep in g if np.isfinite(ep["rate_sig"]))
        got = sum(ep["beats"] for _, ep in g if np.isfinite(ep["rate_sig"]))
        print(f"{name}: {len(g)} episodes, {len({r['patient'] for r, _ in g})} patients. Medians: rate "
              f"signal {_f(med('rate_sig'))}, detector {_f(med('rate_det'))}; unlike fixed "
              f"{_f(med('unlike'), '{:.0%}')}, adaptive {_f(med('unlike_adapt'), '{:.0%}')}; alike "
              f"{_f(med('alike'), '{:.2f}')}. Beats found / expected from the signal rate: "
              + (f"{got / exp:.0%}" if exp else "-"))
    print("Reading: 'rate det' far under 'rate sig' -> the detector misses VT beats; a low 'unlike' -> the"
          "\nbeats keep the template's shape; 'template after' -> calibrated later in the record than"
          "\nthe episode.")

    print(f"\nThe rule swept. VT columns: episodes alerted / episodes ({len(long_)} of 30 s or more, "
          f"{len(short)} shorter). False alerts per 24 h.")
    sets = list(hours)
    print(f"{'gate':<5}{'template':<9}{'R':>4}{'C':>5}{'M':>5}{'VT>=30s':>9}{'VT<30s':>8}{'in VF':>7}"
          + "".join(f"{s:>9}" for s in sets))
    for g, tpl, R, ci, c, M in combos(gated):
        s = sc[(g, tpl, R, ci, M)]
        print(f"{('on' if g else 'off'):<5}{tpl:<9}{R:>4.0f}{c:>5.1f}{('-' if M is None else f'{M:.1f}'):>5}"
              f"{s['long'][0]:>5}/{s['long'][1]:<3}{s['short'][0]:>4}/{s['short'][1]:<3}{s['vf']:>7}"
              + "".join(f"{s['false'][d] / (hours[d] / 24.0):>9.2f}" for d in sets))

    for R in RATES:
        s = sc[(False, "fixed", R, ci_show, SHOW_M)]
        fl = s["false_list"]
        print(f"\nFalse alerts, template fixed, R {R:.0f}, C {SHOW_C}, M {SHOW_M}, no gate: {len(fl)}"
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
