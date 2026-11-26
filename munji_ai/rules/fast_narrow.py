"""Fast narrow rhythm (SVT): the alert and the doctor's report entry, as the app runs them.

    from munji_ai.rules.fast_narrow import FastNarrowTracker
    tracker = FastNarrowTracker(template=stored)      # stored: a beat_match.Template, or None at patch-on
    for chunk in signal_chunks:                       # model path, millivolts, at fs
        for event in tracker.push(chunk):
            ...   # "alert": notify; "report": the doctor's report; "template": store it
    events = tracker.finish()                         # end of the wear: the last report entries

The rules, their basis and every result: docs/fast_narrow_rhythm.md. This is
the streaming form of the two rows chosen there, FINAL_ALERT and FINAL_REPORT
of experiments/fast_narrow_template.py, with nothing retuned:
tests/test_fast_narrow_tracker.py checks that both give the same alerts and
the same report entries at the same times.

What it keeps between windows
-----------------------------
The rate engine reads every 40 s window on its own. These rules need memory:

    the template       the patient's own beat (rules/beat_match.py), learnt
                       once at patch-on or given back from the phone
    the beat stream    the last 3 minutes of beats: the Lorenz score reads 2
                       minutes, the onset test the intervals before a run
    the runs           the run over 100 and the run over 150 in progress: where
                       they started, how sharply, the app's rate just before
    the decisions      the last 2 minutes of (time, rate): "slower before"
    open report runs   a report entry may wait for its run to end, to see
                       whether it ended abruptly

How it runs
-----------
Every 2.5 s (RULES_STRIDE_SEC) it detects beats on the last 40 s, as the rate
engine does. A window is decided one stride later, once the 140 ms after its
last beat have arrived, because a beat's shape needs the samples around it:
an alert reaches the patient 2.5 s after the window it belongs to.

    alert    the decision's rate is 150 or more, 80% of the beats keep the
             template, the run's beats are regular (40 ms), no 3 beats in a
             row are unlike the template, the Lorenz score is not AF, the run
             over 150 began with a cycle-length drop of 9% or more, and the
             app's rate just before it was under 150. Once per episode: a new
             one only after more than 30 s without the condition.
    report   the same rate, shape and VT lock; regular and not AF, or
             organized on RR multiples; and the run over 100 began or ended
             abruptly (9%). When the start was not abrupt, the entry waits
             for the run to end: an entry can come minutes late, and is
             always written with its decision's time.
    template sent once, when it is learnt, so the app can store it.

Times are seconds since the first sample pushed (the patch going on). No
quality gate or lead-off check here: the decision layer owns those.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import config as C
from . import beat_match as bm
from .rate_engine import _trimmed_mean, detect_beats

# The rules' settings: the same values as experiments/fast_narrow_template.py,
# where each one's source is given (docs/fast_narrow_rhythm.md).
FAST_BPM = float(C.FAST_NARROW_BPM)     # 150: alert and report
SVT_BPM = 100.0                         # the SVT definition: the run the report and regularity read
SUSTAIN = float(C.VT_SUSTAINED_SEC)     # 30 s decision window; episodes re-arm after it
SHARE = 0.80                            # beats keeping the template
VT_RUN = 3                              # consecutive beats unlike the template: VT
STABILITY_MS = 40.0                     # regular (Brugada 1998)
ONSET_DROP = 0.09                       # abrupt start, and mirrored, abrupt end (Brugada 1998)
RUN_RR = 4                              # intervals in the running rate that starts a run
ONSET_BEFORE = 8                        # intervals the onset (and the end) is measured against
LZ_SEC, LZ_BIN, LZ_RANGE, LZ_ZERO = 120.0, 0.025, 0.500, 0.075    # Lorenz plot (Sarkar 2008)
LZ_MIN, LZ_RR_MIN, LZ_AF = 30, 0.220, 0.0
ORG_P, ORG_STEP, ORG_TOL, ORG_KMAX = (0.171, 0.400), 0.002, 0.030, 4   # RR multiples (our choices)
ORG_COVER, ORG_SHARE, ORG_MIN_RR = 0.80, 0.15, 12
KEEP_SEC = 180.0                        # beats and decisions kept
KEEP_BEATS = 20                         # and never fewer beats than this (onset: 12 back, end: 4 + 8)
NAN = float("nan")
_ORG_GRID = np.arange(ORG_P[0], ORG_P[1] + 1e-9, ORG_STEP)


# ------------------------------------------------------------ measurements
def af_score(rr: np.ndarray) -> float:
    """Lorenz plot AF score of a series of RR intervals: occupied 25 ms bins
    outside the +-75 ms zero segment minus the points inside it, per point.
    -1: every point at the origin; over 0: AF. NaN with fewer than 30 points."""
    rr = np.asarray(rr, float)
    if rr.size < LZ_MIN + 2:
        return NAN
    d = np.diff(rr)
    x, y = d[1:], d[:-1]
    keep = (rr[:-2] > LZ_RR_MIN) & (rr[1:-1] > LZ_RR_MIN) & (rr[2:] > LZ_RR_MIN)
    x, y = x[keep], y[keep]
    if x.size < LZ_MIN:
        return NAN
    zero = (np.abs(x) <= LZ_ZERO) & (np.abs(y) <= LZ_ZERO)
    lim = LZ_RANGE - 1e-9
    bx = np.floor(np.clip(x[~zero], -lim, lim) / LZ_BIN).astype(np.int64)
    by = np.floor(np.clip(y[~zero], -lim, lim) / LZ_BIN).astype(np.int64)
    occupied = np.unique(bx * 1000 + by).size
    return float(occupied - int(zero.sum())) / x.size


def organized_rr(rr) -> bool:
    """Do the intervals sit on whole multiples (1 to 4) of one atrial cycle of
    171 to 400 ms, within 30 ms, for 80% of them, with at least two multiples
    each holding 15%? (Atrial tachycardia or flutter with variable conduction.)"""
    rr = np.asarray(rr, float)
    if rr.size < ORG_MIN_RR:
        return False
    k = np.rint(rr[None, :] / _ORG_GRID[:, None])
    fit = (k >= 1) & (k <= ORG_KMAX) & (np.abs(rr[None, :] - k * _ORG_GRID[:, None]) <= ORG_TOL)
    cov = fit.mean(axis=1)
    b = int(np.argmax(cov))
    _, cnt = np.unique(k[b][fit[b]], return_counts=True)
    return bool(cov[b] >= ORG_COVER and int((cnt / rr.size >= ORG_SHARE).sum()) >= 2)


def longest_run(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


@dataclass
class Event:
    """What the tracker hands on. kind: "alert", "report" or "template"; t:
    the decision's time (seconds since the patch went on); info: the numbers
    behind it."""
    kind: str
    t: float
    info: dict = field(default_factory=dict)


class _Runs:
    """Runs of the running rate (median of the last 4 intervals) above one
    line, beat by beat: where each started and how sharply (the cycle length
    against the median of the 8 intervals before), the newest fast beat, and,
    for the report, where each ended."""

    def __init__(self, line: float, track_ends: bool):
        self.line, self.track_ends = line, track_ends
        self.s0 = -1                  # start of the run in progress (stream index), -1: none
        self.last_fast = -1           # newest beat that was in a run
        self.last_fast_s0 = -1        # the run it was in
        self.drop: dict = {}          # run start -> drop (NaN: not enough beats before)
        self.ended: dict = {}         # run start -> its last beat, once it has ended

    def add(self, i: int, run_i: float, drop_i) -> None:
        fast = bool(np.isfinite(run_i) and 60.0 / run_i > self.line)
        if fast:
            if self.s0 < 0:
                self.s0 = i
                self.drop[i] = drop_i()
            self.last_fast, self.last_fast_s0 = i, self.s0
        else:
            if self.s0 >= 0 and self.track_ends:
                self.ended[self.s0] = i - 1
            self.s0 = -1

    def newest(self, ts_of, t_end: float, kept_from: int) -> int:
        """Start of the newest run with a beat in the last 30 s, or -1. A beat
        no longer kept (kept_from: the oldest kept) is minutes old: none."""
        if self.last_fast < kept_from or ts_of(self.last_fast) <= t_end - SUSTAIN:
            return -1
        return self.last_fast_s0


class FastNarrowTracker:
    """One patient, one wear. See the module docstring."""

    def __init__(self, fs: int = C.TARGET_FS, step: float = C.RULES_STRIDE_SEC,
                 template: bm.Template | None = None):
        if template is not None and int(template.fs) != int(fs):
            raise ValueError(f"template at {template.fs} Hz, signal at {fs} Hz")
        self.fs, self.step = int(fs), float(step)
        self.template = template
        self.W = int(C.RULES_WINDOW_SEC * fs)
        self.S = int(round(step * fs))
        self.pre, self.L, self.lag = bm.sizes(fs)
        self._dedup = int(C.BEAT_DEDUP_SEC * fs)
        # signal
        self._buf = np.zeros(0)
        self._buf0 = 0                # absolute sample index of _buf[0]
        self._n = 0                   # samples received
        self._next_end = self.W
        self._pending = None          # (end, peaks) of the window not yet decided
        self._first_window = True
        # the beat stream (usable beats, absolute samples) and its running rate
        self._ts: deque = deque()     # (stream index, time s)
        self._ts_i0 = 0               # stream index of the first kept beat
        self._n_stream = 0
        self._last_raw = None         # last raw stream beat, for the duplicate check
        self._runs = {SVT_BPM: _Runs(SVT_BPM, True), FAST_BPM: _Runs(FAST_BPM - 1e-9, False)}
        # template learning
        self._cal = [] if template is None else None   # (time, beat window) while learning
        self._cal_next = 0            # next 10 s stretch to judge
        # decisions
        self._hist: deque = deque()   # (t, ok, bpm)
        self._before: dict = {}       # run over 150 start -> the app's rate just before it
        self._corr: dict = {}         # sample -> (shape, size), once there is a template
        self._last_alert = None
        self._report: deque = deque() # (t, state, run over 100 start, info); state True / None (waits)
        self._last_report = None

    # ---------------------------------------------------------------- input
    def push(self, samples) -> list:
        """Add samples; return the events they complete."""
        x = np.asarray(samples, float)
        self._buf = np.concatenate([self._buf, x])
        self._n += x.size
        out = []
        while self._n >= self._next_end:
            end = self._next_end
            win = self._buf[end - self.W - self._buf0:end - self._buf0]
            peaks = np.asarray(detect_beats(win, self.fs)[0], np.int64) + (end - self.W)
            if self._pending is not None:
                out += self._decide(*self._pending, data_end=self._n)
            self._pending = (end, np.sort(peaks))
            self._next_end += self.S
            keep = self._next_end - self.W - 2 * self.S - self.fs   # what the next decision can need
            if keep > self._buf0:
                self._buf = self._buf[keep - self._buf0:]
                self._buf0 = keep
        return out

    def finish(self) -> list:
        """The wear is over: decide the last window, close every report run."""
        out = []
        if self._pending is not None:
            out += self._decide(*self._pending, data_end=self._n)
            self._pending = None
        self._report = deque((t, False if state is None else state, s0, info)
                             for t, state, s0, info in self._report)
        return out + self._flush_report()

    # ---------------------------------------------------------------- beats
    def _usable(self, p: int, data_end: int) -> bool:
        return p - self.pre - self.lag >= 0 and p - self.pre + self.L + self.lag <= data_end

    def _ext(self, peaks: np.ndarray) -> np.ndarray:
        rel = np.asarray(peaks, np.int64) - self._buf0
        return self._buf[(rel - self.pre - self.lag)[:, None] + np.arange(self.L + 2 * self.lag)[None, :]]

    def _ts_of(self, i: int) -> float:
        k = i - self._ts_i0
        if k < 0:                                    # never wrap round to the newest beats
            raise IndexError(f"stream beat {i} is no longer kept (oldest {self._ts_i0})")
        return self._ts[k][1]

    def _times(self, lo: int, hi: int) -> np.ndarray:
        """Stream times of indices lo to hi - 1 (within what is kept)."""
        a = max(lo, self._ts_i0) - self._ts_i0
        b = max(hi, self._ts_i0) - self._ts_i0
        return np.array([self._ts[k][1] for k in range(a, b)])

    def _add_stream(self, peaks: np.ndarray, data_end: int) -> None:
        for p in peaks:
            p = int(p)
            dup = self._last_raw is not None and p - self._last_raw <= self._dedup
            self._last_raw = p
            if dup or not self._usable(p, data_end):
                continue
            i, t = self._n_stream, p / self.fs
            self._n_stream += 1
            self._ts.append((i, t))
            run_i = NAN
            if i >= RUN_RR:
                tt = self._times(i - RUN_RR, i + 1)
                run_i = float(np.median(np.diff(tt)))

            def drop(i=i, run_i=run_i):
                a = i - RUN_RR - ONSET_BEFORE
                if a < 0:
                    return NAN
                rr = np.diff(self._times(a, i - RUN_RR + 1))
                return 1.0 - run_i / float(np.median(rr))
            for r in self._runs.values():
                r.add(i, run_i, drop)
            if self._cal is not None:
                self._cal.append((t, self._ext(np.array([p]))[0]))
        while len(self._ts) > KEEP_BEATS and self._ts[-1][1] - self._ts[0][1] > KEEP_SEC:
            self._ts.popleft()
            self._ts_i0 += 1

    def _learn(self, final: bool = False) -> list:
        """Judge every 10 s stretch the stream has finished; the first that
        qualifies is the template."""
        if self._cal is None or not self._cal:
            return []
        t_last = self._cal[-1][0]
        while (self._cal_next + 1) * bm.CAL_SEC <= t_last:
            m = self._cal_next
            self._cal_next += 1
            t = np.array([c[0] for c in self._cal])
            i, j = np.searchsorted(t, [m * bm.CAL_SEC, (m + 1) * bm.CAL_SEC])
            w = None
            if j - i >= bm.CAL_MIN_BEATS and t[j - 1] > t[i]:
                bpm = 60.0 * (j - i - 1) / (t[j - 1] - t[i])
                if bm.CAL_BPM[0] <= bpm <= bm.CAL_BPM[1]:
                    w = bm.dominant_template(np.stack([c[1] for c in self._cal[i:j]]), self.L, self.lag)
            if w is not None:
                self.template = bm.Template(waveform=np.asarray(w, float), fs=self.fs,
                                            start=m * bm.CAL_SEC, end=(m + 1) * bm.CAL_SEC)
                self._cal = None
                return [Event("template", self.template.end, {"template": self.template})]
            self._cal = self._cal[j:]
        return []

    def _match(self, peaks: np.ndarray) -> tuple:
        """(shape, size) of these beats against the template; NaN without one."""
        n = len(peaks)
        if self.template is None or n == 0:
            return np.full(n, NAN), np.full(n, NAN)
        new = [int(p) for p in peaks if int(p) not in self._corr]
        if new:
            sh, sz = bm.match(self._ext(np.array(new)), self.template, self.fs)
            self._corr.update({p: (a, b) for p, a, b in zip(new, sh, sz)})
        pairs = [self._corr[int(p)] for p in peaks]
        return np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])

    # ------------------------------------------------------------- decisions
    def _decide(self, end: int, peaks: np.ndarray, data_end: int) -> list:
        fs = self.fs
        out = []
        new = peaks if self._first_window else peaks[peaks >= end - self.S]
        self._first_window = False
        self._add_stream(new, data_end)
        out += self._learn()
        out += self._resolve_ends()

        t_end = end / fs
        b = peaks[peaks >= end - int(SUSTAIN * fs)]
        use = np.array([p for p in b if self._usable(int(p), data_end)], np.int64)
        need = SUSTAIN - 60.0 / C.VT_MIN_BPM
        ok = bool(b.size >= 3 and (b[-1] - b[0]) / fs >= need and use.size >= 3)
        bpm = NAN
        if ok:
            m = _trimmed_mean(np.sort(np.diff(b) / fs), C.RATE_RUN_TRIM)
            bpm = 60.0 / m if m > 0 else NAN
        r100, r150 = self._runs[SVT_BPM], self._runs[FAST_BPM]
        s100 = r100.newest(self._ts_of, t_end, self._ts_i0)
        s150 = r150.newest(self._ts_of, t_end, self._ts_i0)
        if s150 >= 0 and s150 not in self._before:
            self._before = {s150: self._rate_before(self._ts_of(s150))}
        self._hist.append((t_end, ok, bpm))
        while self._hist and self._hist[0][0] < t_end - KEEP_SEC:
            self._hist.popleft()
        for p in [p for p in self._corr if p < end - int(2 * SUSTAIN * fs)]:
            del self._corr[p]
        if not (ok and bpm >= FAST_BPM and b.size >= 4):
            return out + self._flush_report()

        shape, size = self._match(use)
        share = float(np.mean(bm.alike(shape, size)))
        vt_run = longest_run(bm.unlike(shape, size))
        t_all = self._times(self._ts_i0, self._n_stream)
        lz = af_score(np.diff(t_all[(t_all > t_end - LZ_SEC) & (t_all <= t_end)]))
        stab_run, org = NAN, False
        if s100 >= 0:
            lo30 = self._ts_i0 + int(np.searchsorted(t_all, t_end - SUSTAIN, side="left"))
            rr = np.diff(self._times(max(s100, lo30), self._n_stream))
            if rr.size >= 3:
                stab_run = float(np.median(np.abs(np.diff(rr))) * 1000.0)
            org = organized_rr(rr)
        d150 = r150.drop.get(s150, NAN) if s150 >= 0 else NAN
        d100 = r100.drop.get(s100, NAN) if s100 >= 0 else NAN
        before = self._before.get(s150, NAN) if s150 >= 0 else NAN
        not_af = not (lz > LZ_AF)
        regular = stab_run <= STABILITY_MS
        info = {"bpm": bpm, "share": share, "vt_run": vt_run, "regular_ms": stab_run, "lorenz": lz,
                "organized": org, "start_drop": d100, "onset_drop": d150, "rate_before": before}

        alert = (share >= SHARE and regular and bool(np.isfinite(d150) and d150 >= ONSET_DROP)
                 and vt_run < VT_RUN and not_af and before < FAST_BPM)
        if alert:
            if self._last_alert is None or t_end - self._last_alert > SUSTAIN + self.step + 1e-6:
                out.append(Event("alert", t_end, info))
            self._last_alert = t_end

        base = share >= SHARE and ((regular and not_af) or org) and vt_run < VT_RUN
        if base:
            if np.isfinite(d100) and d100 >= ONSET_DROP:
                self._report.append((t_end, True, s100, info))
            elif s100 >= 0:
                self._report.append((t_end, None, s100, info))
        return out + self._flush_report()

    def _rate_before(self, t0: float) -> float:
        """The app's 30 s rate at the last decision that ended at or before t0."""
        for t, ok, bpm in reversed(self._hist):
            if t <= t0:
                return bpm if ok else NAN
        return NAN

    def _resolve_ends(self) -> list:
        """Report entries waiting on a run over 100 that has now ended with
        8 intervals after it: abrupt end or not."""
        r = self._runs[SVT_BPM]
        done = {}
        for s0, j1 in list(r.ended.items()):
            if self._n_stream >= j1 + ONSET_BEFORE + 1 or j1 < RUN_RR:
                rise = NAN
                if j1 >= RUN_RR and j1 - RUN_RR >= self._ts_i0:
                    t = self._times(j1 - RUN_RR, j1 + ONSET_BEFORE + 1)
                    rr = np.diff(t)
                    rise = float(1.0 - np.median(rr[:RUN_RR]) / np.median(rr[RUN_RR:]))
                done[s0] = bool(np.isfinite(rise) and rise >= ONSET_DROP)
                del r.ended[s0]
        if done:
            self._report = deque((t, done[s0] if state is None and s0 in done else state, s0, info)
                                 for t, state, s0, info in self._report)
        for s0 in [s for s in r.drop if s < r.last_fast_s0 and s not in r.ended]:
            del r.drop[s0]
        for s0 in [s for s in self._runs[FAST_BPM].drop if s < self._runs[FAST_BPM].last_fast_s0]:
            del self._runs[FAST_BPM].drop[s0]
        return []

    def _flush_report(self) -> list:
        """Report entries from the decided front of the queue, grouped into
        episodes as the alert is (a new one after more than 30 s without)."""
        out = []
        while self._report and self._report[0][1] is not None:
            t, state, _, info = self._report.popleft()
            if not state:
                continue
            if self._last_report is None or t - self._last_report > SUSTAIN + self.step + 1e-6:
                out.append(Event("report", t, info))
            self._last_report = t
        return out
