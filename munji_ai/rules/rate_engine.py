"""Rate engine: the conditions that are arithmetic, not learning.

    from munji_ai.rules.rate_engine import analyse, sliding_reports
    report = analyse(window_mv)                    # one window
    for t_end, report in sliding_reports(signal_mv):
        alerts = decision_layer.on_rate(t_end, report, usable=..., lead_off=...)

What it reports
---------------
Every number comes from config, where each one sits beside its published
source (2018 ACC/AHA/HRS, 2018 ESC syncope, AHA ACLS, NEWS2, CinC 2015,
ANSI/AAMI EC13, ambulatory telemetry (MCOT) notification criteria).

Two layers, kept apart:

    for the report (graded, never sent to anyone on their own)
    bradycardia           rate held below 60 / 50   for 5 beats   info / warning
    extreme bradycardia   rate held below 40        for 5 beats   critical
    tachycardia           rate held above 100 / 110 for 17 beats  info / warning
    extreme tachycardia   rate held at 150 or more  for 17 beats  critical

    for notification (Finding.notify; MCOT practice)
    sustained bradycardia rate at or below 30, held 30 s          emergent
    sustained tachycardia rate at or above 180, held 30 s         urgent
    pause                 no beat for more than 3 s               urgent
    asystole              no beat for more than 6 s               emergent

"Held for N beats" (or for 30 s) means the rate over that run meets the
condition somewhere in the window. The rate of a run is averaged the way
bedside monitors do it: drop the longest and shortest intervals, average the
rest. One missed beat or one pause cannot make or break a run, an irregular
fast rhythm still counts, and a burst shorter than half the run does not.
Persistence lives here, in the finding, so the decision layer acts on a
single window (config.CONFIRMATION, (1, 1)).

Wide-complex tachycardia is not reported: on real VT this engine called only
14% of windows wide. The "fast narrow" (SVT) label is off as well
(config.FAST_NARROW_LABEL): the width estimate failed against hand-measured
QRS on the QT Database. Width is still measured and reported as narrow,
uncertain or wide, for information only.

Input
-----
One window of the model path (0.5-40 Hz, notch) in MILLIVOLTS, never
normalised: the beat amplitude floor (QRS_MIN_AMP_MV, from EC13) is an
absolute voltage. Windows are RULES_WINDOW_SEC long every RULES_STRIDE_SEC;
sliding_reports() produces them. A window with missing samples must not be
analysed at all: a filled gap is a flat stretch, and a flat stretch reads as
asystole.

Who judges a finding (the decision layer applies this)
------------------------------------------------------
Each finding carries a basis:

    "rate"     brady, tachy (all tiers). Counted only in windows the
               quality gate accepted, because noise invents beats.
    "absence"  pause, asystole. Judged by the electrodes, not the gate: a
               real stopped heart can look like poor signal, and ACLS asks
               for the leads to be checked before a flat line is called
               asystole. Counted when the leads are reported on; when lead
               status is unknown (public recordings), the gate decides.

What it does not do
-------------------
It does not diagnose. Sinus bradycardia and complete heart block can produce
the same rate. The findings are rate observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import (ASYSTOLE_CRITICAL_SEC, FAST_NARROW_BPM,
                      FAST_NARROW_LABEL, HR_BRADY, HR_TACHY, NARROW_QRS_MS,
                      NOTIFY_BRADY_BPM, NOTIFY_SUSTAIN_SEC, NOTIFY_TACHY_BPM,
                      PAUSE_WARNING_SEC, QRS_AMP_HALF_MS, QRS_MIN_AMP_MV,
                      QRS_SEARCH_HALF_MS, QRS_UNCERTAIN_MS, RATE_RUN_BEATS,
                      RATE_RUN_TRIM, RULES_STRIDE_SEC, RULES_WINDOW_SEC,
                      TARGET_FS)
from ..features.sqi import MIN_DYNAMIC_RANGE, detect_r_pantompkins

# Ordered, so max() means "the most urgent".
SEVERITY = ("info", "warning", "critical")

# kind -> (family, basis). A family is one phenomenon at several severities;
# the decision layer confirms per family, so a rate wobbling between two
# severities still adds up to one episode.
KINDS = {
    "bradycardia":           ("brady", "rate"),
    "extreme_bradycardia":   ("brady", "rate"),
    "sustained_bradycardia": ("brady", "rate"),
    "tachycardia":           ("tachy", "rate"),
    "extreme_tachycardia":   ("tachy", "rate"),
    "sustained_tachycardia": ("tachy", "rate"),
    "fast_narrow":           ("fast_narrow", "rate"),
    "pause":                 ("absence", "absence"),
    "asystole":              ("absence", "absence"),
}

# kind -> notification level (MCOT practice). Kinds not listed are for the
# report only and are never sent to anyone.
NOTIFY = {
    "sustained_bradycardia": "emergent",
    "sustained_tachycardia": "urgent",
    "pause":                 "urgent",
    "asystole":              "emergent",
}
NOTIFY_LEVELS = ("urgent", "emergent")       # ordered, mildest first


@dataclass
class Finding:
    """One observation about one window."""
    kind: str
    severity: str
    value: float = float("nan")      # the measurement that triggered it
    detail: str = ""

    @property
    def family(self) -> str:
        return KINDS[self.kind][0]

    @property
    def basis(self) -> str:
        """'rate': needs a gate-accepted window. 'absence': needs leads on."""
        return KINDS[self.kind][1]

    @property
    def notify(self) -> str:
        """'urgent' or 'emergent' if this finding is sent; '' if report only."""
        return NOTIFY.get(self.kind, "")

    def __repr__(self) -> str:
        v = "" if np.isnan(self.value) else f" {self.value:.1f}"
        return f"{self.kind}({self.severity}{v})"


@dataclass
class RateReport:
    """Everything the engine measured on one window."""
    bpm: float = float("nan")               # median rate, for display
    rr_sec: np.ndarray = field(default_factory=lambda: np.array([]))
    slow_run_bpm: float = float("nan")      # slowest rate held for the brady run
    fast_run_bpm: float = float("nan")      # fastest rate held for the tachy run
    slow_sustained_bpm: float = float("nan")  # slowest rate held for 30 s
    fast_sustained_bpm: float = float("nan")  # fastest rate held for 30 s
    qrs_ms: float = float("nan")
    qrs_class: str = ""                     # "narrow" / "uncertain" / "wide" / ""
    n_beats: int = 0
    n_rejected: int = 0                     # detector peaks below the EC13 floor
    longest_gap_sec: float = float("nan")
    flat: bool = False
    findings: list = field(default_factory=list)

    @property
    def severity(self) -> str:
        if not self.findings:
            return ""
        return max((f.severity for f in self.findings), key=SEVERITY.index)

    def kinds(self) -> set:
        return {f.kind for f in self.findings}


# ---------------------------------------------------------------- classify
# Pure functions: numbers in, findings out. Every threshold boundary is tested
# on these directly, because a synthetic signal cannot land on an exact rate
# (250 Hz quantises RR) or an exact width (4 ms steps).

def classify_rate(slow_bpm: float, fast_bpm: float) -> list:
    """Brady and tachy findings from the rates held over each family's run."""
    out = []
    if not np.isnan(slow_bpm):
        if slow_bpm < HR_BRADY["critical"]:
            out.append(Finding("extreme_bradycardia", "critical", slow_bpm))
        elif slow_bpm < HR_BRADY["warning"]:
            out.append(Finding("bradycardia", "warning", slow_bpm))
        elif slow_bpm < HR_BRADY["info"]:
            out.append(Finding("bradycardia", "info", slow_bpm))
    if not np.isnan(fast_bpm):
        if fast_bpm >= HR_TACHY["critical"]:
            out.append(Finding("extreme_tachycardia", "critical", fast_bpm))
        elif fast_bpm > HR_TACHY["warning"]:
            out.append(Finding("tachycardia", "warning", fast_bpm))
        elif fast_bpm > HR_TACHY["info"]:
            out.append(Finding("tachycardia", "info", fast_bpm))
    return out


def classify_sustained(slow_bpm: float, fast_bpm: float) -> list:
    """Notification findings from the rates held for NOTIFY_SUSTAIN_SEC."""
    out = []
    held = f"held {NOTIFY_SUSTAIN_SEC:.0f} s"
    if not np.isnan(slow_bpm) and slow_bpm <= NOTIFY_BRADY_BPM:
        out.append(Finding("sustained_bradycardia", "critical", slow_bpm,
                           detail=f"heart rate {slow_bpm:.0f}, {held}"))
    if not np.isnan(fast_bpm) and fast_bpm >= NOTIFY_TACHY_BPM:
        out.append(Finding("sustained_tachycardia", "critical", fast_bpm,
                           detail=f"heart rate {fast_bpm:.0f}, {held}"))
    return out


def classify_width(qrs_ms: float, band_ms=QRS_UNCERTAIN_MS) -> str:
    """'narrow', 'uncertain' or 'wide' against NARROW_QRS_MS; '' if unmeasured."""
    if np.isnan(qrs_ms):
        return ""
    if band_ms is not None and abs(qrs_ms - NARROW_QRS_MS) < band_ms:
        return "uncertain"
    return "wide" if qrs_ms >= NARROW_QRS_MS else "narrow"


def classify_gap(gap_sec: float) -> Finding | None:
    """Pause or asystole for the longest beat-free stretch in a window."""
    if np.isnan(gap_sec):
        return None
    if gap_sec > ASYSTOLE_CRITICAL_SEC:
        return Finding("asystole", "critical", gap_sec,
                       detail=f"no beat for {gap_sec:.1f} s")
    if gap_sec > PAUSE_WARNING_SEC:
        return Finding("pause", "warning", gap_sec,
                       detail=f"no beat for {gap_sec:.1f} s")
    return None


def _trimmed_mean(sorted_rr: np.ndarray, trim: float) -> float:
    n = sorted_rr.size
    k = int(round(n * trim))
    if n - 2 * k < 1:
        k = (n - 1) // 2
    return float(sorted_rr[k:n - k].mean())


def held_rates(rr_sec: np.ndarray, n_beats: int,
               trim: float = RATE_RUN_TRIM) -> tuple[float, float]:
    """(slowest, fastest) rate held over any n consecutive beats.

    The rate of a run is 60 / the mean of its n RR intervals after dropping
    the longest and the shortest round(n * trim) of them: monitor-style
    averaging. NaN when the window holds fewer than n intervals: a run that
    was not seen is not reported, rather than guessed from fewer beats.
    """
    rr = np.asarray(rr_sec, dtype=np.float64)
    if n_beats < 1 or rr.size < n_beats:
        return float("nan"), float("nan")
    runs = np.sort(np.lib.stride_tricks.sliding_window_view(rr, n_beats), axis=1)
    k = int(round(n_beats * trim))
    if n_beats - 2 * k < 1:
        k = (n_beats - 1) // 2
    mean_rr = runs[:, k:n_beats - k].mean(axis=1)
    mean_rr = mean_rr[mean_rr > 0]
    if mean_rr.size == 0:
        return float("nan"), float("nan")
    return float(60.0 / mean_rr.max()), float(60.0 / mean_rr.min())


def held_rates_for(rr_sec: np.ndarray, seconds: float,
                   trim: float = RATE_RUN_TRIM) -> tuple[float, float]:
    """(slowest, fastest) rate held over any run of beats lasting `seconds`.

    For each beat, the run is the fewest following intervals that add up to
    at least `seconds`; its rate is the same monitor-style trimmed mean. NaN
    when no run that long fits in the window.
    """
    rr = np.asarray(rr_sec, dtype=np.float64)
    if rr.size == 0:
        return float("nan"), float("nan")
    cs = np.concatenate([[0.0], np.cumsum(rr)])
    slow = fast = float("nan")
    for i in range(rr.size):
        j = int(np.searchsorted(cs, cs[i] + seconds, side="left"))
        if j > rr.size:
            break
        m = _trimmed_mean(np.sort(rr[i:j]), trim)
        if m <= 0:
            continue
        r = 60.0 / m
        slow = r if np.isnan(slow) else min(slow, r)
        fast = r if np.isnan(fast) else max(fast, r)
    return slow, fast


def rate_findings(rr_sec: np.ndarray) -> tuple[list, dict]:
    """All rate findings for one window's RR intervals, and the held rates.

    Shared by analyse() and by the evaluation's reference, so both sides of
    every comparison run exactly the same rules.
    """
    slow, _ = held_rates(rr_sec, RATE_RUN_BEATS["brady"])
    _, fast = held_rates(rr_sec, RATE_RUN_BEATS["tachy"])
    s_slow, s_fast = held_rates_for(rr_sec, NOTIFY_SUSTAIN_SEC)
    findings = classify_rate(slow, fast) + classify_sustained(s_slow, s_fast)
    return findings, {"slow_run_bpm": slow, "fast_run_bpm": fast,
                      "slow_sustained_bpm": s_slow, "fast_sustained_bpm": s_fast}


# ----------------------------------------------------------------- measure

def qrs_amplitudes_mv(x: np.ndarray, peaks: np.ndarray, fs: int) -> np.ndarray:
    """Peak-to-valley amplitude around each detected peak (EC13 definition).

    Noise inside the window adds to the reading, so in a noisy window the
    floor is effectively a little lower than 0.15 mV. The error is towards
    counting a beat, not towards a false asystole.
    """
    half = max(1, int(QRS_AMP_HALF_MS / 1000.0 * fs))
    amps = np.empty(len(peaks))
    for i, p in enumerate(peaks):
        seg = x[max(0, p - half):min(len(x), p + half + 1)]
        amps[i] = float(np.ptp(seg)) if seg.size else 0.0
    return amps


def qrs_widths_ms(x: np.ndarray, peaks: np.ndarray, fs: int) -> np.ndarray:
    """Per-beat QRS width estimate: width of the derivative burst.

    Not a caliper measurement. The slope envelope rises and falls with the
    complex; its width at 30% of its own peak tracks QRS duration. The slope
    passes through zero at the R apex, so a complex shows as two lobes; lobes
    closer than 40 ms are joined, anything further (a noise spike, the T wave)
    is not.

    0.3 and 40 ms have no clinical source and cannot have one. What makes them
    acceptable is agreement with expert measurement within IEC 60601-2-25
    limits, which eval_qrs_width_qtdb measures. On synthetic complexes the
    estimate reads about a quarter short.
    """
    d = np.abs(np.diff(x, prepend=x[0]))
    half = max(1, int(QRS_SEARCH_HALF_MS / 1000.0 * fs))
    bridge = max(1, int(0.040 * fs))
    widths = np.full(len(peaks), np.nan)
    for i, p in enumerate(peaks):
        seg = d[max(0, p - half):min(len(d), p + half + 1)]
        if seg.size < 3:
            continue
        top = int(np.argmax(seg))
        thr = 0.3 * float(seg[top])
        if thr <= 0:
            continue
        above = np.flatnonzero(seg >= thr)
        lo = hi = top
        for j in above[above < top][::-1]:
            if lo - j > bridge:
                break
            lo = j
        for j in above[above > top]:
            if j - hi > bridge:
                break
            hi = j
        widths[i] = (hi - lo + 1) / fs * 1000.0
    return widths


def _median_width(x: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    if len(peaks) < 2:
        return float("nan")
    w = qrs_widths_ms(x, peaks, fs)
    w = w[~np.isnan(w)]
    return float(np.median(w)) if w.size else float("nan")


def detect_beats(x: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    """R peaks that pass the EC13 amplitude floor, and how many did not."""
    peaks = detect_r_pantompkins(x, fs)
    if len(peaks) == 0:
        return peaks, 0
    keep = qrs_amplitudes_mv(x, peaks, fs) > QRS_MIN_AMP_MV
    return peaks[keep], int((~keep).sum())


def analyse(x: np.ndarray, fs: int = TARGET_FS) -> RateReport:
    """Measure one window (model path, millivolts) and return its findings."""
    x = np.asarray(x, dtype=np.float64)
    rep = RateReport()
    dur = len(x) / fs

    # Digitally flat: below one step of the converter. A real asystole drifts
    # and carries sensor noise; a constant stream points at the equipment,
    # which ACLS also lists among the causes of a flat line. It is reported as
    # asystole and flagged, so the decision layer can say to check the device.
    if np.ptp(x) < MIN_DYNAMIC_RANGE or np.std(x) < MIN_DYNAMIC_RANGE:
        rep.flat = True
        rep.longest_gap_sec = dur
        f = classify_gap(dur)
        if f:
            f.detail = "flat trace; check the patch and the device"
            rep.findings.append(f)
        return rep

    peaks, rep.n_rejected = detect_beats(x, fs)
    rep.n_beats = len(peaks)

    # Gaps include the window edges: a pause that started before the window
    # is still a pause, measured as at least what the window shows.
    # Known bias, pinned by a test: the detector's peak is not exactly on the
    # R apex, so a gap reads a few tens of ms off, in either direction.
    edges = np.concatenate([[0], peaks, [len(x) - 1]])
    rep.longest_gap_sec = float(np.diff(edges).max() / fs)
    f = classify_gap(rep.longest_gap_sec)
    if f:
        rep.findings.append(f)

    if len(peaks) < 2:
        return rep
    rr = np.diff(peaks) / fs
    rr = rr[rr > 0]
    if rr.size == 0:
        return rep
    rep.rr_sec = rr
    rep.bpm = float(60.0 / np.median(rr))
    findings, held = rate_findings(rr)
    for k, v in held.items():
        setattr(rep, k, v)
    rep.qrs_ms = _median_width(x, peaks, fs)
    rep.qrs_class = classify_width(rep.qrs_ms)
    rep.findings.extend(findings)
    if (FAST_NARROW_LABEL and not np.isnan(rep.fast_run_bpm)
            and rep.fast_run_bpm >= FAST_NARROW_BPM and rep.qrs_class == "narrow"):
        rep.findings.append(Finding("fast_narrow", "warning", rep.fast_run_bpm,
                                    detail=f"QRS {rep.qrs_ms:.0f} ms"))
    return rep


def sliding_reports(signal: np.ndarray, fs: int = TARGET_FS,
                    window_sec: float = RULES_WINDOW_SEC,
                    stride_sec: float = RULES_STRIDE_SEC):
    """Yield (t_end, report) over a recording at the deployment window/stride.

    t_end is the time in seconds at the end of the window, which is when the
    device would have the data to decide.
    """
    win, step = int(window_sec * fs), int(stride_sec * fs)
    for start in range(0, len(signal) - win + 1, step):
        yield (start + win) / fs, analyse(signal[start:start + win], fs)


if __name__ == "__main__":
    # Every branch on a signal built for it. Amplitudes in millivolts.
    from ..data.preprocess import bandpass
    from ..config import MODEL_BAND

    fs = TARGET_FS
    rng = np.random.default_rng(0)

    def beats(bpm, seconds=RULES_WINDOW_SEC, qrs_ms=80, r_mv=1.0, gap=None):
        n = int(seconds * fs)
        k = np.arange(n)
        x = np.zeros(n)
        sd = qrs_ms / 1000.0 * fs / 6.0
        t, times = 0.4, []
        while t < seconds - 0.4:
            times.append(t)
            t += 60.0 / bpm
        if gap:
            g0, glen = gap
            times = [s for s in times if s <= g0] + [g0 + glen] + \
                    [s for s in times if s >= g0 + glen + 60.0 / bpm]
        for s in times:
            c = int(round(s * fs))
            x += r_mv * np.exp(-0.5 * ((k - c) / sd) ** 2)
            x += 0.25 * np.exp(-0.5 * ((k - c - 0.22 * fs) / (0.045 * fs)) ** 2)
        return bandpass(x + rng.normal(0, 0.02, n), MODEL_BAND)

    cases = [
        ("normal 70", beats(70)),
        ("bradycardia 55", beats(55)),
        ("bradycardia 45", beats(45)),
        ("extreme brady 35", beats(35)),
        ("sustained brady 28", beats(28)),
        ("tachycardia 105", beats(105)),
        ("tachycardia 115", beats(115)),
        ("extreme tachy 160 narrow", beats(160, qrs_ms=70)),
        ("extreme tachy 160 wide", beats(160, qrs_ms=220)),
        ("sustained tachy 190", beats(190)),
        ("pause 4 s", beats(70, gap=(5.0, 4.0))),
        ("asystole 7 s", beats(70, gap=(5.0, 7.0))),
        ("quiet line (real asystole)",
         bandpass(rng.normal(0, 0.02, int(RULES_WINDOW_SEC * fs)), MODEL_BAND)),
        ("digitally flat", np.zeros(int(RULES_WINDOW_SEC * fs))),
    ]
    for name, s in cases:
        r = analyse(s, fs)
        bpm = "   n/a" if np.isnan(r.bpm) else f"{r.bpm:6.1f}"
        sent = [f.kind for f in r.findings if f.notify]
        print(f"{name:<28} bpm {bpm}  {r.findings or '-'}  notify: {sent or '-'}")
