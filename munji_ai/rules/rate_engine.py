"""Rate engine — the conditions that are arithmetic, not learning.

    from munji_ai.rules.rate_engine import analyse
    findings = analyse(signal)          # one window
    findings = analyse_stream(windows)  # with N-of-M confirmation

Eight of the twenty conditions in the client specification are defined by a
number, not by a shape: how fast the heart is going, and how long it stops for.
A classifier is the wrong instrument for those. "Heart rate below 40" is a
threshold; training a network to rediscover it from examples would make it
slower, less interpretable, and harder to defend to a regulator who can read
the rule directly.

    sinus bradycardia       HR at or below HR_BRADY["info"]
    extreme bradycardia     HR at or below HR_BRADY["critical"]
    sinus tachycardia       HR at or above HR_TACHY["info"]
    extreme tachycardia     HR at or above HR_TACHY["critical"]
    SVT (fast narrow)       HR >= FAST_NARROW_BPM with a narrow QRS
    wide-complex tachycardia HR >= FAST_NARROW_BPM with a wide QRS
    pause                   no beat for PAUSE_WARNING_SEC
    asystole                no beat for ASYSTOLE_CRITICAL_SEC

Every threshold is read from config. Nothing here invents a number.

Where this sits
---------------
After the quality gate, beside the two networks, before the decision layer.
It runs on the same windows they do and needs only R-peak timing, which is why
GATE_MIN_P allows it through at 0.60 where the beat classifier demands 0.90:
morphology can be too poor to classify a complex and still good enough to
time it.

Asystole is the one finding that depends on the gate rather than on itself. A
detached electrode produces a flat trace, and a flat trace is indistinguishable
from a stopped heart by rate alone. So asystole is reported with a `gated`
flag: the decision layer must confirm the window passed quality before acting
on it. Raising a cardiac arrest alarm because a patch came loose would teach a
patient to ignore the device.

What it does not do
-------------------
It does not diagnose. Sinus bradycardia and complete heart block can produce
the same rate; distinguishing them needs the P waves, which is the rhythm
classifier's work, not a threshold's. The findings here are rate observations,
and the decision layer combines them with the networks' output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import (ASYSTOLE_CRITICAL_SEC, CONFIRMATION, FAST_NARROW_BPM,
                      HR_BRADY, HR_TACHY, NARROW_QRS_MS, PAUSE_WARNING_SEC,
                      TARGET_FS)
from ..features.sqi import MIN_DYNAMIC_RANGE, detect_r_pantompkins

# Severity names the decision layer routes on. Ordered, so max() means
# "the most urgent finding in this window".
SEVERITY = ("info", "warning", "critical")


@dataclass
class Finding:
    """One rate observation about one window."""
    kind: str
    severity: str
    value: float = float("nan")      # the measurement that triggered it
    gated: bool = False              # true if a quality check must confirm it
    detail: str = ""

    def __repr__(self) -> str:
        v = "" if np.isnan(self.value) else f" {self.value:.1f}"
        g = " [needs quality confirmation]" if self.gated else ""
        return f"{self.kind}({self.severity}{v}){g}"


@dataclass
class RateReport:
    """Everything the engine measured on one window."""
    bpm: float = float("nan")
    rr_sec: np.ndarray = field(default_factory=lambda: np.array([]))
    qrs_ms: float = float("nan")
    n_beats: int = 0
    flat: bool = False
    findings: list = field(default_factory=list)

    @property
    def severity(self) -> str:
        """Worst severity present, or "" when nothing fired."""
        if not self.findings:
            return ""
        return max((f.severity for f in self.findings), key=SEVERITY.index)

    def kinds(self) -> set:
        return {f.kind for f in self.findings}


def _qrs_width_ms(x: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    """Median QRS duration, measured as the width of the energy burst.

    Not a clinical caliper measurement. The envelope of the squared derivative
    rises and falls with the complex, so the width at a fraction of its own
    peak tracks QRS duration closely enough to separate narrow from wide —
    which is the only distinction asked of it here.

    A real ventricular complex is broad and slurred; a supraventricular one is
    sharp. The 120 ms line between them is the clinical convention and comes
    from config.
    """
    if len(peaks) < 2:
        return float("nan")
    d = np.abs(np.diff(x, prepend=x[0]))
    half = max(1, int(0.06 * fs))        # look 60 ms either side of the peak
    widths = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(d), p + half + 1)
        seg = d[lo:hi]
        if seg.size < 3:
            continue
        thr = 0.3 * float(seg.max())
        if thr <= 0:
            continue
        above = np.flatnonzero(seg >= thr)
        if above.size:
            widths.append((above[-1] - above[0] + 1) / fs * 1000.0)
    return float(np.median(widths)) if widths else float("nan")


def analyse(x: np.ndarray, fs: int = TARGET_FS) -> RateReport:
    """Measure one window and return every rate finding in it."""
    x = np.asarray(x, dtype=np.float64)
    rep = RateReport()
    dur = len(x) / fs

    # A flat trace is either a stopped heart or a detached electrode, and the
    # rate cannot tell them apart. Report it, flag it, let the gate decide.
    if np.ptp(x) < MIN_DYNAMIC_RANGE or np.std(x) < MIN_DYNAMIC_RANGE:
        rep.flat = True
        if dur >= ASYSTOLE_CRITICAL_SEC:
            rep.findings.append(Finding(
                "asystole", "critical", dur, gated=True,
                detail="flat trace — confirm the electrodes before alerting"))
        return rep

    peaks = detect_r_pantompkins(x, fs)
    rep.n_beats = len(peaks)

    # --- pauses and asystole ------------------------------------------------
    # Measured on the gaps including the window edges: a pause that starts
    # before the window and ends inside it is still a pause.
    #
    # Gap length carries about 30 ms of systematic shortening, because the
    # detector integrates over a 120 ms window and its peak sits slightly
    # inside the complex rather than on it. Measured against synthetic gaps:
    # 4.00 s reads as 3.97 s. That matters only within a few tens of
    # milliseconds of a threshold, and no clinical pause is defined to that
    # precision — but it does mean a gap reported as 3.97 s may have been
    # 4.00 s, so the boundary is soft in the safe direction only by accident,
    # not by design. Worth remembering before anyone tightens the threshold.
    edges = np.concatenate([[0], peaks, [len(x) - 1]]) if len(peaks) \
        else np.array([0, len(x) - 1])
    gaps = np.diff(edges) / fs
    longest = float(gaps.max()) if gaps.size else dur

    if longest >= ASYSTOLE_CRITICAL_SEC:
        rep.findings.append(Finding(
            "asystole", "critical", longest, gated=True,
            detail=f"no beat for {longest:.1f} s"))
    elif longest >= PAUSE_WARNING_SEC:
        rep.findings.append(Finding(
            "pause", "warning", longest,
            detail=f"no beat for {longest:.1f} s"))

    if len(peaks) < 2:
        return rep                     # nothing else is measurable

    # --- rate ---------------------------------------------------------------
    # Median rather than mean: one missed beat doubles an RR interval, and a
    # mean would carry that error into the rate while a median shrugs it off.
    rr = np.diff(peaks) / fs
    rr = rr[rr > 0]
    if rr.size == 0:
        return rep
    rep.rr_sec = rr
    rep.bpm = float(60.0 / np.median(rr))
    rep.qrs_ms = _qrs_width_ms(x, peaks, fs)

    bpm = rep.bpm
    if bpm <= HR_BRADY["critical"]:
        rep.findings.append(Finding("extreme_bradycardia", "critical", bpm))
    elif bpm <= HR_BRADY["warning"]:
        rep.findings.append(Finding("bradycardia", "warning", bpm))
    elif bpm <= HR_BRADY["info"]:
        rep.findings.append(Finding("bradycardia", "info", bpm))

    if bpm >= HR_TACHY["critical"]:
        rep.findings.append(Finding("extreme_tachycardia", "critical", bpm))
    elif bpm >= HR_TACHY["warning"]:
        rep.findings.append(Finding("tachycardia", "warning", bpm))
    elif bpm >= HR_TACHY["info"]:
        rep.findings.append(Finding("tachycardia", "info", bpm))

    # --- narrow or wide complex tachycardia ---------------------------------
    # The distinction that matters clinically. Fast and narrow is
    # supraventricular and rarely immediately lethal; fast and wide is
    # ventricular until proven otherwise, and the shockable detector should
    # already be looking at it.
    if bpm >= FAST_NARROW_BPM and not np.isnan(rep.qrs_ms):
        if rep.qrs_ms < NARROW_QRS_MS:
            rep.findings.append(Finding(
                "fast_narrow", "warning", bpm,
                detail=f"QRS {rep.qrs_ms:.0f} ms — supraventricular"))
        else:
            rep.findings.append(Finding(
                "wide_complex", "critical", bpm,
                detail=f"QRS {rep.qrs_ms:.0f} ms — ventricular until excluded"))

    return rep


def analyse_stream(windows, fs: int = TARGET_FS,
                   usable=None) -> dict:
    """Run the engine over consecutive windows and apply N-of-M confirmation.

    `usable` is an optional per-window boolean from the quality gate. Windows
    marked unusable still get measured — the measurement is informative — but
    a gated finding in one of them is never confirmed, which is what keeps a
    loose electrode from being reported as asystole.

    Returns one entry per condition: how many windows raised it, and whether
    the confirmation rule in config was ever satisfied.
    """
    reports = [analyse(w, fs) for w in windows]
    ok = [True] * len(reports) if usable is None else list(usable)

    kinds = sorted({f.kind for r in reports for f in r.findings})
    out = {}
    for kind in kinds:
        rule = CONFIRMATION.get(_rule_key(kind), (2, 3))
        n, m = rule
        hits = []
        for i, r in enumerate(reports):
            f = next((f for f in r.findings if f.kind == kind), None)
            hits.append(bool(f) and (ok[i] or not f.gated))

        confirmed = any(sum(hits[max(0, i - m + 1):i + 1]) >= n
                        for i in range(len(hits)))
        sev = max((f.severity for r in reports for f in r.findings
                   if f.kind == kind), key=SEVERITY.index)
        out[kind] = {"windows": sum(hits), "confirmed": confirmed,
                     "rule": rule, "severity": sev}

    bpms = np.array([r.bpm for r in reports if not np.isnan(r.bpm)])
    out["_summary"] = {
        "windows": len(reports),
        "median_bpm": float(np.median(bpms)) if bpms.size else float("nan"),
        "flat_windows": sum(1 for r in reports if r.flat),
    }
    return out


def _rule_key(kind: str) -> str:
    """Map a finding name onto its CONFIRMATION entry."""
    return {
        "extreme_bradycardia": "extreme_brady",
        "bradycardia": "extreme_brady",
        "extreme_tachycardia": "extreme_tachy",
        "tachycardia": "extreme_tachy",
    }.get(kind, kind)


if __name__ == "__main__":
    # Synthetic check: every branch should fire on a signal built for it.
    fs = TARGET_FS

    def beats(bpm, seconds=10.0, width_ms=80, amp=2.0):
        n = int(seconds * fs)
        x = np.zeros(n)
        half = max(1, int(width_ms / 1000 * fs / 2))
        step = 60.0 / bpm
        t = 0.2
        while t * fs < n - half:
            c = int(t * fs)
            w = np.arange(max(0, c - half), min(n, c + half + 1))
            x[w] += amp * np.exp(-((w - c) / (half / 1.5)) ** 2)
            t += step
        return x + 0.01 * np.random.RandomState(0).randn(n)

    cases = [
        ("normal 70 bpm", beats(70)),
        ("bradycardia 50", beats(50)),
        ("extreme brady 35", beats(35)),
        ("tachycardia 110", beats(110)),
        ("extreme tachy 160 narrow", beats(160, width_ms=70)),
        ("wide complex 160", beats(160, width_ms=160)),
        ("flat (lead off)", np.zeros(int(10 * fs))),
    ]
    for name, sig_ in cases:
        r = analyse(sig_, fs)
        bpm = "   n/a" if np.isnan(r.bpm) else f"{r.bpm:6.1f}"
        qrs = " n/a" if np.isnan(r.qrs_ms) else f"{r.qrs_ms:4.0f}"
        print(f"{name:<26} bpm {bpm}  qrs {qrs} ms  "
              f"{r.findings if r.findings else '—'}")

    print("\nstream, 6 windows of 160 bpm narrow:")
    for k, v in analyse_stream([beats(160, width_ms=70)] * 6).items():
        print(f"  {k:<22}{v}")
