"""Decision layer: turns model scores and rate findings into the alerts a person sees.

    from munji_ai.decision.decision_layer import DecisionLayer
    dl = DecisionLayer()
    alerts = dl.on_shockable(t, p_shockable, rate_report, lead_off=False)
    # call on_shockable before on_rate for the same window
    alerts = dl.on_rhythm(t, (p_nsr, p_af, p_other), usable=True, lead_off=False)
    alerts = dl.on_rate(t, rate_report, usable=True, lead_off=False, complete=True)
    summary = dl.report()            # AF burden and counts for the doctor's PDF

Why a separate layer
--------------------
The models and the rate engine each answer one narrow question about one
window. Nobody should be paged by a single window, a single model, or two
models saying the same thing twice. This layer owns every rule between a score
and a phone ringing: thresholds, confirmation, naming, priority, suppression,
once-per-episode, and the daily AF burden. It holds no learned parameters, so
every behaviour can be read, tested and defended.

This is the reference implementation. The app runs the same logic in Dart;
tests/test_decision_layer.py is the behaviour both must match.

Rules, and where each comes from
--------------------------------
Shockable (ShockableNet, every 2.5 s)
    A window is positive at P(shockable) >= 0.60; an alert needs 4 positive
    windows in the last 6 (config.CONFIRMATION["vf"]). Both were chosen on
    full-day cross-validation: no sustained episode missed, about 0.5 false
    alarms per patient-day.

    The alert is named VF/VT: a life-threatening ventricular rhythm. Telling
    VF from fast VT was tried four ways on CUDB and VFDB, by patient: rate
    and regularity from the rate engine, dominant frequency and spectral
    peak size from the literature, published complexity measures, and a
    classifier on ShockableNet's own features. None separated them (AUC 0.3
    to 0.6) with VT from only 8 recordings. Both rhythms need the same
    response, an emergency call, so one honest name is safer than a guess.
    Separate names return if more VT data (PhysioNet VTaC) makes them
    reliable.

    The quality gate does NOT silence it. Measured on this project's data,
    artifact-free criteria discarded about 60% of genuine VF windows: VF looks
    like poor signal. Electrode lead-off does silence it, because a detached
    patch cannot show anything.

AF (RhythmNet, every 15 s)
    A window is AF at P(AF) >= 0.60; an alert needs 4 AF windows in the last 6
    (config.CONFIRMATION["af"]): about 0.11 false alerts per day on healthy
    recordings, 91% of 6-minute episodes caught. One alert per episode, and at
    most one per AF_REALERT_S (a day by default), because AF is a warning, not
    an emergency, and repeated alerts for a known rhythm teach people to stop
    reading them. Every AF window still counts towards the burden. Unusable
    windows count neither way. OTHER is never alerted.

Rate rules (rate engine)
    Who is told follows patch-monitoring practice: MCOT notification criteria
    and the ARIC study's alert criteria for the Zio patch in older adults.
    Only these rate findings are ever sent (config, "notification"):

        sustained bradycardia   below 40 held 30 s           urgent   -> warning
        profound bradycardia    30 bpm or less held 30 s     emergent -> critical
        sustained tachycardia   180 bpm or more held 30 s    urgent   -> warning
        pause                   no beat for more than 3 s    urgent   -> warning
        asystole                no beat for more than 6 s    emergent -> critical

    Every other rate finding (below 60 / 50, above 100 / 110, 150-180, and
    the critical tiers when not held 30 s) is graded and counted for the
    doctor's report, never sent. On a healthy NSRDB day those tiers produced about 90
    warnings: the rate really does cross 50 asleep and 110 walking.

    The engine holds each finding over its run (beats, 30 s, or the gap), so
    one window is confirmed (config.CONFIRMATION, (1, 1)). Findings are
    confirmed per family: pause then asystole is one episode, alerted once and
    escalated once.

    Who judges a finding depends on its basis:
      rate findings need a gate-accepted window, because noise invents beats;
      absence findings (pause, asystole) need the leads reported on. The gate
      is not the judge here: a stopped heart can look like poor signal, and
      ACLS asks for the leads to be checked before a flat line is called
      asystole. With lead status unknown (lead_off=None, public recordings),
      the gate decides instead.
    Nothing cardiac is counted while the leads are off or the window has
    missing samples (complete=False); those windows are counted for the
    technical notices.

Priority
    While a VF/VT episode is active, AF and rate warnings are held back: the
    person is already being told the most urgent thing.

    The engine no longer names a fast rhythm narrow (SVT) or wide: its width
    estimate failed on the QT Database. Ventricular rhythms come from the
    ventricular-beat detector. If the SVT label returns, it stays held while
    ShockableNet saw danger in the last 6 windows (ventricular_hold).

Lead-off
    4 lead-off windows in the last 6 raise one "check the patch" notice per
    episode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import config as C
from ..rules.rate_engine import NOTIFY_LEVELS

SEVERITY = ("info", "warning", "critical")


@dataclass
class DecisionConfig:
    shock_threshold: float = 0.60
    shock_rule: tuple = tuple(C.CONFIRMATION["vf"])
    shock_episode_end: int = 12           # negative windows (30 s) that close an episode
    af_threshold: float = 0.60
    af_rule: tuple = tuple(C.CONFIRMATION["af"])
    af_episode_end: int = 40              # windows without AF (10 min) that close an episode
    af_realert_s: float = 24 * 3600.0     # at most one AF alert per this many seconds
    lead_off_rule: tuple = (4, 6)
    gate_shockable_on_quality: bool = False
    ventricular_hold: tuple = ("fast_narrow",)
    rate_episode_end: int = 6             # windows without the finding that close an episode


@dataclass
class Alert:
    t: float
    kind: str
    severity: str
    source: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)


class _Episode:
    """N-of-M confirmation with once-per-episode firing."""

    def __init__(self, rule, end_after):
        self.n, self.m = rule
        self.recent = deque(maxlen=self.m)
        self.end_after = end_after
        self.active = False
        self.quiet = 0

    def step(self, positive: bool) -> bool:
        """True exactly once, when an episode is confirmed."""
        self.recent.append(bool(positive))
        if self.active:
            self.quiet = 0 if positive else self.quiet + 1
            if self.quiet >= self.end_after:
                self.active, self.quiet = False, 0
                self.recent.clear()
            return False
        if sum(self.recent) >= self.n:
            self.active, self.quiet = True, 0
            return True
        return False


class DecisionLayer:
    def __init__(self, cfg: DecisionConfig | None = None):
        self.cfg = cfg or DecisionConfig()
        c = self.cfg
        self._shock = _Episode(c.shock_rule, c.shock_episode_end)
        self._shock_recent: deque = deque(maxlen=c.shock_rule[1])
        self._af = _Episode(c.af_rule, c.af_episode_end)
        self._lead = _Episode(c.lead_off_rule, c.shock_episode_end)
        self._rate: dict = {}
        self._rate_level: dict = {}
        self.rate_report_windows: dict = {}
        self.no_data_windows = 0
        self.lead_off_windows = 0
        self._last_af_alert = -np.inf
        self.af_windows = 0
        self.rhythm_windows = 0
        self.other_windows = 0
        self.alerts: list[Alert] = []

    # ---------------------------------------------------------------- inputs
    def on_shockable(self, t: float, p_shockable: float, rate=None,
                     usable: bool = True, lead_off: bool | None = None) -> list[Alert]:
        out = []
        if self._lead.step(lead_off is True):
            out.append(Alert(t, "check_patch", "info", "device",
                             "the patch seems detached; press it back on"))
        silenced = lead_off is True or (self.cfg.gate_shockable_on_quality and not usable)
        positive = (not silenced) and p_shockable >= self.cfg.shock_threshold
        self._shock_recent.append(positive)
        if self._shock.step(positive):
            bpm = getattr(rate, "bpm", float("nan")) if rate is not None else float("nan")
            out.append(Alert(t, "VF/VT", "critical", "shockable",
                             "life-threatening ventricular rhythm: ventricular "
                             "fibrillation or tachycardia",
                             {"p_shockable": float(p_shockable),
                              "bpm": float(bpm) if np.isfinite(bpm) else None}))
        return self._emit(out)

    def on_rhythm(self, t: float, probs, usable: bool = True,
                  lead_off: bool | None = None) -> list[Alert]:
        p_nsr, p_af, p_other = (float(v) for v in probs)
        if not usable or lead_off is True:
            self._af.step(False)
            return []
        self.rhythm_windows += 1
        is_af = p_af >= self.cfg.af_threshold
        self.af_windows += int(is_af)
        self.other_windows += int(not is_af and p_other >= max(p_nsr, p_af))
        out = []
        if self._af.step(is_af):
            if self._shock.active:
                return []
            if t - self._last_af_alert >= self.cfg.af_realert_s:
                self._last_af_alert = t
                out.append(Alert(t, "AF/AFL", "warning", "rhythm",
                                 "possible atrial fibrillation or flutter; "
                                 "a doctor should confirm on a 12-lead ECG",
                                 {"p_af": p_af}))
        return self._emit(out)

    def on_rate(self, t: float, rate, usable: bool = True,
                lead_off: bool | None = None, complete: bool = True) -> list[Alert]:
        """One rate-engine window. lead_off: True, False, or None if unknown."""
        if not complete:
            self.no_data_windows += 1
        elif lead_off is True:
            self.lead_off_windows += 1
        monitored = complete and lead_off is not True

        worst: dict = {}         # family -> most urgent finding that is sent
        graded: dict = {}        # family -> most severe report-only finding
        if monitored:
            for f in getattr(rate, "findings", []):
                if f.basis == "rate":
                    counts = usable
                else:
                    counts = lead_off is False or (lead_off is None and usable)
                if not counts:
                    continue
                if f.family in self.cfg.ventricular_hold and self.ventricular_activity():
                    continue
                if f.notify:
                    cur = worst.get(f.family)
                    if cur is None or (NOTIFY_LEVELS.index(f.notify)
                                       > NOTIFY_LEVELS.index(cur.notify)):
                        worst[f.family] = f
                else:
                    cur = graded.get(f.family)
                    if cur is None or SEVERITY.index(f.severity) > SEVERITY.index(cur.severity):
                        graded[f.family] = f

        for family, f in graded.items():
            by_sev = self.rate_report_windows.setdefault(family, {})
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1

        out = []
        for family, key in RATE_FAMILIES.items():
            f = worst.get(family)
            alerting = f is not None
            ep = self._rate.get(family)
            if ep is None:
                ep = self._rate[family] = _Episode(tuple(C.CONFIRMATION[key]),
                                                   self.cfg.rate_episode_end)
            started = ep.step(alerting)
            if not ep.active:
                self._rate_level.pop(family, None)
                continue
            level = self._rate_level.get(family)
            escalated = (alerting and level is not None
                         and NOTIFY_LEVELS.index(f.notify) > NOTIFY_LEVELS.index(level))
            if not (started or escalated):
                continue
            self._rate_level[family] = f.notify
            severity = NOTIFY_SEVERITY[f.notify]
            if self._shock.active and severity != "critical":
                continue
            out.append(Alert(t, f.kind, severity, "rate", f.detail,
                             {"value": float(f.value), "notify": f.notify}))
        return self._emit(out)

    def ventricular_activity(self) -> bool:
        """ShockableNet saw danger recently, or its episode is still open."""
        return self._shock.active or any(self._shock_recent)

    # --------------------------------------------------------------- outputs
    def report(self) -> dict:
        n = self.rhythm_windows
        return {"rhythm_windows": n,
                "af_windows": self.af_windows,
                "af_burden": self.af_windows / n if n else 0.0,
                "af_minutes": self.af_windows * C.WINDOW_STRIDE_SEC["rhythm"] / 60.0,
                "other_windows": self.other_windows,
                "rate_report_windows": {k: dict(v) for k, v in self.rate_report_windows.items()},
                "no_data_windows": self.no_data_windows,
                "lead_off_windows": self.lead_off_windows,
                "alerts": [(a.t, a.kind, a.severity) for a in self.alerts]}

    def _emit(self, out: list[Alert]) -> list[Alert]:
        self.alerts.extend(out)
        return out


# notification level -> alert severity
NOTIFY_SEVERITY = {"urgent": "warning", "emergent": "critical"}

# rate family -> its config.CONFIRMATION entry
RATE_FAMILIES = {
    "absence": "asystole",
    "brady": "extreme_brady",
    "tachy": "extreme_tachy",
    "fast_narrow": "fast_narrow",
}
