"""Decision layer: turns model scores and rate findings into the alerts a person sees.

    from munji_ai.decision.decision_layer import DecisionLayer
    dl = DecisionLayer()
    alerts = dl.on_shockable(t, p_shockable, rate_report, lead_off=False)
    # call on_shockable before on_rate for the same window
    alerts = dl.on_rhythm(t, (p_nsr, p_af, p_other), usable=True, lead_off=False)
    alerts = dl.on_rate(t, rate_report, usable=True, lead_off=False)
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
    Each finding is confirmed with its own N-of-M from config.CONFIRMATION and
    alerted once per episode. Asystole is gated: it needs a usable window and
    an attached electrode, since a loose patch also reads flat.

Priority
    While a VF/VT episode is active, AF and rate warnings are held back: the
    person is already being told the most urgent thing.

    Fast-narrow (SVT) and wide-complex findings from the rate engine are held
    whenever ShockableNet saw a shockable window in the last 6. On 429 real
    fast-VT windows the rate engine called 22% "fast narrow" and only 14%
    "wide complex": its beat detector and QRS width are built for normal
    beats. Calling VT "SVT" could mislead treatment, so ShockableNet's view
    wins whenever it sees ventricular danger.

Lead-off
    4 lead-off windows in the last 6 raise one "check the patch" notice per
    episode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import config as C

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
    ventricular_hold: tuple = ("fast_narrow", "wide_complex")
    rate_episode_end: int = 6


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
        self._last_af_alert = -np.inf
        self.af_windows = 0
        self.rhythm_windows = 0
        self.other_windows = 0
        self.alerts: list[Alert] = []

    # ---------------------------------------------------------------- inputs
    def on_shockable(self, t: float, p_shockable: float, rate=None,
                     usable: bool = True, lead_off: bool = False) -> list[Alert]:
        out = []
        if self._lead.step(lead_off):
            out.append(Alert(t, "check_patch", "info", "device",
                             "the patch seems detached; press it back on"))
        silenced = lead_off or (self.cfg.gate_shockable_on_quality and not usable)
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
                  lead_off: bool = False) -> list[Alert]:
        p_nsr, p_af, p_other = (float(v) for v in probs)
        if not usable or lead_off:
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
                lead_off: bool = False) -> list[Alert]:
        out = []
        present = {f.kind: f for f in getattr(rate, "findings", [])}
        kinds = set(present) | set(self._rate)
        for kind in sorted(kinds):
            f = present.get(kind)
            ok = f is not None and (not f.gated or (usable and not lead_off))
            if kind in self.cfg.ventricular_hold and self.ventricular_activity():
                ok = False
            ep = self._rate.get(kind)
            if ep is None:
                rule = tuple(C.CONFIRMATION.get(_rule_key(kind), (2, 3)))
                ep = self._rate[kind] = _Episode(rule, self.cfg.rate_episode_end)
            if ep.step(ok):
                if self._shock.active and f.severity != "critical":
                    continue
                out.append(Alert(t, kind, f.severity, "rate", f.detail,
                                 {"value": float(f.value)}))
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
                "alerts": [(a.t, a.kind, a.severity) for a in self.alerts]}

    def _emit(self, out: list[Alert]) -> list[Alert]:
        self.alerts.extend(out)
        return out


def _rule_key(kind: str) -> str:
    return {"extreme_bradycardia": "extreme_brady", "bradycardia": "extreme_brady",
            "extreme_tachycardia": "extreme_tachy", "tachycardia": "extreme_tachy"
            }.get(kind, kind)
