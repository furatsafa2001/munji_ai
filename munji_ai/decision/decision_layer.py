"""Decision layer: turns model scores and rate findings into the alerts a person sees.

    from munji_ai.decision.decision_layer import DecisionLayer
    dl = DecisionLayer()
    alerts = dl.on_shockable(t, p_shockable, rate_report, lead_off=False)
    # call on_shockable before on_rate for the same window
    alerts = dl.on_rhythm(t, (p_nsr, p_af, p_other), usable=True, lead_off=False)
    alerts = dl.on_rate(t, rate_report, usable=True, lead_off=False, complete=True)
    alerts = dl.on_beats(t, beats_from_mask(mask, t_start), usable=p_usable >= 0.90)
    summary = dl.report()            # AF burden and counts for the doctor's PDF
    saves = dl.take_recording_requests()   # ECG strips the app should keep

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

Beats (BeatSegmenter, every beat window)
    Each beat arrives as (time in seconds, "N" | "PAC" | "PVC");
    beats_from_mask() turns the model's per-sample mask into that, reading it
    the way train_beat.mask_to_events does. Beat windows overlap, so a beat
    closer than BEAT_DEDUP_SEC to the last one is the same beat and is dropped.
    Windows the gate rejects at the morphology line (0.90) count nothing and
    break runs and patterns: a gap is not evidence either way.

    NSVT, report only: 3 or more consecutive PVCs above 100 bpm that end
    within 30 s (2017 AHA/ACC/HRS definition).

    Sustained VT, alert "WCT" (possible VT), emergent (critical): over the
    last 30 s of usable beats, at least 80% are PVCs and the rate is above
    100. Emergent because sustained VT is: in a series of 26,438 MCOT
    patients, wide-complex tachycardia was among the potentially emergent
    events (config, "notification"). The share
    replaces "every beat consecutive", because one PVC the model misses would
    split a long run into short pieces that never reach 30 s
    (eval/vt_run_simulation.py). No WCT alert while a ShockableNet VF/VT
    episode is open: the same event is already being told.

    PVC burden, report only: share of PVCs over the last hour (peak kept) and
    per day, once there are PVC_BURDEN_MIN_BEATS beats; tiers from config
    (2017 AHA/ACC/HRS, 2014 EHRA/HRS/APHRS). PAC count, report only.

    Bigeminy (N PVC) and trigeminy (N N PVC), report only: an episode once the
    unit repeats 3 times in a row; a PAC or a gap breaks it.

ECG saving
    Every emergency alert (VF/VT, asystole, WCT) adds a request to keep the
    ECG from 30 s before to 30 s after it (take_recording_requests). The app
    does the saving.

Naming VF or VT (off: config.VFVT_NAMING_ENABLED)
    From the beats of the last 10 s when ShockableNet fires: PVCs in a row
    (3 or more) -> "VT"; the beat model ran and saw no beat at all -> "VF";
    otherwise, or no beat-model output for that stretch -> "VF/VT". While off,
    the alert stays "VF/VT" and the guess rides along with the recording
    request (shadow mode) for comparison with the doctor's reading.

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
    vfvt_naming: bool = C.VFVT_NAMING_ENABLED


@dataclass
class Alert:
    t: float
    kind: str
    severity: str
    source: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)


@dataclass
class RecordingRequest:
    """Keep the ECG from start to end (seconds); the app does the saving."""
    t: float
    start: float
    end: float
    reason: str
    shadow_name: str | None = None


# Alerts that ask for the ECG around them to be kept.
RECORD_ON = ("VF/VT", "VF", "VT", "asystole", "WCT")


def beats_from_mask(mask, t_start: float, fs: int = C.TARGET_FS) -> list:
    """BeatSegmenter per-sample mask -> [(time_s, "N" | "PAC" | "PVC")].

    Same reading as train.train_beat.mask_to_events: each contiguous run of
    one non-background label is a beat, placed at its centre. Kept here
    without torch, because the app ports this file.
    """
    m = np.asarray(mask).astype(int).ravel()
    out, i, n = [], 0, m.size
    while i < n:
        c = int(m[i])
        if c == 0:
            i += 1
            continue
        j = i
        while j < n and int(m[j]) == c:
            j += 1
        out.append((t_start + ((i + j) // 2) / fs, C.BEAT_CLASSES[c]))
        i = j
    return out


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
        # beats
        self._beats: deque = deque()          # (t, cls), usable windows only
        self._raw: deque = deque()            # (t, cls) as received, for naming
        self._last_beat_t = -np.inf
        self._last_raw_t = -np.inf
        self._last_beats_call = -np.inf
        self._coverage_start = None           # usable beats continuous since
        self.beat_counts = {"N": 0, "PAC": 0, "PVC": 0}
        self.daily_beats: dict = {}           # day index -> [beats, pvcs]
        self.peak_hour_burden = None          # (t, burden, beats)
        self._run: list = []                  # times of the current PVC run
        self.nsvt: list = []                  # (start, end, beats, bpm), under 30 s
        self.long_runs: list = []             # consecutive PVC runs of 30 s or more
        self._since_v = 0                     # normal beats since the last PVC
        self._cycles = {"bigeminy": 0, "trigeminy": 0}
        self.patterns = {"bigeminy": 0, "trigeminy": 0}
        self._vt_active = False
        self._vt_last_true = -np.inf
        self.recording_requests: list = []
        self._recordings_taken = 0

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
            guess = self._vfvt_guess(t)
            name = guess if self.cfg.vfvt_naming else "VF/VT"
            out.append(Alert(t, name, "critical", "shockable", VFVT_DETAIL[name],
                             {"p_shockable": float(p_shockable),
                              "bpm": float(bpm) if np.isfinite(bpm) else None,
                              "shadow_name": guess}))
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

    def on_beats(self, t: float, beats, usable: bool = True) -> list[Alert]:
        """One beat-model window. beats: [(time_s, "N" | "PAC" | "PVC")].

        usable: the gate passed this window at the morphology line
        (config.GATE_MIN_P["morphology"]).
        """
        self._last_beats_call = t
        incoming = sorted((float(bt), str(c)) for bt, c in beats)
        for bt, c in incoming:                              # for naming only
            if bt > self._last_raw_t + C.BEAT_DEDUP_SEC:
                self._raw.append((bt, c))
                self._last_raw_t = bt
        while self._raw and self._raw[0][0] < t - 2 * C.VFVT_NAMING_LOOKBACK_SEC:
            self._raw.popleft()

        if not usable:
            self._break_beats()
            return self._emit(self._vt_step(t, False))

        for bt, c in incoming:
            if bt <= self._last_beat_t + C.BEAT_DEDUP_SEC or c not in self.beat_counts:
                continue
            self._last_beat_t = bt
            if self._coverage_start is None:
                self._coverage_start = bt
            self._add_beat(bt, c)
        keep = max(C.PVC_BURDEN_WINDOW_SEC, C.VT_SUSTAINED_SEC)
        while self._beats and self._beats[0][0] < self._last_beat_t - keep:
            self._beats.popleft()
        self._update_hour_burden()
        return self._emit(self._vt_check(t))

    def take_recording_requests(self) -> list:
        """Recording requests not yet handed to the app."""
        new = self.recording_requests[self._recordings_taken:]
        self._recordings_taken = len(self.recording_requests)
        return new

    # beat helpers ---------------------------------------------------------
    def _add_beat(self, bt: float, c: str) -> None:
        self._beats.append((bt, c))
        self.beat_counts[c] += 1
        day = self.daily_beats.setdefault(int(bt // C.PVC_BURDEN_DAILY_SEC), [0, 0])
        day[0] += 1
        day[1] += int(c == "PVC")
        if c == "PVC":
            self._run.append(bt)
        else:
            self._close_run()
        self._pattern_step(c)

    def _close_run(self) -> None:
        run, self._run = self._run, []
        if len(run) < C.VT_MIN_BEATS:
            return
        dur = run[-1] - run[0]
        bpm = 60.0 * (len(run) - 1) / dur if dur > 0 else float("inf")
        if bpm <= C.VT_MIN_BPM:
            return
        (self.nsvt if dur < C.VT_SUSTAINED_SEC else self.long_runs).append(
            (run[0], run[-1], len(run), bpm))

    def _pattern_step(self, c: str) -> None:
        if c == "PVC":
            # A unit is (N PVC) or (N N PVC). The first unit of a pattern may
            # follow any number of normal beats; the ones after it must keep
            # the exact spacing, or the pattern restarts from this unit.
            for name, normals in (("bigeminy", 1), ("trigeminy", 2)):
                if self._since_v == normals and self._cycles[name] > 0:
                    self._cycles[name] += 1
                elif self._since_v >= normals:
                    self._cycles[name] = 1
                else:
                    self._cycles[name] = 0
                if self._cycles[name] == C.ECTOPY_PATTERN_MIN_CYCLES:
                    self.patterns[name] += 1
            self._since_v = 0
        elif c == "N":
            self._since_v += 1
        else:                                   # a PAC breaks the alternation
            self._since_v = 0
            self._cycles = {k: 0 for k in self._cycles}

    def _break_beats(self) -> None:
        """A gap: close what was seen, start runs and patterns afresh."""
        self._close_run()
        self._since_v = 0
        self._cycles = {k: 0 for k in self._cycles}
        self._coverage_start = None

    def _update_hour_burden(self) -> None:
        if not self._beats:
            return
        end = self._last_beat_t
        lo = end - C.PVC_BURDEN_WINDOW_SEC
        recent = [c for bt, c in self._beats if bt > lo]
        # a full hour of beats (within a minute), and enough of them
        if len(recent) < C.PVC_BURDEN_MIN_BEATS or end - self._beats[0][0] < C.PVC_BURDEN_WINDOW_SEC - 60:
            return
        burden = recent.count("PVC") / len(recent)
        if self.peak_hour_burden is None or burden > self.peak_hour_burden[1]:
            self.peak_hour_burden = (end, burden, len(recent))

    def _vt_check(self, t: float) -> list:
        if self._coverage_start is None or not self._beats:
            return self._vt_step(t, False)
        ref = self._last_beat_t
        lo = ref - C.VT_SUSTAINED_SEC
        if self._coverage_start > lo + 60.0 / C.VT_MIN_BPM:     # not 30 s of usable beats yet
            return self._vt_step(ref, False)
        win = [(bt, c) for bt, c in self._beats if bt >= lo]
        span = win[-1][0] - win[0][0] if len(win) > 1 else 0.0
        if span < C.VT_SUSTAINED_SEC - 60.0 / C.VT_MIN_BPM:
            return self._vt_step(ref, False)
        frac = sum(c == "PVC" for _, c in win) / len(win)
        bpm = 60.0 * (len(win) - 1) / span
        ok = frac >= C.VT_SUSTAINED_PVC_FRACTION and bpm > C.VT_MIN_BPM
        return self._vt_step(ref, ok, frac, bpm)

    def _vt_step(self, t: float, ok: bool, frac=None, bpm=None) -> list:
        out = []
        if ok:
            if not self._vt_active:
                self._vt_active = True
                if not self._shock.active:          # rule: one alert per event
                    out.append(Alert(t, "WCT", "critical", "beats",
                                     "wide-complex tachycardia, possible VT: "
                                     "ventricular beats held for 30 s",
                                     {"pvc_fraction": float(frac), "bpm": float(bpm)}))
            self._vt_last_true = t
        elif self._vt_active and t - self._vt_last_true > C.VT_SUSTAINED_SEC:
            self._vt_active = False
        return out

    def _vfvt_guess(self, t: float) -> str:
        """VT / VF / VF/VT from the beats of the last few seconds."""
        lb = C.VFVT_NAMING_LOOKBACK_SEC
        if self._last_beats_call < t - lb:          # no beat-model output for it
            return "VF/VT"
        recent = [c for bt, c in self._raw if t - lb <= bt <= t]
        if not recent:
            return "VF"
        run = best = 0
        for c in recent:
            run = run + 1 if c == "PVC" else 0
            best = max(best, run)
        return "VT" if best >= C.VT_MIN_BEATS else "VF/VT"

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
                "beats": dict(self.beat_counts),
                "pac_count": self.beat_counts["PAC"],
                "pvc_burden": self._burden_report(),
                "nsvt": len(self.nsvt),
                "nsvt_runs": list(self.nsvt),
                "long_vt_runs": list(self.long_runs),
                "bigeminy_episodes": self.patterns["bigeminy"],
                "trigeminy_episodes": self.patterns["trigeminy"],
                "recording_requests": [(r.t, r.start, r.end, r.reason, r.shadow_name)
                                       for r in self.recording_requests],
                "alerts": [(a.t, a.kind, a.severity) for a in self.alerts]}

    def _burden_report(self) -> dict:
        def tier(b):
            names = [k for k, v in C.PVC_BURDEN.items() if b >= v]
            return names[-1] if names else ""
        daily = {}
        for day, (n, v) in sorted(self.daily_beats.items()):
            if n >= C.PVC_BURDEN_MIN_BEATS:
                daily[day] = {"beats": n, "pvc": v, "burden": v / n, "tier": tier(v / n)}
        peak = None
        if self.peak_hour_burden is not None:
            t, b, n = self.peak_hour_burden
            peak = {"t": t, "burden": b, "beats": n, "tier": tier(b)}
        return {"peak_hour": peak, "daily": daily}

    def _emit(self, out: list[Alert]) -> list[Alert]:
        for a in out:
            if a.kind in RECORD_ON:
                req = RecordingRequest(a.t, a.t - C.ECG_SAVE_BEFORE_SEC,
                                       a.t + C.ECG_SAVE_AFTER_SEC, a.kind,
                                       a.evidence.get("shadow_name"))
                self.recording_requests.append(req)
                a.evidence["save_ecg"] = {"start": req.start, "end": req.end}
        self.alerts.extend(out)
        return out


# what each name of a ShockableNet alert tells the person
VFVT_DETAIL = {
    "VF/VT": "life-threatening ventricular rhythm: ventricular fibrillation or tachycardia",
    "VF": "life-threatening ventricular rhythm: ventricular fibrillation",
    "VT": "life-threatening ventricular rhythm: ventricular tachycardia",
}

# notification level -> alert severity
NOTIFY_SEVERITY = {"urgent": "warning", "emergent": "critical"}

# rate family -> its config.CONFIRMATION entry
RATE_FAMILIES = {
    "absence": "asystole",
    "brady": "extreme_brady",
    "tachy": "extreme_tachy",
    "fast_narrow": "fast_narrow",
}
