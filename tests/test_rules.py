"""Rate engine checks. Synthetic signals in millivolts, no PhysioNet needed.

    python tests/test_rules.py

Threshold boundaries are tested on the pure classify functions, not through a
signal: 250 Hz quantises RR (a "59 bpm" signal measures 59.06) and QRS width
moves in 4 ms steps, so a signal cannot land on a boundary. Signals test what
only a signal can: detection, the amplitude floor, runs, gaps and width.
Confirmation lives in the decision layer: tests/test_decision_layer.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data.preprocess import bandpass
from munji_ai.decision.decision_layer import RATE_FAMILIES
from munji_ai.rules import rate_engine as R

FS = C.TARGET_FS
WIN = C.RULES_WINDOW_SEC


def ecg(bpm=70, seconds=WIN, qrs_ms=80, r_mv=1.0, noise_mv=0.02, p_mv=0.0,
        t_mv=0.25, seed=0, times=None):
    """Deterministic beats on the model path, in mV.

    qrs_ms is the full span of the complex (+/-3 sd). times, if given, are the
    exact beat times in seconds and override bpm.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    k = np.arange(n)
    x = np.zeros(n)
    sd = qrs_ms / 1000.0 * FS / 6.0
    if times is None:
        times, t = [], 0.4
        while t < seconds - 0.4:
            times.append(t)
            t += 60.0 / bpm
    for s in times:
        c = int(round(s * FS))
        if r_mv:
            x += r_mv * np.exp(-0.5 * ((k - c) / sd) ** 2)
        if t_mv:
            x += t_mv * np.exp(-0.5 * ((k - c - 0.22 * FS) / (0.045 * FS)) ** 2)
        if p_mv:
            x += p_mv * np.exp(-0.5 * ((k - c + 0.16 * FS) / (0.02 * FS)) ** 2)
    return bandpass(x + rng.normal(0, noise_mv, n), C.MODEL_BAND)


def regular(start, rr, until):
    out, t = [], start
    while t < until:
        out.append(t)
        t += rr
    return out


def quiet(seconds=WIN, noise_mv=0.02, seed=1):
    """No heartbeat at all, only sensor noise: a real asystole is not flat."""
    rng = np.random.default_rng(seed)
    return bandpass(rng.normal(0, noise_mv, int(seconds * FS)), C.MODEL_BAND)


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def ks(findings):
        return {(f.kind, f.severity) for f in findings}

    nan = float("nan")

    # ------------------------------------------------------------------
    print("\n1. rate boundaries (pure; held rates in, findings out)")
    for slow, want in [(39.9, {("extreme_bradycardia", "critical")}),
                       (40.0, {("bradycardia", "warning")}),
                       (49.9, {("bradycardia", "warning")}),
                       (50.0, {("bradycardia", "info")}),
                       (59.9, {("bradycardia", "info")}),
                       (60.0, set())]:
        got = ks(R.classify_rate(slow, nan))
        check(f"held {slow:5.1f} -> {sorted(want) or 'nothing'}", got == want, str(got))
    for fast, want in [(100.0, set()),
                       (100.1, {("tachycardia", "info")}),
                       (110.0, {("tachycardia", "info")}),
                       (110.1, {("tachycardia", "warning")}),
                       (149.9, {("tachycardia", "warning")}),
                       (150.0, {("extreme_tachycardia", "critical")})]:
        got = ks(R.classify_rate(nan, fast))
        check(f"held {fast:5.1f} -> {sorted(want) or 'nothing'}", got == want, str(got))
    check("no run seen -> nothing", R.classify_rate(nan, nan) == [])

    print("\n2. gap boundaries (pure)")
    for g, want in [(3.0, None), (3.01, "pause"), (6.0, "pause"),
                    (6.01, "asystole"), (nan, None)]:
        f = R.classify_gap(g)
        check(f"gap {g} s -> {want}", (f.kind if f else None) == want)

    print("\n3. width classes (pure)")
    check("119.9 -> narrow", R.classify_width(119.9) == "narrow")
    check("120.0 -> wide", R.classify_width(120.0) == "wide")
    check("NaN -> unmeasured", R.classify_width(nan) == "")
    check("with a 15 ms band: 106 -> narrow, 110 -> uncertain, 134 -> uncertain, 135 -> wide",
          [R.classify_width(w, 15) for w in (105, 110, 134, 135)]
          == ["narrow", "uncertain", "uncertain", "wide"])
    check("band unset in config until QT Database calibrates it", C.QRS_UNCERTAIN_MS is None)

    print("\n4. held rate over a run (monitor-style trimmed mean)")
    rr = np.full(10, 0.8)
    rr[4] = 4.0                                   # one long pause among normal beats
    slow, _ = R.held_rates(rr, 5)
    check("one pause does not make a brady run", slow > 60, f"{slow:.1f}")
    rr = np.full(20, 0.375)
    rr[7] = 0.75                                  # one missed beat in a fast run
    _, fast = R.held_rates(rr, 17)
    check("one missed beat does not break a tachy run", fast >= 150, f"{fast:.1f}")
    rr = np.random.default_rng(2).uniform(0.25, 0.50, 40)
    _, fast = R.held_rates(rr, 17)
    check("irregular fast rhythm (AF around 160) still holds >= 150", fast >= 150, f"{fast:.1f}")
    rr = np.array([0.8] * 9 + [0.375] * 8 + [0.8] * 9)
    _, fast = R.held_rates(rr, 17)
    check("an 8-beat burst at 160 is not a 17-beat run", fast <= 100, f"{fast:.1f}")
    check("fewer intervals than the run -> no rate", np.isnan(R.held_rates(np.full(4, 2.0), 5)[0]))

    print("\n4b. notification thresholds (pure; MCOT practice)")
    got = ks(R.classify_sustained(30.0, nan))
    check("held 30.0 for 30 s -> sustained bradycardia", got == {("sustained_bradycardia", "critical")})
    check("held 30.1 for 30 s -> nothing", R.classify_sustained(30.1, nan) == [])
    got = ks(R.classify_sustained(nan, 180.0))
    check("held 180.0 for 30 s -> sustained tachycardia", got == {("sustained_tachycardia", "critical")})
    check("held 179.9 for 30 s -> nothing", R.classify_sustained(nan, 179.9) == [])
    want = {"sustained_bradycardia": "emergent", "sustained_tachycardia": "urgent",
            "pause": "urgent", "asystole": "emergent", "bradycardia": "",
            "extreme_bradycardia": "", "tachycardia": "", "extreme_tachycardia": "",
            "fast_narrow": ""}
    check("only the four MCOT events are sent; every tier stays report-only",
          all(R.Finding(k, "warning").notify == v for k, v in want.items()))
    rr = np.full(30, 60 / 28)                     # 28 bpm for about 64 s
    slow, _ = R.held_rates_for(rr, 30.0)
    check("28 bpm held for 30 s is found", abs(slow - 28) < 0.5, f"{slow:.1f}")
    rr = np.array([0.8] * 20 + [60 / 28] * 11 + [0.8] * 20)   # about 23 s at 28 bpm
    slow, _ = R.held_rates_for(rr, 30.0)
    check("23 s at 28 bpm is not 30 s held", slow > C.NOTIFY_BRADY_BPM, f"{slow:.1f}")
    check("a window shorter than the run -> no rate", np.isnan(R.held_rates_for(np.full(5, 1.0), 30.0)[0]))

    # ------------------------------------------------------------------
    print("\n5. on signal: runs")
    for bpm, kind, sev in [(35, "extreme_bradycardia", "critical"),
                           (45, "bradycardia", "warning"),
                           (55, "bradycardia", "info"),
                           (105, "tachycardia", "info"),
                           (115, "tachycardia", "warning"),
                           (160, "extreme_tachycardia", "critical")]:
        r = R.analyse(ecg(bpm=bpm))
        check(f"{bpm} bpm -> {kind} {sev}", (kind, sev) in ks(r.findings),
              f"{r.findings}, median {r.bpm:.1f}")
    r = R.analyse(ecg(bpm=72, p_mv=0.12))
    check("72 bpm with P and T waves -> nothing", r.findings == [], str(r.findings))
    r = R.analyse(ecg(bpm=28))
    check("28 bpm across the window -> sustained bradycardia, sent",
          [f.kind for f in r.findings if f.notify] == ["sustained_bradycardia"], str(r.findings))
    r = R.analyse(ecg(bpm=190))
    check("190 bpm across the window -> sustained tachycardia, sent",
          [f.kind for f in r.findings if f.notify] == ["sustained_tachycardia"], str(r.findings))
    r = R.analyse(ecg(bpm=35))
    check("35 bpm -> graded critical for the report, nothing sent",
          "extreme_bradycardia" in r.kinds() and not any(f.notify for f in r.findings))
    burst = regular(0.4, 0.8, 10.0) + regular(10.2, 0.375, 10.2 + 0.375 * 10)
    burst += regular(burst[-1] + 0.8, 0.8, WIN - 0.4)
    r = R.analyse(ecg(times=burst))
    check("a 10-beat burst at 160 is not 17 beats at 150: no extreme tachycardia",
          "extreme_tachycardia" not in r.kinds(), str(r.findings))

    print("\n6. on signal: gaps")
    base = regular(0.4, 0.8, 5.0)
    r = R.analyse(ecg(times=base + regular(base[-1] + 4.0, 0.8, WIN - 0.4)))
    check("a 4.0 s gap -> pause", ("pause", "warning") in ks(r.findings), str(r.findings))
    check("... and no brady from that single gap", not any(f.family == "brady" for f in r.findings))
    bias = r.longest_gap_sec - 4.0
    check("gap bias pinned: within 0.1 s either way", abs(bias) < 0.1, f"read {r.longest_gap_sec:.3f}")
    r = R.analyse(ecg(times=base + regular(base[-1] + 7.0, 0.8, WIN - 0.4)))
    check("a 7.0 s gap -> asystole", ("asystole", "critical") in ks(r.findings), str(r.findings))

    print("\n7. amplitude floor (ANSI/AAMI EC13, 0.15 mV)")
    r = R.analyse(quiet())
    check("quiet, noise-only trace -> no beats, asystole",
          r.n_beats == 0 and "asystole" in r.kinds(),
          f"beats {r.n_beats}, rejected {r.n_rejected}")
    r = R.analyse(ecg(bpm=60, r_mv=0.0, t_mv=0.0, p_mv=0.12))
    check("P waves only (ventricular standstill) -> asystole", "asystole" in r.kinds(),
          f"beats {r.n_beats}")
    r = R.analyse(ecg(bpm=70, r_mv=0.1, t_mv=0.0))
    check("0.1 mV QRS -> not counted", r.n_beats == 0, f"beats {r.n_beats}")
    r = R.analyse(ecg(bpm=70, r_mv=0.2, t_mv=0.0))
    check("0.2 mV QRS -> counted", abs(r.bpm - 70) <= 2, f"{r.bpm:.1f} bpm")
    r = R.analyse(np.zeros(int(WIN * FS)))
    check("digitally flat -> flagged, asystole, says to check the device",
          r.flat and "asystole" in r.kinds() and "device" in r.findings[0].detail)

    print("\n8. QRS width and the fast narrow label")
    w = [R.analyse(ecg(bpm=160, qrs_ms=q)).qrs_ms for q in (60, 120, 200)]
    check("width grows with true width", w[0] < w[1] < w[2], f"{w[0]:.0f} < {w[1]:.0f} < {w[2]:.0f}")
    check("no longer capped at 124 ms", w[2] > 124, f"{w[2]:.0f} ms")
    r = R.analyse(ecg(bpm=160, qrs_ms=70))
    check("SVT label is off (width failed on QT Database): 160 narrow -> extreme tachy only",
          r.kinds() == {"extreme_tachycardia"} and not C.FAST_NARROW_LABEL, str(r.findings))
    check("... width is still measured and reported", r.qrs_class == "narrow", r.qrs_class)
    r = R.analyse(ecg(bpm=160, qrs_ms=240))
    check("160 bpm wide -> extreme tachy only, labelled wide",
          r.kinds() == {"extreme_tachycardia"} and r.qrs_class == "wide",
          f"{r.findings}, {r.qrs_class} {r.qrs_ms:.0f} ms")
    check("wide_complex is never reported", "wide_complex" not in R.KINDS)
    x = ecg(bpm=75, qrs_ms=80)
    x[int(1.2 * FS)] += 0.4
    check("an isolated spike does not stretch a narrow complex",
          R.analyse(x).qrs_class == "narrow", f"{R.analyse(x).qrs_ms:.0f} ms")

    print("\n8b. detector: artifact, tall T waves, quiet gaps")
    from scipy import signal as sg
    rng = np.random.default_rng(11)
    times = regular(0.4, 60 / 72, WIN - 0.4)
    x = ecg(times=times, r_mv=1.9, t_mv=0.4, seed=11)
    sos = sg.butter(2, [1 / 125, 20 / 125], btype="band", output="sos")
    bursts = [(10.0, 14.0, 10.0), (25.0, 27.0, 8.0)]
    for a, b, amp in bursts:
        seg = sg.sosfiltfilt(sos, rng.normal(0, 1, int((b - a) * FS)))
        x[int(a * FS):int(b * FS)] += amp * seg / np.ptp(seg)
    outside = [int(t * FS) for t in times
               if not any(a - 0.3 <= t <= b + 0.3 for a, b, _ in bursts)]
    peaks, _ = R.detect_beats(x, FS)
    got = sum(np.min(np.abs(peaks - c)) <= 0.15 * FS for c in outside)
    check("beats around 8-10 mV motion artifact are recovered (Pan-Tompkins++ rules)",
          got / len(outside) >= 0.85, f"{got}/{len(outside)}")
    r = R.analyse(x)
    check("... and nothing is sent", not any(f.notify for f in r.findings), str(r.findings))
    for bpm in (170, 200):
        rng2 = np.random.default_rng(bpm)
        t, times = 0.4, []
        while t < WIN - 0.4:
            times.append(t)
            t += 60 / bpm * rng2.uniform(0.75, 1.25)          # AF-like irregular
        n2 = int(WIN * FS)
        k2 = np.arange(n2)
        x = np.zeros(n2)
        for tt in times:
            c = int(tt * FS)
            a = rng2.uniform(0.7, 1.3)
            x += a * np.exp(-0.5 * ((k2 - c) / 3.3) ** 2) \
                + 0.3 * np.exp(-0.5 * ((k2 - c - 0.16 * FS) / (0.03 * FS)) ** 2)
        x = bandpass(x + rng2.normal(0, 0.03, n2), C.MODEL_BAND)
        peaks, _ = R.detect_beats(x, FS)
        check(f"fast irregular rhythm at {bpm} bpm: every beat counted, not every other one",
              len(peaks) >= 0.95 * len(times), f"{len(peaks)}/{len(times)}")
    for bpm in (60, 100, 140):
        r = R.analyse(ecg(bpm=bpm, r_mv=1.0, t_mv=1.2))
        check(f"tall T waves at {bpm} bpm are not counted twice", abs(r.bpm - bpm) <= 3,
              f"{r.bpm:.1f} bpm")
    r = R.analyse(ecg(bpm=80, r_mv=1.0, t_mv=1.2))
    check("tall T waves (1.2 mV, the EC13 test) are not counted as beats",
          abs(r.bpm - 80) <= 2, f"{r.bpm:.1f} bpm")
    base = regular(0.4, 0.8, 8.0)
    after = regular(base[-1] + 7.0, 0.8, WIN - 0.4)
    x = ecg(times=base + after)
    a0 = int((base[-1] + 1.0) * FS)
    k = np.arange(int(4.0 * FS))
    x[a0:a0 + k.size] += 0.35 * np.sin(2 * np.pi * 0.5 * k / FS)   # slow wander, 0.7 mV p-p
    r = R.analyse(x)
    check("a 7 s gap with 0.7 mV of slow wander inside is unreadable, not asystole",
          "asystole" not in r.kinds() and r.unreadable_gap_sec > 6,
          f"{r.findings}, unreadable {r.unreadable_gap_sec:.1f} s")
    for p_mv in (0.12, 0.20, 0.25, 0.30):
        x = ecg(times=regular(0.4, 0.8, 8.0))
        p_only = ecg(times=regular(8.4, 0.8, WIN - 0.4), r_mv=0.0, t_mv=0.0, p_mv=p_mv, seed=5)
        x[int(8.3 * FS):] = p_only[int(8.3 * FS):]
        r = R.analyse(x)
        check(f"ventricular standstill, P waves {p_mv:.2f} mV only -> asystole (P waves never counted)",
              "asystole" in r.kinds(), f"{r.n_beats} beats, {r.findings}")
    r = R.analyse(ecg(times=base + after))
    check("the same 7 s gap, quiet, is asystole", "asystole" in r.kinds(), str(r.findings))
    check("absence needs a quiet gap: the line is the EC13 lower QRS detection bound",
          C.ABSENCE_MAX_PTP_MV == 0.5)

    print("\n9. sliding reports")
    sig = np.concatenate([ecg(bpm=70, seconds=60), ecg(bpm=28, seconds=60, seed=3)])
    reps = list(R.sliding_reports(sig))
    check("one report every stride", len(reps) == int((120 - WIN) / C.RULES_STRIDE_SEC) + 1,
          f"{len(reps)} reports")
    first = next((t for t, r in reps if "extreme_bradycardia" in r.kinds()), None)
    check("a slow rate is graded within ~12 s of onset", first is not None and first - 60.0 <= 12.0,
          f"first at {first} s, onset 60 s")
    first = next((t for t, r in reps if "sustained_bradycardia" in r.kinds()), None)
    check("it is sent once held 30 s, within one stride and a beat of the definition",
          first is not None and 30.0 <= first - 60.0 <= 30.0 + C.RULES_STRIDE_SEC + 60 / 28 + 1,
          f"first at {first} s, onset 60 s")

    print("\n10. config consistency")
    check("brady tiers ordered", C.HR_BRADY["critical"] < C.HR_BRADY["warning"] < C.HR_BRADY["info"])
    check("tachy tiers ordered", C.HR_TACHY["info"] < C.HR_TACHY["warning"] < C.HR_TACHY["critical"])
    check("fast narrow on the critical tachy line", C.FAST_NARROW_BPM == C.HR_TACHY["critical"])
    check("pause shorter than asystole", C.PAUSE_WARNING_SEC < C.ASYSTOLE_CRITICAL_SEC)
    check("a gap longer than the asystole line always fits whole in one window",
          C.RULES_WINDOW_SEC - C.RULES_STRIDE_SEC > C.ASYSTOLE_CRITICAL_SEC)
    check("the window holds a brady run down to 15 bpm",
          C.RULES_WINDOW_SEC >= C.RATE_RUN_BEATS["brady"] * 60.0 / 15)
    check("a 30 s sustained run always fits whole in one window",
          C.RULES_WINDOW_SEC - C.RULES_STRIDE_SEC > C.NOTIFY_SUSTAIN_SEC)
    check("notification lines sit beyond the report tiers",
          C.NOTIFY_BRADY_BPM < C.HR_BRADY["critical"] and C.NOTIFY_TACHY_BPM > C.HR_TACHY["critical"])
    fams = {fam for fam, _ in R.KINDS.values()}
    check("every finding family has a decision-layer rule", fams == set(RATE_FAMILIES),
          f"{sorted(fams)} vs {sorted(RATE_FAMILIES)}")
    check("every family's rule exists in CONFIRMATION",
          all(k in C.CONFIRMATION for k in RATE_FAMILIES.values()))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
