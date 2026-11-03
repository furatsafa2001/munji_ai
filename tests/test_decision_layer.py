"""The decision layer's behaviour, one rule per check.

These are the cases the Dart port in the app must reproduce exactly.

    python tests/test_decision_layer.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.decision.decision_layer import DecisionConfig, DecisionLayer, beats_from_mask
from munji_ai.rules.rate_engine import Finding, RateReport

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def organised(bpm=180.0):
    rr = np.full(14, 60.0 / bpm)
    return RateReport(bpm=bpm, rr_sec=rr, qrs_ms=150.0, n_beats=15)


def chaotic():
    rr = np.random.default_rng(0).uniform(0.15, 0.6, 12)
    return RateReport(bpm=float(60 / np.median(rr)), rr_sec=rr, qrs_ms=np.nan, n_beats=13)


def kinds(alerts):
    return [a.kind for a in alerts]


def run_shock(dl, ps, rate=None, lead_off=False, t0=0.0):
    out = []
    for i, p in enumerate(ps):
        out += dl.on_shockable(t0 + i * 2.5, p, rate, lead_off=lead_off)
    return out


def main() -> int:
    print("Shockable: confirmation and naming\n")
    dl = DecisionLayer()
    a = run_shock(dl, [0.9, 0.9, 0.9], chaotic())
    check("3 positive windows are not enough", a == [])
    a = run_shock(dl, [0.9], chaotic(), t0=7.5)
    check("the 4th of 6 fires one VF/VT alert", kinds(a) == ["VF/VT"])
    check("it is critical", a and a[0].severity == "critical")
    a = run_shock(dl, [0.9] * 10, chaotic(), t0=10)
    check("no repeat while the episode lasts", a == [])
    a = run_shock(dl, [0.1] * 12 + [0.9] * 4, chaotic(), t0=40)
    check("a new episode after 30 s of quiet fires again", kinds(a) == ["VF/VT"])
    dl = DecisionLayer()
    check("organised fast beats are still VF/VT, never guessed apart",
          kinds(run_shock(dl, [0.9] * 4, organised())) == ["VF/VT"])
    dl = DecisionLayer()
    check("0.59 never fires", run_shock(dl, [0.59] * 20, chaotic()) == [])

    print("\nLead-off\n")
    dl = DecisionLayer()
    a = run_shock(dl, [0.9] * 8, chaotic(), lead_off=True)
    check("a detached patch silences VF and asks for the patch once",
          kinds(a) == ["check_patch"])
    dl2 = DecisionLayer()
    out = []
    for i in range(4):
        out += dl2.on_shockable(i * 2.5, 0.9, chaotic(), usable=False)
    check("4 positive windows marked unusable still raise VF/VT", kinds(out) == ["VF/VT"])

    print("\nAF\n")
    dl = DecisionLayer()
    out = []
    for i in range(10):
        out += dl.on_rhythm(i * 15, (0.1, 0.8, 0.1))
    check("4 AF windows of 6 raise one AF/AFL alert", kinds(out) == ["AF/AFL"])
    out = []
    for i in range(60):
        out += dl.on_rhythm(150 + i * 15, (0.9, 0.05, 0.05))
    for i in range(6):
        out += dl.on_rhythm(1200 + i * 15, (0.1, 0.8, 0.1))
    check("a new AF episode within 24 h is not re-alerted", out == [])
    out = []
    for i in range(60):
        out += dl.on_rhythm(3000 + i * 15, (0.9, 0.05, 0.05))
    for i in range(6):
        out += dl.on_rhythm(90000 + i * 15, (0.1, 0.8, 0.1))
    check("after 24 h it may alert again", kinds(out) == ["AF/AFL"])
    r = dl.report()
    check("every AF window counts in the burden", r["af_windows"] == 22
          and abs(r["af_burden"] - 22 / r["rhythm_windows"]) < 1e-9,
          f"{r['af_windows']} AF of {r['rhythm_windows']}")

    dl = DecisionLayer()
    out = []
    for i in range(10):
        out += dl.on_rhythm(i * 15, (0.1, 0.1, 0.8))
    check("OTHER is never alerted, only counted", out == [] and dl.report()["other_windows"] == 10)
    dl = DecisionLayer()
    out = []
    for i in range(10):
        out += dl.on_rhythm(i * 15, (0.1, 0.8, 0.1), usable=False)
    check("unusable windows neither alert nor count", out == []
          and dl.report()["rhythm_windows"] == 0)

    print("\nPriority\n")
    dl = DecisionLayer()
    run_shock(dl, [0.9] * 4, chaotic())
    out = []
    for i in range(6):
        out += dl.on_rhythm(10 + i * 15, (0.1, 0.8, 0.1))
    check("no AF alert while a VF episode is active", out == [])

    print("\nRate rules: what is sent and what is only reported\n")

    def rate(*findings, bpm=float("nan")):
        return RateReport(bpm=bpm, findings=list(findings))

    def run_rate(dl, reports, t0=0.0, **kw):
        out = []
        for i, r in enumerate(reports):
            out += dl.on_rate(t0 + i * 2.5, r, **kw)
        return out

    def sev(alerts):
        return [(a.kind, a.severity) for a in alerts]

    extreme = rate(Finding("extreme_bradycardia", "critical", 35))
    dl = DecisionLayer()
    out = run_rate(dl, [extreme] * 20)
    check("a graded finding (35 bpm, critical tier) is never sent", out == [])
    check("... it is counted for the report",
          dl.report()["rate_report_windows"].get("brady", {}).get("critical") == 20)
    dl = DecisionLayer()
    out = run_rate(dl, [rate(Finding("tachycardia", "warning", 115))] * 20)
    check("tachycardia above 110 (warning tier) is never sent", out == [])

    s_brady = rate(Finding("extreme_bradycardia", "critical", 35),
                   Finding("sustained_bradycardia", "critical", 35))
    p_brady = rate(Finding("extreme_bradycardia", "critical", 28),
                   Finding("profound_bradycardia", "critical", 28))
    dl = DecisionLayer()
    out = run_rate(dl, [s_brady] * 6)
    check("below 40 held 30 s is sent once, as warning (urgent)",
          sev(out) == [("sustained_bradycardia", "warning")])
    dl = DecisionLayer()
    out = run_rate(dl, [p_brady] * 6)
    check("30 or less held 30 s is sent once, as critical (emergent)",
          sev(out) == [("profound_bradycardia", "critical")])
    dl = DecisionLayer()
    out = run_rate(dl, [s_brady, s_brady, p_brady, p_brady, s_brady])
    check("35 then 28: one episode, urgent then one escalation to emergent",
          sev(out) == [("sustained_bradycardia", "warning"), ("profound_bradycardia", "critical")])
    s_tachy = rate(Finding("sustained_tachycardia", "critical", 190))
    dl = DecisionLayer()
    out = run_rate(dl, [s_tachy] * 6)
    check("sustained tachycardia (180 or more for 30 s) is sent once, as warning (urgent)",
          sev(out) == [("sustained_tachycardia", "warning")])

    pause = rate(Finding("pause", "warning", 4.0))
    asy = rate(Finding("asystole", "critical", 7.0))
    dl = DecisionLayer()
    out = run_rate(dl, [pause, asy, pause], lead_off=False)
    check("pause then asystole is one episode: urgent, then one escalation to emergent",
          sev(out) == [("pause", "warning"), ("asystole", "critical")])
    dl = DecisionLayer()
    out = run_rate(dl, [asy, pause, asy], lead_off=False)
    check("starting emergent, a milder window does not re-alert", sev(out) == [("asystole", "critical")])
    dl = DecisionLayer()
    out = run_rate(dl, [s_brady] * 3 + [rate()] * 6 + [s_brady])
    check("a new episode after 6 quiet windows alerts again",
          kinds(out) == ["sustained_bradycardia", "sustained_bradycardia"])

    print("\nRate rules: who judges a finding\n")
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=False, lead_off=False)
    check("asystole in gate-rejected windows, leads ON: sent (the gate is not the judge)",
          kinds(out) == ["asystole"])
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=False, lead_off=None)
    check("same, lead status unknown: falls back to the gate, not sent", out == [])
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=True, lead_off=None)
    check("lead status unknown, gate accepts: sent", kinds(out) == ["asystole"])
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=True, lead_off=True)
    check("leads OFF: nothing cardiac", out == [])
    check("... counted as lead-off windows", dl.report()["lead_off_windows"] == 3)
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=True, lead_off=False, complete=False)
    check("missing samples: nothing cardiac", out == [])
    check("... counted as no-data windows", dl.report()["no_data_windows"] == 3)
    dl = DecisionLayer()
    out = run_rate(dl, [s_tachy] * 3, usable=False, lead_off=False)
    check("a rate finding in gate-rejected windows is not sent, leads on or not", out == [])

    print("\nPriority over rate findings\n")
    dl = DecisionLayer()
    out = []
    for i in range(8):
        out += dl.on_shockable(i * 2.5, 0.9, organised())
        out += dl.on_rate(i * 2.5, s_tachy if i >= 4 else rate())
    check("an urgent rate alert during an active VF/VT episode is held back",
          kinds(out) == ["VF/VT"])
    dl = DecisionLayer()
    out = []
    for i in range(8):
        out += dl.on_shockable(i * 2.5, 0.9, organised())
        out += dl.on_rate(i * 2.5, asy if i >= 4 else rate(), lead_off=False)
    check("an emergent rate alert during VF/VT still goes out",
          kinds(out) == ["VF/VT", "asystole"])

    print("\nBeats: reading and de-duplication\n")

    def stream(pattern, t0=0.0):
        """pattern: [(cls, bpm, count)] -> [(t, cls)] in time order."""
        out, t = [], t0
        for cls, bpm, n in pattern:
            for _ in range(n):
                out.append((t, cls))
                t += 60.0 / bpm
        return out

    def feed(dl, beats, win=20.0, stride=10.0, usable=lambda te: True, shift=0.004,
             until=None):
        """Overlapping beat windows, the same beat a few ms apart in each.

        until: stop at this time, so later calls (e.g. a shock alert) come in
        time order, as they do in the app.
        """
        alerts, end = [], beats[-1][0] + stride
        te, k = beats[0][0] + stride, 0
        while te <= (end + win if until is None else until):
            inside = [(bt + (shift if k % 2 else 0.0), c) for bt, c in beats if te - win < bt <= te]
            alerts += dl.on_beats(te, inside, usable=usable(te))
            te += stride
            k += 1
        return alerts

    mask = np.zeros(500, int)
    mask[100:120] = C.BEAT_CLASSES.index("N")
    mask[300:330] = C.BEAT_CLASSES.index("PVC")
    got = beats_from_mask(mask, t_start=10.0)
    check("mask -> beats: centre time and class, background skipped",
          got == [(10.0 + 110 / 250, "N"), (10.0 + 315 / 250, "PVC")], str(got))

    b = stream([("N", 75, 100)])
    dl = DecisionLayer()
    feed(dl, b)
    check("overlapping windows: every beat counted once", dl.beat_counts["N"] == 100,
          str(dl.beat_counts))

    print("\nNSVT (report only)\n")
    dl = DecisionLayer()
    out = feed(dl, stream([("N", 75, 30), ("PVC", 150, 4), ("N", 75, 30)]))
    check("4 PVCs at 150 bpm: one NSVT in the report, no alert",
          dl.report()["nsvt"] == 1 and out == [], str(dl.nsvt))
    dl = DecisionLayer()
    feed(dl, stream([("N", 75, 30), ("PVC", 150, 2), ("N", 75, 30)]))
    check("a couplet (2 PVCs) is not NSVT", dl.report()["nsvt"] == 0)
    dl = DecisionLayer()
    feed(dl, stream([("N", 75, 30), ("PVC", 90, 5), ("N", 75, 30)]))
    check("5 PVCs at 90 bpm is not NSVT (rate must be above 100)", dl.report()["nsvt"] == 0)

    print("\nSustained VT -> WCT alert\n")
    dl = DecisionLayer()
    out = feed(dl, stream([("N", 75, 30), ("PVC", 150, 100), ("N", 75, 30)]))
    sev = [(a.kind, a.severity) for a in out]
    check("40 s of PVCs at 150 bpm: one WCT alert, emergent (critical)",
          sev == [("WCT", "critical")], str(sev))
    check("... recorded as a run of 30 s or more, not NSVT",
          len(dl.long_runs) == 1 and dl.report()["nsvt"] == 0)
    run = []
    for i in range(100):
        run.append(("N" if i % 7 == 6 else "PVC", 150, 1))
    dl = DecisionLayer()
    out = feed(dl, stream([("N", 75, 30)] + run + [("N", 75, 30)]))
    check("1 beat in 7 missed by the model (86% PVC): still one WCT alert",
          kinds(out) == ["WCT"], str(kinds(out)))
    check("... while the strict consecutive rule only saw short pieces",
          len(dl.long_runs) == 0 and dl.report()["nsvt"] > 1, f"nsvt {dl.report()['nsvt']}")
    run = [("N" if i % 3 == 2 else "PVC", 150, 1) for i in range(100)]
    dl = DecisionLayer()
    out = feed(dl, stream([("N", 75, 30)] + run + [("N", 75, 30)]))
    check("67% PVC: no WCT alert", "WCT" not in kinds(out))
    dl = DecisionLayer()
    out = feed(dl, stream([("N", 75, 30), ("PVC", 90, 60), ("N", 75, 30)]))
    check("PVCs held 40 s at 90 bpm: no WCT alert (rate must be above 100)", "WCT" not in kinds(out))
    b = stream([("N", 75, 30), ("PVC", 150, 100), ("N", 75, 30)])
    dl = DecisionLayer()
    out = feed(dl, b, usable=lambda te: not (40 <= te <= 50))
    check("a rejected window inside the run: never 30 s of usable beats, no WCT alert",
          "WCT" not in kinds(out), str(kinds(out)))

    dl = DecisionLayer()
    for i in range(6):
        dl.on_shockable(i * 2.5, 0.9, organised())
    out = feed(dl, stream([("PVC", 150, 100)], t0=20.0))
    check("ShockableNet VF/VT episode open: no second (WCT) alert for the same event",
          "WCT" not in kinds(out), str(kinds(out)))

    print("\nPVC burden, PACs, bigeminy, trigeminy (report only)\n")
    pattern = [("PVC" if i % 8 == 7 else "N", 75, 1) for i in range(75 * 70)]
    dl = DecisionLayer()
    out = feed(dl, stream(pattern))
    pk = dl.report()["pvc_burden"]["peak_hour"]
    check("70 min with 1 PVC in 8 (12.5%): peak hourly burden in the warning tier, no alert",
          pk is not None and pk["tier"] == "warning" and out == [], str(pk))
    check("... daily burden reported with its tier",
          list(dl.report()["pvc_burden"]["daily"].values())[0]["tier"] == "warning")
    dl = DecisionLayer()
    feed(dl, stream([("PVC" if i % 8 == 7 else "N", 75, 1) for i in range(75 * 10)]))
    check("10 minutes only: no hourly burden yet", dl.report()["pvc_burden"]["peak_hour"] is None)

    def pat(units, unit):
        return stream([("N", 75, 20)] + [(c, 75, 1) for _ in range(units) for c in unit]
                      + [("N", 75, 20)])
    dl = DecisionLayer()
    feed(dl, pat(3, ["N", "PVC"]))
    check("N PVC x3: one bigeminy episode", dl.report()["bigeminy_episodes"] == 1)
    dl = DecisionLayer()
    feed(dl, pat(2, ["N", "PVC"]))
    check("N PVC x2: not yet bigeminy", dl.report()["bigeminy_episodes"] == 0)
    dl = DecisionLayer()
    feed(dl, pat(3, ["N", "N", "PVC"]))
    check("N N PVC x3: one trigeminy episode, no bigeminy",
          dl.report()["trigeminy_episodes"] == 1 and dl.report()["bigeminy_episodes"] == 0)
    dl = DecisionLayer()
    feed(dl, stream([("N", 75, 20), ("PVC", 75, 1), ("N", 75, 1), ("PVC", 75, 1),
                     ("PAC", 75, 1), ("N", 75, 1), ("PVC", 75, 1), ("N", 75, 20)]))
    check("a PAC breaks the alternation", dl.report()["bigeminy_episodes"] == 0)
    dl = DecisionLayer()
    feed(dl, stream([("N", 75, 20), ("PAC", 75, 3), ("N", 75, 20), ("PAC", 75, 2), ("N", 75, 20)]))
    check("PACs are counted for the report", dl.report()["pac_count"] == 5)

    print("\nECG saving around emergency alerts\n")
    dl = DecisionLayer()
    for i in range(6):
        dl.on_shockable(100 + i * 2.5, 0.9, organised())
    req = dl.take_recording_requests()
    check("VF/VT alert asks for 30 s before to 30 s after",
          len(req) == 1 and req[0].reason == "VF/VT"
          and req[0].start == req[0].t - 30 and req[0].end == req[0].t + 30, str(req))
    check("... handed over once", dl.take_recording_requests() == [])
    dl = DecisionLayer()
    dl.on_rate(50.0, rate(Finding("asystole", "critical", 7.0)), lead_off=False)
    check("asystole alert asks for the ECG", [r.reason for r in dl.take_recording_requests()] == ["asystole"])
    dl = DecisionLayer()
    dl.on_rate(50.0, rate(Finding("pause", "warning", 4.0)), lead_off=False)
    check("a pause (urgent, not emergency) does not", dl.take_recording_requests() == [])
    dl = DecisionLayer()
    feed(dl, stream([("N", 75, 30), ("PVC", 150, 100)]))
    check("WCT alert asks for the ECG", [r.reason for r in dl.take_recording_requests()] == ["WCT"])

    print("\nNaming VF or VT (off by default: shadow mode)\n")
    check("naming is off in config", C.VFVT_NAMING_ENABLED is False)

    def shock(dl, t0):
        out = []
        for i in range(6):
            out += dl.on_shockable(t0 + i * 2.5, 0.9, organised())
        return out

    vt = stream([("N", 75, 10), ("PVC", 180, 60)])       # PVCs from 8 s to 27.7 s
    dl = DecisionLayer()
    feed(dl, vt, until=30.0)
    out = shock(dl, 22.5)                                # alert at 30 s

    check("PVCs in a row + ShockableNet: alert still 'VF/VT', guess 'VT' saved in shadow",
          kinds(out) == ["VF/VT"] and dl.take_recording_requests()[0].shadow_name == "VT",
          str([(a.kind, a.evidence.get("shadow_name")) for a in out]))
    dl = DecisionLayer()
    for te in (10.0, 20.0, 30.0):
        dl.on_beats(te, [], usable=False)
    out = shock(dl, 17.5)
    check("beat model ran and saw no beats: shadow guess 'VF'",
          out and out[0].evidence["shadow_name"] == "VF", str(out[0].evidence if out else None))
    dl = DecisionLayer()
    out = shock(dl, 100.0)
    check("no beat-model output at all: shadow guess 'VF/VT'",
          out and out[0].evidence["shadow_name"] == "VF/VT")
    dl = DecisionLayer(DecisionConfig(vfvt_naming=True))
    feed(dl, vt, until=30.0)
    out = shock(dl, 22.5)
    check("with naming on, the same case is alerted as 'VT'", kinds(out) == ["VT"], str(kinds(out)))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
