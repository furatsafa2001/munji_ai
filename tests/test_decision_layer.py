"""The decision layer's behaviour, one rule per check.

These are the cases the Dart port in the app must reproduce exactly.

    python tests/test_decision_layer.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.decision.decision_layer import DecisionLayer
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

    print("\nRate rules: confirmation per family\n")

    def rate(*findings, bpm=float("nan")):
        return RateReport(bpm=bpm, findings=list(findings))

    def run_rate(dl, reports, t0=0.0, **kw):
        out = []
        for i, r in enumerate(reports):
            out += dl.on_rate(t0 + i * 2.5, r, **kw)
        return out

    slow = rate(Finding("extreme_bradycardia", "critical", 35))
    dl = DecisionLayer()
    out = run_rate(dl, [slow] * 6)
    check("a rate finding alerts at once (persistence is in the finding) and once",
          kinds(out) == ["extreme_bradycardia"])

    b38 = rate(Finding("extreme_bradycardia", "critical", 38))
    b41 = rate(Finding("bradycardia", "warning", 41))
    dl = DecisionLayer()
    out = run_rate(dl, [b41, b38, b41, b38, b41, b38])
    check("38/41 wobble is one episode: warning, then one escalation to critical",
          [(a.kind, a.severity) for a in out]
          == [("bradycardia", "warning"), ("extreme_bradycardia", "critical")])

    dl = DecisionLayer()
    out = run_rate(dl, [b38, b41, b38, b41])
    check("starting critical, a milder window does not re-alert",
          kinds(out) == ["extreme_bradycardia"])

    info = rate(Finding("bradycardia", "info", 55))
    dl = DecisionLayer()
    out = run_rate(dl, [info] * 10)
    check("info level is never alerted", out == [])
    check("... but is counted for the report",
          dl.report()["rate_info_windows"].get("brady") == 10)

    dl = DecisionLayer()
    out = run_rate(dl, [slow] * 3 + [rate()] * 6 + [slow])
    check("a new episode after 6 quiet windows alerts again",
          kinds(out) == ["extreme_bradycardia", "extreme_bradycardia"])

    print("\nRate rules: who judges a finding\n")
    asy = rate(Finding("asystole", "critical", 7.0))
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=False, lead_off=False)
    check("asystole in gate-rejected windows, leads ON: alerted (the gate is not the judge)",
          kinds(out) == ["asystole"])
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=False, lead_off=None)
    check("same, lead status unknown: falls back to the gate, not alerted", out == [])
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=True, lead_off=None)
    check("lead status unknown, gate accepts: alerted", kinds(out) == ["asystole"])
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=True, lead_off=True)
    check("leads OFF: nothing cardiac", out == [])
    check("... counted as lead-off windows", dl.report()["lead_off_windows"] == 3)
    dl = DecisionLayer()
    out = run_rate(dl, [asy] * 3, usable=True, lead_off=False, complete=False)
    check("missing samples: nothing cardiac", out == [])
    check("... counted as no-data windows", dl.report()["no_data_windows"] == 3)

    tachy = rate(Finding("extreme_tachycardia", "critical", 160))
    dl = DecisionLayer()
    out = run_rate(dl, [tachy] * 3, usable=False, lead_off=False)
    check("a rate finding in gate-rejected windows is not alerted, leads on or not", out == [])

    pause = rate(Finding("pause", "warning", 4.0))
    dl = DecisionLayer()
    out = run_rate(dl, [pause, asy, pause], lead_off=False)
    check("pause then asystole is one absence episode, escalated once",
          [(a.kind, a.severity) for a in out]
          == [("pause", "warning"), ("asystole", "critical")])

    print("\nVentricular hold on rate findings\n")
    narrow = rate(Finding("extreme_tachycardia", "critical", 180),
                  Finding("fast_narrow", "warning", 180))
    dl = DecisionLayer()
    out = []
    for i in range(6):
        out += dl.on_shockable(i * 2.5, 0.9 if i == 0 else 0.2, organised())
        out += dl.on_rate(i * 2.5, narrow)
    check("no SVT label while ShockableNet saw danger in the last 6 windows",
          "fast_narrow" not in kinds(out))
    check("... the critical fast rate itself still alerts",
          "extreme_tachycardia" in kinds(out))
    out = []
    for i in range(6, 20):
        out += dl.on_shockable(i * 2.5, 0.1, organised())
        out += dl.on_rate(i * 2.5, narrow if i >= 14 else rate())
    check("SVT is labelled once ShockableNet has been quiet for 6 windows",
          "fast_narrow" in kinds(out))
    dl = DecisionLayer()
    out = []
    for i in range(8):
        out += dl.on_shockable(i * 2.5, 0.9, organised())
        out += dl.on_rate(i * 2.5, rate(Finding("bradycardia", "warning", 45))
                          if i >= 4 else rate())
    check("a warning-level rate finding during an active VF/VT episode is held back",
          kinds(out) == ["VF/VT"])

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
