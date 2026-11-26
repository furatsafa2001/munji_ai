"""rules/fast_narrow.FastNarrowTracker against experiments/fast_narrow_template.

The tracker is the streaming form of the two rows chosen on 25 November 2026
(FINAL_ALERT, FINAL_REPORT). It must give the same alerts and the same report
entries, at the same times, as the experiment that produced every result:

    template given   the experiment's patch-on template handed to the tracker
                     (as the phone hands back a stored one): identical events
    template learnt  the tracker learns its own: the same template, and the
                     same events (the experiment applies its template to the
                     beats before it as well; the tracker cannot, so records
                     here have a calm start)
    chunks           the same events whatever the size of the pushed chunks
    timing           an alert is out one stride after its window

On synthetic records (no PhysioNet), and on real ones when they are on disk.
Long real records are opt-in:

    python tests/test_fast_narrow_tracker.py
    python tests/test_fast_narrow_tracker.py --records shdbaf:127 ltafdb:42 ltafdb:204
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from munji_ai import config as C  # noqa: E402
from munji_ai.rules import beat_match as bm  # noqa: E402
from munji_ai.rules.fast_narrow import FastNarrowTracker  # noqa: E402

spec = importlib.util.spec_from_file_location("svt_test", ROOT / "tests" / "test_fast_narrow_template.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
G, T = F.G, F.G.T
FS = F.FS
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def times(a) -> list:
    return [round(float(v), 4) for v in a]


def offline(rec, step, span):
    """The experiment: (alerts, report entries, the analysed signal, its template)."""
    r = G.process(rec, span, step)
    b = T.beats_and_template(rec, step, annotated_span=span)
    tpl = None if b["first"] is None else bm.Template(np.asarray(b["first"][2], float), b["fs"],
                                                      b["first"][0], b["first"][1])
    return times(r["alerts"][G.FINAL_ALERT]), times(r["alerts"][G.FINAL_REPORT]), b["x"], tpl


def online(x, step, template=None, chunk=FS, record_when=False):
    """The tracker: (alerts, report entries, template events, alert emit sample counts)."""
    tr = FastNarrowTracker(FS, step, template)
    ev, when, n = [], [], 0
    for i in range(0, len(x), chunk):
        part = x[i:i + chunk]
        n += part.size
        got = tr.push(part)
        ev += got
        when += [(e.t, n) for e in got if e.kind == "alert"]
    ev += tr.finish()
    alerts = times(e.t for e in ev if e.kind == "alert")
    reports = times(sorted(e.t for e in ev if e.kind == "report"))
    learnt = [e for e in ev if e.kind == "template"]
    return alerts, reports, learnt, when, tr


def compare(label, rec, step=None, span=False, learn=True, chunks=(FS,)):
    step = step or G.POS_STEP
    a0, r0, x, tpl = offline(rec, step, span)
    a1, r1, _, when, _ = online(x, step, tpl, chunk=chunks[0])
    check(f"{label}: template given, the same alerts and report entries", (a1, r1) == (a0, r0),
          f"experiment {a0} / {r0}, tracker {a1} / {r1}")
    for c in chunks[1:]:
        a2, r2, _, _, _ = online(x, step, tpl, chunk=c)
        check(f"{label}: the same with chunks of {c} samples", (a2, r2) == (a1, r1))
    if when:
        late = max(n - round(t * FS) for t, n in when)
        check(f"{label}: each alert out within one stride and one chunk of its window",
              late <= int(round(step * FS)) + chunks[0], f"{late} samples")
    if learn:
        a3, r3, learnt, _, tr = online(x, step, None, chunk=chunks[0])
        same_tpl = (tpl is None and not learnt) or (tpl is not None and len(learnt) == 1
                                                    and np.array_equal(tr.template.waveform, tpl.waveform)
                                                    and (tr.template.start, tr.template.end) == (tpl.start, tpl.end))
        check(f"{label}: template learnt, the same template and the same events", same_tpl and (a3, r3) == (a0, r0),
              f"template {None if tr.template is None else (tr.template.start, tr.template.end)}, events {a3} / {r3}")
    return a0, r0


def synthetic() -> None:
    rec = F.record
    print("synthetic records")
    # SVT: abrupt start and end at 180: the alert and the report
    x, _, _ = F.ecg(F.SVT, 3)
    a, r = compare("SVT at 180", rec(x, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu90"), chunks=(FS, 37, 10 * FS))
    check("SVT at 180: one alert and one report entry", len(a) == 1 and len(r) == 1, f"{a} {r}")
    x, _, _ = F.ecg(F.RAMP, 1)
    a, r = compare("sinus speeding up and slowing down", rec(x, [0], ["(N"], "cu91"))
    check("sinus speeding up and slowing down: nothing", not a and not r, f"{a} {r}")
    x, _, _ = F.ecg(F.AF_IRREG, 5)
    compare("irregular AF", rec(x, [0, 151, 241], ["(N", "(AFIB", "(N"], "cu92"))
    x, _, _ = F.ecg(F.VT_NEW, 4)
    a, r = compare("VT with a new shape", rec(x, [0, 151, 241], ["(N", "(VT", "(N"], "cu93"))
    check("VT with a new shape: nothing", not a and not r, f"{a} {r}")
    x, _, _ = F.ecg(F.SLOW, 2)
    compare("SVT at 130", rec(x, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu94"))
    x, _, _ = F.ecg(F.HOVER, 8)
    compare("SVT hovering around 150", rec(x, [0, 151], ["(N", "(SVTA"], "cu95"))
    x, _, _ = F.ecg(F.EXER, 9)
    compare("exercise leap past 100", rec(x, [0], ["(N"], "cu96"))
    x, _, _ = F.ecg(F.IRREG, 12)
    compare("irregular at 170", rec(x, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu97"))
    x, _, _ = F.ecg(F.WITH_RUN, 4)
    compare("fast rhythm with 4 wide beats", rec(x, [0, 151, 156, 157.5, 243], ["(N", "(SVTA", "(VT", "(SVTA", "(N"],
                                                     "cu98"))
    # gradual start, abrupt end: a report entry that has to wait for the end
    x, _, _ = F.ecg([F.REST, (75, 170, 90, 12, 0), (170, 170, 90, 12, 0), (75, 75, 60, 12, 0)], 6)
    a, r = compare("gradual start, abrupt end", rec(x, [0], ["(N"], "cu99"))
    check("gradual start, abrupt end: a report entry (waited for the end), no alert", not a and len(r) == 1, f"{a} {r}")
    # atrial tachycardia with variable conduction: the report through the RR multiples
    rng = np.random.default_rng(21)
    rest = list(np.arange(1.0, 150.0, 0.8))
    k = np.tile([1, 2], 130)
    flip = rng.random(k.size) < 0.1
    k[flip] = 3 - k[flip]
    at = list(rest[-1] + np.cumsum(k * 0.25 + rng.normal(0, 0.005, k.size)))
    bt = np.array(rest + at + list(at[-1] + np.arange(0.8, 60.0, 0.8)))
    a, r = compare("atrial tachycardia, variable conduction",
                   rec(F.ecg_beats(bt, bt[-1] + 1.0, seed=21), [0, rest[-1], at[-1]], ["(N", "(SVTA", "(N"], "cu89"))
    check("atrial tachycardia, variable conduction: a report entry, no alert", not a and len(r) >= 1, f"{a} {r}")
    # two SVTs 20 s apart: one episode; 3 minutes apart: two
    two = [F.REST, (180, 180, 60, 12, 0), (75, 75, 20, 12, 0), (180, 180, 60, 12, 0), (75, 75, 180, 12, 0),
           (180, 180, 60, 12, 0), (75, 75, 60, 12, 0)]
    x, _, _ = F.ecg(two, 11)
    a, r = compare("three SVTs, 20 s and 3 min apart", rec(x, [0], ["(N"], "cu88"))
    # an SVT, then 10 minutes calm: the run's beats leave the tracker's memory (3 minutes) while it is
    # still the newest run; a second SVT after that must be judged as the experiment judges it
    late = [F.REST, (180, 180, 60, 12, 0), (75, 75, 600, 12, 0), (180, 180, 60, 12, 0), (75, 75, 60, 12, 0)]
    x, _, _ = F.ecg(late, 13)
    a, r = compare("SVT, 10 minutes calm, SVT", rec(x, [0], ["(N"], "cu86"))
    check("SVT, 10 minutes calm, SVT: two alerts", len(a) == 2, str(a))
    # the 24 h sets' 5 s stride
    x, _, _ = F.ecg(F.SVT, 3)
    compare("SVT at 180, 5 s stride", rec(x, [0, 151, 241], ["(N", "(SVTA", "(N"], "cu87"), step=5.0)


def pieces() -> None:
    print("pieces")
    from munji_ai.rules import fast_narrow as fn
    check("every rule setting matches fast_narrow_template",
          (fn.FAST_BPM, fn.SVT_BPM, fn.SUSTAIN, fn.SHARE, fn.VT_RUN, fn.STABILITY_MS, fn.ONSET_DROP, fn.RUN_RR,
           fn.ONSET_BEFORE, fn.LZ_SEC, fn.LZ_BIN, fn.LZ_RANGE, fn.LZ_ZERO, fn.LZ_MIN, fn.LZ_RR_MIN, fn.LZ_AF,
           fn.ORG_P, fn.ORG_STEP, fn.ORG_TOL, fn.ORG_KMAX, fn.ORG_COVER, fn.ORG_SHARE, fn.ORG_MIN_RR)
          == (G.FAST_BPM, G.SVT_BPM, G.SUSTAIN, G.SHARE, G.VT_RUN, T.STABILITY_MS, T.ONSET_DROP, T.RUN_RR,
              T.ONSET_BEFORE, G.LZ_SEC, G.LZ_BIN, G.LZ_RANGE, G.LZ_ZERO, G.LZ_MIN, G.LZ_RR_MIN, G.LZ_AF,
              G.ORG_P, G.ORG_STEP, G.ORG_TOL, G.ORG_KMAX, G.ORG_COVER, G.ORG_SHARE, G.ORG_MIN_RR))
    rng = np.random.default_rng(3)
    same = True
    for _ in range(40):
        rr = rng.uniform(0.25, 1.0, int(rng.integers(5, 120)))
        a, b = fn.af_score(rr), G.af_score(rr)
        same &= (np.isnan(a) and np.isnan(b)) or a == b
        same &= fn.organized_rr(rr) == G.organized_rr(rr)[0]
    check("Lorenz score and RR multiples, on 40 random series", same)
    try:
        FastNarrowTracker(FS, 2.5, bm.Template(np.zeros(bm.sizes(128)[1]), 128, 0.0, 10.0))
        check("a template at another fs is refused", False)
    except ValueError:
        check("a template at another fs is refused", True)


def through_decision_layer() -> None:
    """Tracker events into the decision layer, as the app chains them."""
    from munji_ai.decision.decision_layer import DecisionLayer
    print("through the decision layer")
    for label, parts, seed, want in (("SVT at 180", F.SVT, 3, (1, 1)), ("sinus speeding up and slowing down", F.RAMP, 1, (0, 0))):
        x, _, _ = F.ecg(parts, seed)
        tr, dl, alerts = FastNarrowTracker(FS), DecisionLayer(), []
        for i in range(0, len(x), FS):
            alerts += dl.on_fast_narrow(tr.push(x[i:i + FS]))
        alerts += dl.on_fast_narrow(tr.finish())
        r = dl.report()["fast_narrow"]
        got = (sum(a.kind == "fast_narrow" for a in alerts), len(r["entries"]))
        check(f"{label}: fast_narrow alerts and report entries {want}", got == want, str(got))
        if want[0]:
            req = dl.take_recording_requests()
            onset = F.REST[2]                  # the SVT starts after 150 s of rest
            check(f"{label}: the ECG kept holds the start",
                  len(req) == want[0] and all(q.start < onset < q.t for q in req),
                  str([(q.start, q.t, q.end) for q in req]))


def real(keys) -> None:
    from munji_ai.data.loader import load_record
    print("real records (when on disk)")
    done = 0
    for key in keys:
        ds, name = key.split(":", 1)
        try:
            rec = G.read_shdb(Path("data/shdbaf") / name) if ds == "shdbaf" else load_record(ds, name, C.RAW_DIR)
        except Exception as e:                        # noqa: BLE001  not downloaded here
            print(f"  skipped {key}: {type(e).__name__}")
            continue
        step = 5.0 if ds in G.LONG_SETS else G.POS_STEP
        a, r = compare(key, rec, step=step, span=ds in G.SPAN_SETS, learn=False)
        print(f"    {key}: {len(a)} alerts, {len(r)} report entries")
        done += 1
    if not done:
        print("  (no real records found: the synthetic checks above still hold)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="*", default=["cudb:cu01", "vfdb:426", "vfdb:607"],
                    help="dataset:record; shdbaf from data/shdbaf")
    a = ap.parse_args(argv)
    pieces()
    synthetic()
    through_decision_layer()
    real(a.records)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
