"""experiments/template_wct.py and template_vtac.py on synthetic beats, without data.

Normal beats are narrow, ventricular beats (PVCs and VT) wide and of another
shape, as in the VFDB pictures. The checks are about the plumbing: which
stretch the template comes from, that ectopic beats no longer spoil it, that
rhythm labels keep it out of ventricular rhythms, what each new check (size,
wide window, onset, stability) lets through and what it stops, and the VTaC
run search. They are not about how well the rule finds real VT.

    python tests/test_template_wct.py
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tw = load("template_wct")
tv = load("template_vtac")
FS = tw.C.TARGET_FS
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def beat(kind: str, t: np.ndarray, b: float) -> np.ndarray:
    """N narrow QRS, small S, T wave; s the same a third of the size; t the
    same QRS with a large inverted T; V wide and of another shape."""
    if kind in "Nst":
        qrs = 1.0 * np.exp(-((t - b) / 0.012) ** 2) - 0.25 * np.exp(-((t - b - 0.03) / 0.012) ** 2)
        if kind == "t":
            return qrs - 0.7 * np.exp(-((t - b - 0.2) / 0.05) ** 2)
        y = qrs + 0.2 * np.exp(-((t - b - 0.25) / 0.05) ** 2)
        return y / 3.0 if kind == "s" else y
    return -1.2 * np.exp(-((t - b) / 0.04) ** 2) + 0.5 * np.exp(-((t - b - 0.09) / 0.04) ** 2)


def from_beats(times, kinds, total, marks, name="r1", seed=0):
    """A fake Record from beat times and kinds; marks: [(start s, token)]."""
    t = np.arange(int(total * FS)) / FS
    x = np.random.default_rng(seed).normal(0, 0.02, t.size)
    for b, k in zip(times, kinds):
        lo, hi = max(0, int((b - 0.3) * FS)), min(t.size, int((b + 0.5) * FS))
        x[lo:hi] += beat(k, t[lo:hi], b)
    return SimpleNamespace(signal=x, fs=FS, dataset="fake", name=name, patient=f"fake/{name}",
                           rhythm_starts=np.array([int(s * FS) for s, _ in marks]),
                           rhythm_labels=np.array([tok for _, tok in marks], dtype=object),
                           beat_samples=np.array([], int), beat_symbols=np.array([]),
                           peaks=np.round(np.asarray(times) * FS).astype(int), kinds=np.array(list(kinds)))


def record(parts, name="r1", seed=0):
    """[(seconds, bpm, pattern, token)] -> a fake Record. pattern repeats over
    the beats: "N" sinus, "V" ventricular, "NNNV" a PVC every fourth beat."""
    times, kinds, marks, t0 = [], [], [], 0.0
    for sec, bpm, pattern, tok in parts:
        for k, b in enumerate(np.arange(t0 + 0.3, t0 + sec - 0.3, 60.0 / bpm)):
            times.append(b)
            kinds.append(pattern[k % len(pattern)])
        marks.append((t0, tok))
        t0 += sec
    return from_beats(times, kinds, t0, marks, name, seed)


def stretch(pattern: str, bpm: float = 78.0, sec: float = 10.6):
    """The beats of one stretch (10 s, as calibration cuts them), with their kinds."""
    r = record([(sec, bpm, pattern, "(N")])
    pos, ext = tw.extended(r.signal, r.peaks, FS)
    keep = np.isin(r.peaks, pos)
    return r, pos, ext, r.kinds[keep]


def run(parts_or_rec, mode="dominant", name="r1"):
    rec = parts_or_rec if isinstance(parts_or_rec, SimpleNamespace) else record(parts_or_rec, name=name)
    return tw.process(rec, None, annotated_span=False, step=tw.POS_STEP, mode=mode)


def alerted(r, config):
    return tw.score([r], config)["long"][0] == 1


def main() -> int:
    pre, L, lag = tw._sizes(FS)
    print("The template of one stretch\n")
    _, _, ext, kinds = stretch("NNNV")
    shapes = {k: ext[kinds == k][0][None, :] for k in ("N", "V")}
    check("a PVC every fourth beat spoils the strict rule (75% under its 80%)",
          tw.stretch_template(ext, L, lag) is None)
    tpl = tw.dominant_template(ext, L, lag)
    rn = float(tw.best_corr(shapes["N"], tpl)[0]) if tpl is not None else np.nan
    rv = float(tw.best_corr(shapes["V"], tpl)[0]) if tpl is not None else np.nan
    check("... the dominant rule keeps the normal beat as the template, the PVCs left out",
          tpl is not None and rn > 0.95 and rv < 0.5, f"normal {rn:.2f}, PVC {rv:.2f}")
    _, _, ext2, _ = stretch("NV")
    check("bigeminy (half the beats ventricular): no main shape, no template",
          tw.dominant_template(ext2, L, lag) is None)
    _, _, ext3, _ = stretch("N")
    a, b = tw.stretch_template(ext3, L, lag), tw.dominant_template(ext3, L, lag)
    check("a clean stretch calibrates exactly as before", a is not None and np.array_equal(a, b))

    print("\nEvery beat against the template: shape, size, wide\n")
    r, pos, ext, kinds = stretch("NNsstt", sec=40.6)
    Lw, wext = tw.wide_windows(r.signal, pos, FS)
    n = kinds == "N"
    tpl = tw.dominant_template(ext[n], L, lag)
    tplw = tw.wide_template(ext[n], wext[n], tpl, Lw)
    feat = tw.match(ext, wext, tpl, tplw, L, Lw)
    med = {k: {m: float(np.median(feat[m][kinds == k])) for m in feat} for k in "Nst"}
    check("normal beats: shape and wide correlation near 1, size near 1",
          med["N"]["shape"] > 0.98 and med["N"]["wide"] > 0.98 and abs(med["N"]["size"] - 1) < 0.05,
          str({m: round(v, 2) for m, v in med["N"].items()}))
    check("a third of the size, same shape: alike by shape, the size check calls it unlike",
          med["s"]["shape"] > 0.95 and abs(med["s"]["size"] - 1 / 3) < 0.05
          and not tw.unlike(feat, ("shape", False, 0.6))[kinds == "s"].any()
          and tw.unlike(feat, ("shape", True, 0.6))[kinds == "s"].all(), f"size {med['s']['size']:.2f}")
    check("same QRS, inverted T: alike over 100 ms either side of R, unlike over the wide span",
          med["t"]["shape"] > 0.95 and med["t"]["wide"] < 0.6
          and tw.unlike(feat, ("wide", False, 0.6))[kinds == "t"].all(),
          f"shape {med['t']['shape']:.2f}, wide {med['t']['wide']:.2f}")
    check("no template: nothing is unlike", not tw.unlike(tw.match(ext, wext, None, None, L, Lw),
                                                          ("shape", True, 0.6)).any())

    print("\nOnset and stability\n")
    sudden = np.cumsum(np.r_[np.full(30, 0.8), np.full(60, 0.46)])
    start, drop = tw.onset_runs(sudden, 100.0)
    s0 = int(start[-1])
    check("a jump from 75 to 130 bpm: one run, the cycle length down by 40% or more at its start",
          s0 > 0 and drop[s0] > 0.4 and tw.onset_ok(np.array([sudden[-1]]), sudden, start, drop)[0],
          f"drop {drop.get(s0, np.nan):.0%}")
    gradual = np.cumsum(np.r_[np.linspace(0.75, 0.5, 80), np.full(160, 0.5)])
    start, drop = tw.onset_runs(gradual, 100.0)
    s0 = int(start[-1])
    t_early, t_late = gradual[s0] + 30.0, gradual[s0] + tw.ONSET_OVERRIDE_SEC + 5.0
    oks = tw.onset_ok(np.array([t_early, t_late]), gradual, start, drop)
    check("a gradual rise past 100 bpm is held back, and let through once it has lasted 60 s",
          drop[s0] < tw.ONSET_DROP and not oks[0] and oks[1], f"drop {drop[s0]:.1%}")
    tail = np.cumsum(np.r_[np.linspace(0.75, 0.5, 80), np.full(40, 0.5), np.full(50, 0.8)])
    start, drop = tw.onset_runs(tail, 100.0)
    last = int(np.flatnonzero(start >= 0)[-1])
    oks = tw.onset_ok(np.array([tail[last] + 10.0, tail[last] + 31.0]), tail, start, drop)
    check("... its tail too, while the last 30 s still hold a beat of it (then the 30 s rule judges)",
          not oks[0] and oks[1])
    check("no fast run: the onset check never holds anything back",
          tw.onset_ok(np.array([20.0]), np.cumsum(np.full(40, 0.8)), *tw.onset_runs(np.cumsum(np.full(40, 0.8)),
                                                                                     100.0))[0])
    rng = np.random.default_rng(1)
    reg = np.cumsum(0.46 + rng.normal(0, 0.003, 60)) * FS
    irr = np.cumsum(rng.uniform(0.35, 0.65, 60)) * FS
    sr, si_ = tw.instability_ms(reg, FS), tw.instability_ms(irr, FS)
    check("regular VT-like RR is stable, AF-like RR is not (40 ms)",
          sr < tw.STABILITY_MS < si_, f"{sr:.0f} ms against {si_:.0f} ms")

    print("\nRhythm labels and whole records\n")
    vt = [(60, 75, "NNNV", "(HGEA"), (60, 130, "V", "(VT"), (60, 75, "NNNV", "(HGEA")]
    got = {mode: run(vt, mode) for mode in tw.CALS}
    check("strict: sinus rhythm with frequent PVCs (HGEA) gives no template, so the slow VT is missed",
          got["strict"]["first"] is None and not alerted(got["strict"], "run 2 rule"))
    r = got["dominant"]
    check("dominant: the template comes from the HGEA stretch before the VT",
          r["first"] is not None and r["episodes"][0]["template"] == "before", f"ends at {r['first']}")
    sc = {name: tw.score([r], name) for name, _ in tw.configs(False)}
    check("... and the slow VT (130 bpm, sudden, regular) is alerted by every configuration, nothing else",
          all(s["long"] == [1, 1] and not s["false_list"] for s in sc.values()),
          str({n: s["long"][0] for n, s in sc.items()}))

    small = run([(60, 75, "N", "(N"), (90, 130, "s", "(VT"), (30, 75, "N", "(N")], name="small")
    check("VT of the patient's own shape a third of the size: missed by shape alone, caught with size",
          not alerted(small, "run 2 rule") and alerted(small, "+ size")
          and alerted(small, "all: size, onset, stability"),
          f"unlike by shape {small['episodes'][0]['unlike']:.0%}, with size "
          f"{small['episodes'][0]['unlike_size']:.0%}")

    times, kinds, t = [], [], 0.3
    while t < 160.0:                     # 80 bpm, up to 125 over 40 s, held 30 s, then 75
        bpm = 80.0 if t < 30 else 80.0 + 45.0 * min(1.0, (t - 30) / 40) if t < 100 else 75.0
        times.append(t)
        kinds.append("V" if 100 < bpm and t < 100 else "N")
        t += 60.0 / bpm
    exercise = run(from_beats(times, kinds, 160.0, [(0.0, "(N")], name="exercise"))
    al = {n: len(exercise["alerts"][n]) for n in ("run 2 rule", "+ onset")}
    check("sinus speeding up gradually, beats changing shape past 100 bpm: the run-2 rule alerts, "
          "onset holds it back", al["run 2 rule"] >= 1 and al["+ onset"] == 0, str(al))

    rng = np.random.default_rng(2)
    times = list(np.arange(0.3, 60.0, 0.8))
    while times[-1] < 120.0:             # AF with wide beats: irregular, about 120 bpm
        times.append(times[-1] + rng.uniform(0.35, 0.65))
    times += list(np.arange(times[-1] + 0.8, 160.0, 0.8))
    kinds = ["V" if 60.0 <= b < 120.0 else "N" for b in times]
    af = run(from_beats(times, kinds, 160.0, [(0.0, "(N")], name="af"))
    al = {n: len(af["alerts"][n]) for n in ("run 2 rule", "+ stability")}
    check("an irregular fast rhythm of wide beats (AF with aberrancy): stability holds it back",
          al["run 2 rule"] >= 1 and al["+ stability"] == 0, str(al))

    ver = [(p[0], p[1], p[2], "(VER" if p[3] == "(HGEA" else p[3]) for p in vt]
    r2 = run(ver, name="r2")
    n = r2["cal_counts"]
    check("the same beats labelled ventricular escape rhythm (VER): never the template",
          r2["first"] is None and n["shape"] > 0 and n["allowed"] == 0, tw.why_no_template(n))
    r3 = run([(90, 130, "V", "(VT")], name="r3")
    check("a record in VT throughout has no stretch at 40-100 bpm, and says so",
          r3["first"] is None and r3["cal_counts"]["rate"] == 0, tw.why_no_template(r3["cal_counts"]))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tw.report([got["dominant"], small, r2, r3], False)
    text = buf.getvalue()
    check("the report lists records without a template, every episode and every configuration",
          "records without a template: 2 of 4" in text and "fake r2: " in text
          and all(name in text for name, _ in tw.configs(False)) and "all, gate on" not in text
          and "+size" in text and "onset" in text)

    try:
        import matplotlib
    except ImportError:
        print("  SKIP  the picture (no matplotlib)")
    else:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        plot = load("template_wct_plot")
        for rec, label in ((record(ver, name="r2"), "none"), (record(vt), "before")):
            b = tw.beats_and_template(rec, tw.POS_STEP)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "p.pdf"
                with PdfPages(path) as pdf:
                    row = plot.episode_page(pdf, plt, "fake", rec.name, b, 60.0, 120.0)
                check(f"the picture is drawn, template {label}",
                      path.stat().st_size > 10_000 and row["template"] == label)

    print("\nVTaC: the beat test on alarms\n")
    n40 = int(40 * FS)

    def window(parts, seed):
        return record(parts, seed=seed).signal[:n40].astype(np.float32)

    base = {"X": np.stack([window([(40, 75, "N", "(N")], 1)]), "record": np.array(["p1"]),
            "event": np.array(["e1"]), "lead": np.array(["II"])}
    alarms = [
        ([(14, 75, "N", ""), (8, 130, "V", ""), (18, 75, "N", "")], True, "II", "train"),   # VT run
        ([(40, 130, "N", "")], False, "II", "train"),                                        # sinus tachycardia
        ([(14, 75, "N", ""), (8, 130, "V", ""), (18, 75, "N", "")], True, "V", "train"),    # another lead
        ([(14, 75, "N", ""), (8, 130, "V", ""), (18, 75, "N", "")], True, "II", "test"),    # test split
    ]
    win = {"X": np.stack([window(p, 10 + i) for i, (p, *_) in enumerate(alarms)]),
           "record": np.array(["p1"] * 4), "event": np.array(["e1", "e2", "e3", "e4"]),
           "decision": np.array([a[1] for a in alarms]),
           "lead": np.array([a[2] for a in alarms]), "split": np.array([a[3] for a in alarms])}
    res = tv.evaluate(win, base, FS)
    rows = res["rows"]
    k = ("shape", False, 0.6)
    check("two alarms judged; another lead and the test split are set apart",
          len(rows) == 2 and res["skipped"]["no baseline on the alarm's lead"] == 1
          and res["skipped"]["test split"] == 1, dict(res["skipped"]))
    check("the true alarm: a run of VT beats unlike the template is found, at about 130 bpm",
          rows[0]["true"] and rows[0][k] and 120 < rows[0]["rate"] < 140, f"{rows[0]['rate']:.0f} bpm")
    check("the template of the alarm's own recording is marked so; another recording's falls back",
          rows[0]["same"] and not rows[1]["same"])
    check("the false alarm (sinus tachycardia): a fast run, but of the patient's own beats",
          not rows[1]["true"] and np.isfinite(rows[1]["rate"]) and not rows[1][k])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tv.report(res)
    check("the VTaC report prints every match row, for every alarm and for own-recording templates",
          all(f"{m} C {c}" in buf.getvalue() for m, _, c in tv.KEYS)
          and "every alarm: 1 true VT, 1 false" in buf.getvalue()
          and "own recording: 1 true VT, 0 false" in buf.getvalue())

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
