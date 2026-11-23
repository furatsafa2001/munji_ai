"""experiments/template_wct.py's calibration, on synthetic beats, without data.

Normal beats are narrow, ventricular beats (PVCs and VT) wide and of another
shape, as in the VFDB pictures. The checks are about the template: which
stretch it comes from, that ectopic beats no longer spoil it, that rhythm
labels keep it out of ventricular rhythms, and that a record without one says
why. They are not about how well the rule finds real VT.

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


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tw = load("template_wct")
FS = tw.C.TARGET_FS
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def beat(kind: str, t: np.ndarray, b: float) -> np.ndarray:
    if kind == "N":                       # narrow QRS, small S, T wave
        return (1.0 * np.exp(-((t - b) / 0.012) ** 2) - 0.25 * np.exp(-((t - b - 0.03) / 0.012) ** 2)
                + 0.2 * np.exp(-((t - b - 0.25) / 0.05) ** 2))
    return -1.2 * np.exp(-((t - b) / 0.04) ** 2) + 0.5 * np.exp(-((t - b - 0.09) / 0.04) ** 2)


def record(parts, name="r1", seed=0):
    """[(seconds, bpm, pattern, token)] -> a fake Record. pattern repeats over
    the beats: "N" sinus, "V" ventricular, "NNNV" a PVC every fourth beat."""
    total = sum(p[0] for p in parts)
    t = np.arange(int(total * FS)) / FS
    x = np.random.default_rng(seed).normal(0, 0.02, t.size)
    starts, toks, peaks, kinds, t0 = [], [], [], [], 0.0
    for sec, bpm, pattern, tok in parts:
        for k, b in enumerate(np.arange(t0 + 0.3, t0 + sec - 0.3, 60.0 / bpm)):
            kind = pattern[k % len(pattern)]
            x += beat(kind, t, b)
            peaks.append(int(round(b * FS)))
            kinds.append(kind)
        starts.append(int(t0 * FS))
        toks.append(tok)
        t0 += sec
    return SimpleNamespace(signal=x, fs=FS, dataset="fake", name=name, patient=f"fake/{name}",
                           rhythm_starts=np.array(starts), rhythm_labels=np.array(toks, dtype=object),
                           beat_samples=np.array([], int), beat_symbols=np.array([]),
                           peaks=np.array(peaks), kinds=np.array(kinds))


def stretch(pattern: str, bpm: float = 78.0):
    """The beats of one 10 s stretch, cut as calibration cuts them."""
    r = record([(10.6, bpm, pattern, "(N")])
    pre, L, lag = tw._sizes(FS)
    pos, ext = tw.extended(r.signal, r.peaks, FS)
    keep = np.isin(r.peaks, pos)
    return ext, r.kinds[keep], L, lag


def main() -> int:
    print("The template of one stretch\n")
    pre, L, lag = tw._sizes(FS)
    ext, kinds, L, lag = stretch("NNNV")
    shapes = {k: ext[kinds == k][0][None, :] for k in ("N", "V")}
    check("a PVC every fourth beat spoils the strict rule (75% under its 80%)",
          tw.stretch_template(ext, L, lag) is None)
    tpl = tw.dominant_template(ext, L, lag)
    rn = float(tw.best_corr(shapes["N"], tpl)[0]) if tpl is not None else np.nan
    rv = float(tw.best_corr(shapes["V"], tpl)[0]) if tpl is not None else np.nan
    check("... the dominant rule keeps the normal beat as the template, the PVCs left out",
          tpl is not None and rn > 0.95 and rv < 0.5, f"normal {rn:.2f}, PVC {rv:.2f}")
    ext2, _, _, _ = stretch("NV")
    check("bigeminy (half the beats ventricular): no main shape, no template",
          tw.dominant_template(ext2, L, lag) is None)
    ext3, _, _, _ = stretch("N")
    a, b = tw.stretch_template(ext3, L, lag), tw.dominant_template(ext3, L, lag)
    check("a clean stretch calibrates exactly as before", a is not None and np.array_equal(a, b))

    print("\nRhythm labels and a whole record\n")
    vt = [(60, 75, "NNNV", "(HGEA"), (60, 130, "V", "(VT"), (60, 75, "NNNV", "(HGEA")]
    ci, M = tw.CORRS.index(tw.SHOW_C), tw.SHOW_M
    got = {}
    for mode in tw.CALS:
        r = tw.process(record(vt), None, annotated_span=False, step=tw.POS_STEP, mode=mode)
        got[mode] = (r, tw.score([r], False, "fixed", 100.0, ci, M))
    r, s = got["strict"]
    check("strict: sinus rhythm with frequent PVCs (HGEA) gives no template, so the slow VT is missed",
          r["first"] is None and s["long"] == [0, 1], f"VT {s['long']}")
    r, s = got["dominant"]
    check("dominant: the template comes from the HGEA stretch before the VT",
          r["first"] is not None and r["episodes"][0]["template"] == "before", f"ends at {r['first']}")
    check("... and the slow VT (130 bpm) is alerted, nothing else",
          s["long"] == [1, 1] and not s["false_list"], f"VT {s['long']}, false {len(s['false_list'])}")

    ver = [(p[0], p[1], p[2], "(VER" if p[3] == "(HGEA" else p[3]) for p in vt]
    r = tw.process(record(ver, name="r2"), None, annotated_span=False, step=tw.POS_STEP)
    n = r["cal_counts"]
    check("the same beats labelled ventricular escape rhythm (VER): never the template",
          r["first"] is None and n["shape"] > 0 and n["allowed"] == 0, tw.why_no_template(n))
    r3 = tw.process(record([(90, 130, "V", "(VT")], name="r3"), None, annotated_span=False,
                    step=tw.POS_STEP)
    check("a record in VT throughout has no stretch at 40-100 bpm, and says so",
          r3["first"] is None and r3["cal_counts"]["rate"] == 0, tw.why_no_template(r3["cal_counts"]))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tw.report([got["dominant"][0], r, r3], False)
    text = buf.getvalue()
    check("the report lists every record without a template with its reason",
          "records without a template: 2 of 3" in text and "fake r2: " in text and "fake r3: " in text
          and "in an allowed rhythm" in text)

    try:
        import matplotlib
    except ImportError:
        print("  SKIP  the picture (no matplotlib)")
    else:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        plot = load("template_wct_plot")
        rec = record(ver, name="r2")
        b = tw.beats_and_template(rec, tw.POS_STEP)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.pdf"
            with PdfPages(path) as pdf:
                row = plot.episode_page(pdf, plt, "fake", "r2", b, 60.0, 120.0)
            check("the picture of a record without a template is drawn, with its reason",
                  path.stat().st_size > 10_000 and row["template"] == "none")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
