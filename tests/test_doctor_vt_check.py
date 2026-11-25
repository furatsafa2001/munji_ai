"""doctor_vt_check on synthetic records, without PhysioNet.

A fast regular rhythm at 165 bpm that the doctor's report row enters, with a
VT mark inside the entry's 30 s, three ways: ventricular beats marked and
drawn with the patient's own shape (alike), marked but absent from the signal
(missed by the detector), and a VT mark with no ventricular beat marked (no V
beat). The checks are about plumbing: the entries are found, sorted into the
right kind, summed, and drawn. They say nothing about real ECG.

    python tests/test_doctor_vt_check.py
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("doctor_vt_check", ROOT / "experiments" / "doctor_vt_check.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

FS = 250
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def beat_times():
    """Rest at 75 for 150 s, 165 bpm for 120 s, rest for 60 s."""
    t, out = 1.0, []
    for bpm, sec in ((75, 150), (165, 120), (75, 60)):
        t0 = t
        while t < t0 + sec:
            out.append(t)
            t += 60.0 / bpm
    return np.array(out), t + 1.0


def signal(bt, dur, seed=0):
    n = int(dur * FS)
    tt = np.arange(n) / FS
    x = np.zeros(n)
    for c in bt:
        lo, hi = max(0, int((c - 0.3) * FS)), min(n, int((c + 0.3) * FS))
        x[lo:hi] += np.exp(-0.5 * ((tt[lo:hi] - c) / 0.012) ** 2)
    return x + np.random.default_rng(seed).normal(0, 0.01, n)


def record(x, bt, sym, marks, name):
    from munji_ai.data.loader import Record
    starts, labels = zip(*marks)
    return Record(dataset="ltafdb", name=name, patient=f"ltafdb:{name}", signal=x, fs=FS,
                  beat_samples=np.round(bt * FS).astype(int), beat_symbols=np.array(sym, dtype=object),
                  rhythm_starts=np.array([int(s * FS) for s in starts]), rhythm_labels=np.array(labels, dtype=object))


def case(kind, name):
    bt, dur = beat_times()
    onset = bt[np.searchsorted(bt, 151.0)]
    vmask = (bt > onset + 5) & (bt < onset + 6.5)          # 4 beats, about 5 to 6.5 s into the fast run
    sym = np.where(vmask, "V", "N") if kind != "no V beat" else np.full(bt.size, "N")
    drawn = bt[~vmask] if kind == "missed" else bt
    marks = [(0, "(AFIB"), (onset + 4.8, "(VT"), (onset + 6.8, "(AFIB"), (onset + 120, "(N")]
    return record(signal(drawn, dur), bt, sym.tolist(), marks, name)


def main() -> int:
    print("pieces")
    check("longest run of flags", D.longest([1, 1, 0, 1, 1, 1, 0]) == 3 and D.longest([]) == 0)
    check("label at a time", D.label_at([(0, 10, "AF"), (10, 20, "VT")], 12) == "VT"
          and D.label_at([(0, 10, "AF")], 30) == "unlabelled")

    print("records")
    rows = []
    for kind, name in (("alike", "901"), ("missed", "902"), ("no V beat", "903")):
        rec = case(kind, name)
        b, dec, al = D.doctor_entries(rec, D.G.POS_STEP)
        unsafe = [float(t) for t in al if D.G.classify(float(t), b["ivs"]).startswith("unsafe:")]
        check(f"{kind}: the doctor's report row enters the fast rhythm, and the entry touches the VT mark",
              len(al) >= 1 and len(unsafe) >= 1, f"{al} {unsafe}")
        if not unsafe:
            continue
        r = D.look(rec, b, dec, unsafe[0])
        rows.append(r)
        detail = f"kind {r['kind']}, V {r['n_v']}, found {r['n_found']}, unlike {r['n_unlike']}, vt_run {r['vt_run']}"
        check(f"{kind}: sorted as '{kind}'", r["kind"] == kind, detail)
        if kind == "alike":
            check("alike: 4 ventricular beats in a row, all found, none unlike; the VT check saw no run",
                  r["n_v"] == 4 and r["v_run"] == 4 and r["n_found"] == 4 and r["n_unlike"] == 0 and r["vt_run"] < 3
                  and len(r["vt"]) == 1, detail)
        if kind == "missed":
            check("missed: 4 ventricular beats marked, none found by the detector", r["n_v"] == 4 and r["n_found"] == 0,
                  detail)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            D.show("ltafdb", rec.name, b["ivs"], r)
        check(f"{kind}: printed with its kind", f"-> {kind}" in buf.getvalue())
    with contextlib.redirect_stdout(io.StringIO()):
        n = D.summary(rows)
    check("summary counts each kind once", n == {"alike": 1, "missed": 1, "no V beat": 1}, str(n))

    print("plot")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    rec = case("alike", "904")
    b, dec, al = D.doctor_entries(rec, D.G.POS_STEP)
    unsafe = [float(t) for t in al if D.G.classify(float(t), b["ivs"]).startswith("unsafe:")]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "p.pdf"
        with PdfPages(out) as pdf:
            D.plot(pdf, "ltafdb", "904", b, D.look(rec, b, dec, unsafe[0]))
        check("one page per entry is written", out.exists() and out.stat().st_size > 1000)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
