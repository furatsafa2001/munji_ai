"""experiments/qrs_delineation.py on synthetic WFDB records, without PhysioNet.

A QTDB-like record (250 Hz, two leads, q1c marks shared by both) and two
LUDB-like records (500 Hz, twelve leads, one annotation file per lead, in both
naming styles), one with narrow beats and one with wide ones. The checks are
about the plumbing: reading the marks, scoring at the expert's peaks, matching
NeuroKit2's lists by position, the scores themselves, and the report. They are
not about how well NeuroKit2 delineates real ECG.

    python tests/test_qrs_delineation.py
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
spec = importlib.util.spec_from_file_location("qrs_delineation", ROOT / "experiments" / "qrs_delineation.py")
qd = importlib.util.module_from_spec(spec)
sys.modules["qrs_delineation"] = qd
spec.loader.exec_module(qd)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def qd_extent_lo(kind):
    return EXTENT[kind][0]


def g(u, s):
    return np.exp(-0.5 * (u / s) ** 2)


# QRS extent of each kind, seconds from the peak: (onset, end)
EXTENT = {"narrow": (-0.045, 0.045), "wide": (-0.080, 0.095)}


def beat(kind, t, b):
    u = t - b
    if kind == "narrow":
        qrs = -0.15 * g(u + 0.025, 0.008) + g(u, 0.010) - 0.3 * g(u - 0.025, 0.008)
        return qrs + 0.15 * g(u + 0.18, 0.02) + 0.3 * g(u - 0.30, 0.05)
    qrs = -0.15 * g(u + 0.045, 0.015) + g(u, 0.022) - 0.4 * g(u - 0.05, 0.018)
    return qrs + 0.15 * g(u + 0.20, 0.02) - 0.3 * g(u - 0.33, 0.06)


def signal(kind, fs, sec, bpm=70.0, seed=0):
    t = np.arange(int(sec * fs)) / fs
    peaks = np.arange(0.4, sec - 0.4, 60.0 / bpm)
    x = np.random.default_rng(seed).normal(0, 0.01, t.size)
    for b in peaks:
        x += beat(kind, t, b)
    return x, peaks


def marks(kind, fs, peaks):
    """WFDB marks: '(' N ')' for every QRS, and '(' p ')' / '(' t ')' waves."""
    lo, hi = EXTENT[kind]
    smp, sym = [], []
    for b in peaks:
        for s, y in ((b - 0.21, "("), (b - 0.18, "p"), (b - 0.15, ")"), (b + lo, "("), (b, "N"),
                     (b + hi, ")"), (b + 0.22, "("), (b + 0.30, "t"), (b + 0.40, ")")):
            smp.append(int(round(s * fs)))
            sym.append(y)
    return np.array(smp), sym


def wrann(record, ext, samples, symbols, folder: Path):
    """wfdb writes letter-only extensions; QTDB's "q1c" and LUDB 1.0.0's
    "atr_ii" are written under one and renamed (the file does not hold it)."""
    import wfdb
    wfdb.wrann(record, "tmpann", samples, symbols, write_dir=str(folder))
    (folder / f"{record}.tmpann").rename(folder / f"{record}.{ext}")


def write_sets(tmp: Path):
    import wfdb
    q, lu = tmp / "qtdb", tmp / "ludb" / "data"
    q.mkdir(parents=True)
    lu.mkdir(parents=True)
    x, peaks = signal("narrow", 250, 30.0, seed=1)
    wfdb.wrsamp("sel900", fs=250, units=["mV", "mV"], sig_name=["MLII", "V5"],
                p_signal=np.stack([x, 0.8 * x], axis=1), fmt=["16", "16"], write_dir=str(q))
    s, y = marks("narrow", 250, peaks)
    wrann("sel900", "q1c", s, y, q)
    leads = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
    for rec, kind in (("1", "narrow"), ("2", "wide")):
        x, peaks = signal(kind, 500, 10.0, seed=int(rec))
        wfdb.wrsamp(rec, fs=500, units=["mV"] * 12, sig_name=leads, p_signal=np.stack([x] * 12, axis=1),
                    fmt=["16"] * 12, write_dir=str(lu))
        s, y = marks(kind, 500, peaks)
        wrann(rec, "v5", s, y, lu)                     # LUDB 1.0.1 naming
        wrann(rec, "atr_ii", s, y, lu)                 # LUDB 1.0.0 naming
    return q, tmp / "ludb"


def main() -> int:
    if importlib.util.find_spec("neurokit2") is None:
        print("SKIP: NeuroKit2 is not installed (pip install neurokit2)")
        return 0
    print("Matching and scoring\n")
    o, e = qd.bounds_at(np.array([100, 300, 500]), [90, 490], [110.0, float("nan"), 510])
    check("onsets and ends are matched to their beat by position, a dropped entry left empty",
          np.array_equal(o, [90, np.nan, 490], equal_nan=True) and np.array_equal(e, [110, np.nan, 510],
                                                                                   equal_nan=True))
    nw, at, cut, best = qd.separation(np.array([100.0, 100, 130, 140]), np.array([100.0, 125, 130, 150]))
    check("separation: two wide beats, 0.75 at 120 ms, 1.00 at the best cut of 130 ms",
          nw == 2 and abs(at - 0.75) < 1e-9 and cut == 130 and best == 1.0, f"{at:.2f}, {cut:.0f} ms {best:.2f}")
    check("expert marks: '(' N ')' read, P and T waves skipped",
          qd.expert_qrs(["(", "p", ")", "(", "N", ")", "(", "t", ")"], [1, 2, 3, 10, 12, 14, 20, 25, 30]).tolist()
          == [[10, 12, 14]])

    with tempfile.TemporaryDirectory() as tmp:
        qroot, lroot = write_sets(Path(tmp))
        print("\nReading\n")
        q = qd.qtdb_leads(qroot / "sel900.q1c")
        n_beats = len(np.arange(0.4, 29.6, 60 / 70))
        check("QTDB: both leads, the expert's marks shared, beats within 0.5 s of an edge left out",
              [ld for ld, _, _ in q] == ["MLII", "V5"] and 0 < len(q[0][2]) < n_beats
              and np.array_equal(q[0][2], q[1][2]), f"{len(q[0][2])} of {n_beats} beats")
        lu = qd.ludb_leads(lroot / "data" / "2.hea")
        check("LUDB: the leads with a file are read, in either naming style; the rest skipped",
              sorted(ld for ld, _, _ in lu) == ["ii", "v5"])
        first = next(b for b in np.arange(0.4, 9.6, 60 / 70) if b + qd_extent_lo("wide") >= qd.EDGE_SEC)
        check("LUDB: 500 Hz marks rescaled to 250 Hz",
              all(abs(int(mk[0, 1]) - round(first * 250)) <= 1 for _, _, mk in lu), f"first peak {first:.2f} s")
        check("records are found in record order, under data/ too",
              [p.stem for p in qd.records(lroot, "ludb")] == ["1", "2"]
              and [p.stem for p in qd.records(qroot, "qtdb")] == ["sel900"])

        print("\nDelineation\n")
        nar, wid = qd.measure("ludb", lroot / "data" / "1.hea"), qd.measure("ludb", lroot / "data" / "2.hea")
        for meth in qd.METHODS:
            dn = [qd.beat_table(nar["rows"], meth)["meas"], qd.beat_table(wid["rows"], meth)["meas"]]
            found = np.mean(np.isfinite(np.concatenate(dn)))
            med = [float(np.nanmedian(d)) for d in dn]
            check(f"{meth}: runs, marks most beats, and wide beats read much wider than narrow ones",
                  not nar["failed"] and not wid["failed"] and found >= 0.7 and med[1] - med[0] > 40,
                  f"found {found:.0%}, narrow {med[0]:.0f} ms, wide {med[1]:.0f} ms")
        eng = [float(np.nanmedian(r[qd.ENGINE])) for r in (nar["rows"][0], wid["rows"][0])]
        check("the engine's width is scored beside them", eng[1] > eng[0], f"{eng[0]:.0f} and {eng[1]:.0f} ms")

        print("\nScores and report\n")
        rows = nar["rows"] + wid["rows"]
        s = qd.metrics(rows, "dwt")
        check("metrics: every beat counted, the expert's wide beats counted, IEC judged from mean and SD",
              s["n"] == sum(len(r["marks"]) for r in rows) and s["sep"][0] == len(wid["rows"][0]["marks"]) * 2
              and s["iec"] == (abs(s["dur"][0]) <= 10 and s["dur"][1] <= 10))
        check("relative error: a beat-to-beat SD exists; no beat is wider than its own record's usual one",
              np.isfinite(s["rel_sd"]) and s["wider"][0] == 0, f"rel SD {s['rel_sd']:.1f} ms")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = qd.main(["--qtdb", str(qroot), "--ludb", str(lroot), "--workers", "1"])
        text = buf.getvalue()
        check("main: both sets, the literature line, LUDB V5 apart, per-lead tables, IEC verdicts",
              rc == 0 and "QTDB:" in text and "LUDB:" in text and "Martinez 2004" in text
              and "LUDB V5, the lead nearest" in text and "QTDB per lead" in text and "LUDB per lead" in text
              and "IEC 60601-2-25, QTDB all leads" in text and "IEC 60601-2-25, LUDB V5" in text,
              "" if rc == 0 else text[-300:])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = qd.main(["--qtdb", str(Path(tmp) / "none"), "--ludb", str(Path(tmp) / "none"), "--workers", "1"])
        check("no data: says where to get it", rc == 1 and "--download" in buf.getvalue())

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
