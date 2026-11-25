"""Checks data/svt_runs.py on synthetic beat-annotated recordings.

    python tests/test_svt_runs.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data import svt_runs as sr  # noqa: E402
from munji_ai.data.splits import assign  # noqa: E402

ok = True
FS = 250


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def beats(spec):
    """[(symbol, count, bpm)] -> (symbols, samples) at FS."""
    sym, gaps = [], []
    for s, n, bpm in spec:
        sym += [s] * n
        gaps += [int(round(FS * 60.0 / bpm))] * n
    return np.array(sym), np.cumsum(gaps) + 50


def main() -> int:
    print("Runs\n")
    sym, samp = beats([("N", 10, 75), ("S", 2, 150), ("N", 5, 75), ("S", 4, 160), ("N", 5, 75),
                       ("A", 3, 130), ("N", 5, 75), ("S", 5, 90), ("N", 3, 75)])
    runs = sr.find_runs(sym, samp, FS)
    check("finds the 4-beat S run and the 3-beat A run, not the pair or the slow run",
          [r[4] for r in runs] == [4, 3], [r[4:] for r in runs])
    check("measures the run's rate", abs(runs[0][5] - 160) < 2, f"{runs[0][5]:.0f}")

    print("\nRhythm marks\n")
    rh = sr.rhythm_at(np.array([0, 1000, 5000]), ["(N", "(AFIB", ")"], np.array([10, 900, 1200, 6000]))
    check("the last '(' mark before each beat; ')' does not end it", list(rh) == ["(N", "(N", "(AFIB", "(AFIB"],
          list(rh))
    check("no marks: empty", list(sr.rhythm_at(np.array([5]), ["+"], np.array([10]))) == [""])

    print("\nFast sinus\n")
    sym, samp = beats([("N", 60, 75), ("N", 160, 160), ("N", 60, 75)])
    eps, fastest = sr.fast_sinus(sym, samp, FS, np.zeros(sym.size, bool))
    check("a minute of normal beats at 160: one episode, fastest about 160",
          len(eps) == 1 and abs(fastest - 160) < 2 and eps[0][3] >= 155, f"{eps} {fastest:.0f}")
    sym2, samp2 = beats([("N", 60, 75), ("N", 160, 120), ("N", 60, 75)])
    e2, f2 = sr.fast_sinus(sym2, samp2, FS, np.zeros(sym2.size, bool))
    check("at 120: no episode, fastest about 120", not e2 and abs(f2 - 120) < 2, f"{f2:.0f}")
    e3, f3 = sr.fast_sinus(sym, samp, FS, np.ones(sym.size, bool))
    check("inside AF marks: nothing", not e3 and np.isnan(f3))
    mix = np.array(sym)
    mix[60:220:4] = "S"
    e4, _ = sr.fast_sinus(mix, samp, FS, np.zeros(sym.size, bool))
    check("a quarter of the beats premature: not sinus", not e4)

    print("\nEnd to end on a synthetic Icentia\n")
    import wfdb
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        expect = {"svt30": 0, "sinus": 0, "patients": 0, "skipped": 0, "in_af": 0}
        for p in range(1, 13):
            pid = f"p{p:05d}"
            d = tmp / "raw" / "icentia11k" / "p00" / pid
            d.mkdir(parents=True)
            test = assign(f"icentia11k:{pid}") == "test"
            expect["patients"] += 0 if test else 1
            for sgi in range(2):
                name = f"{pid}_s{sgi:02d}"
                if sgi == 0:
                    spec = [("N", 80, 75), ("S", 120, 170), ("N", 80, 75)]       # 42 s of SVT at 170
                else:
                    spec = [("N", 80, 75), ("N", 200, 160), ("N", 80, 75)]       # 75 s of sinus at 160
                sym, samp = beats(spec)
                aux = [""] * sym.size
                aux[0] = "(N"
                if p == 1 and sgi == 0:
                    aux[70] = "(AFIB"                                          # the SVT run sits in an AF mark
                wfdb.wrsamp(name, fs=FS, units=["mV"], sig_name=["ECG"],
                            p_signal=np.zeros((int(samp[-1]) + 100, 1)), fmt=["16"], write_dir=str(d))
                wfdb.wrann(name, "atr", samp, symbol=list(sym), aux_note=aux, write_dir=str(d))
                if test:
                    expect["skipped"] += 1
                elif sgi == 0:
                    expect["in_af" if p == 1 else "svt30"] += 1
                else:
                    expect["sinus"] += 1
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sr.main(["--raw", str(tmp / "raw"), "--cache", str(tmp / "cache")])
        text = buf.getvalue()
        z = np.load(tmp / "cache" / "svt_runs.npz")
        svt = z["kind"] == "svt"
        long_fast = svt & ~z["in_af"] & ((z["end"] - z["start"]) / z["fs"] >= 30) & (z["bpm"] >= 150)
        check("SVT runs of 30 s at 150+ outside AF, one per patient outside the test split",
              int(long_fast.sum()) == expect["svt30"] and int((svt & z["in_af"]).sum()) == expect["in_af"],
              f"{int(long_fast.sum())} vs {expect['svt30']}, in AF {int((svt & z['in_af']).sum())}")
        check("fast sinus episodes, one per patient outside the test split",
              int((z["kind"] == "sinus").sum()) == expect["sinus"], f"{int((z['kind'] == 'sinus').sum())}")
        check("the summary: patients read, test recordings skipped, the scored line",
              rc == 0 and f"from {expect['patients']:,} patients ({expect['skipped']:,} test-split" in text
              and "both (what the label is scored on)" in text and "reached 130" in text, text[-1500:])
        check("test-split patients never appear", not any(assign(f"icentia11k:{q}") == "test"
                                                         for q in {str(x).split(":")[-1] for x in z["patient"]}))

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
