"""Checks data/vt_runs.py on synthetic beat-annotated recordings.

    python tests/test_vt_runs.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data import vt_runs as vr

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    print("Run finding\n")
    fs = 250
    sym = np.array(list("NNNVVVNNVVNNVVVVVNN"))
    gap = {"N": 200, "V": 100}                    # 75 bpm normal, 150 bpm ventricular
    samp = np.cumsum([gap[c] for c in sym])
    runs = vr.find_runs(sym, samp, fs)
    check("finds the 3-beat and 5-beat runs, not the pair", [r[2] for r in runs] == [3, 5],
          [r[2] for r in runs])
    check("measures the run's rate", abs(runs[0][3] - 150) < 1, f"{runs[0][3]:.0f} bpm")
    slow = np.cumsum([{"N": 200, "V": 200}[c] for c in sym])
    check("runs slower than 100 bpm are not VT", vr.find_runs(sym, np.cumsum([{"N": 200, "V": 160}[c] for c in sym]), fs) == [])
    check("min_beats=5 keeps only the long run", [r[2] for r in vr.find_runs(sym, samp, fs, 5)] == [5])
    check("E (ventricular escape) counts as ventricular",
          len(vr.find_runs(np.array(list("NEEEN")), np.cumsum([200, 100, 100, 100, 200]), fs)) == 1)

    print("\nEnd to end on a synthetic dataset\n")
    import wfdb
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        d = tmp / "raw" / "ltafdb"
        d.mkdir(parents=True)
        for k in range(12):
            name = f"{k:02d}"
            s = np.array(list("NNNNVVVVNNNN" * 5))
            t = np.cumsum([{"N": 200, "V": 110}[c] for c in s]) + 50
            wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"],
                        p_signal=np.zeros((int(t[-1]) + 100, 1)), fmt=["16"], write_dir=str(d))
            wfdb.wrann(name, "atr", t, symbol=list(s), write_dir=str(d))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vr.main(["--datasets", "ltafdb", "--raw", str(tmp / "raw"),
                          "--cache", str(tmp / "cache")])
        text = buf.getvalue()
        out = np.load(tmp / "cache" / "vt_runs.npz")
    check("runs and saves", rc == 0 and len(out["beats"]) > 0, text.splitlines()[0])
    check("test-split patients are skipped", "test-split recordings skipped" in text)
    check("every saved run is 4 beats at about 136 bpm",
          set(out["beats"]) == {4} and abs(np.median(out["bpm"]) - 136.4) < 1)
    check("summary table printed", "patients" in text and "100-149" in text)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
