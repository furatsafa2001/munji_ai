"""Checks eval/vt_run_simulation.py on synthetic beat-annotated recordings.

    python tests/test_vt_run_simulation.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval import vt_run_simulation as vs

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    print("Corruption\n")
    rng = np.random.default_rng(0)
    sym = np.array(["V"] * 10000 + ["N"] * 100000)
    out = vs.corrupt(sym, 0.9, 0.01, rng)
    kept = np.mean(out[:10000] == "V")
    fake = np.mean(out[10000:] == "V")
    check("keeps ventricular beats at the given sensitivity", abs(kept - 0.9) < 0.01, f"{kept:.3f}")
    check("adds false ventricular beats at the given rate", abs(fake - 0.01) < 0.002, f"{fake:.4f}")

    print("\nEnd to end on a synthetic dataset\n")
    import wfdb
    fs = 250
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        d = tmp / "raw" / "ltafdb"
        d.mkdir(parents=True)
        for k in range(12):
            s = list("N" * 200 + "VVVV" + "N" * 200 + "VVVVVVVV" + "N" * 200)
            t = np.cumsum([{"N": 200, "V": 110}[c] for c in s]) + 50
            wfdb.wrsamp(f"{k:02d}", fs=fs, units=["mV"], sig_name=["ECG"],
                        p_signal=np.zeros((int(t[-1]) + 100, 1)), fmt=["16"], write_dir=str(d))
            wfdb.wrann(f"{k:02d}", "atr", t, symbol=s, write_dir=str(d))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vs.main(["--datasets", "ltafdb", "--raw", str(tmp / "raw")])
        text = buf.getvalue()
    rows = [l.split() for l in text.splitlines() if l.strip().startswith(("80%", "90%", "95%", "99%"))]
    check("runs and prints a row per setting", rc == 0 and len(rows) == 12, f"{len(rows)} rows")
    best = [r for r in rows if r[0] == "99%" and r[1] == "0.1%"][0]
    worst = [r for r in rows if r[0] == "80%" and r[1] == "1.0%"][0]
    check("near-perfect beats find nearly every run", int(best[2].rstrip("%")) >= 90, best)
    check("worse beats find fewer runs", int(worst[2].rstrip("%")) < int(best[2].rstrip("%")),
          f"{worst[2]} vs {best[2]}")
    sym = np.array(["N"] * 5000)
    samp = np.cumsum(np.full(5000, 100))            # 150 bpm sinus, the worst case
    pred = vs.find_runs(vs.corrupt(sym, 0.9, 0.3, np.random.default_rng(1)), samp, 250)
    _, false = vs.score([], pred)
    check("frequent false PVCs at a fast rate make false runs", false > 0, f"{false} false runs")
    check("long runs survive better than short ones at 80%",
          int(worst[4].rstrip("%")) >= int(worst[3].rstrip("%")), f"5+ {worst[4]}, 3-4 {worst[3]}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
