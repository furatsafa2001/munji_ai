"""The daily-life test of the fast narrow rhythm label (Icentia11k), on
synthetic Icentia-style recordings, without the real data.

Three patients: one whose sinus rhythm climbs slowly to 160 with effort, one
with an SVT that jumps from 75 to 180 although its beats are marked normal
(an SVT the technologists missed), and one at rest. The checks are about
plumbing: the label row, the details, the Poisson bound, and the onset of the
fast 'sinus' episodes read off the annotated beats.

    python tests/test_fast_narrow_icentia.py
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

spec = importlib.util.spec_from_file_location("fast_narrow_icentia", ROOT / "experiments" / "fast_narrow_icentia.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
tspec = importlib.util.spec_from_file_location("tfn", ROOT / "tests" / "test_fast_narrow_template.py")
TT = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(TT)

from munji_ai.data import svt_runs as SR  # noqa: E402
from munji_ai.data.splits import assign  # noqa: E402

ok = True
REST = (75, 75, 150, 12, 0)
EFFORT = [REST, (75, 160, 240, 12, 0), (160, 160, 90, 12, 0), (160, 80, 120, 12, 0)]
HIDDEN = [REST, (180, 180, 90, 12, 0), (75, 75, 90, 12, 0)]
CALM = [REST, (72, 80, 300, 12, 0)]


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def free_patients(n):
    out, k = [], 1
    while len(out) < n:
        pid = f"p{k:05d}"
        if assign(f"icentia11k:{pid}") != "test":
            out.append(pid)
        k += 1
    return out


def write(root: Path, pid: str, parts, seed: int) -> None:
    import wfdb
    x, _, bt = TT.ecg(parts, seed)
    d = root / "icentia11k" / "p00" / pid
    d.mkdir(parents=True, exist_ok=True)
    name = f"{pid}_s00"
    wfdb.wrsamp(name, fs=250, units=["mV"], sig_name=["ECG"], p_signal=x[:, None], fmt=["16"], write_dir=str(d))
    aux = [""] * len(bt)
    aux[0] = "(N"
    wfdb.wrann(name, "atr", np.round(bt * 250).astype(int), symbol=["N"] * len(bt), aux_note=aux, write_dir=str(d))


def main() -> int:
    print("pieces")
    up = E.poisson_upper(0, 10.0)
    check("Poisson: 0 events in 10 days gives an upper bound near 0.37 per day", abs(up - 0.369) < 0.01, f"{up:.3f}")
    check("Poisson: grows with the count", E.poisson_upper(5, 10.0) > E.poisson_upper(1, 10.0) > up)

    print("recordings")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        raw = Path(tmp) / "raw"
        eff, hid, calm = free_patients(3)
        write(raw, eff, EFFORT, 1)
        write(raw, hid, HIDDEN, 2)
        write(raw, calm, CALM, 3)
        with contextlib.redirect_stdout(io.StringIO()):
            SR.main(["--raw", str(raw), "--cache", str(Path(tmp) / "cache")])
        z = np.load(Path(tmp) / "cache" / "svt_runs.npz")
        check("svt_runs sees two fast 'sinus' episodes (the effort and the hidden SVT)",
              int((z["kind"] == "sinus").sum()) == 2, str(list(z["record"][z["kind"] == "sinus"])))

        r_eff = E.run_one(f"{eff}_s00", raw)
        r_hid = E.run_one(f"{hid}_s00", raw)
        check("effort: no label in the PRIMARY row", not r_eff["labels"][E.PRIMARY], str(r_eff["labels"]))
        v = next(iter(r_hid["detail"].values()), {})
        check("the hidden SVT: the PRIMARY row labels it, with its details", len(r_hid["labels"][E.PRIMARY]) == 1
              and v.get("bpm", 0) > 170 and v.get("before", 999) < 150 and v.get("drop", 0) >= 0.09
              and v.get("kind") == "false:NSR" and v.get("near", "").startswith("N "), str(v))

        on_eff = E.onset_on_annotations(raw / "icentia11k" / "p00" / eff / f"{eff}_s00",
                                        *[int(q) for q in (z["start"][z["record"] == f"{eff}_s00"][0],
                                                           z["end"][z["record"] == f"{eff}_s00"][0])])
        on_hid = E.onset_on_annotations(raw / "icentia11k" / "p00" / hid / f"{hid}_s00",
                                        *[int(q) for q in (z["start"][z["record"] == f"{hid}_s00"][0],
                                                           z["end"][z["record"] == f"{hid}_s00"][0])])
        check("onset on the annotations: effort gradual, the hidden SVT abrupt with 150 at once",
              on_eff["drop100"] < 0.09 and on_eff["reach"] > 60 and on_hid["drop100"] >= 0.09 and on_hid["reach"] < 5
              and on_hid["before"] < 100, f"{on_eff} / {on_hid}")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = E.main(["--raw", str(raw), "--svt-runs", str(Path(tmp) / "cache" / "svt_runs.npz"),
                           "--workers", "1"])
        text = buf.getvalue()
        check("end to end: 3 recordings, the label table, the verdict (one label in a short span: FAIL), the episodes",
              "3 recordings from 3 patients" in text and "Labels per row" in text and "->  FAIL" in text
              and code == 2 and "Fast 'sinus' episodes from svt_runs (2)" in text and "1 of 2 started abruptly" in text,
              text[-1800:])
        with contextlib.redirect_stdout(io.StringIO()):
            code = E.main(["--raw", str(Path(tmp) / "none"), "--workers", "1"])
        check("no data: says so", code == 1)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
