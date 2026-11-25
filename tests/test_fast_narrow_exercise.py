"""The exercise test of the fast narrow rhythm label, on synthetic recordings,
without PhysioNet or GitHub.

Gaussian beats written as GUDB's tab-separated files and as an STDB-style
WFDB record: a slow exercise ramp, a fast jog like a fit subject's (a leap
past 100 bpm, then 150 within half a minute), a recording that starts
already fast, and the downloads with fake servers. The checks are about
plumbing; they say nothing about real ECG.

    python tests/test_fast_narrow_exercise.py
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

spec = importlib.util.spec_from_file_location("fast_narrow_exercise", ROOT / "experiments" / "fast_narrow_exercise.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
tspec = importlib.util.spec_from_file_location("tfn", ROOT / "tests" / "test_fast_narrow_template.py")
TT = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(TT)

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


REST = (75, 75, 150, 12, 0)
SLOW = [REST, (75, 160, 240, 12, 0), (160, 160, 60, 12, 0), (160, 80, 120, 12, 0)]
# a fit subject jogging: 82 to 108 in 10 s, 150 some 20 s later, on to 165
JOG = [REST, (82, 108, 10, 12, 0), (108, 152, 20, 12, 0), (152, 165, 60, 12, 0), (165, 165, 30, 12, 0)]
FAST_START = [(130, 130, 60, 12, 0), (130, 165, 30, 12, 0), (165, 165, 60, 12, 0)]
# not exercise: a jump from 75 to 175 at once, the case every row must label
JUMP = [REST, (175, 175, 90, 12, 0), (80, 80, 60, 12, 0)]


def write_gudb(root: Path, subject: int, task: str, parts, seed=0) -> Path:
    x, _, _ = TT.ecg(parts, seed)
    p = root / f"subject_{subject:02d}" / task / "ECG.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = np.zeros((x.size, 6))
    cols[:, 0] = x / 1000.0                       # GUDB stores volts
    np.savetxt(p, cols, delimiter="\t")
    return p


def write_stdb(root: Path, name: str, parts, seed=0) -> Path:
    import wfdb
    from scipy.signal import resample_poly
    x, _, bt = TT.ecg(parts, seed)
    x360 = resample_poly(x, 36, 25)
    root.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(name, fs=360, units=["mV"], sig_name=["ECG0"], p_signal=x360[:, None], fmt=["16"],
                write_dir=str(root))
    sym = ["N"] * len(bt)
    sym[len(bt) // 2] = "V"
    wfdb.wrann(name, "atr", np.round(bt * 360).astype(int), symbol=sym, write_dir=str(root))
    return root / f"{name}.hea"


def main() -> int:
    print("downloads")
    calls = []

    def fake(url, dest, files=None):
        calls.append(url)
        if url.endswith("RECORDS"):
            if files is None:
                return "missing"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(files) + "\n")
        return "fetched"
    with tempfile.TemporaryDirectory() as d:
        n = E.download_stdb(Path(d), fetch=fake)
        check("STDB without RECORDS: 300 to 327, three files each", n == {"fetched": 84}
              and calls[1].endswith("300.hea") and calls[-1].endswith("327.dat"), str(n))
        calls.clear()
        n = E.download_stdb(Path(d), fetch=lambda u, p: fake(u, p, ["301", "305"]))
        check("STDB with RECORDS: only what it lists", n == {"fetched": 6} and len(calls) == 7, str(n))
        calls.clear()

        def flaky(url, dest):
            calls.append(url)
            return "failed: HTTP 503" if "raw.githubusercontent" in url else "fetched"
        n = E.download_gudb(Path(d), tasks=("jogging",), fetch=flaky)
        check("GUDB: the second address when the first fails, 25 subjects", n == {"fetched": 25}
              and calls[0].endswith("subject_00/jogging/ECG.tsv") and "github.io" in calls[1], str(n))

    print("readers")
    with tempfile.TemporaryDirectory() as d:
        p = write_gudb(Path(d), 3, "jogging", JOG, seed=1)
        rec = E.read_gudb(p)
        check("GUDB: chest strap in mV on the model path, sinus, named s03_jogging",
              rec.name == "s03_jogging" and rec.dataset == "gudb" and rec.fs == 250 and abs(rec.signal).max() > 0.3
              and list(rec.rhythm_labels) == ["(N"], f"{rec.name} {abs(rec.signal).max():.2f}")
        check("GUDB: empty files are not listed", len(E.gudb_records(Path(d))) == 1)
        h = write_stdb(Path(d) / "stdb", "390", SLOW, seed=2)
        rs = E.read_stdb(h)
        import wfdb
        hd = wfdb.rdheader(str(h.with_suffix("")))
        check("STDB: 360 Hz read at 250 for the same length, beat labels kept", rs.fs == 250
              and "V" in set(rs.beat_symbols) and abs(len(rs.signal) / 250 - hd.sig_len / hd.fs) < 0.05
              and rs.beat_samples[-1] < len(rs.signal), f"{len(rs.signal) / 250:.2f} vs {hd.sig_len / hd.fs:.2f}")
        check("STDB: beats around a time", "N" in E.beats_near(rs, 300.0) and E.beats_near(rs, 1e6) == "none")
        check("STDB records found", [q.stem for q in E.stdb_records(Path(d) / "stdb")] == ["390"])

    print("recordings")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        r_jog = E.run_one(write_gudb(root, 4, "jogging", JOG, seed=3), "gudb")
        check("a fast jog: informative, an abrupt leap past 100; row 7 labels it, 'slower before' does not",
              r_jog["informative"] and r_jog["drop"] >= 0.09 and r_jog["reach"] <= 30
              and len(r_jog["labels"][E.FIRST]) == 1 and not r_jog["labels"][E.PRIMARY]
              and not r_jog["labels"][E.G.LORENZ_150],
              f"{r_jog['drop']:.2f} {r_jog['reach']:.0f} {r_jog['labels'][E.FIRST]} {r_jog['labels'][E.PRIMARY]}")
        r_jump = E.run_one(write_gudb(root, 7, "jogging", JUMP, seed=8), "gudb")
        check("an abrupt jump to 175 (not exercise): informative, and the PRIMARY labels it",
              r_jump["informative"] and len(r_jump["labels"][E.PRIMARY]) == 1, str(r_jump["labels"][E.PRIMARY]))
        r_slow = E.run_one(write_gudb(root, 5, "jogging", SLOW, seed=4), "gudb")
        check("a slow exercise ramp: informative, no alert in any label row; the doctor's report enters it (its"
              " cost: no onset check)", r_slow["informative"]
              and not any(v for k, v in r_slow["labels"].items() if k.startswith("label:"))
              and len(r_slow["labels"][E.G.DOCTOR]) >= 1, str(r_slow["labels"]))
        r_fs = E.run_one(write_gudb(root, 6, "jogging", FAST_START, seed=5), "gudb")
        check("a recording that starts fast: not informative, no label", not r_fs["informative"]
              and not any(r_fs["labels"].values()), str(r_fs))

        with contextlib.redirect_stdout(io.StringIO()) as out:
            code_fail = E.report([r_jump, r_slow, r_fs])
            code_pass = E.report([r_jog, r_slow, r_fs])
            code_none = E.report([r_fs])
        text = out.getvalue()
        check("verdict: FAIL with the jump, PASS with the jog (row 7 reported as failing), not run with nothing"
              " informative", (code_fail, code_pass, code_none) == (2, 0, 1) and "->  FAIL" in text
              and "->  PASS" in text and "PRIMARY not run" in text and f"reported, '{E.FIRST}': 1 labels" in text,
              str((code_fail, code_pass, code_none)))

        (root / "stdb").mkdir()
        write_stdb(root / "stdb", "391", SLOW, seed=6)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = E.main(["--stdb", str(root / "stdb"), "--gudb", str(root)])
        text = out.getvalue()
        check("end to end: both sets read, the jump fails the verdict", code == 2 and "stdb: 1 recordings" in text
              and "gudb: 4 recordings" in text, text[-600:])
        with contextlib.redirect_stdout(io.StringIO()):
            code = E.main(["--stdb", str(root / "none"), "--gudb", str(root / "none")])
        check("no recordings: says so, not run", code == 1)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
