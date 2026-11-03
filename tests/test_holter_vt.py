"""Checks data/holter_vt.py on synthetic annotated recordings.

    python tests/test_holter_vt.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data import holter_vt as hv

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def beats(x, fs, start, stop, bpm, width):
    t = start
    while t < stop - 0.3:
        c, w = int(t * fs), max(int(width * fs), 1)
        lo, hi = max(c - 4 * w, 0), min(c + 4 * w, len(x))
        x[lo:hi] += np.exp(-0.5 * ((np.arange(lo, hi) - c) / w) ** 2)
        t += 60 / bpm


def main() -> int:
    import wfdb
    from munji_ai.data.windows import config_hash

    fs, rng = 250, np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, folds = tmp / "raw", tmp / "cache", tmp / "folds"
        for d in (raw / "vfdb", raw / "fantasia", cache, folds):
            d.mkdir(parents=True)
        for name, kind in (("418", "VF"), ("419", "VT")):
            n = fs * 90
            x = rng.normal(0, 0.01, n)
            beats(x, fs, 0.5, 30, 70, 0.01)
            seg = np.arange(30 * fs, 70 * fs)
            if kind == "VF":
                x[seg] = np.sin(2 * np.pi * 5 * seg / fs) * rng.random(len(seg))
            else:
                beats(x, fs, 30, 70, 130, 0.03)
            beats(x, fs, 70, 90, 70, 0.01)
            wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                        fmt=["16"], write_dir=str(raw / "vfdb"))
            wfdb.wrann(name, "atr", np.array([10, 30 * fs, 70 * fs]), symbol=["+"] * 3,
                       aux_note=["(N", f"({kind}", "(N"], write_dir=str(raw / "vfdb"))
        x = rng.normal(0, 0.01, fs * 120)
        beats(x, fs, 0.5, 120, 65, 0.01)
        wfdb.wrsamp("f1o01", fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                    fmt=["16"], write_dir=str(raw / "fantasia"))
        rec = np.array(["418", "419", "f1o01"])
        ds = np.array(["vfdb", "vfdb", "fantasia"])
        np.savez(cache / "shockable_train.npz", X=np.zeros((3, 1250), np.float16),
                 y=np.array(["shockable"] * 3), dataset=ds, patient=rec, record=rec,
                 offset=np.arange(3), cfg_hash=config_hash())
        np.savez(cache / "shockable_val.npz", X=np.zeros((0, 1250), np.float16),
                 y=np.array([], dtype=str), dataset=np.array([], dtype=str),
                 patient=np.array([], dtype=str), record=np.array([], dtype=str),
                 offset=np.array([], dtype=int), cfg_hash=config_hash())
        np.savez(folds / "shockable_oof.npz", record=rec, fold=np.zeros(3, int),
                 y=np.array(["shockable"] * 3))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hv.main(["--datasets", "vfdb", "fantasia", "--folds-from", str(folds),
                          "--cache", str(cache), "--raw", str(raw), "--other-per-record", "5"])
        d = np.load(cache / "holter_vt_windows.npz")
    lab = d["label"]
    check("runs and saves", rc == 0 and d["X"].shape[1] == 2500, d["X"].shape)
    check("VT, VF and other all present", {"VT", "VF", "other"} <= set(lab), sorted(set(lab)))
    check("unannotated recording counts as other", set(lab[d["dataset"] == "fantasia"]) == {"other"})
    check("other capped per record", int((lab[d["group"] == "fantasia:f1o01"] == "other").sum()) <= 5)
    check("slow VT measured under 150 bpm", np.nanmedian(d["bpm"][lab == "VT"]) < 150,
          f"{np.nanmedian(d['bpm'][lab == 'VT']):.0f} bpm")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
