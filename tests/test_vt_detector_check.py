"""Checks eval/vt_detector_check.py on synthetic arrest and daily recordings.

    python tests/test_vt_detector_check.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval import vt_detector_check as vc
from munji_ai.models.shockable_net import build

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

    print("Helpers\n")
    check("2 of 3 fires once per episode",
          vc.count_alerts(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], bool), 2, 3) == 2)
    check("isolated positives never fire",
          vc.count_alerts(np.array([1, 0, 0, 1, 0, 0, 1], bool), 2, 3) == 0)

    print("\nEnd to end\n")
    fs, rng = 250, np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, folds = tmp / "raw", tmp / "cache", tmp / "folds"
        for d in (raw / "vfdb", raw / "nsrdb", cache, folds):
            d.mkdir(parents=True)
        for name, kind in (("418", "VF"), ("419", "VT")):
            n = fs * 60
            x = rng.normal(0, 0.01, n)
            beats(x, fs, 0.5, 10, 70, 0.01)
            seg = np.arange(10 * fs, 50 * fs)
            if kind == "VF":
                x[seg] = np.sin(2 * np.pi * 5 * seg / fs) * rng.random(len(seg)) + rng.normal(0, .2, len(seg))
            else:
                beats(x, fs, 10, 50, 180, 0.03)
            beats(x, fs, 50, 60, 70, 0.01)
            wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                        fmt=["16"], write_dir=str(raw / "vfdb"))
            wfdb.wrann(name, "atr", np.array([10, 10 * fs, 50 * fs]), symbol=["+"] * 3,
                       aux_note=["(N", f"({kind}", "(N"], write_dir=str(raw / "vfdb"))
        x = rng.normal(0, 0.01, fs * 600)
        beats(x, fs, 0.5, 600, 72, 0.01)
        wfdb.wrsamp("16265", fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                    fmt=["16"], write_dir=str(raw / "nsrdb"))
        wfdb.wrann("16265", "atr", np.arange(125, fs * 599, 208), symbol=["N"] * len(np.arange(125, fs * 599, 208)),
                   write_dir=str(raw / "nsrdb"))
        rec = np.array(["418", "419", "16265"])
        ds = np.array(["vfdb", "vfdb", "nsrdb"])
        np.savez(cache / "shockable_train.npz", X=np.zeros((3, 1250), np.float16),
                 y=np.array(["shockable"] * 3), dataset=ds, patient=rec, record=rec,
                 offset=np.arange(3), cfg_hash=config_hash())
        np.savez(cache / "shockable_val.npz", X=np.zeros((0, 1250), np.float16),
                 y=np.array([], dtype=str), dataset=np.array([], dtype=str), patient=np.array([], dtype=str),
                 record=np.array([], dtype=str), offset=np.array([], dtype=int), cfg_hash=config_hash())
        np.savez(folds / "shockable_oof.npz", record=rec, fold=np.zeros(3, int), y=np.array(["shockable"] * 3))
        torch.manual_seed(0)
        torch.save({"state_dict": build().state_dict(), "window_s": 10.0}, tmp / "vt.pt")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vc.main(["--model", str(tmp / "vt.pt"), "--folds-from", str(folds),
                          "--cache", str(cache), "--raw", str(raw), "--arrest", "vfdb",
                          "--daily", "nsrdb", "--thresholds", "0.5", "--rules", "2/3"])
        text = buf.getvalue()
    check("runs end to end", rc == 0 and "false alerts per 24 h" in text, text.splitlines()[-1])
    line = [l for l in text.splitlines() if "windows    VF" in l]
    nums = [int(t.replace(",", "")) for t in line[0].split() if t.replace(",", "").isdigit()] if line else []
    check("finds both VF and VT windows", len(nums) >= 2 and nums[0] > 0 and nums[1] > 0, line)
    check("reports daily hours", "nsrdb, 0 h" in text or "nsrdb," in text)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
