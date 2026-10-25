"""Checks eval/af_alarm_rate.py: the counting, and a run on a small fake world.

Fake nsrdb (healthy, no AF) and ltafdb (an annotated AF episode) recordings,
a matching rhythm cache and out-of-fold file, untrained fold models, all in a
temporary folder. Nothing in the repo is read or written.

    python tests/test_af_alarm_rate.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.eval import af_alarm_rate as af
from munji_ai.models.rhythm_net import build

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ecg(seconds, fs, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = np.zeros(n)
    t = 0.4
    while t < seconds - 0.5:
        c = int(t * fs)
        w = max(int(0.01 * fs), 1)
        lo, hi = max(c - 4 * w, 0), min(c + 4 * w, n)
        x[lo:hi] += np.exp(-0.5 * ((np.arange(lo, hi) - c) / w) ** 2)
        t += 0.85
    return x + rng.normal(0, 0.03, n)


def main() -> int:
    import wfdb
    from munji_ai.data.windows import config_hash

    print("Counting\n")
    P = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
    T = np.zeros_like(P)
    c = af.count(P, T, (2, 3))
    check("two separate alerts on a non-AF stretch are two false alarms",
          (c["false"], c["true"]) == (2, 0))
    T2 = T.copy()
    T2[1:4] = True
    c = af.count(P, T2, (2, 3))
    check("an alert during annotated AF is true, and the episode is caught",
          (c["true"], c["caught"], c["episodes"]) == (1, 1, 1))
    check("burden overlap is counted window by window",
          (c["truth_w"], c["overlap_w"]) == (3, 2))
    L = np.zeros(40, bool)
    L[2:30] = True                                   # 28 windows: well over 6 min
    c = af.count(L.copy(), L, (2, 3))
    check("an episode of 6 minutes or more is counted as long",
          (c["long_episodes"], c["long_caught"]) == (1, 1))

    print("\nEnd to end on a fake world\n")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, models = tmp / "raw", tmp / "cache", tmp / "models"
        for d in (raw / "nsrdb", raw / "ltafdb", cache, models):
            d.mkdir(parents=True)
        x = ecg(1200, 128, seed=1)
        wfdb.wrsamp("16265", fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                    p_signal=np.stack([x, x], 1), fmt=["16", "16"],
                    write_dir=str(raw / "nsrdb"))
        x = ecg(1800, 128, seed=2)
        wfdb.wrsamp("00", fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                    p_signal=np.stack([x, x], 1), fmt=["16", "16"],
                    write_dir=str(raw / "ltafdb"))
        wfdb.wrann("00", "atr", np.array([10, 300 * 128, 1200 * 128]),
                   symbol=["+", "+", "+"], aux_note=["(N", "(AFIB", "(N"],
                   write_dir=str(raw / "ltafdb"))

        rec = np.array(["16265"] * 6 + ["00"] * 6)
        dset = np.array(["nsrdb"] * 6 + ["ltafdb"] * 6)
        y = np.array(["NSR"] * 6 + ["AF"] * 3 + ["NSR"] * 3)
        for split, sl in (("train", slice(0, 6)), ("val", slice(6, 12))):
            np.savez(cache / f"rhythm_{split}.npz",
                     X=rng.standard_normal((6, 7500)).astype(np.float16), y=y[sl],
                     dataset=dset[sl], patient=np.char.add(dset[sl], ":" + rec[sl]),
                     record=rec[sl], offset=np.arange(6) * 3750, cfg_hash=config_hash())
        np.savez(models / "rhythm_oof.npz", y=y, pred=y, score=rng.random(12),
                 record=rec, offset=np.tile(np.arange(6) * 3750, 2),
                 patient=rec, fold=np.array([0] * 6 + [1] * 6))
        torch.manual_seed(0)
        for k in (0, 1):
            torch.save(build().state_dict(), models / f"rhythm_fold{k}.pt")

        check("each record maps to the fold that held it out",
              af.fold_of_records(models / "rhythm_oof.npz", cache)
              == {("nsrdb", "16265"): 0, ("ltafdb", "00"): 1})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = af.main(["--models", str(models), "--cache", str(cache), "--raw", str(raw),
                          "--rules", "2/3", "20/24", "--thresholds", "0.3", "0.5"])
        text = buf.getvalue()
        rows = [l for l in text.splitlines() if l.strip().startswith(("0.30", "0.50"))]
        check("runs end to end, one row per threshold and rule",
              rc == 0 and len(rows) == 4, f"{len(rows)} rows")
        check("both sources get their own false-alarm column",
              "nsrdb f/24h" in text and "ltafdb f/24h" in text)
        rec_ = __import__("munji_ai.data.loader", fromlist=["load_record"]).load_record(
            "ltafdb", "00", raw)
        s = np.zeros(len(af.window_scores(build().eval(), rec_.signal, torch.device("cpu"))))
        t = af.truth_flags(rec_, len(s))
        check("the annotated 15-minute AF stretch becomes one episode",
              af.count(t.copy(), t, (2, 3))["episodes"] == 1 and t.sum() > 50,
              f"{int(t.sum())} AF windows")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
