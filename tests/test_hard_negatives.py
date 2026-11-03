"""Checks data/hard_negatives.py (round 2: every record, Icentia + LTAFDB, resumable).

Builds a small fake Icentia11k folder and a fake LTAFDB folder, an untrained
fold model and a tiny shockable cache, all in a temporary folder. Nothing in
the repo is read or written.

    python tests/test_hard_negatives.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data import hard_negatives as hn
from munji_ai.data.splits import assign
from munji_ai.models.shockable_net import build

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
    return x + rng.normal(0, 0.05, n)


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = hn.main(argv)
    return rc, buf.getvalue()


def main() -> int:
    import wfdb
    from munji_ai.data.windows import config_hash

    pats = [f"p{i:05d}" for i in range(10)]
    test_pats = {p for p in pats if assign(f"icentia11k:{p}") == "test"}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, models, cache = tmp / "raw", tmp / "models", tmp / "cache"
        models.mkdir()
        cache.mkdir()
        for k, p in enumerate(pats):
            d = raw / "icentia11k" / "p00" / p
            d.mkdir(parents=True)
            for s in range(3):
                wfdb.wrsamp(f"{p}_s{s:02d}", fs=250, units=["mV"], sig_name=["ECG"],
                            p_signal=ecg(90, 250, seed=k * 10 + s)[:, None],
                            fmt=["16"], write_dir=str(d))
        (raw / "ltafdb").mkdir(parents=True)
        for k, name in enumerate(("00", "01", "03")):
            x = ecg(240, 128, seed=50 + k)
            wfdb.wrsamp(name, fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                        p_signal=np.stack([x, x], 1), fmt=["16", "16"],
                        write_dir=str(raw / "ltafdb"))
            wfdb.wrann(name, "atr", np.array([10, 60 * 128]), symbol=["+", "+"],
                       aux_note=["(N", "(AFIB"], write_dir=str(raw / "ltafdb"))
        torch.manual_seed(0)
        torch.save(build().state_dict(), models / "shockable_fold0.pt")

        print("Selecting recordings\n")
        g, skipped = hn.records_by_patient("icentia11k", raw, 0, None)
        check("every segment of a patient is used by default",
              all(len(v) == 3 for v in g.values()), f"{len(g)} patients")
        check("test-split patients are skipped", skipped == len(test_pats)
              and not any(pid.split(":")[1] in test_pats for pid in g))
        g2, _ = hn.records_by_patient("icentia11k", raw, 2, None)
        check("--segments limits records per patient", all(len(v) == 2 for v in g2.values()))
        gl, _ = hn.records_by_patient("ltafdb", raw, 0, None)
        check("LTAFDB: one patient per record",
              set(gl) == {"ltafdb:00", "ltafdb:01", "ltafdb:03"})

        print("\nMining both sources\n")
        out, work = tmp / "neg.npz", tmp / "parts"
        args = ["--models", str(models), "--raw", str(raw), "--hard", "0.5",
                "--per-patient-hard", "5", "--per-patient-random", "3",
                "--work-dir", str(work), "--out", str(out)]
        rc, text = run(args)
        with np.load(out, allow_pickle=False) as d:
            X, y = d["X"], d["y"].astype(str)
            ds, pat = d["dataset"].astype(str), d["patient"].astype(str)
        check("runs and writes the merged file", rc == 0 and len(X) > 0, f"{len(X)} windows")
        check("windows from both sources", set(ds) == {"icentia11k", "ltafdb"})
        check("every window is labelled not_shockable", set(y) == {"not_shockable"})
        check("stored as the cache dtype, 5 s at 250 Hz",
              X.dtype == np.dtype(C.CACHE_DTYPE) and X.shape[1] == C.window_samples("shockable"))
        per = {p: (pat == p).sum() for p in set(pat)}
        check("no patient gives more than hard + random", max(per.values()) <= 8,
              f"max {max(per.values())}")
        check("no test-split patient in the output",
              not any(p.split(":")[1] in test_pats for p in pat if p.startswith("icentia")))
        check("windows are normalised like the cache",
              np.abs(np.median(X.astype(np.float32), axis=1)).max() < 0.05)

        print("\nResume and merge\n")
        n_parts = len(list(work.glob("*.npz")))
        rc, text = run(args)
        check("a rerun skips every finished patient",
              rc == 0 and f"{n_parts} of {n_parts} patients already done" in text)
        out2 = tmp / "merged.npz"
        rc, _ = run(["--merge-only", "--work-dir", str(work), "--out", str(out2)])
        with np.load(out2, allow_pickle=False) as d:
            n2 = len(d["X"])
        check("--merge-only rebuilds the same file", rc == 0 and n2 == len(X))

        print("\ncross_validate --extra-negatives\n")
        from munji_ai.train import cross_validate as cv
        rng = np.random.default_rng(0)
        for split, n in (("train", 48), ("val", 16)):
            np.savez(cache / f"shockable_{split}.npz",
                     X=rng.standard_normal((n, 1250)).astype(np.float32),
                     y=np.array(["shockable", "not_shockable"] * (n // 2)),
                     patient=np.array([f"cudb:cu{i % 8:02d}" for i in range(n)]),
                     record=np.array([f"cu{i % 8:02d}" for i in range(n)]),
                     offset=np.arange(n) * 625, dataset=np.array(["cudb"] * n),
                     cfg_hash=config_hash())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cv.main(["--stage", "shockable", "--folds", "2", "--epochs", "1",
                          "--batch", "16", "--cache", str(cache), "--device", "cpu",
                          "--out", str(cache / "out"), "--extra-negatives", str(out)])
        check("cross_validate trains with them", rc == 0 and "extra neg" in buf.getvalue())

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
