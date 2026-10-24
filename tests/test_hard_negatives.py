"""Checks the hard-negative mining added by fix_hard_negatives.py.

Builds a small fake Icentia11k folder (WFDB records named like the real
p00000_s00), an untrained fold model, and a tiny shockable cache, all in a
temporary folder. Nothing in the repo is read or written.

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


def ecg(seconds, fs, hr=70.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = np.zeros(n)
    t = 0.4
    while t < seconds - 0.5:
        c = int(t * fs)
        w = max(int(0.01 * fs), 1)
        lo, hi = max(c - 4 * w, 0), min(c + 4 * w, n)
        x[lo:hi] += np.exp(-0.5 * ((np.arange(lo, hi) - c) / w) ** 2)
        t += 60.0 / hr
    return x + rng.normal(0, 0.05, n)


def main() -> int:
    import wfdb
    from munji_ai.data.windows import config_hash

    # Pick fake patient numbers so that at least one lands in the test split.
    pats = [f"p{i:05d}" for i in range(12)]
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
                            p_signal=ecg(120, 250, seed=k * 10 + s)[:, None],
                            fmt=["16"], write_dir=str(d))
        torch.manual_seed(0)
        torch.save(build().state_dict(), models / "shockable_fold0.pt")

        print("Selecting recordings\n")
        groups, skipped = hn.records_by_patient(raw, segments=2, max_patients=None)
        check("test-split patients are skipped", skipped == len(test_pats)
              and not any(pid.split(":")[1] in test_pats for pid in groups),
              f"{skipped} skipped")
        check("at most 2 segments per patient, the first ones",
              all(r == sorted(r)[:2] and len(r) <= 2 for r in groups.values()))

        print("\nMining\n")
        out = tmp / "neg.npz"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hn.mine(models, raw, out, segments=2, hard_thr=0.5, per_patient_hard=5,
                    per_patient_random=3, fold=0, max_patients=None,
                    device=torch.device("cpu"))
        with np.load(out, allow_pickle=False) as d:
            X, y, pat, sc = d["X"], d["y"].astype(str), d["patient"].astype(str), d["score"]
            files = d.files
        check("output looks like a cache split", {"X", "y", "dataset", "patient",
              "record", "offset", "score"} <= set(files))
        check("windows are 5 s at 250 Hz, stored as the cache dtype",
              X.shape[1] == C.window_samples("shockable") and X.dtype == np.dtype(C.CACHE_DTYPE))
        check("every window is labelled not_shockable", set(y) == {"not_shockable"})
        per_pat = {p: (pat == p).sum() for p in set(pat)}
        check("no patient gives more than hard + random windows",
              max(per_pat.values()) <= 8, f"max {max(per_pat.values())}")
        check("no test-split patient in the output",
              not any(p.split(":")[1] in test_pats for p in pat))
        med = np.median(X.astype(np.float32), axis=1)
        check("windows are normalised like the cache", np.abs(med).max() < 0.05)
        check("the summary reports the raw false-positive rate",
              "the raw false-positive rate" in buf.getvalue())

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
        base = ["--stage", "shockable", "--folds", "2", "--epochs", "1",
                "--batch", "16", "--cache", str(cache), "--device", "cpu",
                "--out", str(cache / "out")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cv.main(base + ["--extra-negatives", str(out)])
        check("trains with extra negatives", rc == 0 and "extra neg" in buf.getvalue())
        with np.load(cache / "out" / "shockable_oof.npz", allow_pickle=False) as d:
            n_oof = len(d["y"])
        check("extra windows never enter the held-out predictions", n_oof == 64,
              f"{n_oof} OOF windows")

        leak = tmp / "leak.npz"
        np.savez(leak, X=X[:4], y=y[:4], patient=np.array(["cudb:cu01"] * 4),
                 dataset=np.array(["cudb"] * 4), record=np.array(["cu01"] * 4),
                 offset=np.zeros(4, dtype=np.int64), score=sc[:4])
        refused = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                cv.main(base + ["--extra-negatives", str(leak)])
            except SystemExit:
                refused = True
        check("negatives sharing a patient with the evaluation are refused", refused)

        refused = False
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                cv.main(["--stage", "rhythm", "--extra-negatives", str(out)])
            except SystemExit:
                refused = True
        check("--extra-negatives is refused for the rhythm stage", refused)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
