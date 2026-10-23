"""Checks the false-alarm-rate evaluation added by fix_false_alarm_rate.py.

Builds a tiny fake world in a temporary folder: two short WFDB recordings,
a matching cache and out-of-fold file, and untrained fold models. Nothing
real is read and nothing in the repo is written.

    python tests/test_false_alarm_rate.py
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
from munji_ai.data import preprocess as pp
from munji_ai.eval import false_alarm_rate as far
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
    return x + rng.normal(0, 0.01, n)


def main() -> int:
    rule = C.CONFIRMATION["vf"]

    print("Alarm counting (2 of 3)\n")
    P = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
    T = np.zeros_like(P)
    f_, t_ = far.count_alarms(P, T, rule)
    check("one confirmed run is one alarm, scattered positives fire again",
          (f_, t_) == (2, 0), f"false {f_}, true {t_}")
    T2 = T.copy()
    T2[1:4] = True
    f2, t2 = far.count_alarms(P, T2, rule)
    check("an alarm overlapping a real episode counts as true", (f2, t2) == (1, 1),
          f"false {f2}, true {t2}")
    check("no positives, no alarms",
          far.count_alarms(np.zeros(20, bool), np.zeros(20, bool), rule) == (0, 0))

    print("\nWindowing\n")
    sig = ecg(60, C.TARGET_FS)
    W = np.stack([sig[i:i + 1250] for i in (0, 625, 1250)])
    ref = np.stack([pp.normalize(w) for w in W])
    check("row normalisation matches preprocess.normalize",
          np.allclose(far._normalize_rows(W), ref, atol=1e-9))
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    s = far.window_scores(build().eval(), sig, torch.device("cpu"))
    expected = 1 + (len(sig) - win) // stride
    check("one score per deployment-stride window", len(s) == expected,
          f"{len(s)} windows for 60 s")
    check("scores are probabilities", ((s >= 0) & (s <= 1)).all())

    print("\nEnd to end on a fake world\n")
    import wfdb
    from munji_ai.data.windows import config_hash

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        raw, cache, models = tmp / "raw", tmp / "cache", tmp / "models"
        (raw / "nsrdb").mkdir(parents=True)
        cache.mkdir()
        models.mkdir()
        for i, name in enumerate(("16265", "16272")):
            x = ecg(600, 128, seed=i)
            wfdb.wrsamp(name, fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"],
                        p_signal=np.stack([x, x], axis=1), fmt=["16", "16"],
                        write_dir=str(raw / "nsrdb"))

        recs = np.array(["16265"] * 6 + ["16272"] * 6)
        for split, sl in (("train", slice(0, 8)), ("val", slice(8, 12))):
            n = sl.stop - sl.start
            np.savez(cache / f"shockable_{split}.npz",
                     X=np.zeros((n, 1250), np.float32), y=np.array(["not_shockable"] * n),
                     dataset=np.array(["nsrdb"] * n), patient=recs[sl], record=recs[sl],
                     offset=np.arange(n) * 625, cfg_hash=config_hash())
        np.savez(models / "shockable_oof.npz", y=np.array(["not_shockable"] * 12),
                 pred=np.array(["not_shockable"] * 12), score=np.zeros(12),
                 patient=recs, record=recs, offset=np.arange(12) * 625,
                 fold=np.array([0] * 6 + [1] * 6))
        torch.manual_seed(0)
        for f in (0, 1):
            torch.save(build().state_dict(), models / f"shockable_fold{f}.pt")

        folds = far.fold_of_records(models / "shockable_oof.npz", cache)
        check("each record maps to the fold that held its patient out",
              folds == {("nsrdb", "16265"): 0, ("nsrdb", "16272"): 1})

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = far.main(["--models", str(models), "--cache", str(cache),
                           "--raw", str(raw), "--datasets", "nsrdb",
                           "--thresholds", "0.5", "0.6"])
        text = buf.getvalue()
        check("runs end to end", rc == 0 and "per 24 h" in text)
        check("both recordings scored, 20 minutes in total",
              "nsrdb            2     0.3" in text)
        check("a table per threshold", text.count("threshold 0.") == 2)

    print("\ncross_validate --save-models\n")
    from munji_ai.train import cross_validate as cv
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = Path(tmp)
        for split, n in (("train", 48), ("val", 16)):
            np.savez(cache / f"shockable_{split}.npz",
                     X=rng.standard_normal((n, 1250)).astype(np.float32),
                     y=np.array(["shockable", "not_shockable"] * (n // 2)),
                     patient=np.array([f"p{split}{i % 8}" for i in range(n)]),
                     record=np.array([f"r{split}{i % 8}" for i in range(n)]),
                     offset=np.arange(n) * 625, dataset=np.array(["x"] * n),
                     cfg_hash=config_hash())
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cv.main(["--stage", "shockable", "--folds", "2", "--epochs", "1",
                          "--batch", "16", "--cache", str(cache), "--device", "cpu",
                          "--save-models", "--out", str(cache / "out")])
        saved = sorted(p.name for p in (cache / "out").glob("shockable_fold*.pt"))
        check("one model file per fold", rc == 0 and saved ==
              ["shockable_fold0.pt", "shockable_fold1.pt"], ", ".join(saved))
        m = build()
        m.load_state_dict(torch.load(cache / "out" / "shockable_fold0.pt",
                                     map_location="cpu", weights_only=True))
        check("a saved fold model loads back into ShockableNet", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
