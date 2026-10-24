"""Checks the final-training changes made by fix_final_training.py.

Tiny fake caches, a fake noise bank and fake extra negatives in a temporary
folder, a couple of epochs on CPU. Nothing in the repo is read or written.

    python tests/test_final_training.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data.windows import config_hash
from munji_ai.train import train_rhythm, train_shockable

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def cache_split(path, n, length, labels, prefix, rng):
    y = np.array([labels[i % len(labels)] for i in range(n)])
    np.savez(path, X=rng.standard_normal((n, length)).astype(np.float16), y=y,
             patient=np.array([f"{prefix}:p{i % 6}" for i in range(n)]),
             record=np.array([f"{prefix}{i % 6}" for i in range(n)]),
             offset=np.arange(n) * 625, dataset=np.array([prefix] * n),
             cfg_hash=config_hash())


def run(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(argv)
    return rc, buf.getvalue()


def main() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache"
        cache.mkdir()
        cache_split(cache / "shockable_train.npz", 60, 1250,
                    ["shockable", "not_shockable"], "trn", rng)
        cache_split(cache / "shockable_val.npz", 24, 1250,
                    ["shockable", "not_shockable"], "val", rng)
        cache_split(cache / "rhythm_train.npz", 24, 7500, ["NSR", "AF", "OTHER"], "rtr", rng)
        cache_split(cache / "rhythm_val.npz", 12, 7500, ["NSR", "AF", "OTHER"], "rva", rng)
        np.savez(cache / "noise_bank.npz", fs=250,
                 em=rng.standard_normal(30000).astype(np.float32),
                 ma=rng.standard_normal(30000).astype(np.float32))
        extra = tmp / "neg.npz"
        np.savez(extra, X=rng.standard_normal((20, 1250)).astype(np.float16),
                 y=np.array(["not_shockable"] * 20),
                 patient=np.array([f"icentia11k:p{i:05d}" for i in range(20)]))

        # Random data cannot reach the 0.90 sensitivity floor; the floor is
        # not what these checks are about, so lower it for the run.
        train_shockable.SENSITIVITY_FLOOR = 0.0

        print("train_shockable\n")
        out = tmp / "sh"
        rc, text = run(train_shockable.main,
                       ["--cache", str(cache), "--out", str(out), "--epochs", "2",
                        "--batch", "16", "--device", "cpu", "--augment",
                        "--extra-negatives", str(extra), "--threshold", "0.65"])
        check("trains with augmentation and extra negatives", rc == 0
              and "augment     on" in text and "extra neg   20 windows" in text)
        check("reports at the operating threshold",
              "at the operating threshold 0.65" in text)
        ck = torch.load(out / "shockable_net.pt", map_location="cpu", weights_only=False)
        check("checkpoint records threshold, augmentation, negatives and rule",
              ck.get("threshold") == 0.65 and ck.get("augment") is True
              and ck.get("extra_negatives") == str(extra)
              and tuple(ck.get("confirmation")) == (2, 3))

        rc, text = run(train_shockable.main,
                       ["--cache", str(cache), "--out", str(tmp / "sh0"),
                        "--epochs", "1", "--batch", "16", "--device", "cpu"])
        ck0 = torch.load(tmp / "sh0" / "shockable_net.pt", map_location="cpu",
                         weights_only=False)
        check("still trains without the new flags", rc == 0
              and "augment" not in text and ck0.get("augment") is False)

        leak = tmp / "leak.npz"
        np.savez(leak, X=rng.standard_normal((4, 1250)).astype(np.float16),
                 y=np.array(["not_shockable"] * 4),
                 patient=np.array(["trn:p1"] * 4))
        refused = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                train_shockable.main(["--cache", str(cache), "--out", str(tmp / "x"),
                                      "--epochs", "1", "--device", "cpu",
                                      "--extra-negatives", str(leak)])
            except SystemExit:
                refused = True
        check("negatives sharing a patient with train or val are refused", refused)

        print("\ntrain_rhythm\n")
        out = tmp / "rh"
        rc, text = run(train_rhythm.main,
                       ["--cache", str(cache), "--out", str(out), "--epochs", "1",
                        "--batch", "8", "--device", "cpu", "--augment"])
        ck = torch.load(out / "rhythm_net.pt", map_location="cpu", weights_only=False)
        check("trains with augmentation", rc == 0 and "augment     on" in text)
        check("checkpoint records gamma, augmentation and rule",
              ck.get("augment") is True and ck.get("gamma") == 2.0
              and tuple(ck.get("confirmation")) == (2, 3))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
