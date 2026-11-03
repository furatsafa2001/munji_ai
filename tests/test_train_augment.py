"""Checks the training augmentation added by fix_train_augmentation.py.

Synthetic signals and a synthetic noise bank only: no PhysioNet, no GPU.
The last block runs cross_validate end to end on a tiny fake cache.

    python tests/test_train_augment.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data import preprocess as pp
from munji_ai.data import train_augment as ta

FS = 250
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ecg(seconds=30.0, hr=72.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = np.zeros(n)
    t = 0.4
    while t < seconds - 0.5:
        c = int(t * FS)
        for amp, off, wid in ((0.12, -0.18, 0.022), (1.0, 0.0, 0.008),
                              (-0.18, 0.028, 0.010), (0.28, 0.22, 0.045)):
            k, w = int(off * FS), max(int(wid * FS), 1)
            lo, hi = max(c + k - 4 * w, 0), min(c + k + 4 * w, n)
            if hi > lo:
                x[lo:hi] += amp * np.exp(-0.5 * ((np.arange(lo, hi) - (c + k)) / w) ** 2)
        t += 60.0 / hr * rng.uniform(0.97, 1.03)
    return pp.normalize(x + rng.normal(0, 0.01, n)).astype(np.float32)


def fake_bank(seed=0, minutes=2.0):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    return {"em": rng.standard_normal(n).astype(np.float32),
            "ma": (rng.standard_normal(n) * 0.5).astype(np.float32)}


def snr_db(clean, noisy):
    return float(10 * np.log10(np.mean(clean ** 2) / np.mean((noisy - clean) ** 2)))


def main() -> int:
    X = np.stack([ecg(seed=s) for s in range(8)])
    xt = torch.from_numpy(X)

    print("Noise bank\n")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = Path(tmp) / ta.BANK_NAME
        np.savez(p, fs=FS, **fake_bank())
        bank = ta.load_noise_bank(p)
        check("bank loads both artifact types", set(bank) == {"em", "ma"})
        try:
            ta.load_noise_bank(Path(tmp) / "missing.npz")
            check("a missing bank says how to build it", False)
        except FileNotFoundError as e:
            check("a missing bank says how to build it", "--build" in str(e))
    tr, ev = ta.split_bank(bank)
    n = len(bank["em"])
    check("training part is the first 70%", len(tr["em"]) == int(n * 0.7))
    check("evaluation part is the rest, no overlap",
          len(tr["em"]) + len(ev["em"]) == n
          and np.array_equal(ev["em"], bank["em"][int(n * 0.7):]))

    print("\nMixing and normalisation\n")
    mixer = ta.NoiseMixer(tr, "cpu", seed=1)
    noisy = mixer.mix(xt, torch.full((8,), 10.0)).numpy()
    measured = [snr_db(X[i], noisy[i]) for i in range(8)]
    check("noise lands at the requested SNR", max(abs(m - 10.0) for m in measured) < 0.01,
          f"{min(measured):.3f} to {max(measured):.3f} dB")
    rn = ta.renormalize(torch.from_numpy(noisy)).numpy()
    ref = np.stack([pp.normalize(r) for r in noisy])
    check("renormalise matches preprocess.normalize",
          np.allclose(rn, ref, atol=1e-3), f"max diff {np.abs(rn - ref).max():.2e}")

    print("\nTrainAugment\n")
    saved = (ta.P_NOISE, ta.P_JITTER, ta.P_CUTOUT)
    try:
        ta.P_NOISE = ta.P_JITTER = ta.P_CUTOUT = 0.0
        out = ta.TrainAugment(tr, "cpu", seed=0)(xt).numpy()
        check("all probabilities zero leaves the window unchanged",
              np.allclose(out, X, atol=1e-3))
        ta.P_NOISE = 1.0
        out = ta.TrainAugment(tr, "cpu", seed=0)(xt).numpy()
        check("noise always on changes every window",
              all(not np.allclose(out[i], X[i], atol=1e-2) for i in range(8)))
        q25, q50, q75 = np.percentile(out, [25, 50, 75], axis=1)
        check("output is renormalised (median 0, IQR scale 1)",
              np.allclose(q50, 0, atol=1e-3) and np.allclose((q75 - q25) / 1.349, 1, atol=1e-3))
        ta.P_NOISE, ta.P_CUTOUT = 0.0, 1.0
        out = ta.TrainAugment(tr, "cpu", seed=3)(xt).numpy()
        flat = [np.max(np.convolve(np.diff(r) == 0, np.ones(5), "valid")) >= 5 for r in out]
        check("cutout leaves a flat stretch", sum(flat) >= 4, f"{sum(flat)}/8 windows")
    finally:
        ta.P_NOISE, ta.P_JITTER, ta.P_CUTOUT = saved
    out3 = ta.TrainAugment(tr, "cpu", seed=0)(xt.unsqueeze(1))
    check("accepts (batch, 1, length) too", tuple(out3.shape) == (8, 1, 7500))

    print("\nHeld-out noisy evaluation\n")
    a = ta.add_noise_fixed(X, ev, 6.0, seed=1000)
    b = ta.add_noise_fixed(X, ev, 6.0, seed=1000)
    c = ta.add_noise_fixed(X, ev, 6.0, seed=1001)
    check("same seed, same noisy copy", np.array_equal(a, b))
    check("different seed, different copy", not np.array_equal(a, c))
    check("noisy copy keeps the shape", a.shape == X.shape)

    print("\ncross_validate end to end (tiny fake cache)\n")
    from munji_ai.data.windows import config_hash
    from munji_ai.train import cross_validate as cv
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = Path(tmp)
        for split, n in (("train", 48), ("val", 16)):
            y = np.array(["shockable", "not_shockable"] * (n // 2))
            np.savez(cache / f"shockable_{split}.npz",
                     X=rng.standard_normal((n, 1250)).astype(np.float32), y=y,
                     patient=np.array([f"p{split}{i % 8}" for i in range(n)]),
                     record=np.array([f"r{split}{i % 8}" for i in range(n)]),
                     offset=np.arange(n) * 625, dataset=np.array(["x"] * n),
                     cfg_hash=config_hash())
        np.savez(cache / ta.BANK_NAME, fs=FS, **fake_bank())
        base = ["--stage", "shockable", "--folds", "2", "--epochs", "1",
                "--batch", "16", "--cache", str(cache), "--device", "cpu"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cv.main(base + ["--out", str(cache / "plain")])
        check("runs as before without the new flags", rc == 0
              and "augment     off" in buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cv.main(base + ["--augment", "--noisy-eval", "6",
                                 "--out", str(cache / "aug")])
        text = buf.getvalue()
        check("runs with --augment and --noisy-eval", rc == 0
              and "augment     on" in text and "under NSTDB noise at 6 dB" in text)
        with np.load(cache / "aug" / "shockable_oof.npz", allow_pickle=False) as d:
            files = d.files
        check("noisy predictions are saved with the OOF file",
              "pred_noisy" in files and "score_noisy" in files)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
