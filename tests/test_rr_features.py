"""Checks the RR-feature fusion added by fix_rhythm_rr_features.py.

Synthetic ECG only: each rhythm is built from a known list of beat intervals,
so every feature has an expected direction. No PhysioNet access, no GPU.

    python tests/test_rr_features.py
"""

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.features.rr import (CLIP, RR_FEATURES, cached_rr_features,
                                  rr_features)
from munji_ai.models.rhythm_net import RhythmNet

FS = 250
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ecg(rr_list, seconds=30.0, seed=0):
    """Gaussian P-QRS-T complexes placed at the given beat intervals."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = np.zeros(n)
    t, i = 0.4, 0
    while t < seconds - 0.5:
        c = int(t * FS)
        for amp, off, wid in ((0.12, -0.18, 0.022), (1.0, 0.0, 0.008),
                              (-0.18, 0.028, 0.010), (0.28, 0.22, 0.045)):
            k, w = int(off * FS), max(int(wid * FS), 1)
            lo, hi = max(c + k - 4 * w, 0), min(c + k + 4 * w, n)
            if hi > lo:
                x[lo:hi] += amp * np.exp(-0.5 * ((np.arange(lo, hi) - (c + k)) / w) ** 2)
        t += rr_list[i % len(rr_list)]
        i += 1
    return x + rng.normal(0, 0.01, n)


def feats(rr_list, seed=0):
    return dict(zip(RR_FEATURES, rr_features(ecg(rr_list, seed=seed), FS)))


def main() -> int:
    rng = np.random.default_rng(1)
    nsr = feats([60 / 70 * rng.uniform(0.98, 1.02) for _ in range(60)])
    big = feats([0.55, 1.05] * 30)
    af = feats(list(rng.uniform(0.45, 1.1, 80)))
    pause = feats([0.85] * 10 + [2.6] + [0.85] * 20)

    print("RR features\n")
    out = rr_features(ecg([0.8] * 40), FS)
    check("12 features, float32", out.shape == (12,) and out.dtype == np.float32)
    flat = rr_features(np.zeros(7500), FS)
    check("flat trace gives all zeros and valid = 0", not flat.any())
    check("sinus rhythm is valid", nsr["valid"] == 1.0)
    check("sinus rate read correctly", abs(nsr["hr"] - 0.70) < 0.03,
          f"{nsr['hr'] * 100:.0f} bpm")
    check("sinus rhythm is regular", nsr["cv"] < 0.05, f"cv {nsr['cv']:.3f}")
    check("sinus rhythm has no premature beats", nsr["premature"] == 0.0)
    check("bigeminy: half the beats premature", abs(big["premature"] - 0.5) < 0.05,
          f"{big['premature']:.2f}")
    check("bigeminy: RR alternates every beat", big["tpr"] > 0.9, f"tpr {big['tpr']:.2f}")
    check("AF entropy well above sinus", af["entropy"] > nsr["entropy"] + 0.4,
          f"{af['entropy']:.2f} vs {nsr['entropy']:.2f}")
    check("AF is irregular", af["cv"] > 0.15, f"cv {af['cv']:.3f}")
    check("a pause shows as a long interval", pause["long"] > 0,
          f"{pause['long']:.3f}")
    allv = np.array([list(d.values()) for d in (nsr, big, af, pause)])
    check("every feature stays inside [0, CLIP]",
          (allv >= 0).all() and (allv <= CLIP).all())

    print("\nRhythmNet fusion\n")
    base = RhythmNet()
    check("n_rr = 0 keeps the original 337,875 parameters",
          base.n_params() == 337_875, f"{base.n_params():,}")
    check("n_rr = 0 adds no new weights to the checkpoint",
          not any(k.startswith("rr_norm") for k in base.state_dict()))
    fused = RhythmNet(n_rr=len(RR_FEATURES))
    x = torch.randn(4, 7500)
    r = torch.rand(4, len(RR_FEATURES))
    check("fused model forward gives (batch, 3)", tuple(fused(x, r).shape) == (4, 3))
    check("fused predict_proba gives P(AF) per window",
          tuple(fused.predict_proba(x, r).shape) == (4,))
    try:
        fused(x)
        check("fused model refuses to run without rr", False)
    except ValueError:
        check("fused model refuses to run without rr", True)

    print("\nFeature cache\n")
    X = np.stack([ecg([0.8] * 40, seed=s) for s in range(3)])
    rec = np.array(["a", "a", "b"])
    off = np.array([0, 3750, 0])
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "rhythm_trainval_rr.npz"
        F1 = cached_rr_features(X, rec, off, p)
        F2 = cached_rr_features(X, rec, off, p)
        check("second call reuses the saved file", np.array_equal(F1, F2))
        F3 = cached_rr_features(X[::-1].copy(), rec[::-1], off[::-1], p)
        check("different window order is rebuilt, not reused",
              np.array_equal(F3, F1[::-1]))

    print("\ncross_validate\n")
    from munji_ai.train import cross_validate as cv
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((48, 7500)).astype(np.float32)
    ytr = np.array(["NSR", "AF", "OTHER"] * 16)
    Xva = rng.standard_normal((6, 7500)).astype(np.float32)
    a = argparse.Namespace(batch=16, lr=1e-3, epochs=1, penalty=1.3, gamma=2.0,
                           device="cpu", _y_fold=np.array(["NSR", "AF", "OTHER"] * 2))
    classes = list(cv.STAGES["rhythm"]["classes"])
    Ftr = rng.random((48, len(RR_FEATURES))).astype(np.float32)
    Fva = rng.random((6, len(RR_FEATURES))).astype(np.float32)
    m, P = cv.train_one_fold("rhythm", Xtr, ytr, Xva, torch.device("cpu"), a,
                             classes, Ftr, Fva)
    check("rhythm fold trains with RR features", P.shape == (6, 3) and m.n_rr == 12)
    m, P = cv.train_one_fold("rhythm", Xtr, ytr, Xva, torch.device("cpu"), a, classes)
    check("rhythm fold still trains without them", P.shape == (6, 3) and m.n_rr == 0)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            cv.main(["--stage", "shockable", "--rr"])
        check("--rr is refused for the shockable stage", False)
    except SystemExit:
        check("--rr is refused for the shockable stage", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
