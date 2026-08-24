"""Exercises preprocessing, augmentation, and splitting on a synthetic ECG.

No PhysioNet access required. Verifies the numerics are correct before the
real corpora arrive.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data import augment, preprocess as pp, splits


def synth_ecg(seconds=60.0, hr=72.0, fs=360, seed=0):
    """Crude but adequate ECG: gaussian P, QRS, and T per beat."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr = 60.0 / hr
    peaks = []
    tb = 0.35
    while tb < seconds - 0.5:
        c = int(tb * fs)
        peaks.append(c)
        for amp, off, wid in ((0.12, -0.18, 0.022), (1.0, 0.0, 0.008),
                              (-0.18, 0.028, 0.010), (0.28, 0.22, 0.045)):
            k = int(off * fs)
            w = max(int(wid * fs), 1)
            lo, hi = max(c + k - 4 * w, 0), min(c + k + 4 * w, n)
            if hi > lo:
                g = np.exp(-0.5 * ((np.arange(lo, hi) - (c + k)) / w) ** 2)
                x[lo:hi] += amp * g
        tb += rr * rng.uniform(0.97, 1.03)
    x += 0.05 * np.sin(2 * np.pi * 0.25 * t)          # baseline wander
    x += rng.normal(0, 0.01, n)                        # sensor noise
    return x, fs, np.array(peaks)


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    print("\n1. resampling")
    x, fs, peaks = synth_ecg(seconds=60, fs=360)
    y = pp.resample_to(x, fs)
    check("length ratio", abs(len(y) / len(x) - C.TARGET_FS / fs) < 1e-3,
          f"{len(x)}@{fs} -> {len(y)}@{C.TARGET_FS}")
    check("256 Hz device rate maps cleanly",
          abs(len(pp.resample_to(x, 256)) / len(x) - 250 / 256) < 1e-3)

    print("\n2. filter paths")
    m = pp.model_path(x, fs)
    d = pp.diagnostic_path(x, fs)
    check("model path finite", np.isfinite(m).all())
    check("diagnostic path finite", np.isfinite(d).all())
    check("model path removes more drift than diagnostic",
          np.std(m - np.mean(m)) <= np.std(d - np.mean(d)) * 1.05,
          f"std model={np.std(m):.3f} diag={np.std(d):.3f}")

    # 0.05 Hz vs 0.5 Hz: the diagnostic path must retain low-frequency content.
    lo = np.fft.rfft(d)[: int(0.4 * len(d) / C.TARGET_FS) + 1]
    lo_m = np.fft.rfft(m)[: int(0.4 * len(m) / C.TARGET_FS) + 1]
    check("diagnostic retains sub-0.4 Hz energy (ST preserved)",
          np.sum(np.abs(lo)) > np.sum(np.abs(lo_m)) * 1.5,
          f"ratio={np.sum(np.abs(lo)) / (np.sum(np.abs(lo_m)) + 1e-9):.1f}x")

    print("\n3. R-peak alignment survives filtering (zero-phase)")
    ym = pp.model_path(x, fs)
    scale = C.TARGET_FS / fs
    exp = np.round(peaks * scale).astype(int)
    found = []
    for p in exp[2:-2]:
        w = ym[max(p - 20, 0): p + 20]
        found.append(np.argmax(w) + max(p - 20, 0))
    err = np.abs(np.array(found) - exp[2:-2])
    check("peak drift under 4 samples (16 ms)", err.max() <= 4,
          f"max={err.max()} mean={err.mean():.2f}")

    print("\n4. normalisation robustness")
    spike = ym.copy()
    spike[len(spike) // 2] = 200.0
    nz, ns = pp.normalize(ym), pp.normalize(spike)
    check("single spike barely moves the scale",
          abs(np.std(nz[:1000]) - np.std(ns[:1000])) < 0.05,
          f"{np.std(nz[:1000]):.3f} vs {np.std(ns[:1000]):.3f}")

    print("\n5. windowing")
    for stage in ("quality", "beat", "rhythm", "shockable"):
        w = pp.sliding_windows(ym, C.window_samples(stage), C.stride_samples(stage))
        check(f"{stage:<9} windows", w.shape[1] == C.window_samples(stage),
              f"{w.shape} ({C.WINDOW_SEC[stage]}s)")

    print("\n6. noise augmentation at controlled SNR")
    rng = np.random.default_rng(1)
    fake = {"em": rng.normal(0, 1, 200_000), "bw": np.sin(np.linspace(0, 900, 200_000))}
    clean = pp.sliding_windows(ym, C.window_samples("quality"),
                               C.stride_samples("quality"))
    for target in (18.0, 6.0, 0.0):
        noisy = augment.mix(clean[3], fake["em"], target, rng)
        resid = noisy - clean[3]
        got = 20 * np.log10(np.sqrt(np.mean(clean[3] ** 2)) /
                            np.sqrt(np.mean(resid ** 2)))
        check(f"SNR {target:>5.1f} dB achieved", abs(got - target) < 0.5,
              f"measured {got:.2f} dB -> {augment.quality_from_snr(got)}")

    X, yq, snr = augment.make_quality_set(clean[:80], fake, seed=2)
    dist = {k: int((yq == k).sum()) for k in C.QUALITY_CLASSES}
    check("quality set covers every class", all(v > 0 for v in dist.values()), str(dist))
    check("labels agree with the injected SNR",
          all(augment.quality_from_snr(s) == l
              for s, l in zip(snr, yq) if np.isfinite(s)))

    print("\n7. lead-off simulation")
    lo_sig = augment.simulate_lead_off(ym[:5000], rng=rng)
    check("flat segment present", np.std(lo_sig) < np.std(ym[:5000]),
          f"std {np.std(ym[:5000]):.3f} -> {np.std(lo_sig):.3f}")

    print("\n8. patient-level split determinism and leak check")
    pats = [f"icentia11k:p{i:05d}" for i in range(4000)]
    a = [splits.assign(p) for p in pats]
    b = [splits.assign(p) for p in pats]
    check("deterministic across calls", a == b)
    frac = {s: a.count(s) / len(a) for s in ("train", "val", "test")}
    check("fractions within 2% of target",
          all(abs(frac[s] - C.SPLIT_FRACTIONS[s]) < 0.02 for s in frac),
          str({k: round(v, 3) for k, v in frac.items()}))

    manifest = {
        "mitdb/201": {"dataset": "mitdb", "record": "201", "patient": "mitdb:201",
                      "split": "test"},
        "mitdb/202": {"dataset": "mitdb", "record": "202", "patient": "mitdb:201",
                      "split": "test"},
    }
    try:
        splits._assert_disjoint(manifest)
        check("same-subject records share a split (201/202)", True)
    except AssertionError:
        check("same-subject records share a split (201/202)", False)

    manifest["mitdb/202"]["split"] = "train"
    try:
        splits._assert_disjoint(manifest)
        check("leak is detected", False)
    except AssertionError:
        check("leak is detected", True, "raised as expected")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
