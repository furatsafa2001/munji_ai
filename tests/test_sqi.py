"""SQI feature, baseline-model, and rhythm-bias checks. No PhysioNet needed."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C
from munji_ai.data import augment, preprocess as pp
from munji_ai.eval import metrics as M
from munji_ai.features import sqi
from test_pipeline import synth_ecg


def noise_bank(rng):
    n = 400_000
    return {"em": rng.normal(0, 1, n),
            "bw": np.sin(np.linspace(0, 1500, n)),
            "ma": rng.normal(0, 1, n) * np.abs(np.sin(np.linspace(0, 9000, n)))}


def irregular_ecg(seconds=120, seed=7):
    """AF-like: same beats, wildly irregular spacing."""
    rng = np.random.default_rng(seed)
    fs = C.TARGET_FS
    n = int(seconds * fs)
    x = np.zeros(n)
    t = 0.4
    while t < seconds - 0.5:
        c = int(t * fs)
        for amp, off, wid in ((1.0, 0.0, 0.008), (-0.2, 0.028, 0.010),
                              (0.3, 0.22, 0.045)):
            k, w = int(off * fs), max(int(wid * fs), 1)
            lo, hi = max(c + k - 4 * w, 0), min(c + k + 4 * w, n)
            if hi > lo:
                g = np.exp(-0.5 * ((np.arange(lo, hi) - (c + k)) / w) ** 2)
                x[lo:hi] += amp * g
        t += rng.uniform(0.45, 1.25)          # irregularly irregular
    return x + rng.normal(0, 0.01, n)


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    rng = np.random.default_rng(0)
    noises = noise_bank(rng)
    clean_raw, fs0, _ = synth_ecg(seconds=120, fs=250)
    clean = pp.model_path(clean_raw, fs0)
    win = C.window_samples("quality")
    W = pp.sliding_windows(clean, win, win)

    print("\n1. SQI features separate clean from noisy")
    good = W[5]
    bad = augment.mix(good, noises["em"], -3.0, rng)
    fg, fb = sqi.features(good), sqi.features(bad)
    names = list(sqi.FEATURE_NAMES)
    for k in ("qSQI", "pSQI", "basSQI"):
        i = names.index(k)
        check(f"{k} drops with noise", fg[i] > fb[i], f"{fg[i]:.3f} -> {fb[i]:.3f}")

    print("\n2. qSQI needs two genuinely different detectors")
    a = sqi.detect_r_pantompkins(good)
    b = sqi.detect_r_envelope(good)
    check("both detectors find beats on clean signal", len(a) > 5 and len(b) > 5,
          f"pan={len(a)} env={len(b)}")
    check("high agreement on clean", sqi.q_sqi(good) > 0.7, f"{sqi.q_sqi(good):.3f}")
    check("agreement collapses on noise", sqi.q_sqi(bad) < sqi.q_sqi(good),
          f"{sqi.q_sqi(bad):.3f}")

    print("\n3. lead-off and flat-signal detection")
    flat = np.zeros(win) + rng.normal(0, 0.001, win)
    ff = sqi.features(flat)
    check("flat signal yields no plausible rate",
          ff[names.index("rate_bpm")] == 0 or ff[names.index("qSQI")] == 0,
          f"rate={ff[names.index('rate_bpm')]:.0f} qSQI={ff[names.index('qSQI')]:.3f}")
    check("qSQI is 0 on flat signal", ff[names.index("qSQI")] == 0.0,
          f"{ff[names.index('qSQI')]:.3f}")

    print("\n4. feature matrix shape and finiteness")
    F = sqi.feature_matrix(W[:20])
    check("shape correct", F.shape == (20, len(sqi.FEATURE_NAMES)), str(F.shape))
    check("all finite", np.isfinite(F).all())

    print("\n5. class balance after widening the SNR range")
    long_raw, lfs, _ = synth_ecg(seconds=1800, fs=250, seed=11)
    LW = pp.sliding_windows(pp.model_path(long_raw, lfs), win, win // 2)
    X, y, _ = augment.make_quality_set(LW, noises, seed=5)
    frac = {c: float((y == c).mean()) for c in C.QUALITY_CLASSES}
    check("no class below 15%", min(frac.values()) > 0.15,
          str({k: round(v, 3) for k, v in frac.items()}))

    print("\n6. baseline classifier hits target")
    try:
        from munji_ai.models.sqi_baseline import SQIBaseline
        perm = np.random.default_rng(0).permutation(len(X))
        Xs, ys = X[perm], y[perm]
        n_tr = int(len(Xs) * 0.7)
        clf = SQIBaseline(n_estimators=200).fit(Xs[:n_tr], ys[:n_tr], verbose=False)
        pred = clf.predict(Xs[n_tr:])
        passed, detail = M.passes_target(ys[n_tr:], pred)
        for k, v in detail.items():
            check(f"{k} >= {v['target']:.2f}", v["pass"], f"got {v['got']:.3f}")
        check("sensitivity is not zero — gate actually rejects",
              detail["sensitivity"]["got"] > 0.0)
        check("round-trips through save/load",
              (SQIBaseline.load(clf.save(Path("/tmp/_sqi.pkl")))
               .predict(Xs[n_tr:n_tr + 20]) == pred[:20]).all())
    except ImportError:
        check("scikit-learn available", False, "pip install scikit-learn")

    print("\n6b. graded score gives distinct operating points")
    # The detail the old qrs_only class carried now lives in P(usable): a
    # strict threshold for morphology, a permissive one for rhythm timing.
    yt = np.array(["usable"] * 70 + ["unusable"] * 30)
    pu = np.concatenate([rng.uniform(0.55, 1.0, 70), rng.uniform(0.0, 0.7, 30)])
    sc = M.gate_views_scored(yt, pu)
    check("both operating points reported", set(sc) == set(C.GATE_MIN_P), str(set(sc)))
    check("permissive point passes more windows",
          sc["rhythm"]["pass_rate"] > sc["morphology"]["pass_rate"],
          f"rhythm {sc['rhythm']['pass_rate']:.2f} vs "
          f"morphology {sc['morphology']['pass_rate']:.2f}")
    check("strict point rejects more",
          sc["morphology"]["sensitivity"] >= sc["rhythm"]["sensitivity"])

    print("\n7. rhythm bias detector — the critical audit")
    n = 600
    src = np.array(["nsrdb"] * 300 + ["ltafdb"] * 200 + ["cudb"] * 100)

    fair = np.array(["good"] * 540 + ["unusable"] * 60)
    rng.shuffle(fair)
    b1 = M.rhythm_rejection_bias(fair, src)
    check("unbiased gate passes audit", not b1["biased"], f"ratio={b1['ratio']:.2f}")

    biased = np.where(src == "nsrdb", "good", "unusable")
    b2 = M.rhythm_rejection_bias(biased, src)
    check("biased gate is caught", b2["biased"], f"ratio={b2['ratio']:.2f}")
    check("per-dataset breakdown present",
          set(b2["per_dataset"]) == {"nsrdb", "ltafdb", "cudb"})

    print("\n8. irregular rhythm must not look like noise to the features")
    irr = irregular_ecg()
    iw = pp.sliding_windows(pp.model_path(irr, C.TARGET_FS), win, win)
    q_reg = np.mean([sqi.q_sqi(w) for w in W[:20]])
    q_irr = np.mean([sqi.q_sqi(w) for w in iw[:20]])
    check("qSQI stays high on irregular-but-clean signal", q_irr > 0.6,
          f"regular={q_reg:.3f} irregular={q_irr:.3f}")
    check("qSQI gap between regular and irregular is small",
          abs(q_reg - q_irr) < 0.25, f"gap={abs(q_reg - q_irr):.3f}")

    print("\n9. quality carriers now include arrhythmic sources")
    from munji_ai.data.registry import by_stage
    carriers = {d.key for d in by_stage("quality")}
    check("AF carrier present", "ltafdb" in carriers, str(sorted(carriers)))
    check("VF carrier present", "cudb" in carriers or "vfdb" in carriers)
    check("ectopic carrier present", "svdb" in carriers)

    print("\n10. external benchmark registered")
    from munji_ai.data.registry import get
    ch = get("challenge2011")
    check("challenge2011 is validation-only", ch.role == "val")
    check("prefers a V5-like lead", "V5" in ch.channel_names, str(ch.channel_names))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
