"""Fit the VF-or-VT namer on cardiologists' labels and export it for the app.

    python -m munji_ai.train.train_vfvt_namer
    python -m munji_ai.train.train_vfvt_namer --fragments data/raw/ecg_fragments --out exports

Needs the frag folders of PhysioNet's ECG Fragment Database
(ecg-fragment-high-risk-label) and our VFDB copy. No PyTorch, no GPU: a few
minutes on a laptop.

Steps
-----
1. Windows: every cardiologist fragment found in our VFDB copy
   (data/fragments.locate), one 5 s window around each, normalised as
   ShockableNet's input (data/preprocess.normalize): the namer reads the
   window the app already has. Truth as in eval/vf_vt_cardiologist.py: VF
   against VT (VTHR, VTLR, VTTdP). Flutter (VFL) is never learnt from; its
   P(VT) is only reported.
2. Check, out of fold by patient (GroupKFold, 5): the AUC with its 95%
   interval (patients resampled), and what the adopted zone
   (config.VFVT_NAME_ZONE) would name. Under AUC 0.85, the bar set before
   the experiment, it stops and writes nothing.
3. Fit the same pipeline (vfvt_common.make_clf("logistic")) on every VF and
   VT window.
4. Check that the numpy runtime (models/vfvt_namer.py), the arithmetic the
   app repeats in Dart, gives scikit-learn's P(VT) to 1e-9 on every window.
   If not, it writes nothing.
5. Write, in --out:
       vfvt_namer.json         the parameters, how each measure is computed,
                               the input, the decision rule, provenance and
                               the check
       vfvt_namer_golden.json  windows with their measures and P(VT), for the
                               app's Dart test; tests/test_vfvt_namer.py runs
                               them in Python too

The out-of-fold check is the estimate of how the fitted namer does on new
patients: the model fitted on everyone has no held-out score of its own.
The windows come from 17 VFDB patients at most, so that estimate is wide
(its interval says how wide), and shadow mode on real patients is the test
that decides whether names are ever shown. Test-split records are skipped
(data/fragments.locate). Nothing else is written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.preprocess import normalize
from ..eval.vf_vt_cardiologist import SHOWN, TRUTH, auc_ci, build_windows, naming, oof
from ..eval.vfvt_common import auc, make_clf
from ..features.vfvt import FEATURE_NAMES, features
from ..models.vfvt_namer import VFVTNamer, params_from_pipeline

NAME = "vfvt_namer"
MIN_AUC = 0.85                 # set before the experiment (vf_vt_feasibility.py)
MAX_DIFF = 1e-9                # numpy runtime against scikit-learn
GOLDEN_PER_CLASS = 10
GOLDEN_DECIMALS = 6
FS = C.TARGET_FS
WIN = C.window_samples("shockable")

# How each measure is computed (features/vfvt.py), for the app's Dart port.
# x is the input window; z = (x - mean(x)) / (sd(x) + 1e-8), sd with divisor n.
DEFINITIONS = {
    "lz_complexity": "b = 1 where z > median(z), else 0. Parse b left to right into phrases, "
                     "each the shortest run from there that is not yet a phrase (the last one "
                     "may repeat one). c = number of phrases; value c * log2(n) / n",
    "perm_entropy": "pattern of (z[i], z[i+2], z[i+4]) for every i: the stable argsort, coded "
                    "as a0 + 3 a1 + 9 a2 (ties keep sample order); Shannon entropy of the "
                    "pattern frequencies in bits, divided by log2(6)",
    "spectral_entropy": "Welch spectrum of z: periodic Hann window of 512 samples, overlap 256 "
                        "(segments 0-511, 256-767, 512-1023; the rest unused), each segment's "
                        "mean removed, one-sided, segments averaged. Bins f = k * 250 / 512 "
                        "with 1 <= f <= 15 Hz; q = power / sum of power over those bins; value "
                        "-sum(q * log2(q + 1e-12)) / log2(number of bins)",
    "dominant_hz": "f of the bin with the largest power among those (the first if tied)",
    "concentration": "sum of q over the bins within 0.5 Hz of dominant_hz",
    "periodicity": "r(k) = sum over i of z[i] * z[i+k], divided by n; the largest r(k) for "
                   "k = 50 to 149 samples (0.2 to 0.6 s)",
    "kurtosis": "m4 / m2^2 - 3, central moments of z with divisor n (Fisher, biased)",
    "skewness": "m3 / m2^1.5, central moments of z with divisor n (biased)",
    "zero_crossings": "share of the n - 1 steps where sign(z) changes (sign(0) = 0)",
    "bpm_signal": "on x itself, not z: a = |x - median(x)|, s = 1.4826 * median(a); peaks of a "
                  "as scipy.signal.find_peaks(a, distance=50, prominence=2 * s) finds them "
                  "(plateaus at their middle, distance applied before prominence); value 60 / "
                  "median of the peak-to-peak intervals in seconds. Missing (null) under 3 "
                  "peaks or when s is 0",
}


# ----------------------------------------------------------------- the data
def dataset(found: list, load) -> dict:
    """The windows of build_windows, normalised as ShockableNet's input, with
    their measures and truth (VF 0, VT 1, flutter -1)."""
    w = build_windows(found, load)
    Xn = np.stack([normalize(x) for x in w["X"]]) if len(w["X"]) else np.zeros((0, WIN))
    F = np.stack([features(x) for x in Xn]) if len(Xn) else np.zeros((0, len(FEATURE_NAMES)))
    y = np.array([TRUTH.get(k, -1) for k in w["label"]], dtype=int)
    return {**w, "Xn": Xn, "F": F, "y": y}


def group_of(label: str) -> str:
    return "VFL" if label in SHOWN else "VT" if TRUTH.get(label) == 1 else "VF"


# ---------------------------------------------------------------- the check
def cross_check(d: dict, zone) -> dict:
    """Out of fold by patient, exactly as eval/vf_vt_cardiologist scores it."""
    y, F, groups = d["y"], d["F"], d["record"]
    p = oof(F, y, groups, "logistic")
    scored = (y >= 0) & np.isfinite(p)
    lo, hi = auc_ci(y, p, groups)
    fl = p[np.isin(d["label"], SHOWN) & np.isfinite(p)]
    return {"p": p, "auc": auc(y[scored], p[scored]), "ci": (float(lo), float(hi)),
            "zone": naming(y, p, *zone),
            "flutter": {"windows": int(fl.size), "named_VT": int((fl >= zone[1]).sum()),
                        "named_VF": int(((fl <= zone[0]) & (fl < zone[1])).sum())}}


# --------------------------------------------------------------- the export
def golden_case(namer: VFVTNamer, x: np.ndarray, label: str) -> dict:
    """One window as stored (rounded), with what the namer computes from exactly that."""
    x = np.round(np.asarray(x, np.float64), GOLDEN_DECIMALS)
    f = features(x)
    return {"label": label, "input": x.tolist(),
            "features": [float(v) if np.isfinite(v) else None for v in f],
            "p_vt": float(namer.p_vt_from_features(f)[0])}


def golden(namer: VFVTNamer, d: dict, zone, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = np.array([group_of(k) for k in d["label"]])
    cases = []
    for g in ("VF", "VT", "VFL"):
        idx = np.flatnonzero(groups == g)
        for i in sorted(rng.choice(idx, min(GOLDEN_PER_CLASS, len(idx)), replace=False)):
            cases.append(golden_case(namer, d["Xn"][i], f"{g} ({d['label'][i]}, VFDB {d['record'][i]})"))
    t = np.arange(WIN) / FS
    cases.append(golden_case(namer, np.sin(2 * np.pi * 4.0 * t),
                             "synthetic: 4 Hz sine, no rate measurable (the fill value is used)"))
    return {"model": f"{NAME}.json", "features": list(FEATURE_NAMES), "zone": list(zone),
            "note": f"input: {WIN} samples at {FS} Hz, normalised as ShockableNet's input, "
                    f"rounded to {GOLDEN_DECIMALS} decimals; features and p_vt are computed "
                    "from the input as stored, in float64; null: not measurable (the "
                    "namer uses its fill value)",
            "cases": cases}


def metadata(params: dict, d: dict, chk: dict, zone, diff: float) -> dict:
    import sklearn
    y = d["y"]
    count = Counter(group_of(k) for k in d["label"])
    vt_kinds = Counter(k for k in d["label"] if TRUTH.get(k) == 1)
    z = chk["zone"]
    return {
        "model": NAME,
        "exported": date.today().isoformat(),
        "what": "P(VT) against VF, for a window ShockableNet calls shockable",
        "input": {"shape": [WIN], "dtype": "float64", "sample_rate_hz": FS,
                  "window_s": C.WINDOW_SEC["shockable"],
                  "what": "the window ShockableNet scores, after its preprocessing and "
                          "normalisation (shockable_net.json)",
                  "run_on": "windows at or above ShockableNet's threshold only; the decision "
                            "layer reads no other"},
        "standardise": "z = (x - mean(x)) / (sd(x) + 1e-8), sd with divisor n",
        "features": [{"name": n, "how": DEFINITIONS[n]} for n in FEATURE_NAMES],
        "pipeline": {**params,
                     "formula": "v = x, or fill where x is missing; z = (v - mean) / scale; "
                                "p_vt = 1 / (1 + exp(-(intercept + sum(coef * z))))"},
        "output": {"name": "p_vt", "range": [0, 1], "meaning": "probability of VT rather than VF"},
        "decision": {"zone": list(zone),
                     "rule": f"each window that confirmed the alert is named VF at p_vt <= "
                             f"{zone[0]}, VT at p_vt >= {zone[1]}, else nothing; the alert is "
                             "named only when all of them got the same name, else VF/VT",
                     "naming_enabled": bool(C.VFVT_NAMING_ENABLED),
                     "meaning": "while naming is off the alert says VF/VT and the name is kept "
                                "with the ECG recording only (shadow mode)"},
        "training": {
            "labels": "ECG Fragment Database (PhysioNet ecg-fragment-high-risk-label), "
                      "cardiologists of the Almazov Centre",
            "signal": "our VFDB copy, each fragment found by its waveform (data/fragments.py); "
                      "test-split records skipped",
            "truth": {"VF": ["VF"], "VT": sorted(k for k, v in TRUTH.items() if v == 1)},
            "never_learnt_from": list(SHOWN),
            "windows": {"VF": count["VF"], "VT": count["VT"],
                        "VT_by_kind": dict(sorted(vt_kinds.items())),
                        "VFL_scored_only": count["VFL"]},
            "records": int(len(set(d["record"][y >= 0]))),
            "classifier": "mean imputation, standard scaling, logistic regression "
                          "(C 0.1, balanced classes)",
            "scikit_learn": sklearn.__version__},
        "check": {"method": "out of fold, GroupKFold 5 by patient",
                  "auc": float(chk["auc"]), "auc_95ci": list(chk["ci"]), "bar": MIN_AUC,
                  "zone": {"windows": z["windows"], "named": z["named"],
                           "VF_named_VT": z["vf_as_vt"], "VT_named_VF": z["vt_as_vf"]},
                  "flutter": chk["flutter"],
                  "note": "the estimate for new patients; the model fitted on everyone has no "
                          "held-out score of its own"},
        "verification": {"windows": int(len(d["F"])), "max_abs_diff_vs_scikit_learn": diff,
                         "bar": MAX_DIFF, "passed": bool(diff <= MAX_DIFF)},
    }


# -------------------------------------------------------------------- main
def run(found: list, load, out: Path, zone=tuple(C.VFVT_NAME_ZONE), min_auc: float = MIN_AUC) -> int:
    d = dataset(found, load)
    y = d["y"]
    if len(np.unique(y[y >= 0])) < 2:
        print("need both VF and VT windows: nothing written")
        return 1
    count = Counter(group_of(k) for k in d["label"])
    vt_kinds = Counter(k for k in d["label"] if TRUTH.get(k) == 1)
    print(f"windows     VF {count['VF']}, VT {count['VT']} ("
          + ", ".join(f"{k} {v}" for k, v in sorted(vt_kinds.items()))
          + f"), VFL {count['VFL']} scored only, from {len(set(d['record']))} VFDB records"
          + ("; skipped " + ", ".join(f"{k} {v}" for k, v in d["skipped"].items())
             if d["skipped"] else ""))

    chk = cross_check(d, zone)
    a, (lo, hi), z = chk["auc"], chk["ci"], chk["zone"]
    reached = bool(np.isfinite(a) and a >= min_auc)
    print("\ncheck, out of fold by patient (GroupKFold 5)")
    print(f"  AUC {a:.3f} (95% CI {lo:.2f}-{hi:.2f}), bar {min_auc}: "
          + ("reached" if reached else "NOT reached"))
    print(f"  zone {zone[0]}/{zone[1]}: named {z['named']} of {z['windows']} windows, "
          f"wrong {z['vf_as_vt'] + z['vt_as_vf']} (VF named VT {z['vf_as_vt']}, "
          f"VT named VF {z['vt_as_vf']})")
    f = chk["flutter"]
    print(f"  flutter: {f['windows']} windows, would be named VT {f['named_VT']}, VF {f['named_VF']}")
    if not reached:
        print(f"\nAUC under {min_auc}: nothing written")
        return 1

    keep = y >= 0
    clf = make_clf("logistic")
    clf.fit(d["F"][keep], y[keep])
    params = params_from_pipeline(clf)
    namer = VFVTNamer(params)
    diff = float(np.max(np.abs(namer.p_vt_from_features(d["F"]) - clf.predict_proba(d["F"])[:, 1])))
    print(f"\nfitted      on {int(keep.sum())} windows (VF {int((y == 0).sum())}, VT {int((y == 1).sum())})")
    print(f"verified    numpy runtime against scikit-learn on {len(d['F'])} windows: "
          f"largest difference {diff:.1e} (bar {MAX_DIFF:.0e})")
    if not diff <= MAX_DIFF:
        print("FAILED: the runtime the app copies disagrees with the fitted model; nothing written")
        return 1
    order = np.argsort(-np.abs(params["coef"]))
    print("weights     (on standardised measures; + leans VT)  "
          + ", ".join(f"{FEATURE_NAMES[i]} {params['coef'][i]:+.2f}" for i in order))

    out.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN would make the files unreadable to strict JSON
    # parsers (Dart's among them); missing measures are written as null.
    meta = json.dumps(metadata(params, d, chk, zone, diff), indent=2, allow_nan=False)
    gold = json.dumps(golden(namer, d, zone), allow_nan=False)
    (out / f"{NAME}.json").write_text(meta)
    (out / f"{NAME}_golden.json").write_text(gold)
    print(f"\nwrote {NAME}.json and {NAME}_golden.json in {out}")
    print("Naming stays off (config.VFVT_NAMING_ENABLED): the name is kept in shadow mode.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", type=Path, default=None, help="default: <raw>/ecg_fragments")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--out", type=Path, default=Path("exports"))
    a = ap.parse_args(argv)

    from ..data.fragments import locate
    from ..data.loader import load_record
    root = a.fragments or a.raw / "ecg_fragments"
    if not root.exists():
        print(f"no {root}: download the frag folders of PhysioNet ecg-fragment-high-risk-label")
        return 1
    found, skipped, _, _ = locate(root, a.raw)
    print(f"fragments   found {len(found)} in our VFDB copy (best correlation 0.8 or more)"
          + ("; skipped " + ", ".join(f"{k} {v}" for k, v in skipped.items()) if skipped else ""))
    return run(found, lambda name: load_record("vfdb", name, a.raw), a.out)


if __name__ == "__main__":
    raise SystemExit(main())
