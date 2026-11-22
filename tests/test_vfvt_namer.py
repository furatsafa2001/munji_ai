"""The VF-or-VT namer without PyTorch: its measures, its runtime, its training.

Synthetic VF (band-limited noise) and VT (a periodic wave), as in
test_vf_vt_cardiologist.py; the checks are about the plumbing: the measures
do not depend on gain or offset, the numpy runtime the app copies is
scikit-learn's pipeline to the last digits, the export is refused under the
bar, the golden windows give their values again, and a window's P(VT)
reaches the decision layer's shadow name.

    python tests/test_vfvt_namer.py
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from munji_ai import config as C  # noqa: E402
from munji_ai.data.preprocess import normalize  # noqa: E402
from munji_ai.decision.decision_layer import DecisionLayer  # noqa: E402
from munji_ai.eval.vfvt_common import make_clf  # noqa: E402
from munji_ai.features.vfvt import FEATURE_NAMES, features  # noqa: E402
from munji_ai.models.vfvt_namer import VFVTNamer, params_from_pipeline  # noqa: E402
from munji_ai.train import train_vfvt_namer as T  # noqa: E402

ok = True
FS = C.TARGET_FS
WIN = C.window_samples("shockable")


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def wave(kind: str, seconds: float, rng) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    if kind == "VF":
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(4, [3.0, 7.0], btype="band", fs=FS, output="sos"),
                        rng.normal(0, 1, t.size))
        return x / np.ptp(x)
    hz = {"VT": rng.uniform(2.8, 3.5), "VFL": 4.2}[kind]
    x = np.sin(2 * np.pi * hz * t) + 0.4 * np.sin(4 * np.pi * hz * t + 0.5)
    return x / np.ptp(x) + rng.normal(0, 0.02, t.size)


def record(parts, rng):
    """[(kind, seconds)] -> a fake Record with its rhythm annotation."""
    sig = np.concatenate([wave(k, s, rng) for k, s in parts])
    starts = np.cumsum([0] + [int(s * FS) for _, s in parts[:-1]])
    token = {"VF": "(VF", "VT": "(VT", "VFL": "(VFL"}
    return SimpleNamespace(signal=sig, fs=FS, rhythm_starts=starts,
                           rhythm_labels=np.array([token[k] for k, _ in parts]),
                           beat_samples=np.array([], int), beat_symbols=np.array([]))


def fake_fragments(rng):
    recs = {f"6{i:02d}": record([("VF", 60), ("VT", 60), ("VFL", 30)], rng) for i in range(10)}
    found = []
    for name in recs:
        found += [{"record": name, "label": "VF", "centre_s": c, "r": 0.99} for c in (15, 30, 45)]
        found += [{"record": name, "label": "VTHR", "centre_s": c, "r": 0.99} for c in (75, 90, 105)]
        found += [{"record": name, "label": "VFL", "centre_s": 135.0, "r": 0.99}]
    return recs, found


def main() -> int:
    rng = np.random.default_rng(0)

    print("The measures\n")
    vf, vt = wave("VF", 5, rng)[:WIN], wave("VT", 5, rng)[:WIN]
    worst = max(np.nanmax(np.abs(features(x) - features(g)))
                for x in (vf, vt) for g in (normalize(x), 3.0 * x + 0.5))
    check("the same values off the raw window, ShockableNet's normalised one, or 3x + 0.5",
          worst < 1e-6, f"largest difference {worst:.1e}")
    f = features(np.sin(2 * np.pi * 4.0 * np.arange(WIN) / FS))
    check("a smooth wave has no measurable rate (NaN); every other measure is read",
          np.isnan(f[FEATURE_NAMES.index("bpm_signal")]) and np.isfinite(np.delete(f, 9)).all(),
          str(np.round(f, 3)))

    print("\nThe runtime the app copies\n")
    X = [wave("VF", 5, rng)[:WIN] for _ in range(30)] + [wave("VT", 5, rng)[:WIN] for _ in range(30)]
    F = np.stack([features(normalize(x)) for x in X])
    F[::7, FEATURE_NAMES.index("bpm_signal")] = np.nan       # some rates not measurable
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    clf = make_clf("logistic")
    clf.fit(F, y)
    params = params_from_pipeline(clf)
    namer = VFVTNamer(params)
    diff = np.max(np.abs(namer.p_vt_from_features(F) - clf.predict_proba(F)[:, 1]))
    check("P(VT) is scikit-learn's to 1e-12, missing measures filled the same way",
          diff < 1e-12, f"largest difference {diff:.1e}")
    check("p_vt(window) reads the window's own measures",
          abs(namer.p_vt(X[0]) - namer.p_vt_from_features(features(X[0]))[0]) < 1e-15)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "namer.json"
        path.write_text(json.dumps({"model": "x", "pipeline": params}))
        again = VFVTNamer.load(path)
        check("loaded back from JSON, the same P(VT)",
              np.array_equal(again.p_vt_from_features(F), namer.p_vt_from_features(F)))
    refused = []
    for broken in ({**params, "features": params["features"][::-1]},
                   {**params, "coef": params["coef"][:-1]},
                   {**params, "scale": [0.0] + params["scale"][1:]},
                   {**params, "intercept": float("nan")}):
        try:
            VFVTNamer(broken)
        except ValueError:
            refused.append(True)
    check("parameters for other measures, of the wrong length, zero scale or NaN are refused",
          len(refused) == 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        far = namer.p_vt_from_features(np.full((2, len(FEATURE_NAMES)), 1e6) * [[1], [-1]])
    check("a far-out window gives 0 or 1, no NaN and no warning",
          np.isfinite(far).all() and set(np.round(far, 6)) <= {0.0, 1.0}, str(far))

    print("\nTraining and export\n")
    recs, found = fake_fragments(rng)
    load = recs.__getitem__
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        code = T.run(found, load, out)
        meta = json.loads((out / "vfvt_namer.json").read_text()) if code == 0 else {}
        gold = json.loads((out / "vfvt_namer_golden.json").read_text()) if code == 0 else {}
    check("separable windows: the check reaches the bar and both files are written",
          code == 0 and meta and gold)
    if code == 0:
        check("the check is the out-of-fold one, by patient, with its interval",
              meta["check"]["auc"] >= T.MIN_AUC and len(meta["check"]["auc_95ci"]) == 2
              and "GroupKFold" in meta["check"]["method"], str(meta["check"]["auc"]))
        check("the runtime matches scikit-learn on every window, flutter included",
              meta["verification"]["passed"] and meta["verification"]["windows"] == 70)
        check("the export says what the app needs: each measure in order and how, the zone, the input",
              [f["name"] for f in meta["features"]] == FEATURE_NAMES
              and all(f["how"] for f in meta["features"])
              and meta["decision"]["zone"] == list(C.VFVT_NAME_ZONE)
              and meta["input"]["shape"] == [WIN] and meta["decision"]["naming_enabled"] is False)
        d = T.dataset(found, load)
        alone = make_clf("logistic").fit(d["F"][d["y"] >= 0], d["y"][d["y"] >= 0])
        check("flutter is never learnt from: the fit on VF and VT alone has the same weights",
              np.allclose(params_from_pipeline(alone)["coef"], meta["pipeline"]["coef"], atol=1e-12)
              and meta["training"]["windows"]["VFL_scored_only"] == 10)
        shipped = VFVTNamer(meta["pipeline"])
        bad = [c["label"] for c in gold["cases"] if not golden_matches(shipped, c)]
        check("every golden window gives its stored measures and P(VT) again",
              not bad and len(gold["cases"]) == 31,
              f"{len(gold['cases'])} cases" + (f", off: {bad}" if bad else ""))
        check("... one of them with a measure missing, so the app's test reaches the fill value",
              any(None in c["features"] for c in gold["cases"]))

        print("\nFrom a window to the shadow name\n")
        names = {}
        for kind in ("VT", "VF"):
            dl = DecisionLayer()
            out = []
            for i in range(6):
                x = normalize(wave(kind, 5, rng)[:WIN])
                out += dl.on_shockable(i * 2.5, 0.9, None, p_vt=shipped.p_vt(x))
            names[kind] = [(a.kind, a.evidence["shadow_name"]) for a in out]
        check("synthetic VT windows scored one by one: alert 'VF/VT', named 'VT' in shadow; VF 'VF'",
              names == {"VT": [("VF/VT", "VT")], "VF": [("VF/VT", "VF")]}, str(names))

    with tempfile.TemporaryDirectory() as tmp:
        code = T.run(found, load, Path(tmp), min_auc=1.01)
        check("under the bar nothing is written", code == 1 and not any(Path(tmp).iterdir()))

    print("\nThe shipped namer (exports/)\n")
    meta_path, gold_path = REPO / "exports" / "vfvt_namer.json", REPO / "exports" / "vfvt_namer_golden.json"
    if meta_path.exists() and gold_path.exists():
        shipped = VFVTNamer.load(meta_path)
        gold = json.loads(gold_path.read_text())
        bad = [c["label"] for c in gold["cases"] if not golden_matches(shipped, c)]
        check("features/vfvt.py and the runtime still give every golden value the namer was "
              "exported with", not bad, f"{len(gold['cases'])} cases" + (f", off: {bad}" if bad else ""))
    else:
        print("  SKIP  not exported yet (python -m munji_ai.train.train_vfvt_namer)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def golden_matches(namer: VFVTNamer, case: dict, tol: float = 1e-9) -> bool:
    f = features(np.asarray(case["input"], np.float64))
    want = np.array([np.nan if v is None else v for v in case["features"]], np.float64)
    same = np.array_equal(np.isnan(f), np.isnan(want)) and np.allclose(f, want, atol=tol, equal_nan=True)
    return bool(same and abs(namer.p_vt_from_features(f)[0] - case["p_vt"]) <= tol)


if __name__ == "__main__":
    raise SystemExit(main())
