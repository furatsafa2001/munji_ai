"""eval/vf_vt_cardiologist.py without PyTorch, on synthetic VF, VT and flutter.

VF is band-limited noise (no repeat), VT and flutter are periodic waves, so
the rate features must separate them; the checks are about the plumbing:
windows cut at the right time, patients never on both sides of a fold,
flutter never learnt from, label noise lowering the score, the abstain zone,
and the beat rule read the way the app reads it.

    python tests/test_vf_vt_cardiologist.py
"""

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai import config as C  # noqa: E402
from munji_ai.eval import vf_vt_cardiologist as V  # noqa: E402

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


def main() -> int:
    rng = np.random.default_rng(0)
    print("Windows\n")
    recs = {f"6{i:02d}": record([("VF", 60), ("VT", 60), ("VFL", 30)], rng) for i in range(10)}
    found = []
    for name in recs:
        # every window has the 20 s of history a beat window needs
        found += [{"record": name, "label": "VF", "centre_s": c, "r": 0.99} for c in (25, 35, 50)]
        found += [{"record": name, "label": "VTHR", "centre_s": c, "r": 0.99} for c in (70, 90, 110)]
        found += [{"record": name, "label": "VFL", "centre_s": 135.0, "r": 0.99},
                  {"record": name, "label": "N", "centre_s": 20.0, "r": 0.99}]
    found.append({"record": "600", "label": "VF", "centre_s": 1.0, "r": 0.99})     # too near the start
    w = V.build_windows(found, lambda name: recs[name])
    labs = dict(zip(*np.unique(w["label"], return_counts=True)))
    check("one 5 s window per fragment of a scored or shown class; N and edges left out",
          w["X"].shape == (70, WIN) and labs == {"VF": 30, "VTHR": 30, "VFL": 10}
          and w["skipped"]["at the record's edge"] == 1, str(labs))
    i = int(np.flatnonzero(w["label"] == "VTHR")[0])
    lo = int(70 * FS) - WIN // 2
    check("the window is cut from the record's own signal at the fragment's time",
          np.array_equal(w["X"][i], recs[w["record"][i]].signal[lo:lo + WIN]))
    check("... and carries VFDB's own label for it",
          set(w["vfdb"][w["label"] == "VF"]) == {"VF"} and set(w["vfdb"][w["label"] == "VTHR"]) == {"VT"})

    print("\nScoring\n")
    y = np.array([V.TRUTH.get(k, -1) for k in w["label"]])
    y_vfdb = y.copy()
    flip = np.flatnonzero(y == 0)[::3]                     # VFDB calls a third of the VF "VT"
    y_vfdb[flip] = 1
    F = {"rate": V.rate_features(w["X"]), "published": V.published_features(w["X"])}
    res = V.evaluate(F, y, w["record"], w["label"], y_vfdb)
    r = res[("rate", "logistic")]
    check("periodic VT against noise-like VF: the rate features separate them",
          r["auc"] > 0.95, f"AUC {r['auc']:.3f}")
    check("the interval comes from resampling patients and holds the estimate",
          r["ci"][0] <= r["auc"] <= r["ci"][1], f"{r['ci'][0]:.3f}-{r['ci'][1]:.3f}")
    check("flutter is scored by the fold holding its patient, never learnt from",
          np.isfinite(r["p"][w["label"] == "VFL"]).all() and np.isfinite(r["vfl_median"]))
    check("VFDB's noisier labels score lower on the very same windows",
          r["auc_vfdb"] < r["auc_same_windows"], f"{r['auc_vfdb']:.3f} < {r['auc_same_windows']:.3f}")
    p = np.zeros(len(y))
    p[y == 1] = 0.95
    p[y == 0] = 0.05
    p[np.flatnonzero(y == 0)[0]] = 0.6                     # one VF that looks like VT
    n = V.naming(y, p, 0.3, 0.7)
    check("abstain zone: the VF at 0.6 gets no name, nothing is named wrong",
          n["named"] == 59 and n["vf_as_vt"] == 0 and n["vt_as_vf"] == 0, str(n))
    n = V.naming(y, p, 0.5, 0.5)
    check("no zone: every window named, that VF named VT", n["named"] == 60 and n["vf_as_vt"] == 1)

    print("\nThe beat rule\n")
    win = C.window_samples("beat")

    class FakeLabeller:
        """3 PVCs in a row in the last 10 s for VT windows, no beat at all for VF."""
        def run(self, rec, starts):
            m = np.zeros((len(starts), win), np.uint8)
            for i, s in enumerate(starts):
                t_end = (s + win) / FS
                if 60 <= t_end - 2.5 < 120:                # a VT window of the fake records
                    for k in range(5):
                        c = win - int((1 + 0.35 * k) * FS)
                        m[i, c - 8:c + 8] = C.BEAT_CLASSES.index("PVC")
            return m, np.ones(len(starts), bool)

    guess = V.beat_rule(w, FakeLabeller())
    check("PVCs in a row -> VT, no beat -> VF, the decision layer's own guess",
          set(guess[w["label"] == "VTHR"]) == {"VT"} and set(guess[w["label"] == "VF"]) == {"VF"},
          str(dict(zip(*np.unique(guess, return_counts=True)))))
    early = V.build_windows([{"record": "600", "label": "VF", "centre_s": 10.0, "r": 0.99}],
                            lambda name: recs[name])
    check("less than 20 s of signal before the window's end: no beat window, no name",
          list(V.beat_rule(early, FakeLabeller())) == ["VF/VT"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        V.show(w, y, res, guess)
    text = buf.getvalue()
    check("the report prints every part and the verdict",
          all(s in text for s in ("1. VF or VT", "2. Naming", "3. The beat model", "best AUC")))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
