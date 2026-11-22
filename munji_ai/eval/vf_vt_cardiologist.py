"""VF or VT as cardiologists label them: can a second stage name them?

    python -m munji_ai.eval.vf_vt_cardiologist
    python -m munji_ai.eval.vf_vt_cardiologist --no-beats --no-embeddings   # features only

Why again
---------
vf_vt_feasibility.py and the naming experiments found VF and fast VT
inseparable on CUDB and VFDB (AUC 0.3 to 0.6). Their truth was the
databases' own rhythm annotation, and VT was split into fast and slow by one
rate per episode. Both are wrong for this question: the episode rate misread
47% of "slow" VT windows (experiments/vt_rate_check.py), and VFDB's
annotation disagrees with cardiologists on which rhythm is which: of 162
windows the ECG Fragment Database's cardiologists call VF, VFDB calls 60 VT
(experiments/shockable_labels_check.py). A classifier taught and scored on
that truth cannot separate the two even where the signal does. The earlier
test on the fragments themselves (experiments/fragment_external.py) read
them at 360 Hz, 1.44 times too fast. This asks again on the cardiologists'
labels, every window cut from our own VFDB copy at the fragment's true time
(data/fragments.py).

Truth
-----
    VF    cardiologist VF
    VT    cardiologist VTHR and VTLR (monomorphic) and VTTdP (torsade de
          pointes, polymorphic VT)
    VFL   ventricular flutter: never trained on or scored; its P(VT) is shown,
          to decide which name it would get
One 5 s window (ShockableNet's input) centred on each fragment.

Candidates, each scored out of fold by patient (GroupKFold, 5), logistic
regression and random forest as in vf_vt_feasibility:
    rate         autocorrelation rate, regularity and "no rate"
                 (data/shockable_labels.acf_rates)
    published    features/vfvt.py: complexity, entropies, dominant
                 frequency and its concentration, periodicity, shape
    embeddings   ShockableNet's pooled features, from the fold model that
                 never saw the record (--shock, cross-validation folds)
    all          the three together
    beat rule    the decision layer's shadow naming, as the app runs it: the
                 beat model over the 20 s before the window's end, 3 PVCs in
                 a row in the last 10 s -> VT, no beat -> VF, else no name
The same windows are also scored against VFDB's own label, to show what the
labels alone did to the earlier answer.

Naming keeps an abstain zone: "VT" at P(VT) >= hi, "VF" at P(VT) <= lo, else
"VF/VT", the name the app gives today. A wrong name (VF named VT, VT named
VF) is the error that matters; the share of windows that get a name at all is
what naming would add.

Verdict, as before: a pooled out-of-fold AUC of 0.85 or more is worth a
second stage, and naming goes on only with wrong names near zero, first in
shadow mode (config.VFVT_NAMING_ENABLED).

Reads only. Writes nothing. Test-split records are skipped.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.shockable_labels import acf_rates
from .vfvt_common import auc, features, make_clf

FS = C.TARGET_FS
WIN = C.window_samples("shockable")
TRUTH = {"VF": 0, "VTHR": 1, "VTLR": 1, "VTTDP": 1}
SHOWN = ("VFL",)
KINDS = ("logistic", "forest")
ZONES = ((0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9))
BOOTSTRAP = 1000


# ----------------------------------------------------------------- windows
def build_windows(found: list, load) -> dict:
    """One 5 s window of our own VFDB signal around each located fragment of a
    class scored or shown, with VFDB's own label for that window.

    load: record name -> Record (load_record("vfdb", name, raw) in use).
    """
    from ..data.windows import dominant_label, rhythm_intervals
    recs, rows, skipped = {}, [], Counter()
    for f in found:
        if f["label"] not in TRUTH and f["label"] not in SHOWN:
            continue
        name = f["record"]
        if name not in recs:
            try:
                r = load(name)
                recs[name] = (r, rhythm_intervals(r))
            except Exception:
                recs[name] = None
        if recs[name] is None:
            skipped["record not loaded"] += 1
            continue
        r, iv = recs[name]
        lo = int(round(f["centre_s"] * FS)) - WIN // 2
        if lo < 0 or lo + WIN > len(r.signal):
            skipped["at the record's edge"] += 1
            continue
        rows.append((np.asarray(r.signal[lo:lo + WIN], np.float64), f["label"], name,
                     (lo + WIN) / FS, dominant_label(iv, lo, lo + WIN, 0.5) if iv else None))
    X = np.stack([x for x, *_ in rows]) if rows else np.zeros((0, WIN))
    return {"X": X,
            "label": np.array([r[1] for r in rows], dtype="<U8"),
            "record": np.array([r[2] for r in rows], dtype="<U16"),
            "end_s": np.array([r[3] for r in rows], dtype=float),
            "vfdb": np.array([r[4] or "mixed" for r in rows], dtype="<U8"),
            "records": {k: v[0] for k, v in recs.items() if v is not None},
            "skipped": skipped}


def rate_features(X: np.ndarray) -> np.ndarray:
    """Autocorrelation rate (NaN when irregular), regularity, and the no-rate flag."""
    rate, reg = acf_rates(X, FS)
    return np.column_stack([rate, reg, (~np.isfinite(rate)).astype(float)])


def published_features(X: np.ndarray) -> np.ndarray:
    return np.stack([features(x) for x in X]) if len(X) else np.zeros((0, 10))


# ---------------------------------------------------------------- scoring
def oof(F: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str) -> np.ndarray:
    """Out-of-fold P(VT) for every row, by patient. Rows with y < 0 (flutter)
    are never learnt from, but are scored by the fold that holds their patient."""
    from sklearn.model_selection import GroupKFold

    p = np.full(len(y), np.nan)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return p
    for tr, te in GroupKFold(n_splits).split(F, y, groups):
        tr = tr[y[tr] >= 0]
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf(kind)
        clf.fit(F[tr], y[tr])
        p[te] = clf.predict_proba(F[te])[:, 1]
    return p


def auc_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int = BOOTSTRAP,
           seed: int = 0) -> tuple:
    """95% interval of the AUC, resampling patients."""
    keep = (y >= 0) & np.isfinite(p)
    y, p, groups = y[keep], p[keep], groups[keep]
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    vals = []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(ids, len(ids), replace=True)])
        a = auc(y[idx], p[idx])
        if np.isfinite(a):
            vals.append(a)
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def naming(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> dict:
    """Names given with an abstain zone, and the wrong ones."""
    ok = (y >= 0) & np.isfinite(p)
    vt = ok & (p >= hi)
    vf = ok & (p <= lo) & ~vt
    named = vt | vf
    return {"windows": int(ok.sum()), "named": int(named.sum()),
            "vf_as_vt": int((vt & (y == 0)).sum()), "vt_as_vf": int((vf & (y == 1)).sum())}


def beat_rule(w: dict, labeller) -> np.ndarray:
    """The decision layer's shadow guess at each window's end, as the app makes it."""
    from ..decision.decision_layer import DecisionLayer, beats_from_mask
    win = C.window_samples("beat")
    out = []
    for name, t_end in zip(w["record"], w["end_s"]):
        rec = w["records"][name]
        start = int(round(t_end * FS)) - win
        if start < 0:
            out.append("VF/VT")
            continue
        masks, usable = labeller.run(rec, [start])
        dl = DecisionLayer()
        dl.on_beats(t_end, beats_from_mask(masks[0], start / FS, FS), usable=bool(usable[0]))
        out.append(dl._vfvt_guess(t_end))          # the guess VF/VT alerts carry
    return np.array(out)


def evaluate(F: dict, y: np.ndarray, groups: np.ndarray, label: np.ndarray,
             y_vfdb: np.ndarray | None = None, ci: bool = True) -> dict:
    """Every candidate, out of fold. F: name -> feature matrix (rows as y)."""
    res = {}
    for name, M in F.items():
        for kind in KINDS:
            p = oof(M, y, groups, kind)
            r = {"p": p, "auc": auc(y[y >= 0], p[y >= 0])}
            r["ci"] = auc_ci(y, p, groups) if ci else (np.nan, np.nan)
            fl = p[np.isin(label, SHOWN) & np.isfinite(p)]
            r["vfl_median"] = float(np.median(fl)) if fl.size else np.nan
            r["vfl_vt"] = float((fl >= 0.5).mean()) if fl.size else np.nan
            if y_vfdb is not None:
                both = (y >= 0) & (y_vfdb >= 0)
                po = oof(M[both], y_vfdb[both], groups[both], kind)
                pc = oof(M[both], y[both], groups[both], kind)
                r["auc_vfdb"] = auc(y_vfdb[both], po)
                r["auc_same_windows"] = auc(y[both], pc)
            res[(name, kind)] = r
    return res


# ------------------------------------------------------------------ report
def show(w: dict, y: np.ndarray, res: dict, guess: np.ndarray | None) -> None:
    lab = w["label"]
    print("\nwindows     " + ", ".join(f"{k} {v}" for k, v in Counter(lab).most_common())
          + f"  from {len(set(w['record']))} VFDB records")
    agree = Counter((("VF" if t == 0 else "VT"), v) for t, v in zip(y, w["vfdb"]) if t >= 0)
    print("VFDB's own label of the same windows, by cardiologist class:")
    for c in ("VF", "VT"):
        print(f"  {c}: " + ", ".join(f"{v} {n}" for (cc, v), n in agree.most_common() if cc == c))

    print("\n1. VF or VT by cardiologists: out-of-fold AUC by patient (95% CI, patients resampled)")
    print(f"  {'candidate':<12}{'model':<10}{'AUC':>6}{'95% CI':>15}   {'VFDB labels':>11}"
          f"{'same windows':>14}   {'flutter P(VT)':>13}")
    for (name, kind), r in res.items():
        lo, hi = r["ci"]
        print(f"  {name:<12}{kind:<10}{r['auc']:>6.3f}{f'{lo:.2f}-{hi:.2f}':>15}   "
              f"{r.get('auc_vfdb', np.nan):>11.3f}{r.get('auc_same_windows', np.nan):>14.3f}   "
              f"{r['vfl_median']:>8.2f} ({r['vfl_vt']:.0%})")
    print("  VFDB labels: the same features taught and scored on VFDB's label, on the windows\n"
          "  where VFDB says VF or VT; same windows: cardiologists' label on those windows.\n"
          "  Flutter: median P(VT), and the share that would be named VT.")

    best = max(res, key=lambda k: res[k]["auc"] if np.isfinite(res[k]["auc"]) else -1)
    p = res[best]["p"]
    print(f"\n2. Naming with an abstain zone ({best[0]}, {best[1]}: the best AUC above, so a little"
          f" optimistic)")
    print(f"  {'VF below':>9}{'VT from':>9}{'named':>12}{'wrong':>9}{'VF named VT':>13}{'VT named VF':>13}")
    for lo, hi in ZONES:
        n = naming(y, p, lo, hi)
        wrong = n["vf_as_vt"] + n["vt_as_vf"]
        print(f"  {lo:>9.2f}{hi:>9.2f}{n['named']:>6}/{n['windows']:<5}"
              f"{wrong / max(n['named'], 1):>9.1%}{n['vf_as_vt']:>13}{n['vt_as_vf']:>13}")

    if guess is not None:
        print("\n3. The beat model's shadow rule (decision_layer, 3 PVCs in a row -> VT, no beat -> VF)")
        print(f"  {'truth':<6}{'n':>5}{'VT':>6}{'VF':>6}{'VF/VT':>7}")
        for t, c in ((0, "VF"), (1, "VT")):
            g = guess[y == t]
            print(f"  {c:<6}{len(g):>5}" + "".join(f"{(g == k).sum():>{wd}}"
                                                     for k, wd in (("VT", 6), ("VF", 6), ("VF/VT", 7))))
        fl = guess[np.isin(w["label"], SHOWN)]
        if fl.size:
            print(f"  {'VFL':<6}{fl.size:>5}" + "".join(f"{(fl == k).sum():>{wd}}"
                                                        for k, wd in (("VT", 6), ("VF", 6), ("VF/VT", 7))))
        wrong = ((guess == "VT") & (y == 0)).sum() + ((guess == "VF") & (y == 1)).sum()
        named = np.isin(guess, ("VT", "VF")) & (y >= 0)
        print(f"  wrong names {wrong} of {named.sum()} named")

    a = res[best]["auc"]
    print("\n" + "=" * 70)
    print(f"best AUC {a:.3f} ({best[0]}, {best[1]}); the bar for a second stage is 0.85: "
          + ("reached." if a >= 0.85 else "not reached."))
    print("Naming goes on only with wrong names near zero, and first in shadow mode.")


# -------------------------------------------------------------------- main
def embeddings_out_of_fold(w: dict, shock: Path, cache: Path) -> np.ndarray:
    """ShockableNet's pooled features of every window, from the fold model that
    never saw its record (fold 0 for records outside the cross-validation)."""
    import torch

    from ..data.preprocess import normalize
    from .false_alarm_rate import fold_of_records, load_fold_models
    from .vf_vt_feasibility import embeddings
    models = load_fold_models(shock, torch.device("cpu"))
    fold_of = fold_of_records(shock / "shockable_oof.npz", cache)
    Z = np.stack([normalize(x) for x in w["X"]]).astype(np.float32)
    folds = np.array([fold_of.get(("vfdb", r), 0) for r in w["record"]])
    out = None
    for f in np.unique(folds):
        idx = np.flatnonzero(folds == f)
        E = embeddings(models.get(int(f), models[0]), Z[idx])
        out = np.zeros((len(Z), E.shape[1]), np.float32) if out is None else out
        out[idx] = E
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fragments", type=Path, default=None, help="default: <raw>/ecg_fragments")
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--cache", type=Path, default=C.CACHE_DIR)
    ap.add_argument("--shock", type=Path, default=Path("checkpoints/shock_cv_aha"),
                    help="cross-validation fold models (shockable_fold*.pt + shockable_oof.npz)")
    ap.add_argument("--beat-ckpt", type=Path, default=Path("checkpoints/beat_segmenter.pt"))
    ap.add_argument("--gate-ckpt", type=Path, default=Path("checkpoints/quality_gate.pt"))
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-beats", action="store_true", help="skip the beat model's rule")
    ap.add_argument("--no-embeddings", action="store_true", help="skip ShockableNet's features")
    a = ap.parse_args(argv)

    from ..data.fragments import locate
    from ..data.loader import load_record
    root = a.fragments or a.raw / "ecg_fragments"
    if not root.exists():
        print(f"no {root}: download the frag folders of PhysioNet ecg-fragment-high-risk-label")
        return 1
    found, skipped, best_r, _ = locate(root, a.raw)
    print(f"fragments   found {len(found)} in our VFDB copy (best correlation 0.8 or more)"
          + ("; skipped " + ", ".join(f"{k} {v}" for k, v in skipped.items()) if skipped else ""))
    w = build_windows(found, lambda name: load_record("vfdb", name, a.raw))
    y = np.array([TRUTH.get(k, -1) for k in w["label"]])
    if len(np.unique(y[y >= 0])) < 2:
        print("need both VF and VT windows")
        return 1
    y_vfdb = np.array([{"VF": 0, "VT": 1}.get(k, -1) for k in w["vfdb"]])
    F = {"rate": rate_features(w["X"]), "published": published_features(w["X"])}
    if not a.no_embeddings:
        F["embeddings"] = embeddings_out_of_fold(w, a.shock, a.cache)
    F["all"] = np.hstack(list(F.values()))
    guess = None
    if not a.no_beats:
        from .eval_beats_model import Labeller
        guess = beat_rule(w, Labeller(a.beat_ckpt, None if a.no_gate else a.gate_ckpt,
                                      gate=not a.no_gate))
    res = evaluate(F, y, w["record"], w["label"], y_vfdb)
    show(w, y, res, guess)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
