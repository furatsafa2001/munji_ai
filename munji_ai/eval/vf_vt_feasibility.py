"""Can a second stage tell VF from fast VT? An honest feasibility test.

    python -m munji_ai.eval.vf_vt_feasibility \
        --checkpoint checkpoints/shockable_final/shockable_net.pt \
        --models checkpoints/shock_cv_v3

Why
---
The client lists VF and wide-complex tachycardia as separate outputs.
ShockableNet only says "shockable", and simple rate and regularity measures
from the rate engine could not separate the two on this data (AUC about 0.5).
Published work does separate them on CUDB and VFDB, with complexity and shape
measures or with learned features, often at very high reported accuracy. Much
of that work splits segments at random, so one patient sits on both sides.

This script asks the same question honestly: every split is by patient, so
the score is for people the classifier never saw. It compares three
candidates for a second stage that would only run after ShockableNet fires:

    features     published measures: Lempel-Ziv complexity, permutation and
                 spectral entropy, dominant frequency and its concentration,
                 periodicity, kurtosis, skewness, zero-crossing rate, rate
                 measured from the signal
    embeddings   the 384 pooled features ShockableNet already computes
    both         the two together

The embeddings come from the final ShockableNet, which saw these patients'
windows during training, but only ever with shockable / not labels, never VF
or VT; the VF/VT classifier on top is still cross-validated by patient.

With --vtac (data/cache/vtac_windows.npz from munji_ai.data.vtac), confirmed
VT from ICU monitors joins the training side in every fold. The score that
counts is still on held-out CUDB and VFDB patients only, where VF and VT come
from the same recordings: a classifier that had merely learnt "ICU monitor"
against "Holter" would score badly there, because those VT windows are not
from an ICU monitor. VTaC's own test split is scored separately, as the share
of its confirmed VT the classifier calls VT.

Verdict: a pooled out-of-fold AUC of 0.85 or more on CUDB and VFDB makes a
second stage worth building.

Reads only. Writes nothing.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .. import config as C
from ..config import CACHE_DIR, RAW_DIR
from ..data.loader import load_record
from ..data.preprocess import normalize
from ..data.windows import _episode_bpm, dominant_label, rhythm_intervals, split_vt_by_rate
from ..models.shockable_net import build
from .false_alarm_rate import fold_of_records

FS = C.TARGET_FS
FEATURE_NAMES = ["lz_complexity", "perm_entropy", "spectral_entropy", "dominant_hz",
                 "concentration", "periodicity", "kurtosis", "skewness",
                 "zero_crossings", "bpm_signal"]


def window_truth(rec, n: int) -> list:
    """Per 5 s window: 'VF', 'VT' (150 bpm or more) or None, by the annotation."""
    iv = split_vt_by_rate(rec, rhythm_intervals(rec))
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")
    out = []
    for i in range(n):
        lab = dominant_label(iv, i * stride, i * stride + win, C.SHOCKABLE_PURITY,
                             excluded=C.SHOCKABLE_EXCLUDED) if iv else None
        out.append("VF" if lab == "VF" else "VT" if lab == "VT_FAST" else None)
    return out


def lempel_ziv(x: np.ndarray) -> float:
    """Normalised Lempel-Ziv complexity of the median-binarised signal."""
    s = (x > np.median(x)).astype(np.uint8).tobytes()
    n, i, c, seen = len(s), 0, 0, set()
    while i < n:
        k = 1
        while i + k <= n and s[i:i + k] in seen:
            k += 1
        seen.add(s[i:i + k])
        c += 1
        i += k
    return c * np.log2(n) / n


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 2) -> float:
    n = len(x) - (order - 1) * delay
    pats = np.stack([x[i * delay:i * delay + n] for i in range(order)], axis=1)
    codes = np.argsort(pats, axis=1) @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(math.factorial(order)))


def features(x: np.ndarray) -> np.ndarray:
    z = (x - x.mean()) / (x.std() + 1e-8)
    f, p = welch(z, FS, nperseg=512)
    m = (f >= 1) & (f <= 15)
    pm = p[m] / p[m].sum()
    k = int(np.argmax(p[m]))
    band = np.abs(f[m] - f[m][k]) <= 0.5
    ac = np.correlate(z, z, "full")[len(z) - 1:] / len(z)
    bpm = _episode_bpm(x, FS)
    return np.array([
        lempel_ziv(z),
        permutation_entropy(z),
        float(-(pm * np.log2(pm + 1e-12)).sum() / np.log2(len(pm))),
        float(f[m][k]),
        float(pm[band].sum()),
        float(ac[int(0.2 * FS):int(0.6 * FS)].max()),
        float(kurtosis(z)),
        float(skew(z)),
        float(np.mean(np.abs(np.diff(np.sign(z))) > 0)),
        float(bpm) if bpm else np.nan,
    ])


@torch.no_grad()
def embeddings(model, X: np.ndarray) -> np.ndarray:
    grab = {}
    h = model.head.register_forward_hook(lambda mod, inp, out: grab.update(z=inp[0]))
    out = []
    for i in range(0, len(X), 256):
        model(torch.from_numpy(X[i:i + 256].astype(np.float32)))
        out.append(grab["z"].numpy())
    h.remove()
    return np.concatenate(out)


def vtac_windows(path: Path):
    """Confirmed-VT windows from VTaC: 5 s ending at, around and after the alarm.

    The bedside monitor alarms once it has seen VT, so the run is in progress
    at the alarm; three overlapping windows from 15 to 25 s of the 40 s slice
    (alarm at 20 s) keep close to it without leaning on any model to choose.
    """
    d = np.load(path)
    keep = d["decision"].astype(bool)
    win = C.window_samples("shockable")
    starts = [int(s * FS) for s in (15.0, 17.5, 20.0)]
    raw, groups, split = [], [], []
    for x, rec, sp in zip(d["X"][keep], d["record"][keep], d["split"][keep]):
        for s0 in starts:
            raw.append(np.asarray(x[s0:s0 + win], dtype=np.float64))
            groups.append(f"vtac:{rec}")
            split.append(str(sp))
    return raw, np.array(groups), np.array(split)


def make_clf(kind: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "forest":
        return make_pipeline(SimpleImputer(), RandomForestClassifier(
            300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1))
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=5000))


def cv_scores(F: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str,
              F_extra: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold P(VT) by patient; F_extra (all VT) joins every training fold."""
    from sklearn.model_selection import GroupKFold

    n_splits = min(5, len(np.unique(groups)))
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(F, y, groups):
        Ftr, ytr = F[tr], y[tr]
        if F_extra is not None and len(F_extra):
            Ftr = np.vstack([Ftr, F_extra])
            ytr = np.concatenate([ytr, np.ones(len(F_extra), dtype=int)])
        if len(np.unique(ytr)) < 2:
            continue
        clf = make_clf(kind)
        clf.fit(Ftr, ytr)
        p[te] = clf.predict_proba(F[te])[:, 1]
    return p


def auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recalls(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> tuple:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok] >= thr
    return float((p[y == 1]).mean()), float((~p[y == 0]).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/shockable_final/shockable_net.pt"))
    ap.add_argument("--models", type=Path, default=Path("checkpoints/shock_cv_v3"),
                    help="folder whose out-of-fold file lists the train/val recordings")
    ap.add_argument("--datasets", nargs="+", default=["cudb", "vfdb"])
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--vtac", type=Path, default=None,
                    help="data/cache/vtac_windows.npz: add confirmed VT from ICU monitors")
    a = ap.parse_args(argv)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    folds = fold_of_records(a.models / "shockable_oof.npz", a.cache)
    win, stride = C.window_samples("shockable"), C.stride_samples("shockable")

    raw, norm, y, groups = [], [], [], []
    t0 = time.time()
    for ds in a.datasets:
        for (_, name) in sorted(k for k in folds if k[0] == ds):
            rec = load_record(ds, name, a.raw)
            n = 1 + (len(rec.signal) - win) // stride
            for i, t in enumerate(window_truth(rec, n)):
                if t is None:
                    continue
                x = np.asarray(rec.signal[i * stride:i * stride + win], dtype=np.float64)
                raw.append(x)
                norm.append(normalize(x))
                y.append(1 if t == "VT" else 0)
                groups.append(f"{ds}:{name}")
        print(f"  {ds:<6} read   {time.time() - t0:5.0f} s", flush=True)
    y, groups = np.array(y), np.array(groups)
    n_vt_p = len(set(groups[y == 1]))
    print(f"\nwindows     VF {int((y == 0).sum()):,}   VT {int((y == 1).sum()):,}")
    print(f"recordings  with VF {len(set(groups[y == 0]))}   with VT {n_vt_p}")
    if n_vt_p < 2:
        print("too few recordings with VT to cross-validate")
        return 1

    print("computing features ...", flush=True)
    Ff = np.stack([features(x) for x in raw])
    Fe = embeddings(model, np.stack(norm))
    both = np.hstack([Ff, Fe])

    extra = {}
    if a.vtac is not None:
        v_raw, v_groups, v_split = vtac_windows(a.vtac)
        dev = np.isin(v_split, ["train", "val"])
        print(f"VTaC        {int(dev.sum()):,} VT windows for training from "
              f"{len(set(v_groups[dev])):,} patients, {int((~dev).sum()):,} held out (its test split)")
        vf_ = np.stack([features(x) for x in v_raw])
        ve_ = embeddings(model, np.stack([normalize(x) for x in v_raw]))
        extra = {"features": (vf_[dev], vf_[~dev]), "embeddings": (ve_[dev], ve_[~dev]),
                 "both": (np.hstack([vf_, ve_])[dev], np.hstack([vf_, ve_])[~dev])}

    sets = {"features": Ff, "embeddings": Fe, "both": both}
    plan = [("features", "forest"), ("features", "logistic"),
            ("embeddings", "logistic"), ("both", "logistic")]
    candidates = {}
    for fs_name, kind in plan:
        candidates[f"{fs_name}, {kind}"] = cv_scores(sets[fs_name], y, groups, kind)
        if extra:
            candidates[f"{fs_name}, {kind} + VTaC"] = cv_scores(
                sets[fs_name], y, groups, kind, F_extra=extra[fs_name][0])

    print(f"\nheld-out CUDB and VFDB patients, out of fold (VT is the positive class)")
    print(f"  {'candidate':<34}{'AUC':>7}{'VT recall':>11}{'VF recall':>11}")
    best = None
    for name, p in candidates.items():
        a_ = auc(y, p)
        r_vt, r_vf = recalls(y, p)
        print(f"  {name:<34}{a_:>7.3f}{r_vt:>11.0%}{r_vf:>11.0%}")
        if best is None or a_ > best[1]:
            best = (name, a_)

    if extra:
        print(f"\nVTaC test split: share of its confirmed VT called VT")
        for fs_name, kind in plan:
            clf = make_clf(kind)
            Xtr = np.vstack([sets[fs_name], extra[fs_name][0]])
            ytr = np.concatenate([y, np.ones(len(extra[fs_name][0]), dtype=int)])
            clf.fit(Xtr, ytr)
            if len(extra[fs_name][1]):
                r = float((clf.predict_proba(extra[fs_name][1])[:, 1] >= 0.5).mean())
                print(f"  {fs_name + ', ' + kind:<34}{r:>7.0%}")

    print(f"\nsingle features (AUC, VT above VF)")
    for j, nm in enumerate(FEATURE_NAMES):
        print(f"  {nm:<18}{auc(y, Ff[:, j]):.3f}")

    print(f"\nbest on CUDB and VFDB: {best[0]}, AUC {best[1]:.3f}")
    print("worth building a second stage" if best[1] >= 0.85 else
          "not yet: more VT data first (PhysioNet VTaC), then test again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
