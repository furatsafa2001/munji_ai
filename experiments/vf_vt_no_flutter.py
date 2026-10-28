"""VT against VF without ventricular flutter, on CUDB and VFDB together.

    python experiments/vf_vt_no_flutter.py

The Helsinki thesis (Siddiqui 2025) excluded ventricular flutter (VFL) as
ambiguous. Earlier MUNJI tests counted it as VF, and flutter is regular and
sine-like, much like fast VT, so it may have blurred the comparison. This
keeps MUNJI's larger data (CUDB and VFDB, train and validation records only)
and adopts that correction:

    VF   annotated (VF / (VFIB, CUDB's bracketed VF
    VT   annotated (VT
    VFL  dropped

5 s windows every 2.5 s, as in the thesis, a window taking the rhythm that
covers at least half of it. Full magnitude spectra, scored by patient (5
folds), with a 95% patient-bootstrap interval.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munji_ai.data.loader import load_record
from munji_ai.eval.false_alarm_rate import fold_of_records

FS, WIN, STRIDE = 250, 1250, 625
VF_TOKENS, VT_TOKENS = {"(VF", "(VFIB"}, {"(VT"}


def spectrum(x):
    x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-8)
    f, p = welch(x, FS, nperseg=512)
    m = (f >= 0.5) & (f <= 25)
    p = p[m] / (p[m].sum() + 1e-12)
    return np.log(p + 1e-8)


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootstrap(y, p, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in pats}
    out = []
    for _ in range(n):
        sel = np.concatenate([idx[k] for k in rng.choice(pats, len(pats), replace=True)])
        a = auc(y[sel], p[sel])
        if np.isfinite(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5])


def windows(rec):
    starts = np.asarray(rec.rhythm_starts, dtype=int)
    toks = [str(t).split("\x00")[0].strip() for t in rec.rhythm_labels]
    n = len(rec.signal)
    ends = np.append(starts[1:], n)
    out = []
    for i in range(0, max(0, (n - WIN) // STRIDE + 1)):
        lo, hi = i * STRIDE, i * STRIDE + WIN
        cover = {}
        for s, e, t in zip(starts, ends, toks):
            ov = min(hi, e) - max(lo, s)
            if ov > 0:
                cover[t] = cover.get(t, 0) + ov
        if not cover:
            continue
        tok, ov = max(cover.items(), key=lambda kv: kv[1])
        if ov < WIN / 2:
            continue
        lab = "VF" if tok in VF_TOKENS else "VT" if tok in VT_TOKENS else None
        if lab:
            out.append((lab, rec.signal[lo:hi]))
    return out


def main():
    folds = fold_of_records(Path("checkpoints/shock_cv_v3/shockable_oof.npz"), Path("data/cache"))
    X, y, g = [], [], []
    for ds in ("cudb", "vfdb"):
        for (d, name) in sorted(k for k in folds if k[0] == ds):
            for lab, x in windows(load_record(d, name)):
                X.append(spectrum(np.asarray(x, dtype=np.float64)))
                y.append(int(lab == "VT"))
                g.append(f"{d}:{name}")
    X, y, g = np.array(X), np.array(y), np.array(g)
    print(f"windows    VT {y.sum():,}, VF {(y == 0).sum():,}   (flutter dropped)")
    print(f"patients   with VT {len(set(g[y == 1]))}, with VF {len(set(g[y == 0]))}, total {len(set(g))}")
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        "random forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    print("\nVT against VF, out of fold by patient")
    for name, make in models.items():
        p = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            if len(set(y[tr])) < 2:
                continue
            clf = make()
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        ok = np.isfinite(p)
        lo, hi = bootstrap(y[ok], p[ok], g[ok])
        print(f"  {name:<15} AUC {auc(y[ok], p[ok]):.3f}   95% interval {lo:.2f} to {hi:.2f}")


if __name__ == "__main__":
    main()
