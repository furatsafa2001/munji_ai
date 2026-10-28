"""VT against VF on VFDB alone, the Helsinki set-up, with a confidence interval.

    python vfdb_only_check.py

A 2025 Helsinki thesis reached AUC 0.735 for VT against VF on VFDB, split by
patient. VFDB has 22 recordings, so a test set holds a handful of patients and
the true AUC could lie far from the measured one. This scores every patient
out of fold (leave one patient out) and bootstraps over patients for a 95%
interval: the measured AUC says how good it looked; the interval says how far
it can be trusted.
"""

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 250


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


def patient_bootstrap(y, p, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in pats}
    out = []
    for _ in range(n):
        pick = rng.choice(pats, len(pats), replace=True)
        sel = np.concatenate([idx[k] for k in pick])
        a = auc(y[sel], p[sel])
        if np.isfinite(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5])


d = np.load("data/cache/holter_vt_windows.npz")
lab, grp, ds = d["label"].astype(str), d["group"].astype(str), d["dataset"].astype(str)
keep = (ds == "vfdb") & np.isin(lab, ["VT", "VF"])
X, y, g = d["X"][keep].astype(np.float64), (lab[keep] == "VT").astype(int), grp[keep]
print(f"VFDB windows   VT {y.sum()}, VF {(y == 0).sum()}")
print(f"patients       with VT {len(set(g[y == 1]))}, with VF {len(set(g[y == 0]))}, total {len(set(g))}")
S = np.stack([spectrum(x) for x in X])

models = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
    "random forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1),
}
print(f"\nleave one patient out, VT against VF")
for name, make in models.items():
    p = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(S, y, g):
        if len(set(y[tr])) < 2:
            continue
        clf = make()
        clf.fit(S[tr], y[tr])
        p[te] = clf.predict_proba(S[te])[:, 1]
    ok = np.isfinite(p)
    lo, hi = patient_bootstrap(y[ok], p[ok], g[ok])
    print(f"  {name:<15} AUC {auc(y[ok], p[ok]):.3f}   95% interval {lo:.2f} to {hi:.2f}")
print("\nThe interval is how far the true AUC could be from the measured one,")
print("given how few patients there are. A wide interval means the number")
print("cannot be trusted either way.")
