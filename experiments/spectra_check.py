"""VT against VF with full magnitude spectra (after Alwan, Cvetkovic and Curtis, 2018).

    python spectra_check.py

Earlier tests used one or two numbers from the spectrum (dominant frequency,
peak size). This uses the whole shape: the normalised power spectrum from 0.5
to 25 Hz, about 50 values per window, and lets the classifier find the
pattern. Scored by patient, like every other test.
"""

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


d = np.load("data/cache/holter_vt_windows.npz")
lab, grp = d["label"].astype(str), d["group"].astype(str)
rng = np.random.default_rng(0)
other = np.flatnonzero(lab == "other")
keep = np.concatenate([np.flatnonzero(lab != "other"), rng.choice(other, min(2000, len(other)), replace=False)])
X, lab, grp = d["X"][keep].astype(np.float64), lab[keep], grp[keep]
print(f"windows  VT {(lab == 'VT').sum()}, VF {(lab == 'VF').sum()}, other {(lab == 'other').sum()}")
S = np.stack([spectrum(x) for x in X])
print(f"spectrum values per window: {S.shape[1]}")

models = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
    "random forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1),
}


def honest(pos, neg, title):
    m = pos | neg
    y = pos[m].astype(int)
    print(f"\n{title}  ({len(set(grp[m & pos]))} recordings with the positive class)")
    for name, make in models.items():
        p = np.full(m.sum(), np.nan)
        for tr, te in GroupKFold(5).split(S[m], y, grp[m]):
            if len(set(y[tr])) < 2:
                continue
            clf = make()
            clf.fit(S[m][tr], y[tr])
            p[te] = clf.predict_proba(S[m][te])[:, 1]
        ok = np.isfinite(p)
        # Context: one score per recording and class, the mean over its windows.
        # If single windows carry a weak signal, pooling many sharpens it; if
        # they carry none, pooling cannot create one.
        g, yy, pp = grp[m][ok], y[ok], p[ok]
        keys = sorted(set(zip(g, yy)))
        rec_y = np.array([k[1] for k in keys])
        rec_p = np.array([pp[(g == k[0]) & (yy == k[1])].mean() for k in keys])
        print(f"  {name:<15} by patient   AUC {auc(yy, pp):.3f}   "
              f"whole recording ({len(keys)})   AUC {auc(rec_y, rec_p):.3f}")


honest(lab == "VT", lab == "VF", "VT against VF (the naming question)")
honest(lab == "VT", lab == "other", "VT against other rhythms")
honest(lab == "VF", lab == "other", "VF against other rhythms (sanity check: should be high)")
