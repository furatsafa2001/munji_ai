import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 250
SEG, HOP = int(0.4 * FS), int(0.04 * FS)          # 0.4 s pieces, every 40 ms
MIN_LAG, MAX_LAG = int(0.2 * FS), int(2.0 * FS)   # compare pieces 0.2 to 2 s apart

def similarity_features(x):
    x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-8)
    starts = np.arange(0, len(x) - SEG, HOP)
    P = np.stack([x[s:s + SEG] for s in starts])
    P = P - P.mean(1, keepdims=True)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ P.T                                    # the similarity map
    lag = np.abs(starts[:, None] - starts[None, :])
    band = (lag >= MIN_LAG) & (lag <= MAX_LAG)
    Sb = np.where(band, S, -np.inf)
    best = Sb.max(1)
    best_lag = starts[Sb.argmax(1)] - starts
    lags = np.arange(MIN_LAG, MAX_LAG + 1, HOP)
    prof = np.array([np.mean(np.diagonal(S, k // HOP)) for k in lags if k // HOP < len(S)])
    return np.array([best.mean(), (best > 0.8).mean(), (best > 0.9).mean(),
                     S[band].mean(), prof.max(), np.std(np.abs(best_lag)) / FS])

NAMES = ["mean best match", "share above 0.8", "share above 0.9",
         "mean similarity", "strongest repeat", "spread of repeat time"]

def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

d = np.load("data/cache/holter_vt_windows.npz")
lab, grp = d["label"].astype(str), d["group"].astype(str)
rng = np.random.default_rng(0)
other = np.flatnonzero(lab == "other")
keep = np.concatenate([np.flatnonzero(lab != "other"), rng.choice(other, 2000, replace=False)])
X, lab, grp = d["X"][keep].astype(np.float64), lab[keep], grp[keep]
print(f"windows  VT {(lab == 'VT').sum()}, VF {(lab == 'VF').sum()}, other {(lab == 'other').sum()}")
F = np.stack([similarity_features(x) for x in X])

def honest(mask_pos, mask_neg, title):
    m = mask_pos | mask_neg
    y = mask_pos[m].astype(int)
    print(f"\n{title}  ({len(set(grp[m & mask_pos]))} recordings with the positive class)")
    for j, nm in enumerate(NAMES):
        print(f"  {nm:<24}{auc(y, F[m][:, j]):.3f}")
    p = np.full(m.sum(), np.nan)
    for tr, te in GroupKFold(5).split(F[m], y, grp[m]):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        clf.fit(F[m][tr], y[tr])
        p[te] = clf.predict_proba(F[m][te])[:, 1]
    ok = np.isfinite(p)
    print(f"  all six together, by patient   AUC {auc(y[ok], p[ok]):.3f}")

honest(lab == "VT", lab == "VF", "VT against VF (the naming question)")
honest(lab == "VT", lab == "other", "VT against other rhythms")
honest(lab == "VF", lab == "other", "VF against other rhythms (sanity check: should be high)")
