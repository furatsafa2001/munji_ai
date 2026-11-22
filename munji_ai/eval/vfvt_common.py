"""Shared by the VF-or-VT experiments, and free of PyTorch.

vf_vt_feasibility.py, vt_detect_feasibility.py and vf_vt_cardiologist.py read
the same published measures off a window and score them the same way. The
measures themselves live in features/vfvt.py, beside the namer that ships
them (models/vfvt_namer.py), so the experiments and the app read one
definition.

    features(x)      features/vfvt.features, re-exported (FEATURE_NAMES)
    make_clf(kind)   "logistic" (scaled, C 0.1) or "forest", balanced classes
    auc(y, p)        rank AUC, NaN scores ignored
"""

from __future__ import annotations

import numpy as np

from ..features.vfvt import FEATURE_NAMES, features, lempel_ziv, permutation_entropy

__all__ = ["FEATURE_NAMES", "features", "lempel_ziv", "permutation_entropy", "make_clf", "auc"]


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


def auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
