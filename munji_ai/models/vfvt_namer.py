"""The VF-or-VT namer: the second stage after ShockableNet.

    from munji_ai.models.vfvt_namer import VFVTNamer
    namer = VFVTNamer.load("exports/vfvt_namer.json")
    p_vt = namer.p_vt(window)      # the normalised 5 s window ShockableNet scored
    alerts = dl.on_shockable(t, p_shockable, rate, p_vt=p_vt)

ShockableNet says "shockable"; this says which: P(VT) against VF, from the
ten measures of features/vfvt.py and a logistic regression that
train/train_vfvt_namer.py fits on cardiologists' labels (the ECG Fragment
Database). It gives a probability and nothing else: the decision layer turns
it into a name (config.VFVT_NAME_ZONE) and keeps that name in shadow mode
until doctors' readings agree with it (config.VFVT_NAMING_ENABLED).

No PyTorch and no scikit-learn here. The fitted pipeline (mean imputation,
standard scaling, logistic regression) is read from the exported JSON and
applied with numpy, the same arithmetic the app repeats in Dart:

    v_i   = x_i, or fill_i when x_i is missing
    z_i   = (v_i - mean_i) / scale_i
    P(VT) = 1 / (1 + exp(-(intercept + sum_i coef_i * z_i)))

Run it only on windows ShockableNet calls positive: the decision layer reads
P(VT) of those windows and no other, so nothing is computed while nothing
fires.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..features.vfvt import FEATURE_NAMES, features

VECTORS = ("fill", "mean", "scale", "coef")


def params_from_pipeline(clf) -> dict:
    """The numbers of a fitted vfvt_common.make_clf("logistic") pipeline."""
    steps = [s for _, s in clf.steps]
    kinds = [type(s).__name__ for s in steps]
    if kinds != ["SimpleImputer", "StandardScaler", "LogisticRegression"]:
        raise ValueError(f"expected imputer, scaler, logistic regression; got {kinds}")
    imp, sc, lr = steps
    n = len(FEATURE_NAMES)
    fill = np.asarray(imp.statistics_, np.float64)
    if fill.shape != (n,) or not np.isfinite(fill).all():
        raise ValueError("every measure needs a fill value: one was missing in every "
                         "training window, and the imputer dropped it")
    if [int(c) for c in lr.classes_] != [0, 1] or lr.coef_.shape != (1, n):
        raise ValueError("expected one logistic regression of VF (0) against VT (1)")
    return {"features": list(FEATURE_NAMES), "fill": fill.tolist(),
            "mean": np.asarray(sc.mean_, np.float64).tolist(),
            "scale": np.asarray(sc.scale_, np.float64).tolist(),
            "coef": np.asarray(lr.coef_[0], np.float64).tolist(),
            "intercept": float(lr.intercept_[0])}


class VFVTNamer:
    def __init__(self, params: dict):
        names = list(params.get("features", ()))
        if names != list(FEATURE_NAMES):
            raise ValueError(f"this namer reads {names}; features/vfvt.py computes "
                             f"{list(FEATURE_NAMES)}")
        vec = {}
        for k in VECTORS:
            v = np.asarray(params.get(k, ()), np.float64)
            if v.shape != (len(names),) or not np.isfinite(v).all():
                raise ValueError(f"{k}: needs {len(names)} finite numbers")
            vec[k] = v
        if (vec["scale"] <= 0).any():
            raise ValueError("scale: every value must be positive")
        self.intercept = float(params.get("intercept", np.nan))
        if not np.isfinite(self.intercept):
            raise ValueError("intercept: needs a finite number")
        self.fill, self.mean, self.scale, self.coef = (vec[k] for k in VECTORS)

    @classmethod
    def load(cls, path) -> "VFVTNamer":
        """From exports/vfvt_namer.json (its "pipeline"), or bare parameters."""
        meta = json.loads(Path(path).read_text())
        return cls(meta.get("pipeline", meta))

    def p_vt_from_features(self, F) -> np.ndarray:
        """P(VT) for each row of measures (FEATURE_NAMES order; NaN when missing)."""
        F = np.atleast_2d(np.asarray(F, np.float64))
        v = np.where(np.isfinite(F), F, self.fill)
        s = self.intercept + ((v - self.mean) / self.scale) @ self.coef
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(-s))

    def p_vt(self, window) -> float:
        """P(VT) of one normalised 5 s window."""
        return float(self.p_vt_from_features(features(window))[0])
