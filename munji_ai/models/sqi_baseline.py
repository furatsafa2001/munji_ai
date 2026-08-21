"""Baseline quality gate: SQI features into a gradient-boosted tree.

This is the number the CNN has to beat. Published fused-SQI methods reach
roughly 94-97% accuracy while published deep models reach about 94% with 91%
sensitivity and 95% specificity — so the margin is genuinely narrow, and a
CNN that only matches this is not worth its cost.

Class weighting is asymmetric on purpose. Rejecting a usable window blinds
every downstream model and can hide a real cardiac event; passing a slightly
noisy one costs a little downstream accuracy. Those errors are not equal and
the loss should not treat them as if they were.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import QUALITY_CLASSES
from ..features.sqi import FEATURE_NAMES, feature_matrix

# Penalty for wrongly rejecting a usable window, relative to wrongly passing an
# unusable one. Drives specificity, the gate's governing metric.
#
# Tuned down from 3.0: at that weight the model stopped predicting 'unusable'
# at all — specificity hit target while sensitivity went to zero, which is a
# gate that passes everything and gates nothing. Specificity-first does not
# mean specificity-only.
FALSE_REJECT_PENALTY = 1.5


class SQIBaseline:
    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 random_state: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.classes_ = list(QUALITY_CLASSES)
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators, max_depth=max_depth,
            random_state=random_state, early_stopping=True,
            validation_fraction=0.15,
        )

    # ------------------------------------------------------------------ fit
    def _weights(self, y: np.ndarray) -> np.ndarray:
        """Balance classes, then add the asymmetric false-rejection penalty."""
        y = np.asarray(y, dtype=str)
        w = np.ones(len(y))
        for c in self.classes_:
            n = (y == c).sum()
            if n:
                w[y == c] = len(y) / (len(self.classes_) * n)
        w[np.isin(y, ["good", "qrs_only"])] *= FALSE_REJECT_PENALTY
        return w

    def fit(self, X_raw: np.ndarray, y: np.ndarray, fs: int | None = None,
            verbose: bool = True) -> "SQIBaseline":
        if verbose:
            print(f"extracting {len(FEATURE_NAMES)} features from "
                  f"{len(X_raw):,} windows")
        F = feature_matrix(X_raw, fs, verbose=verbose) if fs else \
            feature_matrix(X_raw, verbose=verbose)
        self.model.fit(F, np.asarray(y, dtype=str), sample_weight=self._weights(y))
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X_raw: np.ndarray, fs: int | None = None) -> np.ndarray:
        F = feature_matrix(X_raw, fs) if fs else feature_matrix(X_raw)
        return self.model.predict(F)

    def predict_from_features(self, F: np.ndarray) -> np.ndarray:
        return self.model.predict(F)

    # ------------------------------------------------------------- persist
    def save(self, path: Path) -> Path:
        import pickle

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"model": self.model, "classes": self.classes_,
                         "features": FEATURE_NAMES}, f)
        return p

    @classmethod
    def load(cls, path: Path) -> "SQIBaseline":
        import pickle

        with open(Path(path), "rb") as f:
            blob = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model, obj.classes_ = blob["model"], blob["classes"]
        return obj
