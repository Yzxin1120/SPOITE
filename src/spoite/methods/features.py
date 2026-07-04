"""Shared linear-in-parameters feature class."""

from __future__ import annotations

import numpy as np


class QuadraticFeaturizer:
    @staticmethod
    def _raw(X: np.ndarray) -> np.ndarray:
        # x3*x4 is required so the Section 8.1 near-linear CATE belongs to the
        # model class; the indicator/trigonometric nonlinear CATE does not.
        interaction = (X[:, 2] * X[:, 3])[:, None]
        return np.column_stack((X, X**2, interaction))

    def fit(self, X: np.ndarray) -> "QuadraticFeaturizer":
        Z = self._raw(X)
        self.mean_ = np.nanmean(Z, axis=0)
        self.scale_ = np.nanstd(Z, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._raw(X)
        Z = np.where(np.isnan(Z), self.mean_, Z)
        return np.column_stack((np.ones(len(X)), (Z - self.mean_) / self.scale_))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
