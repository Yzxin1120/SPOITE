"""Leakage-safe nuisance estimation and standard DR pseudo-outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NuisanceConfig:
    folds: int = 5
    ridge_alpha: float = 1.0
    numerical_guard: float = 1e-6
    fixed_clip: float | None = None
    seed: int = 0


def _models(cfg: NuisanceConfig):
    reg0 = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=cfg.ridge_alpha)
    )
    reg1 = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=cfg.ridge_alpha)
    )
    prop = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.seed),
    )
    return reg0, reg1, prop


def _bound_propensity(e: np.ndarray, cfg: NuisanceConfig) -> np.ndarray:
    eps = cfg.numerical_guard if cfg.fixed_clip is None else cfg.fixed_clip
    return np.clip(e, eps, 1.0 - eps)


def dr_pseudo_outcome(
    A: np.ndarray, Y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray
) -> np.ndarray:
    return mu1 - mu0 + A * (Y - mu1) / e - (1 - A) * (Y - mu0) / (1 - e)


def crossfit_dr(
    X: np.ndarray, A: np.ndarray, Y: np.ndarray, cfg: NuisanceConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(A)
    mu0, mu1, e = (np.full(n, np.nan) for _ in range(3))
    fold_id = np.full(n, -1, dtype=int)
    splitter = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    for fold, (fit, held) in enumerate(splitter.split(X, A)):
        r0, r1, prop = _models(cfg)
        if np.unique(A[fit]).size != 2:
            raise ValueError("both treatment arms are required in every nuisance fold")
        r0.fit(X[fit][A[fit] == 0], Y[fit][A[fit] == 0])
        r1.fit(X[fit][A[fit] == 1], Y[fit][A[fit] == 1])
        prop.fit(X[fit], A[fit])
        mu0[held], mu1[held] = r0.predict(X[held]), r1.predict(X[held])
        e[held] = prop.predict_proba(X[held])[:, 1]
        fold_id[held] = fold
    e = _bound_propensity(e, cfg)
    gamma = dr_pseudo_outcome(A, Y, mu0, mu1, e)
    return gamma, {"mu0": mu0, "mu1": mu1, "e": e, "fold_id": fold_id}


def validation_dr(
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    A_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: NuisanceConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    r0, r1, prop = _models(cfg)
    r0.fit(X_train[A_train == 0], Y_train[A_train == 0])
    r1.fit(X_train[A_train == 1], Y_train[A_train == 1])
    prop.fit(X_train, A_train)
    mu0, mu1 = r0.predict(X_val), r1.predict(X_val)
    e = _bound_propensity(prop.predict_proba(X_val)[:, 1], cfg)
    return dr_pseudo_outcome(A_val, Y_val, mu0, mu1, e), {
        "mu0": mu0, "mu1": mu1, "e": e
    }
