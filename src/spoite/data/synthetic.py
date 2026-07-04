"""Synthetic DGP specified in SPOITE Section 8.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .base import DatasetBundle


@dataclass(frozen=True)
class SyntheticConfig:
    p: int = 10
    ar_rho: float = 0.5
    gamma_e: float = 2.0
    cate_regime: str = "near_linear"
    noise_sd: float = 1.0
    tail: str = "gaussian"
    rare_probability: float = 0.05
    rare_shift: float = 4.0


def propensity_direction(p: int) -> np.ndarray:
    if p < 4:
        raise ValueError("p must be at least four")
    alpha = np.zeros(p)
    alpha[:4] = (1.0, 1.0, -1.0, 0.5)
    return alpha / np.linalg.norm(alpha)


def cate(X: np.ndarray, regime: str) -> np.ndarray:
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    if regime == "near_linear":
        return 1.0 + 0.5 * x1 - 0.5 * x2 + 0.3 * x3 * x4
    if regime == "nonlinear":
        return (
            1.0 + 0.5 * x1 - 0.5 * x2
            + 0.9 * (x3 > 0.0)
            - 0.6 * np.cos(x4)
            + 0.45 * np.sin(x2 * x3)
        )
    raise ValueError(f"unknown CATE regime: {regime}")


def baseline(X: np.ndarray) -> np.ndarray:
    return X[:, 0] + 0.5 * X[:, 1] + X[:, 2] ** 2


def generate_synthetic(
    n: int, cfg: SyntheticConfig, seed: int, dataset_id: str | None = None
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    cov = cfg.ar_rho ** np.abs(
        np.subtract.outer(np.arange(cfg.p), np.arange(cfg.p))
    )
    X = rng.multivariate_normal(np.zeros(cfg.p), cov, size=n)
    mixture = np.zeros(n, dtype=bool)
    if cfg.tail == "rare_mixture":
        mixture = rng.random(n) < cfg.rare_probability
        X[mixture] += cfg.rare_shift * propensity_direction(cfg.p)
    elif cfg.tail != "gaussian":
        raise ValueError(f"unknown covariate tail: {cfg.tail}")

    e_true = expit(cfg.gamma_e * (X @ propensity_direction(cfg.p)))
    A = rng.binomial(1, e_true)
    tau = cate(X, cfg.cate_regime)
    mu0 = baseline(X)
    # Independent potential-outcome disturbances, as stated in the source DGP.
    y0 = mu0 + cfg.noise_sd * rng.standard_normal(n)
    y1 = mu0 + tau + cfg.noise_sd * rng.standard_normal(n)
    Y = np.where(A == 1, y1, y0)
    return DatasetBundle(
        X=X,
        A=A,
        Y=Y,
        tau_true=tau,
        e_true=e_true,
        dataset_id=dataset_id or f"synthetic-{cfg.cate_regime}-{cfg.tail}-s{seed}",
        metadata={
            "seed": seed,
            "gamma_e": cfg.gamma_e,
            "tail": cfg.tail,
            "rare_count": int(mixture.sum()),
            "true_propensity_clipped": False,
        },
    )
