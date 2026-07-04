"""Evaluation metrics for causal prediction and constrained decisions."""

from __future__ import annotations

import numpy as np

from spoite.optimization.allocation import AllocationOracle


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def decision_metrics(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    cost: np.ndarray,
    batches: np.ndarray,
    oracle: AllocationOracle,
) -> dict[str, float]:
    regrets, normalized, agreements, crossings = [], [], [], []
    for idx in batches:
        true_b = tau_true[idx] - cost[idx]
        pred_b = tau_hat[idx] - cost[idx]
        true_obj, w_true = oracle(true_b)
        _, w_hat = oracle(pred_b)
        regret = max(0.0, true_obj - float(true_b @ w_hat))
        regrets.append(regret)
        normalized.append(regret / max(abs(true_obj), 1e-8))
        agreements.append(float(np.array_equal(w_true, w_hat)))
        crossings.append(float(not np.array_equal(w_true, w_hat)))
    return {
        "regret": float(np.mean(regrets)),
        "normalized_regret": float(np.mean(normalized)),
        "decision_agreement": float(np.mean(agreements)),
        "boundary_crossing": float(np.mean(crossings)),
    }


def exact_dfi(
    bootstrap_tau: np.ndarray,
    tau_hat: np.ndarray,
    cost: np.ndarray,
    batches: np.ndarray,
    oracle: AllocationOracle,
    vertices: np.ndarray,
    radius: float = 1.0,
) -> tuple[float, float]:
    """Definition 5.1 over every feasible competitor with positive margin."""
    values: list[float] = []
    coverage: list[float] = []
    for idx in batches:
        b_hat = tau_hat[idx] - cost[idx]
        _, w_star = oracle(b_hat)
        D = w_star - vertices
        margins = D @ b_hat
        keep = margins > 1e-10
        if not keep.any():
            values.append(np.nan)
            coverage.append(0.0)
            continue
        D = D[keep]
        margins = margins[keep]
        covariance = np.cov(bootstrap_tau[:, idx], rowvar=False, ddof=1)
        variance = np.einsum("ij,jk,ik->i", D, covariance, D)
        sd = np.sqrt(np.maximum(variance, 0.0))
        values.append(float(np.max(radius * sd / margins)))
        coverage.append(float(len(margins) / max(len(vertices) - 1, 1)))
    return float(np.nanmean(values)), float(np.mean(coverage))
