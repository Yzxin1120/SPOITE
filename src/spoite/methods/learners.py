"""Shared MSE, MAE, CPO+, and decision-calibrated CPO learners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spoite.optimization.allocation import AllocationOracle


@dataclass(frozen=True)
class DecisionInstances:
    phi: np.ndarray
    gamma: np.ndarray
    cost: np.ndarray
    batches: np.ndarray

    def predicted_benefit(self, theta: np.ndarray, k: int) -> np.ndarray:
        idx = self.batches[k]
        return self.phi[idx] @ theta - self.cost[idx]

    def pseudo_benefit(self, k: int) -> np.ndarray:
        idx = self.batches[k]
        return self.gamma[idx] - self.cost[idx]


def make_instances(
    phi: np.ndarray,
    gamma: np.ndarray,
    cost: np.ndarray,
    X: np.ndarray,
    m: int,
    seed: int,
) -> DecisionInstances:
    rng = np.random.default_rng(seed)
    n = len(phi) // m * m
    batches = rng.permutation(len(phi))[:n].reshape(-1, m)
    # Sorting makes the approved within-batch X1 median groups fixed.
    batches = np.take_along_axis(
        batches, np.argsort(X[batches, 0], axis=1, kind="stable"), axis=1
    )
    return DecisionInstances(phi, gamma, cost, batches)


def fit_mse(phi: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    penalty = np.eye(phi.shape[1]) * (len(phi) * l2)
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ target)


def fit_mae(
    phi: np.ndarray,
    target: np.ndarray,
    l2: float,
    lr: float,
    epochs: int,
    seed: int,
) -> np.ndarray:
    theta = fit_mse(phi, target, l2)
    avg = theta.copy()
    rng = np.random.default_rng(seed)
    steps = 0
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(len(phi)), max(1, len(phi) // 128)):
            grad = phi[idx].T @ np.sign(phi[idx] @ theta - target[idx]) / len(idx)
            grad += 2 * l2 * np.r_[0.0, theta[1:]]
            steps += 1
            theta -= lr / np.sqrt(steps) * grad
            avg += (theta - avg) / steps
    return avg


def _cpo_gradient(
    inst: DecisionInstances,
    theta: np.ndarray,
    ks: np.ndarray,
    oracle: AllocationOracle,
    pseudo_decisions: np.ndarray,
) -> np.ndarray:
    grad = np.zeros_like(theta)
    for k in ks:
        idx = inst.batches[k]
        pseudo = inst.pseudo_benefit(k)
        pred = inst.predicted_benefit(theta, k)
        _, w_perturbed = oracle(2 * pred - pseudo)
        grad += inst.phi[idx].T @ (2 * (w_perturbed - pseudo_decisions[k]))
    return grad / len(ks)


def _differences_and_margins(
    b_hat: np.ndarray,
    oracle: AllocationOracle,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, w_star = oracle(b_hat)
    differences = w_star - vertices
    margins = differences @ b_hat
    positive = margins > 1e-10
    if not positive.any():
        return np.empty((0, len(b_hat))), np.empty(0)
    return differences[positive], margins[positive]


def fit_cpo(
    inst: DecisionInstances,
    oracle: AllocationOracle,
    l2: float,
    lr: float,
    epochs: int,
    seed: int,
    theta0: np.ndarray | None = None,
    boundary_penalty: float = 0.0,
    band_quantile: float = 0.1,
    outer_rounds: int = 1,
    vertices: np.ndarray | None = None,
) -> np.ndarray:
    theta = fit_mse(inst.phi, inst.gamma, l2) if theta0 is None else theta0.copy()
    rng = np.random.default_rng(seed)
    rounds = outer_rounds if boundary_penalty > 0 else 1
    epochs_per_round = max(1, epochs // rounds)
    pseudo_decisions = np.stack(
        [oracle(inst.pseudo_benefit(k))[1] for k in range(len(inst.batches))]
    )
    for outer in range(rounds):
        active: list[np.ndarray] | None = None
        if boundary_penalty > 0:
            if vertices is None:
                raise ValueError("DC-CPO requires the complete vertex set")
            raw = [
                _differences_and_margins(
                    inst.predicted_benefit(theta, k), oracle, vertices
                )
                for k in range(len(inst.batches))
            ]
            pools = [margins for _, margins in raw if len(margins)]
            pooled = np.concatenate(pools) if pools else np.empty(0)
            rho = float(np.quantile(pooled, band_quantile)) if len(pooled) else -np.inf
            active = [
                differences[margins <= rho]
                for differences, margins in raw
            ]
        avg = theta.copy()
        steps = 0
        for _ in range(epochs_per_round):
            order = rng.permutation(len(inst.batches))
            for ks in np.array_split(order, max(1, len(order) // 4)):
                grad = _cpo_gradient(inst, theta, ks, oracle, pseudo_decisions)
                if active is not None:
                    for k in ks:
                        D = active[k]
                        if len(D):
                            idx = inst.batches[k]
                            error = inst.predicted_benefit(theta, k) - inst.pseudo_benefit(k)
                            grad += (
                                2 * boundary_penalty
                                * inst.phi[idx].T @ (D.T @ (D @ error))
                                / (len(ks) * len(D))
                            )
                grad += 2 * l2 * np.r_[0.0, theta[1:]]
                norm = np.linalg.norm(grad)
                if norm > 50:
                    grad *= 50 / norm
                steps += 1
                theta -= lr / np.sqrt(steps) * grad
                avg += (theta - avg) / steps
        theta = avg
    return theta


def predict(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return phi @ theta
