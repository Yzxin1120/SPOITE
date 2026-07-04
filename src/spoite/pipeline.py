"""One auditable fit/tune/evaluate pipeline shared by all experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spoite.causal import NuisanceConfig, crossfit_dr, validation_dr
from spoite.methods import (
    QuadraticFeaturizer,
    fit_cpo,
    fit_mae,
    fit_mse,
    make_instances,
)
from spoite.optimization import AllocationOracle, enumerate_vertices, make_problem


METHODS = ("MSE-DR", "MAE-DR", "CPO+", "DC-CPO(band)")


@dataclass
class Prepared:
    X_train: np.ndarray
    A_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    A_val: np.ndarray
    Y_val: np.ndarray
    gamma_train: np.ndarray
    gamma_val: np.ndarray
    cost_train: np.ndarray
    cost_val: np.ndarray
    phi_train: np.ndarray
    phi_val: np.ndarray
    featurizer: QuadraticFeaturizer
    inst_train: object
    inst_val: object
    oracle: AllocationOracle
    vertices: np.ndarray


def prepare(
    train,
    validation,
    *,
    m: int,
    budget_fraction: float,
    cost_low: float,
    cost_high: float,
    nuisance: NuisanceConfig,
    seed: int,
) -> Prepared:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701]))
    cost_train = rng.uniform(cost_low, cost_high, train.n)
    cost_val = rng.uniform(cost_low, cost_high, validation.n)
    gamma_train, _ = crossfit_dr(train.X, train.A, train.Y, nuisance)
    gamma_val, _ = validation_dr(
        train.X, train.A, train.Y,
        validation.X, validation.A, validation.Y,
        nuisance,
    )
    feat = QuadraticFeaturizer()
    phi_train = feat.fit_transform(train.X)
    phi_val = feat.transform(validation.X)
    inst_train = make_instances(
        phi_train, gamma_train, cost_train, train.X, m, seed + 11
    )
    inst_val = make_instances(
        phi_val, gamma_val, cost_val, validation.X, m, seed + 13
    )
    problem = make_problem(m, budget_fraction)
    return Prepared(
        train.X, train.A, train.Y,
        validation.X, validation.A, validation.Y,
        gamma_train, gamma_val,
        cost_train, cost_val,
        phi_train, phi_val, feat,
        inst_train, inst_val,
        AllocationOracle(problem), enumerate_vertices(problem),
    )


def _prediction_loss(phi, gamma, theta, kind: str) -> float:
    residual = phi @ theta - gamma
    return float(np.mean(residual**2 if kind == "mse" else np.abs(residual)))


def _pseudo_regret(inst, theta, oracle: AllocationOracle) -> float:
    values = []
    for k in range(len(inst.batches)):
        pseudo = inst.pseudo_benefit(k)
        pred = inst.predicted_benefit(theta, k)
        optimum, _ = oracle(pseudo)
        _, decision = oracle(pred)
        values.append(max(0.0, optimum - pseudo @ decision))
    return float(np.mean(values))


def tune(
    p: Prepared,
    *,
    l2_grid: list[float],
    lr_grid: list[float],
    epochs: int,
    boundary_grid: list[float],
    band_grid: list[float],
    outer_rounds: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    fits: dict[str, np.ndarray] = {}
    selected: dict[str, dict[str, float]] = {}

    candidates = []
    for l2 in l2_grid:
        theta = fit_mse(p.phi_train, p.gamma_train, l2)
        candidates.append((_prediction_loss(p.phi_val, p.gamma_val, theta, "mse"), l2, theta))
    _, l2, fits["MSE-DR"] = min(candidates, key=lambda x: (x[0], x[1]))
    selected["MSE-DR"] = {"l2": l2}

    candidates = []
    for l2 in l2_grid:
        for lr in lr_grid:
            theta = fit_mae(p.phi_train, p.gamma_train, l2, lr, epochs, seed)
            loss = _prediction_loss(p.phi_val, p.gamma_val, theta, "mae")
            candidates.append((loss, l2, lr, theta))
    _, l2, lr, fits["MAE-DR"] = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    selected["MAE-DR"] = {"l2": l2, "lr": lr, "epochs": epochs}

    candidates = []
    for l2 in l2_grid:
        for lr in lr_grid:
            theta = fit_cpo(p.inst_train, p.oracle, l2, lr, epochs, seed)
            candidates.append((_pseudo_regret(p.inst_val, theta, p.oracle), l2, lr, theta))
    _, l2, lr, fits["CPO+"] = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    selected["CPO+"] = {"l2": l2, "lr": lr, "epochs": epochs}

    candidates = []
    for boundary in boundary_grid:
        for band in band_grid:
            theta = fit_cpo(
                p.inst_train, p.oracle, l2, lr, epochs, seed,
                theta0=fits["CPO+"],
                boundary_penalty=boundary,
                band_quantile=band,
                outer_rounds=outer_rounds,
                vertices=p.vertices,
            )
            candidates.append(
                (_pseudo_regret(p.inst_val, theta, p.oracle), boundary, band, theta)
            )
    _, boundary, band, fits["DC-CPO(band)"] = min(
        candidates, key=lambda x: (x[0], x[1], x[2])
    )
    selected["DC-CPO(band)"] = {
        "l2": l2, "lr": lr, "epochs": epochs,
        "boundary_penalty": boundary,
        "band_quantile": band,
        "outer_rounds": outer_rounds,
    }
    return fits, selected


def fit_selected(inst, phi, oracle, vertices, selected, seed):
    out = {}
    out["MSE-DR"] = fit_mse(phi, inst.gamma, selected["MSE-DR"]["l2"])
    m = selected["MAE-DR"]
    out["MAE-DR"] = fit_mae(phi, inst.gamma, m["l2"], m["lr"], int(m["epochs"]), seed)
    c = selected["CPO+"]
    out["CPO+"] = fit_cpo(inst, oracle, c["l2"], c["lr"], int(c["epochs"]), seed)
    d = selected["DC-CPO(band)"]
    out["DC-CPO(band)"] = fit_cpo(
        inst, oracle, d["l2"], d["lr"], int(d["epochs"]), seed,
        theta0=out["CPO+"],
        boundary_penalty=d["boundary_penalty"],
        band_quantile=d["band_quantile"],
        outer_rounds=int(d["outer_rounds"]),
        vertices=vertices,
    )
    return out
