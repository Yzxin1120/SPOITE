#!/usr/bin/env python3
"""Run Experiment 1 (core synthetic Q1-Q3) from the frozen configuration."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spoite.causal import NuisanceConfig, crossfit_dr
from spoite.config import current_git_commit, load_config
from spoite.data import DatasetBundle, SyntheticConfig, generate_synthetic
from spoite.evaluation import decision_metrics, exact_dfi, pehe
from spoite.methods import QuadraticFeaturizer, make_instances
from spoite.pipeline import fit_selected, prepare, tune
from spoite.provenance import validate_result_rows


ROOT = Path(__file__).resolve().parents[1]


def _combine(a: DatasetBundle, b: DatasetBundle) -> DatasetBundle:
    e_true = (
        None if a.e_true is None or b.e_true is None
        else np.r_[a.e_true, b.e_true]
    )
    return DatasetBundle(
        X=np.vstack((a.X, b.X)),
        A=np.r_[a.A, b.A],
        Y=np.r_[a.Y, b.Y],
        tau_true=np.r_[a.tau_true, b.tau_true],
        e_true=e_true,
        dataset_id=f"{a.dataset_id}+{b.dataset_id}",
        metadata={"role": "final_training"},
    )


def _cfgs(
    cfg, regime: str, gamma_e: float | None = None, tail: str = "gaussian"
) -> tuple[SyntheticConfig, NuisanceConfig]:
    s = cfg["synthetic"]
    return SyntheticConfig(
        p=s["p"], ar_rho=s["ar_rho"],
        gamma_e=cfg["experiment_1"]["gamma_e"] if gamma_e is None else gamma_e,
        cate_regime=regime, noise_sd=s["noise_sd"], tail=tail,
        rare_probability=s["rare_mixture_probability"],
        rare_shift=s["rare_mixture_shift"],
    ), NuisanceConfig(
        folds=cfg["nuisance"]["folds"],
        ridge_alpha=cfg["nuisance"]["ridge_alpha"],
        numerical_guard=cfg["nuisance"]["estimated_propensity_numerical_guard"],
    )


def _bootstrap_predictions(
    final_train, test, cost_train, selected, p, nuisance, cfg, seed, n_boot
):
    names = tuple(selected)
    predictions = {name: [] for name in names}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    indices = rng.integers(0, final_train.n, size=(n_boot, final_train.n))
    for rep, idx in enumerate(indices):
        X, A, Y, cost = (
            final_train.X[idx], final_train.A[idx], final_train.Y[idx], cost_train[idx]
        )
        ncfg = NuisanceConfig(
            folds=nuisance.folds, ridge_alpha=nuisance.ridge_alpha,
            numerical_guard=nuisance.numerical_guard,
            fixed_clip=nuisance.fixed_clip, seed=seed * 1000 + rep,
        )
        gamma, _ = crossfit_dr(X, A, Y, ncfg)
        feat = QuadraticFeaturizer()
        phi = feat.fit_transform(X)
        phi_test = feat.transform(test.X)
        inst = make_instances(phi, gamma, cost, X, 16, seed * 1000 + rep)
        fitted = fit_selected(
            inst, phi, p.oracle, p.vertices, selected, seed * 1000 + rep
        )
        for name in names:
            pred = phi_test @ fitted[name]
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"non-finite bootstrap prediction: {name}")
            predictions[name].append(pred)
    return {name: np.stack(value) for name, value in predictions.items()}


def run_one(
    cfg, regime: str, seed: int, n_boot: int, *,
    gamma_e: float | None = None,
    tail: str = "gaussian",
    budget_fraction: float | None = None,
    experiment: str = "experiment_1",
    fixed_clip: float | None = None,
    dr_variant: str = "standard",
    report_methods: tuple[str, ...] = (
        "Oracle", "MSE-DR", "MAE-DR", "CPO+", "DC-CPO(band)"
    ),
) -> list[dict]:
    started = time.perf_counter()
    scfg, nuisance = _cfgs(cfg, regime, gamma_e, tail)
    if fixed_clip is not None:
        nuisance = NuisanceConfig(**{**nuisance.__dict__, "fixed_clip": fixed_clip})
    budget_fraction = (
        cfg["experiment_1"]["budget_fraction"]
        if budget_fraction is None else budget_fraction
    )
    nuisance = NuisanceConfig(**{**nuisance.__dict__, "seed": seed})
    sizes = cfg["synthetic"]
    train = generate_synthetic(sizes["n_train"], scfg, seed * 10 + 1, f"e1-{regime}-s{seed}-train")
    val = generate_synthetic(sizes["n_validation"], scfg, seed * 10 + 2, f"e1-{regime}-s{seed}-val")
    test = generate_synthetic(sizes["n_test"], scfg, seed * 10 + 3, f"e1-{regime}-s{seed}-test")
    d = cfg["decision"]
    p = prepare(
        train, val, m=d["m"], budget_fraction=budget_fraction,
        cost_low=d["cost_low"], cost_high=d["cost_high"],
        nuisance=nuisance, seed=seed,
    )
    t = cfg["training"]
    _, selected = tune(
        p, l2_grid=t["l2_grid"], lr_grid=t["learning_rate_grid"],
        epochs=t["epochs"], boundary_grid=t["dc_boundary_penalty_grid"],
        band_grid=t["dc_band_quantile_grid"], outer_rounds=t["dc_outer_rounds"],
        seed=seed,
    )

    final_train = _combine(train, val)
    gamma, nuis_diag = crossfit_dr(final_train.X, final_train.A, final_train.Y, nuisance)
    feat = QuadraticFeaturizer()
    phi = feat.fit_transform(final_train.X)
    phi_test = feat.transform(test.X)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701]))
    # Advance exactly as prepare did, then append validation costs.
    cost_train = rng.uniform(d["cost_low"], d["cost_high"], train.n)
    cost_val = rng.uniform(d["cost_low"], d["cost_high"], val.n)
    cost_final = np.r_[cost_train, cost_val]
    cost_test = np.random.default_rng(
        np.random.SeedSequence([seed, 703])
    ).uniform(d["cost_low"], d["cost_high"], test.n)
    inst = make_instances(phi, gamma, cost_final, final_train.X, d["m"], seed + 17)
    test_inst = make_instances(
        phi_test, np.zeros(test.n), cost_test, test.X, d["m"], seed + 19
    )
    fitted = fit_selected(inst, phi, p.oracle, p.vertices, selected, seed)
    predictions = {name: phi_test @ theta for name, theta in fitted.items()}
    predictions["Oracle"] = test.tau_true.copy()

    boot = {}
    if n_boot:
        boot = _bootstrap_predictions(
            final_train, test, cost_final, selected, p, nuisance, cfg, seed, n_boot
        )
        boot["Oracle"] = np.repeat(test.tau_true[None, :], n_boot, axis=0)

    elapsed = time.perf_counter() - started
    rows = []
    for name in report_methods:
        tau_hat = predictions[name]
        dm = decision_metrics(
            tau_hat, test.tau_true, cost_test, test_inst.batches, p.oracle
        )
        dfi, coverage = (np.nan, np.nan)
        if n_boot:
            dfi, coverage = exact_dfi(
                boot[name], tau_hat, cost_test, test_inst.batches, p.oracle, p.vertices
            )
        rows.append({
            "experiment": experiment,
            "config_id": cfg["_config_sha256"][:12],
            "git_commit": current_git_commit(ROOT),
            "dataset": "synthetic",
            "regime": regime,
            "gamma_e": scfg.gamma_e,
            "covariate_tail": tail,
            "budget_fraction": budget_fraction,
            "dr_variant": dr_variant,
            "seed": seed,
            "split_id": f"generated-{seed*10+1}-{seed*10+2}-{seed*10+3}",
            "method": name,
            "pehe": pehe(tau_hat, test.tau_true),
            **dm,
            "dfi": dfi,
            "dfi_competitor_coverage": coverage,
            "nuisance": json.dumps(nuisance.__dict__, sort_keys=True),
            "method_config": json.dumps(selected.get(name, {"oracle": True}), sort_keys=True),
            "decision_config": json.dumps({
                "m": d["m"], "budget_fraction": budget_fraction,
                "groups": "within_batch_X1_median",
                "cost": [d["cost_low"], d["cost_high"]],
            }, sort_keys=True),
            "bootstrap_replicates": n_boot,
            "metric_config": json.dumps({
                "dfi": "exact_all_positive_margin_competitors",
                "dfi_radius": 1.0,
                "pipeline_bootstrap_replicates": n_boot,
                "pehe": "root_mean_squared_cate_error",
            }, sort_keys=True),
            "estimated_e_min": float(nuis_diag["e"].min()),
            "estimated_e_max": float(nuis_diag["e"].max()),
            "runtime_seconds_scenario": elapsed,
            "failure_status": "ok",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    e = cfg["experiment_1"]
    start = e["seeds"]["start"] if args.seed_start is None else args.seed_start
    stop = e["seeds"]["stop_inclusive"] if args.seed_stop is None else args.seed_stop
    n_boot = (
        cfg["inference"]["pipeline_bootstrap_replicates"]
        if args.bootstrap is None else args.bootstrap
    )
    output = args.output or ROOT / "outputs/final/experiment_1_rows.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in e["cate_regimes"]:
        for seed in range(start, stop + 1):
            print(f"experiment_1 regime={regime} seed={seed} bootstrap={n_boot}", flush=True)
            rows.extend(run_one(cfg, regime, seed, n_boot))
            validate_result_rows(rows)
            pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    # Apple's Accelerate backend can leave spurious IEEE flags after correct
    # BLAS operations under NumPy 2.2. Every public boundary checks finiteness.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    main()
