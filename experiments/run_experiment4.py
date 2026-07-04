#!/usr/bin/env python3
"""Run Experiment 4 semi-synthetic benchmark validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spoite.causal import NuisanceConfig, crossfit_dr
from spoite.config import load_config
from spoite.data import (
    acic_outcome_files, load_acic_split, load_ihdp_split, load_twins_split,
)
from spoite.evaluation import decision_metrics, exact_dfi, pehe
from spoite.methods import QuadraticFeaturizer, make_instances
from spoite.pipeline import fit_selected, prepare, tune
from run_experiment1 import _bootstrap_predictions, _combine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parent / "experiment/8.3/semi/data"


def evaluate(cfg, train, val, test, scenario: str, seed: int, n_boot: int):
    start = time.perf_counter()
    d, t = cfg["decision"], cfg["training"]
    nuisance = NuisanceConfig(
        folds=cfg["nuisance"]["folds"],
        ridge_alpha=cfg["nuisance"]["ridge_alpha"],
        numerical_guard=cfg["nuisance"]["estimated_propensity_numerical_guard"],
        seed=seed,
    )
    p = prepare(
        train, val, m=d["m"], budget_fraction=0.30,
        cost_low=d["cost_low"], cost_high=d["cost_high"],
        nuisance=nuisance, seed=seed,
    )
    _, selected = tune(
        p, l2_grid=t["l2_grid"], lr_grid=t["learning_rate_grid"],
        epochs=t["epochs"], boundary_grid=t["dc_boundary_penalty_grid"],
        band_grid=t["dc_band_quantile_grid"], outer_rounds=t["dc_outer_rounds"],
        seed=seed,
    )
    final_train = _combine(train, val)
    gamma, diag = crossfit_dr(final_train.X, final_train.A, final_train.Y, nuisance)
    feat = QuadraticFeaturizer()
    phi = feat.fit_transform(final_train.X)
    phi_test = feat.transform(test.X)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701]))
    cost_final = np.r_[
        rng.uniform(d["cost_low"], d["cost_high"], train.n),
        rng.uniform(d["cost_low"], d["cost_high"], val.n),
    ]
    cost_test = np.random.default_rng(
        np.random.SeedSequence([seed, 703])
    ).uniform(d["cost_low"], d["cost_high"], test.n)
    inst = make_instances(phi, gamma, cost_final, final_train.X, d["m"], seed + 17)
    test_inst = make_instances(
        phi_test, np.zeros(test.n), cost_test, test.X, d["m"], seed + 19
    )
    fitted = fit_selected(inst, phi, p.oracle, p.vertices, selected, seed)
    pred = {name: phi_test @ theta for name, theta in fitted.items()}
    pred["Oracle"] = test.tau_true
    boot = _bootstrap_predictions(
        final_train, test, cost_final, selected, p, nuisance, cfg, seed, n_boot
    )
    boot["Oracle"] = np.repeat(test.tau_true[None, :], n_boot, axis=0)
    rows = []
    for method in ("Oracle", "MSE-DR", "MAE-DR", "CPO+", "DC-CPO(band)"):
        dm = decision_metrics(
            pred[method], test.tau_true, cost_test, test_inst.batches, p.oracle
        )
        dfi, coverage = exact_dfi(
            boot[method], pred[method], cost_test, test_inst.batches, p.oracle, p.vertices
        )
        rows.append({
            "experiment": "experiment_4",
            "config_id": cfg["_config_sha256"][:12],
            "dataset": test.metadata.get("outcome_file", scenario.split(":")[0]),
            "scenario": scenario,
            "seed": seed,
            "split_id": test.dataset_id,
            "method": method,
            "pehe": pehe(pred[method], test.tau_true),
            **dm,
            "dfi": dfi,
            "dfi_competitor_coverage": coverage,
            "nuisance": json.dumps(nuisance.__dict__, sort_keys=True),
            "method_config": json.dumps(selected.get(method, {"oracle": True}), sort_keys=True),
            "bootstrap_replicates": n_boot,
            "estimated_e_min": float(diag["e"].min()),
            "estimated_e_max": float(diag["e"].max()),
            "runtime_seconds_scenario": time.perf_counter() - start,
            "failure_status": "ok",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset", choices=("all", "ihdp", "acic", "twins"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final/experiment_4_rows.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n_boot = cfg["inference"]["pipeline_bootstrap_replicates"] if args.bootstrap is None else args.bootstrap
    jobs = []
    if args.dataset in ("all", "ihdp"):
        for r in range(100):
            jobs.append((f"IHDP:r{r}", r, lambda r=r: load_ihdp_split(args.data_root, r)))
    if args.dataset in ("all", "acic"):
        for i, path in enumerate(acic_outcome_files(args.data_root)):
            jobs.append((f"ACIC:{path.stem}", 1000 + i, lambda path=path, i=i: load_acic_split(args.data_root, path, 20260704 + i)))
    if args.dataset in ("all", "twins"):
        for temperature in cfg["experiment_4"]["twins_temperatures"]:
            for seed in range(20):
                jobs.append((
                    f"Twins:t{temperature:g}:s{seed}", 2000 + seed,
                    lambda temperature=temperature, seed=seed: load_twins_split(
                        args.data_root, temperature, seed,
                        cfg["experiment_4"]["twins_subsample"],
                    ),
                ))
    if args.limit is not None:
        jobs = jobs[:args.limit]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario, seed, loader in jobs:
        print(f"experiment_4 {scenario} bootstrap={n_boot}", flush=True)
        rows.extend(evaluate(cfg, *loader(), scenario, seed, n_boot))
        pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    main()
