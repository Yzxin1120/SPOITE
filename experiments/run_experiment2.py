#!/usr/bin/env python3
"""Exact and Monte Carlo verification of SPOITE Theorem 5.3 geometry."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spoite.config import current_git_commit, load_config
from spoite.provenance import validate_result_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final/experiment_2_geometry.csv")
    parser.add_argument("--figure", type=Path, default=ROOT / "outputs/final/experiment_2_geometry.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    reps = cfg["experiment_2"]["monte_carlo_replicates"]
    radius = cfg["experiment_2"]["radius"]
    rng = np.random.default_rng(82026)
    rows = []
    for a in cfg["experiment_2"]["a"]:
        for orientation, covariance in (
            ("Sigma_1", np.diag([1 / a, a])),
            ("Sigma_2", np.diag([a, 1 / a])),
        ):
            started = time.perf_counter()
            draw = rng.multivariate_normal([1.0, 0.0], covariance, size=reps)
            crossing = draw[:, 0] < 0  # choose v=(-1,0) instead of w0=(0,0)
            rows.append({
                "experiment": "experiment_2",
                "config_id": cfg["_config_sha256"][:12],
                "git_commit": current_git_commit(ROOT),
                "dataset": "appendix_A4_geometry",
                "regime": orientation,
                "seed": 82026,
                "split_id": "not_applicable",
                "nuisance": "not_applicable",
                "method_config": "analytic_geometry",
                "method": "exact_geometry",
                "decision_config": "S=conv{(0,0),(-1,0)};b0=(1,0)",
                "metric_config": "exact_DFI_radius_1;Monte_Carlo",
                "a": a,
                "orientation": orientation,
                "determinant": float(np.linalg.det(covariance)),
                "trace": float(np.trace(covariance)),
                "exact_dfi": float(radius * np.sqrt(covariance[0, 0])),
                "mc_crossing_frequency": float(crossing.mean()),
                "mc_regret": float(crossing.mean()),
                "mc_replicates": reps,
                "runtime_seconds_scenario": time.perf_counter() - started,
                "failure_status": "ok",
            })
    validate_result_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    for ax, a in zip(axes, cfg["experiment_2"]["a"]):
        for label, cov, color in (
            ("$\\Sigma_1$", np.diag([1 / a, a]), "#2878B5"),
            ("$\\Sigma_2$", np.diag([a, 1 / a]), "#C82423"),
        ):
            ax.add_patch(Ellipse(
                (1, 0), 2 * np.sqrt(cov[0, 0]), 2 * np.sqrt(cov[1, 1]),
                fill=False, lw=2, color=color, label=label,
            ))
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.scatter([1], [0], color="black", s=20)
        ax.set_title(f"a={a}")
        ax.set_xlabel("$b_1$")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("$b_2$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=200)
    print(f"wrote {args.output} and {args.figure}")


if __name__ == "__main__":
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    main()
