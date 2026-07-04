#!/usr/bin/env python3
"""Run the Experiment 3 Q4 hard-regime factorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spoite.config import load_config
from run_experiment1 import run_one

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final/experiment_3_rows.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    e = cfg["experiment_3"]
    start = e["seeds"]["start"] if args.seed_start is None else args.seed_start
    stop = e["seeds"]["stop_inclusive"] if args.seed_stop is None else args.seed_stop
    final_boot = cfg["inference"]["pipeline_bootstrap_replicates"]
    if args.bootstrap is not None:
        final_boot = args.bootstrap
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for gamma in e["gamma_e"]:
        for tail in e["tails"]:
            for budget in e["budget_fraction"]:
                extreme = (
                    gamma in (0.5, 4.0)
                    and tail in ("gaussian", "rare_mixture")
                    and budget in (0.15, 0.45)
                )
                n_boot = final_boot if extreme else 0
                for seed in range(start, stop + 1):
                    print(
                        f"experiment_3 gamma={gamma} tail={tail} "
                        f"budget={budget} seed={seed} bootstrap={n_boot}",
                        flush=True,
                    )
                    rows.extend(run_one(
                        cfg, "nonlinear", seed, n_boot,
                        gamma_e=gamma, tail=tail, budget_fraction=budget,
                        experiment="experiment_3",
                        report_methods=("CPO+", "DC-CPO(band)"),
                    ))
                    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    main()
