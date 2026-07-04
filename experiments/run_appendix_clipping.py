#!/usr/bin/env python3
"""Pre-declared fixed estimated-propensity clipping sensitivity appendix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spoite.config import load_config
from run_experiment1 import run_one
from spoite.provenance import validate_result_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final/appendix_clipping_rows.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    e = cfg["appendix_clipping"]
    start = e["seeds"]["start"] if args.seed_start is None else args.seed_start
    stop = e["seeds"]["stop_inclusive"] if args.seed_stop is None else args.seed_stop
    n_boot = cfg["inference"]["pipeline_bootstrap_replicates"] if args.bootstrap is None else args.bootstrap
    variants = [("standard", None)] + [(f"fixed_clip_{x:g}", x) for x in e["epsilon"]]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for budget in e["budget_fraction"]:
        for label, epsilon in variants:
            for seed in range(start, stop + 1):
                print(f"appendix budget={budget} variant={label} seed={seed}", flush=True)
                rows.extend(run_one(
                    cfg, "nonlinear", seed, n_boot,
                    gamma_e=e["gamma_e"], tail=e["tail"],
                    budget_fraction=budget, experiment="appendix_clipping",
                    fixed_clip=epsilon, dr_variant=label,
                    report_methods=("CPO+", "DC-CPO(band)"),
                ))
                validate_result_rows(rows)
                pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    main()
