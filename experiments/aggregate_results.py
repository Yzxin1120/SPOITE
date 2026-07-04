#!/usr/bin/env python3
"""Create frozen summary tables, paired contrasts, and diagnostic figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spoite.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/final"


def paired_ci(values: np.ndarray, reps: int, seed: int = 20260704):
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def summarize_experiment1(df: pd.DataFrame, cfg):
    metrics = ["pehe", "regret", "normalized_regret", "dfi", "boundary_crossing"]
    table = df.groupby(["regime", "method"])[metrics].agg(["mean", "median", "std"])
    table.to_csv(FINAL / "experiment_1_summary.csv")
    contrasts = []
    wide = df.pivot(index=["regime", "seed"], columns="method")
    for regime in df["regime"].unique():
        for metric, left, right in (
            ("regret", "CPO+", "MSE-DR"),
            ("regret", "CPO+", "MAE-DR"),
            ("regret", "DC-CPO(band)", "CPO+"),
            ("dfi", "DC-CPO(band)", "CPO+"),
            ("boundary_crossing", "DC-CPO(band)", "CPO+"),
        ):
            # lower-is-better: positive means the right-hand baseline is worse.
            values = (
                wide.loc[regime, (metric, right)]
                - wide.loc[regime, (metric, left)]
            ).to_numpy()
            mean, low, high = paired_ci(
                values, cfg["inference"]["paired_seed_bootstrap_replicates"]
            )
            contrasts.append({
                "regime": regime, "metric": metric, "method": left,
                "baseline": right, "baseline_minus_method": mean,
                "ci_low": low, "ci_high": high,
                "win_rate": float(np.mean(values > 0)),
            })
    pd.DataFrame(contrasts).to_csv(FINAL / "experiment_1_paired_contrasts.csv", index=False)

    pairs = []
    methods = [m for m in df.method.unique() if m != "Oracle"]
    for regime in df.regime.unique():
        sub = df[df.regime == regime].set_index(["seed", "method"])
        for a in methods:
            for b in methods:
                if a >= b:
                    continue
                pa, pb = sub.xs(a, level="method"), sub.xs(b, level="method")
                reversal = ((pa.pehe < pb.pehe) & (pa.regret > pb.regret)) | (
                    (pb.pehe < pa.pehe) & (pb.regret > pa.regret)
                )
                pairs.append({
                    "regime": regime, "method_a": a, "method_b": b,
                    "ranking_reversal_rate": float(reversal.mean()),
                })
    pd.DataFrame(pairs).to_csv(FINAL / "experiment_1_ranking_reversals.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = df[df.method != "Oracle"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, sub in plot.groupby("method"):
        ax.scatter(sub.pehe, sub.regret, s=18, alpha=.65, label=method)
    ax.set(xlabel="PEHE", ylabel="Decision regret")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FINAL / "experiment_1_pehe_regret.png", dpi=200)


def main():
    cfg = load_config(ROOT / "configs/final.json")
    e1 = pd.read_csv(FINAL / "experiment_1_rows.csv")
    summarize_experiment1(e1, cfg)
    for stem, groups in (
        ("experiment_3", ["gamma_e", "covariate_tail", "budget_fraction", "method"]),
        ("experiment_4", ["dataset", "method"]),
        ("appendix_clipping", ["budget_fraction", "dr_variant", "method"]),
    ):
        path = FINAL / f"{stem}_rows.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        metrics = [c for c in ("pehe", "regret", "normalized_regret", "dfi", "boundary_crossing") if c in df]
        df.groupby(groups)[metrics].agg(["mean", "median", "std"]).to_csv(
            FINAL / f"{stem}_summary.csv"
        )
    manifest = {
        p.name: {"bytes": p.stat().st_size}
        for p in sorted(FINAL.iterdir()) if p.is_file()
    }
    (FINAL / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("final summaries and manifest written")


if __name__ == "__main__":
    main()
