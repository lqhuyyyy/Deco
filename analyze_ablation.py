"""Summarize per-client ablation results and paired block/seed bootstrap intervals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    "DeCo-Full", "DeCo-noEWC", "DeCo-noReplay", "DeCo-noSimilarity",
    "DeCo-noGate", "Matched-EWC", "Matched-Replay", "Matched-Similarity",
]
METRICS = ["AE-RMSE", "AE-MAE", "BWT-RMSE", "BWT-MAE"]


def read_seed_outputs(paths):
    frames = []
    for path in paths:
        csv_path = Path(path) / "per_client_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing per-client metrics file: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing = set(["LCLid", "source_block", "Seed", "Method", *METRICS]) - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    unknown = set(results["Method"].unique()) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown ablation methods: {sorted(unknown)}")
    return results


def block_seed_bootstrap_delta(frame, metric, comparator, n_boot, rng):
    paired = frame.pivot_table(
        index=["Seed", "source_block", "LCLid"],
        columns="Method",
        values=metric,
        aggfunc="first",
    )
    if "DeCo-Full" not in paired or comparator not in paired:
        raise ValueError(f"Missing paired metric values for {metric}: {comparator}")
    paired = paired[["DeCo-Full", comparator]].dropna()
    paired["delta"] = paired["DeCo-Full"] - paired[comparator]

    values_by_seed = {}
    for seed, seed_frame in paired.reset_index().groupby("Seed", sort=True):
        values_by_seed[seed] = {
            block: group["delta"].to_numpy(dtype=np.float64)
            for block, group in seed_frame.groupby("source_block", sort=False)
        }
    seeds = list(values_by_seed)
    if not seeds:
        raise ValueError("No paired seed/block/client rows remain after matching.")

    estimates = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        total = 0.0
        count = 0
        for seed in sampled_seeds:
            block_map = values_by_seed[seed]
            blocks = list(block_map)
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            for block in sampled_blocks:
                block_values = block_map[block]
                total += float(block_values.sum())
                count += len(block_values)
        estimates[iteration] = total / max(1, count)

    return (
        float(paired["delta"].mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        len(seeds),
        int(paired.reset_index()["source_block"].nunique()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dirs", nargs="+", required=True,
                        help="Local result folders, one per random seed.")
    parser.add_argument("--output-dir", default="ablation_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024,
                        help="Random seed for the bootstrap sampler.")
    args = parser.parse_args()

    frame = read_seed_outputs(args.seed_dirs)
    expected = set(METHODS)
    for seed, group in frame.groupby("Seed"):
        present = set(group["Method"].unique())
        missing = expected - present
        if missing:
            raise ValueError(f"Seed {seed} is missing methods: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_means = frame.groupby(["Seed", "Method"], as_index=False)[METRICS].mean()
    summary = seed_means.groupby("Method")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "ablation_summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    comparisons = []
    for comparator in METHODS:
        if comparator == "DeCo-Full":
            continue
        for metric in METRICS:
            mean_delta, ci_low, ci_high, n_seeds, n_blocks = block_seed_bootstrap_delta(
                frame, metric, comparator, args.bootstrap_replicates, rng
            )
            full_mean = float(frame.loc[frame["Method"] == "DeCo-Full", metric].mean())
            comparator_mean = float(frame.loc[frame["Method"] == comparator, metric].mean())
            comparisons.append({
                "Metric": metric,
                "Comparator": comparator,
                "DeCo-Full mean": full_mean,
                "Comparator mean": comparator_mean,
                "Delta DeCo-minus-comparator": mean_delta,
                "95% CI low": ci_low,
                "95% CI high": ci_high,
                "Seeds": n_seeds,
                "Blocks": n_blocks,
                "Bootstrap replicates": args.bootstrap_replicates,
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "paired_block_bootstrap.csv", index=False)

    print("Per-seed summary:")
    print(seed_means.groupby("Method")[METRICS].agg(["mean", "std"]).to_string())
    print(f"\nSaved: {output_dir / 'ablation_summary.csv'}")
    print(f"Saved: {output_dir / 'paired_block_bootstrap.csv'}")
    print("For AE metrics, a negative DeCo-minus-comparator delta favors DeCo-Full.")
    print("For BWT error deltas, a more negative value means more improvement on earlier seasons.")


if __name__ == "__main__":
    main()
