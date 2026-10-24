"""Combine one-method result folders produced by separate Kaggle runs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["Local-FT", "Dec-Avg", "Dec-EWC", "Dec-Replay", "SimDec-FT", "DeCo-LSTM"]
EXPECTED_METHODS = set(METHOD_ORDER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path,
                        help="Folder containing one subfolder per method.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Write a table even if one or more methods are missing.")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    rmse_by_method = {}
    for result_file in sorted(results_dir.glob("*/table1_main_results.csv")):
        frame = pd.read_csv(result_file)
        if "Method" not in frame or frame.empty:
            raise ValueError(f"No method result found in {result_file}")
        frames.append(frame)
        for method in frame["Method"].astype(str):
            rmse_file = result_file.parent / f"{method.replace('-', '_')}_R_rmse.npy"
            if rmse_file.is_file():
                rmse_by_method[method] = np.load(rmse_file)

    if not frames:
        raise FileNotFoundError(
            f"No */table1_main_results.csv files found under {results_dir}"
        )

    results = pd.concat(frames, ignore_index=True)
    duplicates = results[results["Method"].duplicated(keep=False)]["Method"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate method results found: {sorted(set(duplicates))}")

    found = set(results["Method"].astype(str))
    missing = EXPECTED_METHODS - found
    unexpected = found - EXPECTED_METHODS
    if unexpected:
        raise ValueError(f"Unexpected method names: {sorted(unexpected)}")
    if missing and not args.allow_incomplete:
        raise ValueError(
            "Missing method results: " + ", ".join(sorted(missing))
            + ". Use --allow-incomplete to write a partial table."
        )

    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    results["_order"] = results["Method"].map(order)
    results = results.sort_values("_order").drop(columns="_order")
    results.to_csv(output_dir / "table1_main_results.csv", index=False)
    if rmse_by_method:
        np.save(output_dir / "R_rmse_by_method.npy", rmse_by_method, allow_pickle=True)

    print(results.to_string(index=False))
    print(f"Saved combined table to {output_dir / 'table1_main_results.csv'}")


if __name__ == "__main__":
    main()
