#!/usr/bin/env python3
"""
Step 7: Execute 10,000-draw Monte Carlo simulations across Scenarios A, B, and C.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.experiments.monte_carlo import run_monte_carlo_simulation

CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")
OUTPUT_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/processed")

SCENARIOS = ["precision_component", "machined_metal", "high_value_component"]


def main():
    print("=== Step 07: Executing 10,000-Draw Monte Carlo Simulations (seed=42) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_out = {}
    dist_dfs = []

    for sc in SCENARIOS:
        sc_path = CONFIG_DIR / f"{sc}.yaml"
        print(f"Running 10,000 draws for {sc}...")
        res = run_monte_carlo_simulation(sc_path, n_draws=10000, seed=42)

        # Extract draws
        draws = res.pop("draws")
        df_draws = pd.DataFrame(draws)
        df_draws["scenario"] = sc
        dist_dfs.append(df_draws)

        summary_out[sc] = res
        print(f"  {sc}: P(Delta C > 0) = {res['p_delta_c_positive']*100:.1f}%, P(SQI > 0) = {res['p_sqi_positive']*100:.1f}%")
        print(f"  Delta C: p50 = {res['delta_c']['p50']:.2f} kgCO2e [p05: {res['delta_c']['p05']:.2f}, p95: {res['delta_c']['p95']:.2f}]")

    with open(OUTPUT_DIR / "monte_carlo_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    df_all_draws = pd.concat(dist_dfs, ignore_index=True)
    df_all_draws.to_csv(OUTPUT_DIR / "monte_carlo_distributions.csv", index=False)

    print(f"Exported Monte Carlo summaries to {OUTPUT_DIR / 'monte_carlo_summary.json'}")
    print(f"Exported raw distribution draws to {OUTPUT_DIR / 'monte_carlo_distributions.csv'}")
    print("Step 07 completed successfully.\n")


if __name__ == "__main__":
    main()