#!/usr/bin/env python3
"""
Step 6: Execute 11-parameter tornado sensitivity analysis across Scenarios A, B, and C.
"""

from pathlib import Path
import pandas as pd

from src.experiments.sensitivity import run_tornado_sensitivity

CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")
OUTPUT_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/processed")

SCENARIOS = ["precision_component", "machined_metal", "high_value_component"]


def main():
    print("=== Step 06: Running 11-Parameter Tornado Sensitivity Sweeps ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for sc in SCENARIOS:
        sc_path = CONFIG_DIR / f"{sc}.yaml"
        df_sc = run_tornado_sensitivity(sc_path, swing_fraction=0.25)
        all_dfs.append(df_sc)
        print(f"Scenario {sc} Top 3 Sensitivity Drivers:")
        for idx, row in df_sc.head(3).iterrows():
            print(f"  {idx+1}. {row['parameter']:35s} (Swing: {row['swing']:.2f} kgCO2e)")

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = OUTPUT_DIR / "sensitivity_rankings.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Saved complete sensitivity rankings to {out_csv}")
    print("Step 06 completed successfully.\n")


if __name__ == "__main__":
    main()