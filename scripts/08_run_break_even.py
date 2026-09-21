#!/usr/bin/env python3
"""
Step 8: Solve analytical and numerical break-even frontiers (pi*, e_edge*, d*, gamma*).
"""

from pathlib import Path
import json
import pandas as pd

from src.experiments.break_even import solve_break_even_frontiers

CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")
OUTPUT_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/processed")

SCENARIOS = ["precision_component", "machined_metal", "high_value_component"]


def main():
    print("=== Step 08: Solving Break-Even Frontiers ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for sc in SCENARIOS:
        sc_path = CONFIG_DIR / f"{sc}.yaml"
        res = solve_break_even_frontiers(sc_path)
        records.append(res)
        print(f"Scenario: {sc}")
        print(f"  pi* (Defect Prevalence)             : {res['pi_star'] if res['pi_star'] else 'N/A'}")
        print(f"  e_edge* (Compute Budget kWh/1k)     : {res['e_edge_star_kwh_per_1k']:.2f} kWh/1k ({res['e_edge_star_kwh_per_1k']:.2f} Wh/frame)" if res['e_edge_star_kwh_per_1k'] else "  e_edge*: N/A")
        print(f"  d* (Max Allowable Delay)            : {res['d_star_seconds']:.2f} s ({res['d_star_frames']:.1f} frames)" if res['d_star_seconds'] else "  d*: N/A")
        print(f"  gamma* (Grid Carbon Factor Boundary): {res['gamma_star']:.3f} kgCO2e/kWh" if res['gamma_star'] else "  gamma*: N/A")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "break_even_table.csv", index=False)
    with open(OUTPUT_DIR / "break_even_results.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Exported break-even results to {OUTPUT_DIR / 'break_even_results.json'}")
    print("Step 08 completed successfully.\n")


if __name__ == "__main__":
    main()