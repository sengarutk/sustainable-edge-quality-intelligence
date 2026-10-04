#!/usr/bin/env python3
"""Step 06: Monte Carlo propagation (10,000 draws, seed 42) and PRCC global sensitivity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.monte_carlo import run_monte_carlo  # noqa: E402
from src.io_utils import write_csv, write_json  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402

N_DRAWS, SEED = 10000, 42


def main():
    summaries, draws, prcc = {}, [], []
    for sc in SCENARIOS:
        s, d, p = run_monte_carlo(load_scenario(sc), n_draws=N_DRAWS, seed=SEED)
        summaries[sc] = s
        draws.append(d)
        prcc.append(p)
        c = s["comparisons"]["B3_vs_L0"]
        print(f"{sc}: P(dC>0)={c['p_delta_c_positive']:.3f}, median dC={c['delta_c_kgco2e']['p50']:.3g} kgCO2e")
    write_json(summaries, PROCESSED_DIR / "monte_carlo_summary.json")
    write_csv(pd.concat(draws, ignore_index=True), PROCESSED_DIR / "monte_carlo_draws.csv")
    write_csv(pd.concat(prcc, ignore_index=True), PROCESSED_DIR / "prcc_sensitivity.csv")


if __name__ == "__main__":
    main()
