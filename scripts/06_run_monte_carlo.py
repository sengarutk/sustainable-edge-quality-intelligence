#!/usr/bin/env python3
"""Step 06: Monte Carlo propagation (10,000 draws, seed 42), PRCC and Sobol global sensitivity, the same Monte
Carlo analysis with Paper A's correlated-score alert measurements, and the parameter-correlation stress test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.correlation_stress import stress_test  # noqa: E402
from src.experiments.monte_carlo import run_monte_carlo  # noqa: E402
from src.experiments.sobol import sobol_indices  # noqa: E402
from src.io_utils import write_csv, write_json  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402

N_DRAWS, SEED = 10000, 42
SOBOL_LOG2_N = 14  # 16,384 base rows per matrix


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
    for variant in ("correlated_scores", "visa_scores", "visa_correlated_scores"):
        out = {sc: run_monte_carlo(load_scenario(sc, variant=variant), n_draws=N_DRAWS, seed=SEED)[0] for sc in SCENARIOS}
        write_json(out, PROCESSED_DIR / f"monte_carlo_summary_{variant}.json")
    sob = pd.concat([sobol_indices(load_scenario(sc), m=SOBOL_LOG2_N, seed=SEED) for sc in SCENARIOS], ignore_index=True)
    write_csv(sob, PROCESSED_DIR / "sobol_indices.csv")
    print(sob.groupby("scenario").S1.sum().rename("sum of first-order indices"))
    stress = pd.concat([stress_test(load_scenario(sc), n=N_DRAWS, seed=SEED) for sc in SCENARIOS], ignore_index=True)
    write_csv(stress, PROCESSED_DIR / "correlation_stress.csv")
    print(stress.groupby("scenario")[["shift_B3_vs_L0", "shift_B3_vs_N0"]].agg(["min", "max"]))


if __name__ == "__main__":
    main()
