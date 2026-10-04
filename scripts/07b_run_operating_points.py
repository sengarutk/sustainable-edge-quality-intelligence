#!/usr/bin/env python3
"""Step 07b: detector operating curves, carbon-optimal thresholds and the product-population study."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.decision_rule import evaluate_population  # noqa: E402
from src.experiments.detector_eval import CATEGORIES  # noqa: E402
from src.experiments.operating_points import curve, optimal_thresholds, reference_points  # noqa: E402
from src.io_utils import write_csv  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402

POPULATION_SIZE, POPULATION_SEED = 20000, 7


def main():
    write_csv(reference_points(), PROCESSED_DIR / "detector_reference_points.csv")
    write_csv(pd.concat([curve(c) for c in CATEGORIES], ignore_index=True), PROCESSED_DIR / "detector_curves.csv")
    opt = optimal_thresholds([load_scenario(s) for s in SCENARIOS])
    write_csv(opt, PROCESSED_DIR / "carbon_optimal_thresholds.csv")
    print(opt.groupby("scenario")[["fpr_opt", "recall_opt", "saving_vs_q99", "saving_vs_budget"]].median())
    pop = evaluate_population(POPULATION_SIZE, POPULATION_SEED)
    write_csv(pd.DataFrame(pop), PROCESSED_DIR / "population.csv")
    print(f"population: B3 beats N0 in {100 * (pop['dc_N0'] > 0).mean():.1f}%, L0 in {100 * (pop['dc_L0'] > 0).mean():.1f}%")


if __name__ == "__main__":
    main()
