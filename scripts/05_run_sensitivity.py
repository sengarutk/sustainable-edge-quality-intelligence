#!/usr/bin/env python3
"""Step 05: one-way sensitivity of Delta C (B4 vs L0) over registry bounds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.sensitivity import run_tornado  # noqa: E402
from src.io_utils import write_csv  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402


def main():
    df = pd.concat([run_tornado(load_scenario(sc)) for sc in SCENARIOS], ignore_index=True)
    write_csv(df, PROCESSED_DIR / "tornado_sensitivity.csv")
    for sc in SCENARIOS:
        top = df[df.scenario == sc].head(3)
        print(sc, ", ".join(f"{p} ({s:.3g})" for p, s in zip(top.parameter, top.swing)))


if __name__ == "__main__":
    main()
