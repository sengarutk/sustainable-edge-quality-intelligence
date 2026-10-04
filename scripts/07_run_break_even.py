#!/usr/bin/env python3
"""Step 07: break-even frontiers at central parameter values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.break_even import solve_frontiers  # noqa: E402
from src.io_utils import write_json  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402


def main():
    out = {sc: solve_frontiers(load_scenario(sc)) for sc in SCENARIOS}
    write_json(out, PROCESSED_DIR / "break_even.json")
    for sc, fr in out.items():
        print(sc, {k: (v["status"], v["value"]) for k, v in fr.items()})


if __name__ == "__main__":
    main()
