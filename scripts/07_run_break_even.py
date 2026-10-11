#!/usr/bin/env python3
"""Step 07: break-even frontiers, the cry-wolf (reviewer compliance) sweep, the alarm-design comparison
(independent and correlated scores) and the structural (recycled-content substitution) sensitivity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.alarm_designs import alarm_design_table  # noqa: E402
from src.experiments.break_even import solve_frontiers  # noqa: E402
from src.experiments.cry_wolf import sweep  # noqa: E402
from src.experiments.structural import substitution_sensitivity  # noqa: E402
from src.io_utils import write_csv, write_json  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402


def main():
    out = {sc: solve_frontiers(load_scenario(sc)) for sc in SCENARIOS}
    write_json(out, PROCESSED_DIR / "break_even.json")
    for sc, fr in out.items():
        print(sc, {k: (v["status"], v["value"]) for k, v in fr.items()})
    curves, summary = sweep()
    write_csv(curves, PROCESSED_DIR / "cry_wolf_curves.csv")
    write_csv(summary, PROCESSED_DIR / "cry_wolf_summary.csv")
    ad = alarm_design_table()
    write_csv(ad, PROCESSED_DIR / "alarm_designs.csv")
    print(ad[["score_model", "scenario", "regime", "rho", "ppv", "delta_C_vs_L0", "omega_star"]].to_string(index=False))
    st = substitution_sensitivity()
    write_csv(st, PROCESSED_DIR / "substitution_sensitivity.csv")
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
