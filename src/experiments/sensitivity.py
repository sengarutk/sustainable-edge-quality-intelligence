"""
One-way (tornado) sensitivity of Delta C (B4 vs L0 manual inspection).
Each uncertain registry parameter is moved to its registry low and high bound
with all others at central values, so swings reflect documented uncertainty
rather than an arbitrary common percentage.
"""

from __future__ import annotations

import pandas as pd

from src.models.pipeline_evaluator import delta_carbon
from src.params import Scenario


def run_tornado(scenario: Scenario, base="L0_Manual", policy="B3_Full_Policy") -> pd.DataFrame:
    central = scenario.central()

    def dc(params):
        return delta_carbon(params, scenario.class_shares, base, policy)

    nominal = dc(central)
    rows = []
    for key in scenario.uncertain_keys():
        r = scenario.records[key]
        lo = dc({**central, key: r.low})
        hi = dc({**central, key: r.high})
        rows.append({"scenario": scenario.name, "parameter": key, "symbol": r.symbol, "low": r.low,
                     "central": r.central, "high": r.high, "delta_c_low": lo, "delta_c_high": hi,
                     "delta_c_nominal": nominal, "swing": abs(hi - lo)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
