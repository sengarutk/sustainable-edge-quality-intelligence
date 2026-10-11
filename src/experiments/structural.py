"""
Structural sensitivity: model choices that are scenario definitions rather than uncertain inputs.

Recycled-content substitution. The reference model treats recovered scrap as closed-loop recycling
that displaces virgin material one-to-one (s = 1). Recycled material is often not fully
substitutable (Sanabria Garcia et al. 2023), so the analysis is repeated with s = 0.5 and with s = 0
(recovered scrap displaces nothing and only adds its processing burden), at central values and in
the Monte Carlo analysis (same draws and seed as the main analysis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.monte_carlo import COMPARISONS, sample_parameters
from src.models.pipeline_evaluator import compare, evaluate_all
from src.params import load_scenario
from src.paths import SCENARIOS

SUBSTITUTION = (1.0, 0.5, 0.0)


def substitution_sensitivity(n_draws: int = 10000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        s = load_scenario(sc)
        draws = sample_parameters(s, n_draws, seed)
        for sub in SUBSTITUTION:
            cen = evaluate_all({**s.central(), "recycled_substitution_ratio": sub}, s.class_shares)
            mc = evaluate_all({**draws, "recycled_substitution_ratio": np.full(n_draws, sub)}, s.class_shares)
            row = {"scenario": sc, "substitution": sub}
            for name in ("B3_vs_L0", "B3_vs_N0"):
                base, pol = COMPARISONS[name]
                row[f"delta_C_{name}"] = float(compare(cen[base], cen[pol]).delta["C"])
                row[f"p_{name}"] = float(np.mean(compare(mc[base], mc[pol]).delta["C"] > 0))
            rows.append(row)
    return pd.DataFrame(rows)
