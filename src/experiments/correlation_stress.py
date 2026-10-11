"""
Correlation stress test of the Monte Carlo analysis.

The main Monte Carlo analysis samples every uncertain parameter independently. The ledger gives no
empirical basis for joint distributions, but several inputs could plausibly co-vary (for example,
defects that are hard for the detector to see may also be hard for inspectors to see, and part mass,
material factor, prevalence and recovery may move together across products). Instead of asserting
correlation values, this stress test imposes a Spearman rank correlation of -0.8, -0.5, +0.5 or +0.8
on one pair of parameters at a time (Gaussian copula: Pearson rho = 2 sin(pi rho_s / 6) on the normal
scores, then the same triangular marginals as the main analysis; all other parameters independent;
same draw count and seed) and records how far the probability that B3 beats manual inspection and no
inspection moves from its independent value. Sobol indices are not recomputed under dependence,
because they lose their variance-decomposition meaning there (Iooss and Prieur 2019).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.experiments.monte_carlo import COMPARISONS, sample_parameters
from src.experiments.sobol import triangular_ppf
from src.models.pipeline_evaluator import compare, evaluate_all
from src.params import Scenario

PAIR_PARAMETERS = ("ai_recall", "manual_inspection_recall", "defect_prevalence", "part_mass", "material_carbon_factor",
                   "material_recovery_fraction", "recycling_burden", "rework_energy", "base_reworkability",
                   "collateral_multiplier")
RANK_CORRELATIONS = (-0.8, -0.5, 0.5, 0.8)
TARGETS = ("B3_vs_L0", "B3_vs_N0")


def correlated_draws(scenario: Scenario, a: str, b: str, rank_corr: float, n: int, seed: int) -> Dict[str, np.ndarray]:
    """Independent draws as in the main analysis, with parameters a and b replaced by a rank-correlated pair."""
    p = sample_parameters(scenario, n, seed)
    rng = np.random.Generator(np.random.PCG64(seed + 1))
    r = 2.0 * np.sin(np.pi * rank_corr / 6.0)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, r], [r, 1.0]], size=n)
    u = norm.cdf(z)
    for j, key in enumerate((a, b)):
        rec = scenario.records[key]
        p[key] = triangular_ppf(u[:, j], rec.low, rec.central, rec.high)
    return p


def _probabilities(scenario: Scenario, params) -> Dict[str, float]:
    res = evaluate_all(params, scenario.class_shares)
    out = {}
    for t in TARGETS:
        base, pol = COMPARISONS[t]
        out[t] = float(np.mean(compare(res[base], res[pol]).delta["C"] > 0))
    return out


def stress_test(scenario: Scenario, n: int = 10000, seed: int = 42) -> pd.DataFrame:
    keys = [k for k in PAIR_PARAMETERS if scenario.records[k].high > scenario.records[k].low]
    rows: List[Dict] = []
    ref = _probabilities(scenario, sample_parameters(scenario, n, seed))
    rows.append({"scenario": scenario.name, "a": "", "b": "", "rank_corr": 0.0,
                 **{f"p_{t}": v for t, v in ref.items()}})
    for a, b in combinations(keys, 2):
        for rc in RANK_CORRELATIONS:
            pr = _probabilities(scenario, correlated_draws(scenario, a, b, rc, n, seed))
            rows.append({"scenario": scenario.name, "a": a, "b": b, "rank_corr": rc, **{f"p_{t}": v for t, v in pr.items()}})
    df = pd.DataFrame(rows)
    for t in TARGETS:
        df[f"shift_{t}"] = df[f"p_{t}"] - ref[t]
    return df
