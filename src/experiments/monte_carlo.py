"""
Monte Carlo uncertainty propagation and global sensitivity (PRCC).

Every registry parameter with low < high is drawn independently from a
triangular(low, central, high) distribution using a seeded PCG64 generator;
the model is evaluated once on the whole draw vector (vectorised).
Partial rank correlation coefficients (Marino et al., 2008) rank the parameters
by their monotone influence on Delta C with all other parameters controlled.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.models.pipeline_evaluator import compare, evaluate_all
from src.params import Scenario

PRIMARY_COMPARISON = "B3_vs_L0"  # PRCC target
COMPARISONS = {
    "B3_vs_L0": ("L0_Manual", "B3_Full_Policy"),
    "B3_vs_N0": ("N0_NoInspection", "B3_Full_Policy"),
    "B3_vs_B0": ("B0_Raw", "B3_Full_Policy"),
}


def sample_parameters(scenario: Scenario, n_draws: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = {}
    for key in sorted(scenario.records):
        r = scenario.records[key]
        if r.high > r.low:
            draws[key] = rng.triangular(r.low, r.central, r.high, size=n_draws)
        else:
            draws[key] = np.full(n_draws, r.central)
    return draws


def _summ(x: np.ndarray) -> Dict[str, float]:
    return {"p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "mean": float(np.mean(x))}


def prcc(inputs: Dict[str, np.ndarray], output: np.ndarray, keys: List[str]) -> pd.DataFrame:
    ranks = np.column_stack([stats.rankdata(inputs[k]) for k in keys])
    ry = stats.rankdata(output)
    n, p = ranks.shape
    rows = []
    for i, k in enumerate(keys):
        others = np.column_stack([np.ones(n), np.delete(ranks, i, axis=1)])
        bx, *_ = np.linalg.lstsq(others, ranks[:, i], rcond=None)
        by, *_ = np.linalg.lstsq(others, ry, rcond=None)
        rx, rr = ranks[:, i] - others @ bx, ry - others @ by
        r = float(np.corrcoef(rx, rr)[0, 1])
        dof = n - 2 - (p - 1)
        t = r * np.sqrt(dof / max(1e-300, 1.0 - r * r))
        rows.append({"parameter": k, "prcc": r, "p_value": float(2 * stats.t.sf(abs(t), dof))})
    return pd.DataFrame(rows).sort_values("prcc", key=np.abs, ascending=False).reset_index(drop=True)


def run_monte_carlo(scenario: Scenario, n_draws: int = 10000, seed: int = 42) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    params = sample_parameters(scenario, n_draws, seed)
    results = evaluate_all(params, scenario.class_shares)
    summary = {"scenario": scenario.name, "n_draws": n_draws, "seed": seed, "comparisons": {}}
    draws = {}
    for name, (base, policy) in COMPARISONS.items():
        c = compare(results[base], results[policy])
        dC, sqi = c.delta["C"], c.sqi.scores["balanced"]
        summary["comparisons"][name] = {
            "p_delta_c_positive": float(np.mean(dC > 0)),
            "p_sqi_balanced_positive": float(np.mean(sqi > 0)),
            "delta_m_kg": _summ(c.delta["M"]), "delta_e_kwh": _summ(c.delta["E"]),
            "delta_c_kgco2e": _summ(dC), "delta_h_h": _summ(c.delta["H"]),
            "sqi_balanced": _summ(sqi),
        }
        for dim in ("M", "E", "C", "H"):
            draws[f"{name}_delta_{dim}"] = c.delta[dim]
        draws[f"{name}_sqi_balanced"] = sqi
    edge_c = results["B3_Full_Policy"].carbon
    summary["edge_carbon_share_of_primary_total"] = _summ((edge_c.edge + edge_c.hardware) / edge_c.total)
    df_draws = pd.DataFrame(draws)
    df_draws.insert(0, "draw", np.arange(n_draws))
    df_draws.insert(0, "scenario", scenario.name)
    sens = prcc(params, draws[f"{PRIMARY_COMPARISON}_delta_C"], scenario.uncertain_keys())
    sens.insert(0, "scenario", scenario.name)
    return summary, df_draws, sens
