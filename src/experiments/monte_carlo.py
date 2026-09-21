"""
10,000-draw Monte Carlo simulation engine across Scenarios A, B, and C.
Computes empirical percentiles (5th, 50th, 95th) and P(Delta C > 0), P(SQI > 0).
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import triang

from src.models.pipeline_evaluator import ScenarioPipelineEvaluator


def run_monte_carlo_simulation(
    scenario_path: Path,
    n_draws: int = 10000,
    seed: int = 42,
    policy_tier: str = "B4_Full_Cascade",
) -> Dict[str, Any]:
    np.random.seed(seed)
    evaluator = ScenarioPipelineEvaluator.from_yaml(scenario_path)

    def draw_triangular(nominal: float, low_scale: float = 0.8, high_scale: float = 1.2, size: int = n_draws):
        low = nominal * low_scale
        high = nominal * high_scale
        c = (nominal - low) / (high - low)
        return triang.rvs(c, loc=low, scale=high - low, size=size)

    masses = draw_triangular(evaluator.m_kg, 0.8, 1.2)
    prevalences = draw_triangular(evaluator.pi, 0.75, 1.25)
    ef_mats = draw_triangular(evaluator.ef_mat, 0.85, 1.15)
    gammas = draw_triangular(evaluator.gamma, 0.85, 1.15)
    e_rws = draw_triangular(evaluator.e_rw, 0.8, 1.2)
    c_escs = draw_triangular(evaluator.c_esc, 0.8, 1.2)
    betas = draw_triangular(evaluator.beta, 0.8, 1.2)
    recalls_b0 = np.clip(draw_triangular(0.88, 0.95, 1.05), 0.80, 0.92)
    recalls_b4 = np.clip(draw_triangular(0.99, 0.98, 1.01), 0.95, 1.00)
    delays_b4 = np.clip(draw_triangular(3.0, 0.6, 1.4), 1.0, 6.0)

    delta_ms = np.zeros(n_draws)
    delta_es = np.zeros(n_draws)
    delta_cs = np.zeros(n_draws)
    delta_hs = np.zeros(n_draws)
    sqis_balanced = np.zeros(n_draws)

    cfg_base = dict(evaluator.cfg)

    for i in range(n_draws):
        cfg = dict(cfg_base)
        cfg["part_mass_kg"] = masses[i]
        cfg["defect_prevalence"] = prevalences[i]
        cfg["material_carbon_factor_kgco2e_per_kg"] = ef_mats[i]
        cfg["grid_carbon_factor"] = gammas[i]
        cfg["rework_energy_kwh_per_unit"] = e_rws[i]
        cfg["escape_carbon_penalty_kgco2e"] = c_escs[i]
        cfg["reworkability_decay_per_sec"] = betas[i]

        ev = ScenarioPipelineEvaluator(cfg)
        b0 = ev.evaluate_policy("B0_Raw", recalls_b0[i], 180.0, 150.0, edge_active=False)
        b4 = ev.evaluate_policy(policy_tier, recalls_b4[i], 12.0, delays_b4[i], baseline_result=b0, edge_active=True)

        delta_ms[i] = b4.material.net_savings_kg
        delta_es[i] = b4.energy.net_energy_diff_kwh
        delta_cs[i] = b4.carbon.net_carbon_benefit_kgco2e
        delta_hs[i] = b4.workload.avoided_hours
        sqis_balanced[i] = b4.sqi.profile_scores["balanced"]

    results = {
        "scenario": evaluator.name,
        "n_draws": n_draws,
        "seed": seed,
        "p_delta_c_positive": float(np.mean(delta_cs > 0)),
        "p_sqi_positive": float(np.mean(sqis_balanced > 0)),
        "delta_m": {
            "p05": float(np.percentile(delta_ms, 5)),
            "p50": float(np.percentile(delta_ms, 50)),
            "p95": float(np.percentile(delta_ms, 95)),
        },
        "delta_e": {
            "p05": float(np.percentile(delta_es, 5)),
            "p50": float(np.percentile(delta_es, 50)),
            "p95": float(np.percentile(delta_es, 95)),
        },
        "delta_c": {
            "p05": float(np.percentile(delta_cs, 5)),
            "p50": float(np.percentile(delta_cs, 50)),
            "p95": float(np.percentile(delta_cs, 95)),
        },
        "delta_h": {
            "p05": float(np.percentile(delta_hs, 5)),
            "p50": float(np.percentile(delta_hs, 50)),
            "p95": float(np.percentile(delta_hs, 95)),
        },
        "sqi_balanced": {
            "p05": float(np.percentile(sqis_balanced, 5)),
            "p50": float(np.percentile(sqis_balanced, 50)),
            "p95": float(np.percentile(sqis_balanced, 95)),
        },
        "draws": {
            "delta_m": delta_ms,
            "delta_e": delta_es,
            "delta_c": delta_cs,
            "delta_h": delta_hs,
            "sqi_balanced": sqis_balanced,
        }
    }
    return results