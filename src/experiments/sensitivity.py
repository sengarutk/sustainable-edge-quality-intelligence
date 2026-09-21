"""
One-way 11-parameter sensitivity analysis and tornado ranking for Delta C.
Parameters: pi, m, EF_mat, eta, gamma, e_edge, t_review, e_rw, r_p, d, beta.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.models.pipeline_evaluator import ScenarioPipelineEvaluator

PARAM_KEYS = [
    "defect_prevalence",            # pi
    "part_mass_kg",                 # m
    "material_carbon_factor_kgco2e_per_kg", # EF_mat
    "material_recovery_fraction",   # eta
    "grid_carbon_factor",           # gamma
    "edge_energy_kwh_per_1k",       # e_edge
    "review_duration_seconds",      # t_review
    "rework_energy_kwh_per_unit",   # e_rw
    "actionable_recall",            # r_p
    "policy_delay_frames",          # d
    "reworkability_decay_per_sec",  # beta
]


def run_tornado_sensitivity(
    scenario_path: Path,
    policy_tier: str = "B4_Full_Cascade",
    swing_fraction: float = 0.25,
) -> pd.DataFrame:
    evaluator = ScenarioPipelineEvaluator.from_yaml(scenario_path)

    # Baseline policy: B0_Raw with late detection
    baseline_policy = ("B0_Raw", 0.88, 180.0, 150.0)
    eval_policy = ("B4_Full_Cascade", 0.99, 12.0, 3.0)

    b0_base = evaluator.evaluate_policy(*baseline_policy, edge_active=False)
    b4_base = evaluator.evaluate_policy(*eval_policy, baseline_result=b0_base, edge_active=True)
    delta_c_nominal = b4_base.carbon.net_carbon_benefit_kgco2e

    results = []

    for p in PARAM_KEYS:
        if p == "actionable_recall":
            nom = 0.99
            low = max(0.85, nom * (1.0 - swing_fraction))
            high = min(1.00, nom * (1.0 + swing_fraction))
        elif p == "policy_delay_frames":
            nom = 3.0
            low = max(0.0, nom * (1.0 - swing_fraction))
            high = nom * (1.0 + swing_fraction)
        else:
            nom = float(getattr(evaluator, {
                "defect_prevalence": "pi",
                "part_mass_kg": "m_kg",
                "material_carbon_factor_kgco2e_per_kg": "ef_mat",
                "material_recovery_fraction": "eta",
                "grid_carbon_factor": "gamma",
                "edge_energy_kwh_per_1k": "e_edge",
                "review_duration_seconds": "t_review",
                "rework_energy_kwh_per_unit": "e_rw",
                "reworkability_decay_per_sec": "beta",
            }[p]))
            low = max(0.0, nom * (1.0 - swing_fraction))
            high = nom * (1.0 + swing_fraction)
            if p == "material_recovery_fraction":
                high = min(1.0, high)

        # Evaluate at low
        cfg_low = dict(evaluator.cfg)
        r_p_low = eval_policy[1]
        d_low = eval_policy[3]

        if p == "actionable_recall":
            r_p_low = low
        elif p == "policy_delay_frames":
            d_low = low
        else:
            cfg_low[p] = low

        ev_low = ScenarioPipelineEvaluator(cfg_low)
        b0_l = ev_low.evaluate_policy(baseline_policy[0], baseline_policy[1], baseline_policy[2], baseline_policy[3], edge_active=False)
        b4_l = ev_low.evaluate_policy(eval_policy[0], r_p_low, eval_policy[2], d_low, baseline_result=b0_l, edge_active=True)
        dc_low = b4_l.carbon.net_carbon_benefit_kgco2e

        # Evaluate at high
        cfg_high = dict(evaluator.cfg)
        r_p_high = eval_policy[1]
        d_high = eval_policy[3]

        if p == "actionable_recall":
            r_p_high = high
        elif p == "policy_delay_frames":
            d_high = high
        else:
            cfg_high[p] = high

        ev_high = ScenarioPipelineEvaluator(cfg_high)
        b0_h = ev_high.evaluate_policy(baseline_policy[0], baseline_policy[1], baseline_policy[2], baseline_policy[3], edge_active=False)
        b4_h = ev_high.evaluate_policy(eval_policy[0], r_p_high, eval_policy[2], d_high, baseline_result=b0_h, edge_active=True)
        dc_high = b4_h.carbon.net_carbon_benefit_kgco2e

        swing = abs(dc_high - dc_low)
        results.append({
            "scenario": evaluator.name,
            "parameter": p,
            "nominal": nom,
            "low": low,
            "high": high,
            "delta_c_low": dc_low,
            "delta_c_high": dc_high,
            "delta_c_nominal": delta_c_nominal,
            "swing": swing,
        })

    df = pd.DataFrame(results).sort_values(by="swing", ascending=False).reset_index(drop=True)
    return df