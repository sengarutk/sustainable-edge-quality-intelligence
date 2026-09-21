"""
Analytical and numerical break-even frontier solvers for Paper B.
Solves for pi* (prevalence), e_edge* (compute energy), d* (delay), and gamma* (grid carbon).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from scipy.optimize import root_scalar

from src.models.pipeline_evaluator import ScenarioPipelineEvaluator


def solve_break_even_frontiers(
    scenario_path: Path,
    policy_tier: str = "B4_Full_Cascade",
    baseline_tier: str = "B0_Raw",
    recall_b0: float = 0.88,
    recall_b4: float = 0.99,
    alert_rate_b0: float = 180.0,
    alert_rate_b4: float = 12.0,
    delay_frames_b0: float = 150.0,
    delay_frames_b4: float = 3.0,
) -> Dict[str, Any]:
    evaluator = ScenarioPipelineEvaluator.from_yaml(scenario_path)

    # 1. Total Defect Prevalence Break-Even (pi*)
    # At pi = 0, Delta C = -0.071 kgCO2e < 0 (net-negative regime)
    def delta_c_vs_pi(pi_val: float) -> float:
        cfg = dict(evaluator.cfg)
        cfg["defect_prevalence"] = float(pi_val)
        ev = ScenarioPipelineEvaluator(cfg)
        b0 = ev.evaluate_policy(baseline_tier, recall_b0, alert_rate_b0, delay_frames_b0, edge_active=False)
        b4 = ev.evaluate_policy(policy_tier, recall_b4, alert_rate_b4, delay_frames_b4, baseline_result=b0, edge_active=True)
        return b4.carbon.net_carbon_benefit_kgco2e

    pi_star = None
    try:
        f_0 = delta_c_vs_pi(0.0)
        f_hi = delta_c_vs_pi(0.10)
        if f_0 * f_hi <= 0:
            sol = root_scalar(delta_c_vs_pi, bracket=[0.0, 0.10], method="brentq")
            if sol.converged:
                pi_star = float(sol.root)
    except Exception:
        pass

    # 1b. Pure Material Scrap Break-Even (pi*_scrap, ignoring field escape penalties)
    def delta_c_scrap_only(pi_val: float) -> float:
        cfg = dict(evaluator.cfg)
        cfg["defect_prevalence"] = float(pi_val)
        cfg["escape_carbon_penalty_kgco2e"] = 0.0
        ev = ScenarioPipelineEvaluator(cfg)
        b0 = ev.evaluate_policy(baseline_tier, recall_b0, alert_rate_b0, delay_frames_b0, edge_active=False)
        b4 = ev.evaluate_policy(policy_tier, recall_b4, alert_rate_b4, delay_frames_b4, baseline_result=b0, edge_active=True)
        return b4.carbon.net_carbon_benefit_kgco2e

    pi_star_scrap = None
    try:
        f_s0 = delta_c_scrap_only(0.0)
        f_shi = delta_c_scrap_only(0.20)
        if f_s0 * f_shi <= 0:
            sol_s = root_scalar(delta_c_scrap_only, bracket=[0.0, 0.20], method="brentq")
            if sol_s.converged:
                pi_star_scrap = float(sol_s.root)
    except Exception:
        pass

    # 2. Maximum Allowable Compute Energy Budget (e_edge*) [kWh per 1,000 units]
    def delta_c_vs_eedge(e_val: float) -> float:
        cfg = dict(evaluator.cfg)
        cfg["edge_energy_kwh_per_1k"] = float(e_val)
        ev = ScenarioPipelineEvaluator(cfg)
        b0 = ev.evaluate_policy(baseline_tier, recall_b0, alert_rate_b0, delay_frames_b0, edge_active=False)
        b4 = ev.evaluate_policy(policy_tier, recall_b4, alert_rate_b4, delay_frames_b4, baseline_result=b0, edge_active=True)
        return b4.carbon.net_carbon_benefit_kgco2e

    e_edge_star_kwh = None
    try:
        f0 = delta_c_vs_eedge(0.0)
        f_max = delta_c_vs_eedge(10000.0)
        if f0 * f_max <= 0:
            sol = root_scalar(delta_c_vs_eedge, bracket=[0.0, 10000.0], method="brentq")
            if sol.converged:
                e_edge_star_kwh = float(sol.root)
    except Exception:
        pass

    # 3. Maximum Allowable Policy Delay (d*)
    def delta_c_vs_delay(d_val: float) -> float:
        b0 = evaluator.evaluate_policy(baseline_tier, recall_b0, alert_rate_b0, delay_frames_b0, edge_active=False)
        b4 = evaluator.evaluate_policy(policy_tier, recall_b4, alert_rate_b4, float(d_val), baseline_result=b0, edge_active=True)
        return b4.carbon.net_carbon_benefit_kgco2e

    d_star_frames = None
    d_star_seconds = None
    try:
        f_0 = delta_c_vs_delay(0.0)
        f_1000 = delta_c_vs_delay(1000.0)
        if f_0 * f_1000 <= 0:
            sol = root_scalar(delta_c_vs_delay, bracket=[0.0, 1000.0], method="brentq")
            if sol.converged:
                d_star_frames = float(sol.root)
                d_star_seconds = d_star_frames / evaluator.fps
    except Exception:
        pass

    # 4. Grid Carbon Factor Boundary (gamma*)
    def delta_c_vs_gamma(gamma_val: float) -> float:
        cfg = dict(evaluator.cfg)
        cfg["grid_carbon_factor"] = float(gamma_val)
        ev = ScenarioPipelineEvaluator(cfg)
        b0 = ev.evaluate_policy(baseline_tier, recall_b0, alert_rate_b0, delay_frames_b0, edge_active=False)
        b4 = ev.evaluate_policy(policy_tier, recall_b4, alert_rate_b4, delay_frames_b4, baseline_result=b0, edge_active=True)
        return b4.carbon.net_carbon_benefit_kgco2e

    gamma_star = None
    try:
        g_low = delta_c_vs_gamma(0.001)
        g_high = delta_c_vs_gamma(100.0)
        if g_low * g_high <= 0:
            sol = root_scalar(delta_c_vs_gamma, bracket=[0.001, 100.0], method="brentq")
            if sol.converged:
                gamma_star = float(sol.root)
    except Exception:
        pass

    return {
        "scenario": evaluator.name,
        "pi_star": pi_star,
        "pi_star_scrap_only": pi_star_scrap,
        "e_edge_star_kwh_per_1k": e_edge_star_kwh,
        "d_star_frames": d_star_frames,
        "d_star_seconds": d_star_seconds,
        "gamma_star": gamma_star,
    }