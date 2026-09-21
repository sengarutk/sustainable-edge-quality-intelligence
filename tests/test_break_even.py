"""
Unit tests for analytical break-even frontiers and identification of net-negative regimes.
"""

from pathlib import Path
from src.models.pipeline_evaluator import ScenarioPipelineEvaluator
from src.experiments.break_even import solve_break_even_frontiers

SCENARIO_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")


def test_delta_c_monotonic_with_prevalence():
    """Delta C increases monotonically with defect prevalence pi."""
    ev = ScenarioPipelineEvaluator.from_yaml(SCENARIO_DIR / "machined_metal.yaml")
    prevalences = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080]
    delta_cs = []

    for pi_val in prevalences:
        cfg = dict(ev.cfg)
        cfg["defect_prevalence"] = pi_val
        e = ScenarioPipelineEvaluator(cfg)
        b0 = e.evaluate_policy("B0_Raw", 0.88, 180.0, 0.0)
        b4 = e.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0)
        delta_cs.append(b4.carbon.net_carbon_benefit_kgco2e)

    for i in range(len(prevalences) - 1):
        assert delta_cs[i] < delta_cs[i + 1], f"Delta C not strictly increasing: {delta_cs[i]} vs {delta_cs[i+1]}"


def test_scenario_a_net_negative_carbon_regime():
    """Proves edge AI is not universally green when mass is small, compute energy is high, or recall drops."""
    ev = ScenarioPipelineEvaluator.from_yaml(SCENARIO_DIR / "precision_component.yaml")

    # High compute power scenario / slightly lower recall
    cfg = dict(ev.cfg)
    cfg["edge_energy_kwh_per_1k"] = 150.0  # Excessive compute power
    e_heavy = ScenarioPipelineEvaluator(cfg)

    b0 = e_heavy.evaluate_policy("B0_Raw", 0.88, 180.0, 0.0, edge_active=False)
    b4 = e_heavy.evaluate_policy("B4_Heavy", 0.99, 12.0, 3.0, baseline_result=b0, edge_active=True)

    # Must be net-negative carbon (edge compute carbon exceeds all material & scrap savings)
    assert b4.carbon.net_carbon_benefit_kgco2e < 0.0, "Expected net-negative carbon under high compute budget"


def test_break_even_solver_convergence():
    """Validates break-even solvers return finite values for plausible frontiers."""
    sol_h = solve_break_even_frontiers(SCENARIO_DIR / "high_value_component.yaml")
    assert sol_h["d_star_seconds"] is not None
    assert sol_h["d_star_seconds"] > 0.5  # Bounded positive allowable delay
    assert sol_h["e_edge_star_kwh_per_1k"] is not None
    assert sol_h["e_edge_star_kwh_per_1k"] > 100.0  # High allowable budget