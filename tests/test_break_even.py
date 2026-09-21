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
        b0 = e.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
        b4 = e.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0, edge_active=True)
        delta_cs.append(b4.carbon.net_carbon_benefit_kgco2e)

    for i in range(len(prevalences) - 1):
        assert delta_cs[i] < delta_cs[i + 1], f"Delta C not strictly increasing: {delta_cs[i]} vs {delta_cs[i+1]}"


def test_scenario_a_net_negative_carbon_regime():
    """Proves edge AI is not universally green: at pi=0, compute carbon dominates, causing Delta C < 0."""
    ev = ScenarioPipelineEvaluator.from_yaml(SCENARIO_DIR / "precision_component.yaml")

    cfg = dict(ev.cfg)
    cfg["defect_prevalence"] = 0.0
    e_zero = ScenarioPipelineEvaluator(cfg)

    b0 = e_zero.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
    b4 = e_zero.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0, edge_active=True)

    # Must be net-negative carbon when no defects occur
    assert b4.carbon.net_carbon_benefit_kgco2e < 0.0, (
        f"Expected net-negative carbon at pi=0, got {b4.carbon.net_carbon_benefit_kgco2e}"
    )


def test_break_even_solver_convergence():
    """Validates break-even solvers return finite values for plausible frontiers."""
    sol_a = solve_break_even_frontiers(SCENARIO_DIR / "precision_component.yaml")
    assert sol_a["pi_star"] is not None
    assert 1e-6 < sol_a["pi_star"] < 0.01  # Realistic pi* in industrial regime
    assert sol_a["e_edge_star_kwh_per_1k"] is not None
    assert sol_a["e_edge_star_kwh_per_1k"] > 50.0