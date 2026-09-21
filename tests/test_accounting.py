"""
Unit tests for non-negativity, material savings, and monotonicity of accounting components.
"""

import pytest
from pathlib import Path
from src.models.pipeline_evaluator import ScenarioPipelineEvaluator

CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")


@pytest.mark.parametrize("scenario_file", [
    "precision_component.yaml",
    "machined_metal.yaml",
    "high_value_component.yaml",
])
def test_accounting_non_negativity(scenario_file):
    sc_path = CONFIG_DIR / scenario_file
    ev = ScenarioPipelineEvaluator.from_yaml(sc_path)

    res = ev.evaluate_policy("TEST_POLICY", recall=0.95, alert_rate_per_hr=20.0, delay_frames=2.0)

    # Invariants
    assert res.material.gross_scrap_kg >= 0.0
    assert res.material.recovered_scrap_kg >= 0.0
    assert res.material.net_loss_kg >= 0.0

    assert res.workload.total_review_hours >= 0.0
    assert res.workload.queue_utilization_rho >= 0.0

    assert res.energy.edge_compute_kwh >= 0.0
    assert res.energy.human_review_kwh >= 0.0
    assert res.energy.rework_kwh >= 0.0
    assert res.energy.total_energy_kwh >= 0.0

    assert res.carbon.material_embodied_carbon_kgco2e >= 0.0
    assert res.carbon.electricity_carbon_kgco2e >= 0.0
    assert res.carbon.escape_penalty_carbon_kgco2e >= 0.0
    assert res.carbon.total_carbon_kgco2e >= 0.0


def test_positive_material_savings_under_cascade():
    """Validates that timely cascade inspection yields strictly positive Delta M (material saved from scrap)."""
    for sc in ["precision_component", "machined_metal", "high_value_component"]:
        ev = ScenarioPipelineEvaluator.from_yaml(CONFIG_DIR / f"{sc}.yaml")
        b0 = ev.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
        b4 = ev.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0, edge_active=True)

        assert b4.material.net_savings_kg > 0.0, f"Scenario {sc} expected Delta M > 0, got {b4.material.net_savings_kg}"


def test_escape_penalty_monotonic_with_declining_recall():
    """Validates that as recall drops, escape penalty and total carbon increase monotonically."""
    sc_path = CONFIG_DIR / "machined_metal.yaml"
    ev = ScenarioPipelineEvaluator.from_yaml(sc_path)

    recalls = [1.00, 0.98, 0.95, 0.90, 0.70, 0.40, 0.00]
    escapes = []
    carbon_totals = []

    for r in recalls:
        res = ev.evaluate_policy(f"POL_{r}", recall=r, alert_rate_per_hr=15.0, delay_frames=3.0)
        escapes.append(res.carbon.escape_penalty_carbon_kgco2e)
        carbon_totals.append(res.carbon.total_carbon_kgco2e)

    for i in range(len(recalls) - 1):
        assert escapes[i] <= escapes[i + 1] + 1e-9
        assert carbon_totals[i] <= carbon_totals[i + 1] + 1e-9