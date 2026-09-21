"""
Unit tests for deterministic reproducibility.
Validates that seed=42 produces bit-identical Monte Carlo and scenario outputs.
"""

from pathlib import Path
from src.experiments.monte_carlo import run_monte_carlo_simulation

SCENARIO_PATH = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios/precision_component.yaml")


def test_monte_carlo_seed_reproducibility():
    run1 = run_monte_carlo_simulation(SCENARIO_PATH, n_draws=200, seed=42)
    run2 = run_monte_carlo_simulation(SCENARIO_PATH, n_draws=200, seed=42)

    assert run1["p_delta_c_positive"] == run2["p_delta_c_positive"]
    assert run1["p_sqi_positive"] == run2["p_sqi_positive"]

    assert run1["delta_c"]["p50"] == run2["delta_c"]["p50"]
    assert run1["delta_m"]["p50"] == run2["delta_m"]["p50"]
    assert run1["delta_e"]["p50"] == run2["delta_e"]["p50"]
    assert run1["delta_h"]["p50"] == run2["delta_h"]["p50"]
    assert run1["sqi_balanced"]["p50"] == run2["sqi_balanced"]["p50"]