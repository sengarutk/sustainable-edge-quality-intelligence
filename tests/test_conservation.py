"""
Unit and invariant tests for strict mass and count conservation.
"""

import pytest
import numpy as np
from src.quality.intervention_model import QualityInterventionModel
from src.sustainability.material_accounting import MaterialAccountingEngine


def test_randomized_defect_count_conservation():
    """Validates that across 1,000 randomized configurations, D == N_rework + N_scrap + N_escape."""
    np.random.seed(12345)
    for _ in range(1000):
        total_defects = float(np.random.uniform(0.1, 500.0))
        recall = float(np.random.uniform(0.0, 1.0))
        delay_frames = float(np.random.uniform(0.0, 60.0))
        q0 = float(np.random.uniform(0.5, 0.99))
        beta = float(np.random.uniform(0.05, 0.40))
        fps = float(np.random.choice([24.0, 30.0, 60.0]))

        model = QualityInterventionModel(
            base_reworkability=q0,
            reworkability_decay_per_sec=beta,
            sampling_rate_fps=fps,
        )
        res = model.evaluate(total_defects, recall, delay_frames)

        sum_outcomes = res.n_rework + res.n_scrap + res.n_escape
        assert abs(total_defects - sum_outcomes) < 1e-6, (
            f"Conservation violated: D={total_defects}, sum={sum_outcomes}"
        )
        assert res.n_rework >= -1e-9
        assert res.n_scrap >= -1e-9
        assert res.n_escape >= -1e-9


def test_material_recovery_upper_bound_invariant():
    """Validates that circular material recovery mass never exceeds gross scrap mass."""
    np.random.seed(67890)
    for _ in range(1000):
        n_scrap = float(np.random.uniform(0.0, 100.0))
        m = float(np.random.uniform(0.001, 50.0))
        eta = float(np.random.uniform(0.0, 1.0))

        engine = MaterialAccountingEngine(part_mass_kg=m, recovery_fraction=eta)
        out = engine.evaluate(n_scrap)

        assert out.recovered_scrap_kg <= out.gross_scrap_kg + 1e-9
        assert out.net_loss_kg >= -1e-9
        assert abs(out.gross_scrap_kg - (out.recovered_scrap_kg + out.net_loss_kg)) < 1e-9