"""
Unit tests for the Multi-Profile Sustainability Quality Index (SQI).
"""

from pathlib import Path
from src.sustainability.sqi import SQIEngine, DEFAULT_PROFILES
from src.models.pipeline_evaluator import ScenarioPipelineEvaluator

CONFIG_PATH = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/sqi_weights.yaml")
SCENARIO_PATH = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios/precision_component.yaml")


def test_sqi_weight_sum_to_unity():
    engine = SQIEngine(config_path=CONFIG_PATH)
    for profile, weights in engine.profiles.items():
        total_w = sum(weights.values())
        assert abs(total_w - 1.0) < 1e-4, f"Profile {profile} weights do not sum to 1.0: {total_w}"


def test_baseline_identity_sqi_zero():
    """Validates that a policy identical to baseline yields SQI = 0.0."""
    ev = ScenarioPipelineEvaluator.from_yaml(SCENARIO_PATH)
    b0 = ev.evaluate_policy("B0_Raw", recall=1.0, alert_rate_per_hr=180.0, delay_frames=0.0)
    identical = ev.evaluate_policy("B0_Identical", recall=1.0, alert_rate_per_hr=180.0, delay_frames=0.0, baseline_result=b0)

    for prof, score in identical.sqi.profile_scores.items():
        assert abs(score) < 1e-6, f"SQI for identical baseline policy should be 0.0, got {score} in {prof}"


def test_monotonic_sqi_response():
    engine = SQIEngine(config_path=CONFIG_PATH)
    # Higher savings should strictly increase SQI
    eval_low = engine.evaluate(
        delta_m=1.0, m_baseline_loss=10.0,
        delta_e=1.0, e_baseline_total=10.0,
        delta_c=1.0, c_baseline_total=10.0,
        delta_h=1.0, h_baseline_hours=10.0,
    )
    eval_high = engine.evaluate(
        delta_m=5.0, m_baseline_loss=10.0,
        delta_e=5.0, e_baseline_total=10.0,
        delta_c=5.0, c_baseline_total=10.0,
        delta_h=5.0, h_baseline_hours=10.0,
    )
    for prof in engine.profiles:
        assert eval_high.profile_scores[prof] > eval_low.profile_scores[prof]