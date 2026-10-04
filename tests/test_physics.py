"""Units, conservation and closure of the accounting model; integrity of the energy measurement."""

import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from src.experiments.energy_benchmark import NvmlMeter
from src.experiments.monte_carlo import sample_parameters
from src.models.pipeline_evaluator import compare, evaluate_all, load_regimes
from src.paths import ENERGY_SUMMARY, REPO_ROOT
from src.quality.intervention_model import reworkability, route_defects
from src.sustainability.carbon_accounting import COMPONENTS
from src.sustainability.energy_accounting import edge_energy_kwh
from src.sustainability.workload_accounting import ai_review_workload, functional_unit_hours, manual_inspection_workload


def test_edge_energy_units():
    # 10 W for 1 h = 10 Wh = 0.01 kWh
    assert edge_energy_kwh(2.0, 3.0, 5.0, 1.0) == pytest.approx(0.01)
    # functional unit of 1000 parts at 1800 parts/h lasts 0.5556 h
    assert functional_unit_hours(1000, 1800) == pytest.approx(1000 / 1800)


def test_per_frame_energy_consistency():
    # Every measured window: E_frame = P * duration / frames, and at the target rate E_frame ~= P / fps.
    summary = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    w = pd.read_csv(REPO_ROOT / summary["trace_dir"] / "windows.csv")
    active = w[w.stage != "BASELINE_IDLE"]
    assert np.allclose(active.e_frame_total_j, active.p_total_w * active.duration_s / active.frames, rtol=1e-12)
    assert np.allclose(active.e_frame_active_j, active.p_active_w * active.duration_s / active.frames, rtol=1e-12)
    assert np.allclose(active.e_frame_total_j, active.p_total_w / summary["target_fps"], rtol=0.01)
    idle = w[w.stage == "BASELINE_IDLE"].set_index("repeat").p_total_w
    assert np.allclose(active.p_active_w, active.p_total_w - idle.loc[active.repeat].values, rtol=1e-12)


def test_power_meter_refuses_failed_or_empty_traces():
    class FlakySensor:
        calls = 0

        def nvmlDeviceGetPowerUsage(self, handle):
            FlakySensor.calls += 1
            if FlakySensor.calls > 3:
                raise OSError("sensor lost")
            return 10_000

        def nvmlDeviceGetTotalEnergyConsumption(self, handle):
            raise OSError("unsupported")

    meter = NvmlMeter.__new__(NvmlMeter)
    meter.nv, meter.handle, meter.dt = FlakySensor(), None, 0.001
    meter._stop, meter._trace, meter._error, meter._thread = threading.Event(), [], None, None
    meter.start()
    time.sleep(0.05)
    with pytest.raises(RuntimeError, match="sampling failed"):
        meter.stop()


def test_reworkability_decay():
    assert reworkability(0.0, 0.9, 30.0) == pytest.approx(0.9)
    assert reworkability(30.0, 0.9, 30.0) == pytest.approx(0.9 / np.e)
    d = np.linspace(0, 300, 50)
    assert np.all(np.diff(reworkability(d, 0.8, 20.0)) < 0)
    with pytest.raises(ValueError):
        reworkability(1.0, 1.2, 10.0)


def test_count_conservation_randomised(rng):
    n = 5000
    shares = rng.dirichlet([1, 1, 1])
    out = route_defects(rng.uniform(0, 500, n), rng.uniform(0, 1, n), rng.uniform(0, 600, n),
                        rng.uniform(0.3, 1, n), rng.uniform(5, 200, n), tuple(shares))
    assert np.allclose(out.total_defects, out.n_rework + out.n_scrap + out.n_escape, atol=1e-9)
    assert (out.n_rework >= 0).all() and (out.n_scrap >= 0).all() and (out.n_escape >= 0).all()
    assert np.allclose(out.n_escape_class_c, out.n_escape * shares[2])


def test_invalid_recall_rejected():
    with pytest.raises(ValueError):
        route_defects(10.0, 1.2, 0.0, 0.9, 30.0, (0.7, 0.2, 0.1))


def test_workload_definitions():
    # Paper A load model: false alerts over T_FU plus a alerts per detected defect
    w = ai_review_workload(false_alarm_rate_per_h=12.0, true_detections=40.0, alerts_per_defect=1.5, t_fu_h=0.5,
                           review_time_s=60.0)
    assert w.events == pytest.approx(12 * 0.5 + 1.5 * 40)
    assert w.operator_hours == pytest.approx(66 * 60 / 3600)
    assert w.utilisation_rho == pytest.approx((66 / 0.5) / (3600 / 60))
    m = manual_inspection_workload(1000, 1800, 2.0)
    assert m.operator_hours == pytest.approx(1000 * 2 / 3600)
    assert m.utilisation_rho == pytest.approx(1.0)


def test_waterfall_closure_and_nonnegativity_over_draws(scenario):
    p = sample_parameters(scenario, 2000, seed=3)
    res = evaluate_all(p, scenario.class_shares)
    for r in res.values():
        for c in COMPONENTS:
            assert (getattr(r.carbon, c) >= 0).all()
    for base in ("N0_NoInspection", "L0_Manual", "B0_Raw"):
        c = compare(res[base], res["B3_Full_Policy"])
        assert np.allclose(sum(c.waterfall[k] for k in COMPONENTS), c.delta["C"], rtol=1e-12, atol=1e-9)


def test_identical_regime_gives_zero_deltas(scenario):
    res = evaluate_all(scenario.central(), scenario.class_shares)
    c = compare(res["B3_Full_Policy"], res["B3_Full_Policy"])
    assert all(abs(float(v)) < 1e-12 for v in c.delta.values())
    assert all(abs(float(v)) < 1e-12 for v in c.sqi.scores.values())


def test_edge_terms_only_for_ai_regimes(scenario):
    res = evaluate_all(scenario.central(), scenario.class_shares)
    for rid, r in res.items():
        is_ai = load_regimes()[rid].kind == "ai"
        assert (float(r.energy.edge_kwh) > 0) == is_ai
        assert (float(r.carbon.hardware) > 0) == is_ai


def test_no_inspection_means_every_defect_escapes(scenario):
    r = evaluate_all(scenario.central(), scenario.class_shares)["N0_NoInspection"]
    assert float(r.routing.n_escape) == pytest.approx(float(r.routing.total_defects))
    assert float(r.workload.operator_hours) == 0.0


@pytest.mark.parametrize("tier,persistent", [("B0_Raw", False), ("B1_EMA", False), ("B2_EMA_kofN", True),
                                             ("B3_Full_Policy", True)])
def test_ai_operating_point_uses_paper_a_measurements(scenario, tier, persistent):
    p = {**scenario.central(), "persistence_recall_loss": 0.01}
    r = evaluate_all(p, scenario.class_shares)[tier]
    expected_fa = p[f"nominal_false_alarm_rate@{tier}"] + p["glare_burst_rate"] * p[f"glare_alerts_per_burst@{tier}"]
    assert float(r.false_alarm_rate) == pytest.approx(expected_fa)
    assert float(r.delay_s) == pytest.approx(p[f"detection_delay@{tier}"] / p["camera_fps"])
    assert float(r.recall) == pytest.approx(p["ai_recall"] - (0.01 if persistent else 0.0))
    alerts = expected_fa * float(r.t_fu_h) + p[f"alerts_per_defect@{tier}"] * float(r.routing.routed)
    assert float(r.workload.events) == pytest.approx(alerts)


def test_paper_a_policy_rates_reproduce_its_table():
    """The imported registry values must equal the means printed in the Paper A ablation table."""
    from src.validation.source_registry import load_registry
    c = {r.key: r.central for r in load_registry()}
    assert round(c["nominal_false_alarm_rate@B0_Raw"]) == 3971
    assert round(c["nominal_false_alarm_rate@B3_Full_Policy"], 1) == 0.6
    assert round(c["alerts_per_defect@B0_Raw"], 2) == 144.23
    assert round(c["alerts_per_defect@B3_Full_Policy"], 2) == 1.12
    assert round(c["detection_delay@B3_Full_Policy"], 1) == 3.4
    assert c["review_time"] == 60.0


def test_throughput_scales_edge_energy_linearly(scenario):
    p = scenario.central()
    e1 = evaluate_all(p, scenario.class_shares)["B3_Full_Policy"].energy.edge_kwh
    e2 = evaluate_all({**p, "line_throughput": p["line_throughput"] / 2}, scenario.class_shares)["B3_Full_Policy"].energy.edge_kwh
    assert float(e2) == pytest.approx(2 * float(e1))
