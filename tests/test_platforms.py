"""Embedded (primary) and workstation edge platforms."""

import json

import pytest

from src.experiments.energy_benchmark import measured_registry_bounds
from src.paths import ENERGY_SUMMARY, JETSON_SUMMARY, PROCESSED_DIR
from src.validation.source_registry import load_registry, platform_records, scenario_records


def _rows(scope):
    return {r.parameter: r for r in load_registry() if r.scope == scope}


def test_registry_holds_both_measured_platforms():
    js = json.loads(JETSON_SUMMARY.read_text())
    ws = json.loads(ENERGY_SUMMARY.read_text())
    for scope, summary in (("all", js["runs"][js["primary"]]), ("platform:workstation", ws)):
        rows = _rows(scope)
        for key, bounds in measured_registry_bounds(summary).items():
            assert (rows[key].low, rows[key].central, rows[key].high) == bounds


def test_platform_records_stay_out_of_scenarios():
    recs = scenario_records("machined_metal")
    assert not any(k.endswith("@workstation") for k in recs)
    assert set(platform_records("workstation")) <= {r.parameter for r in load_registry() if r.scope == "all"}


def test_jetson_runs_keep_line_rate_and_decisions():
    js = json.loads(JETSON_SUMMARY.read_text())
    for cfg, run in js["runs"].items():
        assert run["stages"]["STAGE_FULL_PIPELINE"]["achieved_fps"]["min"] > 29.5, cfg
        assert run["workload_detail"]["fidelity_vs_fp32_reference"]["decision_agreement"] == 1.0, cfg


def test_jetson_summaries_match_raw_windows():
    import pandas as pd
    from src.paths import REPO_ROOT
    js = json.loads(JETSON_SUMMARY.read_text())
    for cfg, run in js["runs"].items():
        w = pd.read_csv(REPO_ROOT / run["trace_dir"] / "windows.csv")
        full = w[w.stage == "STAGE_FULL_PIPELINE"]
        assert run["stages"]["STAGE_FULL_PIPELINE"]["p_active_w"]["median"] == pytest.approx(full.p_active_w.median())
        assert len(full) == run["repeats"]


def test_platform_choice_does_not_change_the_benefit():
    import pandas as pd
    d = pd.read_csv(PROCESSED_DIR / "platform_comparison.csv")
    for sc, sub in d.groupby("scenario"):
        assert sub.delta_C_vs_L0.max() - sub.delta_C_vs_L0.min() < 0.02 * sub.delta_C_vs_L0.max(), sc
        ws = sub[sub.configuration.str.startswith("workstation")].edge_cell_carbon.iloc[0]
        assert (sub[sub.configuration.str.startswith("jetson")].edge_cell_carbon < ws).all()


def test_unknown_platform_raises():
    with pytest.raises(KeyError):
        platform_records("does_not_exist")
