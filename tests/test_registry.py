"""Source registry integrity and consistency with measured / imported evidence."""

import json

import pandas as pd
import pytest
import yaml

from src.experiments.energy_benchmark import measured_registry_bounds
from src.imports.import_paper_a import load_snapshot, snapshot_rows
from src.params import load_scenario
from src.paths import ENERGY_SUMMARY, REPO_ROOT, SCENARIOS
from src.validation.source_registry import ParameterRecord, load_registry, validate_registry_file


def test_registry_validates():
    assert validate_registry_file() == len(load_registry())


def test_every_non_assumption_has_citation():
    for r in load_registry():
        if r.classification.value != "Scenario assumption":
            assert r.citation_key != "none", r.key


def test_bounds_violation_rejected():
    with pytest.raises(ValueError):
        ParameterRecord(parameter="x", symbol="x", scope="all", low=2, central=1, high=3, unit="-",
                        classification="Scenario assumption", citation_key="none", notes="n")


def test_unknown_scope_rejected():
    with pytest.raises(ValueError):
        ParameterRecord(parameter="x", symbol="x", scope="nowhere", low=1, central=1, high=1, unit="-",
                        classification="Scenario assumption", citation_key="none", notes="n")


def test_workstation_rows_match_energy_summary():
    recs = {r.key: r for r in load_registry()}
    for key, expect in measured_registry_bounds(json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))).items():
        r = recs[f"{key}@workstation"]
        assert (r.low, r.central, r.high) == expect
        assert r.classification.value == "Measured by this study"


def test_measured_summary_matches_raw_windows():
    s = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    w = pd.read_csv(REPO_ROOT / s["trace_dir"] / "windows.csv")
    full = w[w.stage == "STAGE_FULL_PIPELINE"]
    assert s["stages"]["STAGE_FULL_PIPELINE"]["p_active_w"]["median"] == pytest.approx(full.p_active_w.median())
    assert len(full) == s["repeats"]
    assert (full.achieved_fps > 0.99 * s["target_fps"]).all()


def test_paper_a_rows_match_snapshot():
    by_scope = {f"{r.parameter}@{r.scope}": r for r in load_registry()}
    rows = snapshot_rows()
    for key, (lo, c, hi) in rows.items():
        r = by_scope[key]
        assert (r.low, r.central, r.high) == (lo, c, hi), key
        assert r.citation_key == "sengar2026paperA"
    derived = {k for k, r in by_scope.items() if r.classification.value == "Derived from Paper A"}
    assert derived <= set(rows)


def test_paper_a_snapshot_is_pinned_and_complete():
    snap = load_snapshot()
    assert len(snap["source_commit"]) == 40
    assert set(snap["tier_policy"]) == {"B0_Raw", "B1_EMA", "B2_EMA_kofN", "B3_Full_Policy"}
    assert all(v == 1.0 for v in snap["sustained_defect_recall"].values())


def test_scenario_yaml_contains_no_numeric_parameters(tmp_path, monkeypatch):
    for sc in SCENARIOS:
        assert sum(load_scenario(sc).class_shares) == pytest.approx(1.0)
    import src.params as params
    bad = yaml.safe_load((params.SCENARIO_DIR / "machined_metal.yaml").read_text())
    bad["part_mass_kg"] = 1.0
    (tmp_path / "machined_metal.yaml").write_text(yaml.safe_dump(bad))
    monkeypatch.setattr(params, "SCENARIO_DIR", tmp_path)
    with pytest.raises(ValueError, match="source registry"):
        params.load_scenario("machined_metal")


def test_citation_keys_exist_in_bibliography():
    bib = (REPO_ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    for r in load_registry():
        for key in r.citation_key.split(";"):
            if key not in ("none", "this_study"):
                assert "{" + key + "," in bib, key
