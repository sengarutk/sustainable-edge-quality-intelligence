"""Measured detector operating points, carbon-optimal thresholds and the general decision rule."""

import json

import numpy as np
import pytest

from src.experiments.decision_rule import affine_coefficients, pi_star
from src.experiments.detector_eval import SCORE_DIR, SEEDS, score_path, units
from src.experiments.operating_points import curve, recall_ledger_values, reference_points
from src.models.pipeline_evaluator import compare, evaluate_all
from src.paths import PROCESSED_DIR
from src.validation.source_registry import load_registry


def test_scores_committed_for_every_unit_and_seed():
    meta = json.loads((SCORE_DIR / "run_meta.json").read_text())
    for u in units():
        for seed in SEEDS:
            z = np.load(score_path(u, seed))
            assert set(z["role"]) == {"calibration", "heldout_good", "test_good", "test_defect"}
            assert meta["runs"][f"{u.label}/seed{seed}"]["n_scores"] == len(z["scores"])


def test_registry_recall_is_the_measured_one():
    r = next(x for x in load_registry() if x.key == "ai_recall")
    assert (r.low, r.central, r.high) == tuple(round(v, 4) for v in recall_ledger_values(reference_points()))
    assert r.classification.value == "Measured by this study"


@pytest.mark.parametrize("unit", units(), ids=lambda u: u.label)
def test_operating_curve_is_monotone(unit):
    cv = curve(unit).sort_values("q")
    assert (np.diff(cv.recall) <= 1e-12).all() and (np.diff(cv.fpr) <= 1e-12).all()
    assert cv.recall.between(0, 1).all() and cv.fpr.between(0, 1).all()


def test_carbon_optimal_threshold_dominates_reference_thresholds():
    import pandas as pd
    opt = pd.read_csv(PROCESSED_DIR / "carbon_optimal_thresholds.csv")
    assert (opt.carbon_opt <= opt.carbon_q99 + 1e-9).all()
    assert (opt.carbon_opt <= opt.carbon_budget + 1e-9).all()


@pytest.mark.parametrize("base", ["N0_NoInspection", "L0_Manual"])
def test_closed_form_break_even_matches_brent(scenario, base):
    be = json.loads((PROCESSED_DIR / "break_even.json").read_text())[scenario.name]
    key = "pi_star_vs_N0" if base == "N0_NoInspection" else "pi_star_vs_L0"
    closed = float(pi_star(scenario.central(), scenario.class_shares, base))
    if be[key]["status"] == "none_positive":  # ahead at every prevalence: fixed carbon F < 0, so pi* < 0
        assert closed <= 0
    else:
        assert closed == pytest.approx(be[key]["value"], rel=1e-8)


def test_delta_c_is_affine_in_defects(scenario, rng):
    p = scenario.central()
    beta, f = affine_coefficients(p, scenario.class_shares, "N0_NoInspection")
    for pi_ in rng.uniform(0, 0.2, 5):
        r = evaluate_all({**p, "defect_prevalence": pi_}, scenario.class_shares)
        dc = float(compare(r["N0_NoInspection"], r["B3_Full_Policy"]).delta["C"])
        assert dc == pytest.approx(p["functional_unit"] * pi_ * float(beta) - float(f), rel=1e-10, abs=1e-12)
