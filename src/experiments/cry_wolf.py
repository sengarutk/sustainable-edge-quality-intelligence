"""
Cry-wolf analysis: how reviewer distrust of unreliable alerts changes the carbon outcome.

Operators learn alert reliability and comply with alerts roughly in proportion to it
(probability matching; Bliss et al. 1995), although trained controllers can stay largely
compliant (Wickens et al. 2009). In the detect-and-divert cell a non-compliant reviewer releases a
truly defective diverted part back to the line, where it escapes. With alert precision
PPV = true alerts / all alerts, the released share of true detections is omega * (1 - PPV);
omega = 0 is full compliance (the registry value) and omega = 1 full probability matching.
The sweep evaluates every AI regime against manual inspection over omega in [0, 1], for the
independent-score (primary) and the correlated-score alert measurements of Paper A.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.models.pipeline_evaluator import AI_TIERS, EXTERNAL_TIERS, ai_operating_point, compare, evaluate_all, load_regimes
from src.params import load_scenario
from src.paths import SCENARIOS
from src.sustainability.workload_accounting import functional_unit_hours

OMEGA = np.linspace(0.0, 1.0, 101)
SCORE_MODELS = {"independent": None, "correlated": "correlated_scores", "visa_independent": "visa_scores",
                "visa_correlated": "visa_correlated_scores"}  # alert-measurement sets of Paper A (MVTec AD unless prefixed)


def alert_precision(p: Dict, tier: str) -> float:
    regime = load_regimes()[tier]
    recall, _, fa, per_defect = ai_operating_point(p, regime)
    true_alerts = per_defect * recall * p["functional_unit"] * p["defect_prevalence"]
    false_alerts = fa * functional_unit_hours(p["functional_unit"], p["line_throughput"])
    return float(true_alerts / (true_alerts + false_alerts))


def _delta_vs_l0(p: Dict, shares, tier: str, omega) -> np.ndarray:
    res = evaluate_all({**p, "cry_wolf_strength": omega}, shares)
    return np.asarray(compare(res["L0_Manual"], res[tier]).delta["C"], dtype=float)


def omega_star(p: Dict, shares, tier: str) -> Tuple[float | None, str]:
    """Reviewer distrust omega at which the tier's advantage over manual inspection vanishes."""
    d0, d1 = float(_delta_vs_l0(p, shares, tier, 0.0)), float(_delta_vs_l0(p, shares, tier, 1.0))
    if d0 > 0 > d1:
        return float(brentq(lambda w: float(_delta_vs_l0(p, shares, tier, w)), 0.0, 1.0, xtol=1e-10)), "root"
    if d0 <= 0:
        return None, "negative_at_full_compliance"
    return None, "none_positive"


def sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    curves, summary = [], []
    for model, variant in SCORE_MODELS.items():
        for sc in SCENARIOS:
            s = load_scenario(sc, variant=variant)
            p = s.central()
            for tier in AI_TIERS + EXTERNAL_TIERS:
                dc = _delta_vs_l0(p, s.class_shares, tier, OMEGA)
                curves.append(pd.DataFrame({"score_model": model, "scenario": sc, "tier": tier, "omega": OMEGA,
                                            "delta_C_vs_L0": dc}))
                star, status = omega_star(p, s.class_shares, tier)
                summary.append({"score_model": model, "scenario": sc, "tier": tier, "ppv": alert_precision(p, tier),
                                "delta_C_omega0": dc[0], "delta_C_omega1": dc[-1], "omega_star": star, "status": status})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)
