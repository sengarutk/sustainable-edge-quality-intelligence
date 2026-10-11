"""
Alarm designs and score correlation.

Paper A measured its alert policies twice: with frame scores drawn independently (the primary
values of this study) and with temporally correlated scores (AR(1) Gaussian copula, rho = 0.9; the
'correlated_scores' variant of the registry), and it added three textbook alarm baselines
(delay-timer, EMA with hysteresis, decision-level fusion). For every scenario, score model and AI
regime this module reports, at central parameter values: the carbon saving against manual
inspection and against no inspection, the single-reviewer utilisation and number of reviewers,
the alert precision (share of alerts that concern a defect) and the cry-wolf break-even omega*
(reviewer distrust at which the advantage over manual inspection vanishes; src.experiments.cry_wolf).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.cry_wolf import alert_precision, omega_star
from src.models.pipeline_evaluator import AI_TIERS, EXTERNAL_TIERS, compare, evaluate_all
from src.params import load_scenario
from src.paths import SCENARIOS

SCORE_MODELS = {"independent": None, "correlated": "correlated_scores", "visa_independent": "visa_scores",
                "visa_correlated": "visa_correlated_scores"}  # alert-measurement sets of Paper A (MVTec AD unless prefixed)


def alarm_design_table() -> pd.DataFrame:
    rows = []
    for model, variant in SCORE_MODELS.items():
        for sc in SCENARIOS:
            s = load_scenario(sc, variant=variant)
            p = s.central()
            res = evaluate_all(p, s.class_shares)
            for t in AI_TIERS + EXTERNAL_TIERS:
                r = res[t]
                star, status = omega_star(p, s.class_shares, t)
                # Paper A counts a defect first flagged more than 15 frames after onset as missed; if such late
                # flags still divert the part, the regime keeps the detector's recall:
                late = evaluate_all({**p, f"recall_loss@{t}": 0.0}, s.class_shares)
                rows.append({"score_model": model, "scenario": sc, "regime": t, "recall": float(r.recall),
                             "rho": float(r.workload.utilisation_rho),
                             "reviewers": int(np.ceil(float(r.workload.utilisation_rho) - 1e-9)),
                             "operator_hours": float(r.workload.operator_hours), "ppv": alert_precision(p, t),
                             "delta_C_vs_L0": float(compare(res["L0_Manual"], r).delta["C"]),
                             "delta_C_vs_N0": float(compare(res["N0_NoInspection"], r).delta["C"]),
                             "delta_C_vs_L0_late_flags_divert": float(compare(late["L0_Manual"], late[t]).delta["C"]),
                             "omega_star": star, "omega_status": status})
    return pd.DataFrame(rows)
