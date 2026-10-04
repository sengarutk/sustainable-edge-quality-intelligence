"""
Human workload: operator hours per functional unit and single-server utilisation.

AI regimes: every alert reaches the operator queue (the load model of Paper A).
  alerts = lambda_FA * T_FU + a * r * D       (false alerts + a alerts per detected defect)
  lambda_FA = lambda_nominal + g * b          (good-part false alerts + glare bursts x alerts/burst)
  H      = alerts * t_review / 3600           [h / FU]
  rho    = (alerts / T_FU) / mu,  mu = 3600 / t_review  [per operator]
Manual end-of-line regime: every part is inspected.
  H = N * t_L0 / 3600,   rho = Theta * t_L0 / 3600
where T_FU = N / Theta [h] is the production time of one functional unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkloadOutcome:
    operator_hours: np.ndarray
    utilisation_rho: np.ndarray
    events: np.ndarray


def functional_unit_hours(n_units, throughput_per_h):
    if np.any(np.asarray(throughput_per_h) <= 0):
        raise ValueError("line throughput must be positive")
    return np.asarray(n_units, dtype=float) / throughput_per_h


def ai_review_workload(false_alarm_rate_per_h, true_detections, alerts_per_defect, t_fu_h, review_time_s) -> WorkloadOutcome:
    events = np.asarray(false_alarm_rate_per_h, dtype=float) * t_fu_h + np.asarray(alerts_per_defect) * true_detections
    hours = events * review_time_s / 3600.0
    rho = (events / t_fu_h) * review_time_s / 3600.0
    return WorkloadOutcome(operator_hours=hours, utilisation_rho=rho, events=events)


def manual_inspection_workload(n_units, throughput_per_h, inspection_time_s) -> WorkloadOutcome:
    events = np.asarray(n_units, dtype=float) * np.ones_like(np.asarray(inspection_time_s, dtype=float))
    hours = events * inspection_time_s / 3600.0
    rho = np.asarray(throughput_per_h, dtype=float) * inspection_time_s / 3600.0
    return WorkloadOutcome(operator_hours=hours, utilisation_rho=rho, events=events)


def no_workload(like) -> WorkloadOutcome:
    z = np.zeros_like(np.asarray(like, dtype=float))
    return WorkloadOutcome(operator_hours=z, utilisation_rho=z, events=z)
