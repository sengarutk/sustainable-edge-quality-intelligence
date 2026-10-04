"""
Operational electricity per functional unit [kWh/FU].

  E_edge   = (P_idle + P_active + P_host) * T_FU / 1000   (edge AI regimes only)
  E_review = H * P_station / 1000
  E_rework = N_rework * e_rw
P_idle and P_active are measured GPU powers at the line capture rate
(results/raw/energy_summary.json); P_host covers host/camera/lighting power
that NVML cannot observe. T_FU [h] = N / Theta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyOutcome:
    edge_kwh: np.ndarray
    review_kwh: np.ndarray
    rework_kwh: np.ndarray

    @property
    def total_kwh(self):
        return self.edge_kwh + self.review_kwh + self.rework_kwh


def edge_energy_kwh(p_idle_w, p_active_w, p_host_w, t_fu_h):
    return (np.asarray(p_idle_w) + p_active_w + p_host_w) * t_fu_h / 1000.0


def energy_flows(edge_active: bool, p_idle_w, p_active_w, p_host_w, t_fu_h,
                 operator_hours, station_power_w, n_rework, rework_kwh_per_part) -> EnergyOutcome:
    edge = edge_energy_kwh(p_idle_w, p_active_w, p_host_w, t_fu_h)
    if not edge_active:
        edge = np.zeros_like(np.asarray(edge, dtype=float))
    review = np.asarray(operator_hours, dtype=float) * station_power_w / 1000.0
    rework = np.asarray(n_rework, dtype=float) * rework_kwh_per_part
    return EnergyOutcome(edge_kwh=edge, review_kwh=review, rework_kwh=rework)
