"""
Energy sustainability accounting: edge compute, human workstation review, and machine rework.
Dimensionally reconciles physical frame energy e_frame (Wh/frame, mWh/frame, J/frame)
and functional unit energy E_edge = (N * n_f * e_frame) / 1000 [kWh/FU].
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyOutcomes:
    edge_compute_kwh: float
    human_review_kwh: float
    rework_kwh: float
    total_energy_kwh: float
    net_energy_diff_kwh: float = 0.0
    frames_per_part: float = 1.0
    e_frame_wh: float = 0.171
    e_frame_joules: float = 615.6


class EnergyAccountingEngine:
    def __init__(
        self,
        edge_energy_kwh_per_1k: float = 0.171,
        review_workstation_power_w: float = 85.0,
        rework_energy_kwh_per_unit: float = 0.05,
        functional_unit_units: int = 1000,
        frames_per_part: float = 1.0,
        edge_energy_wh_per_frame: float = None,
    ):
        self.n_units = functional_unit_units
        self.n_f = frames_per_part

        if edge_energy_wh_per_frame is not None:
            self.e_frame_wh = edge_energy_wh_per_frame
        elif edge_energy_kwh_per_1k is not None:
            # E_edge [kWh/FU] = (N * n_f * e_frame [Wh/frame]) / 1000
            # => e_frame [Wh/frame] = (E_edge [kWh/FU] * 1000) / (N * n_f)
            self.e_frame_wh = (edge_energy_kwh_per_1k * 1000.0) / (self.n_units * self.n_f)
        else:
            self.e_frame_wh = 0.171

        self.e_frame_joules = self.e_frame_wh * 3600.0
        self.e_frame_mwh = self.e_frame_wh * 1000.0
        self.e_edge_kwh_per_1k = (self.n_units * self.n_f * self.e_frame_wh) / 1000.0
        self.p_station = review_workstation_power_w
        self.e_rw = rework_energy_kwh_per_unit

    def evaluate(
        self,
        review_hours: float,
        n_rework: float,
        baseline_total_kwh: float = None,
        edge_active: bool = True,
        frames_per_part: float = None,
    ) -> EnergyOutcomes:
        n_f = self.n_f if frames_per_part is None else frames_per_part
        # Physical unit: E_edge = (N * n_f * e_frame_wh) / 1000 [kWh/FU]
        # where e_frame_wh is in Wh/frame (or J/frame / 3600).
        e_comp = (self.n_units * n_f * self.e_frame_wh / 1000.0) if edge_active else 0.0

        # Human review workstation electrical energy [kWh] = Hours * Watts / 1000
        e_rev = max(0.0, review_hours) * (self.p_station / 1000.0)

        # Machine rework electrical energy [kWh]
        e_rework = max(0.0, n_rework) * self.e_rw

        total = e_comp + e_rev + e_rework

        diff = 0.0
        if baseline_total_kwh is not None:
            diff = baseline_total_kwh - total

        return EnergyOutcomes(
            edge_compute_kwh=e_comp,
            human_review_kwh=e_rev,
            rework_kwh=e_rework,
            total_energy_kwh=total,
            net_energy_diff_kwh=diff,
            frames_per_part=n_f,
            e_frame_wh=self.e_frame_wh,
            e_frame_joules=self.e_frame_joules,
        )