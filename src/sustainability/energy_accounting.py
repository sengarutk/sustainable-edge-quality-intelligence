"""
Energy sustainability accounting: edge compute, human workstation review, and machine rework.
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
        if edge_energy_wh_per_frame is not None:
            self.e_frame_wh = edge_energy_wh_per_frame
        else:
            self.e_frame_wh = edge_energy_kwh_per_1k
        self.e_edge_kwh_per_1k = edge_energy_kwh_per_1k
        self.p_station = review_workstation_power_w
        self.e_rw = rework_energy_kwh_per_unit
        self.n_units = functional_unit_units
        self.n_f = frames_per_part

    def evaluate(
        self,
        review_hours: float,
        n_rework: float,
        baseline_total_kwh: float = None,
        edge_active: bool = True,
        frames_per_part: float = None,
    ) -> EnergyOutcomes:
        n_f = self.n_f if frames_per_part is None else frames_per_part
        # Physical unit: E_edge = N * n_f * e_frame / 1000 [kWh/FU]
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
        )
