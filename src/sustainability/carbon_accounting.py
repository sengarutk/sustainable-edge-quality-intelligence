"""
Carbon sustainability accounting: material embodied carbon, electricity footprint, and escape penalties.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class CarbonOutcomes:
    material_embodied_carbon_kgco2e: float
    electricity_carbon_kgco2e: float
    escape_penalty_carbon_kgco2e: float
    total_carbon_kgco2e: float
    net_carbon_benefit_kgco2e: float = 0.0
    waterfall_components: Dict[str, float] = field(default_factory=dict)


class CarbonAccountingEngine:
    def __init__(
        self,
        material_carbon_factor_kgco2e_per_kg: float = 3.50,
        grid_carbon_factor_kgco2e_per_kwh: float = 0.417,
        escape_carbon_penalty_kgco2e: float = 12.0,
    ):
        self.ef_mat = material_carbon_factor_kgco2e_per_kg
        self.gamma = grid_carbon_factor_kgco2e_per_kwh
        self.c_escape = escape_carbon_penalty_kgco2e

    def evaluate(
        self,
        net_material_loss_kg: float,
        total_energy_kwh: float,
        n_escape: float,
        baseline_carbon_kgco2e: float = None,
        baseline_material_loss_kg: float = None,
        baseline_rework_kwh: float = None,
        baseline_review_kwh: float = None,
        current_rework_kwh: float = None,
        current_review_kwh: float = None,
        current_edge_kwh: float = None,
        baseline_n_escape: float = None,
    ) -> CarbonOutcomes:
        # Material embodied carbon loss
        c_mat = max(0.0, net_material_loss_kg) * self.ef_mat

        # Grid electricity carbon footprint
        c_elec = max(0.0, total_energy_kwh) * self.gamma

        # Field escape penalty carbon
        c_esc = max(0.0, n_escape) * self.c_escape

        total = c_mat + c_elec + c_esc

        benefit = 0.0
        if baseline_carbon_kgco2e is not None:
            benefit = baseline_carbon_kgco2e - total

        waterfall = {}
        if baseline_carbon_kgco2e is not None and baseline_material_loss_kg is not None:
            delta_c_mat = (baseline_material_loss_kg - net_material_loss_kg) * self.ef_mat
            b_rw = baseline_rework_kwh if baseline_rework_kwh is not None else 0.0
            c_rw = current_rework_kwh if current_rework_kwh is not None else 0.0
            delta_c_rw = (b_rw - c_rw) * self.gamma

            b_rev = baseline_review_kwh if baseline_review_kwh is not None else 0.0
            c_rev = current_review_kwh if current_review_kwh is not None else 0.0
            delta_c_rev = (b_rev - c_rev) * self.gamma

            c_edge = current_edge_kwh if current_edge_kwh is not None else 0.0
            c_edge_carbon = c_edge * self.gamma

            b_esc = baseline_n_escape if baseline_n_escape is not None else 0.0
            delta_c_esc = (b_esc - n_escape) * self.c_escape

            waterfall = {
                "delta_c_mat": delta_c_mat,
                "delta_c_rw": delta_c_rw,
                "delta_c_rev": delta_c_rev,
                "c_edge": -c_edge_carbon,
                "delta_c_esc": delta_c_esc,
                "delta_c_total": benefit,
            }

        return CarbonOutcomes(
            material_embodied_carbon_kgco2e=c_mat,
            electricity_carbon_kgco2e=c_elec,
            escape_penalty_carbon_kgco2e=c_esc,
            total_carbon_kgco2e=total,
            net_carbon_benefit_kgco2e=benefit,
            waterfall_components=waterfall,
        )
