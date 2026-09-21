"""
Carbon sustainability accounting: material embodied carbon, electricity footprint, and escape penalties.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CarbonOutcomes:
    material_embodied_carbon_kgco2e: float
    electricity_carbon_kgco2e: float
    escape_penalty_carbon_kgco2e: float
    total_carbon_kgco2e: float
    net_carbon_benefit_kgco2e: float = 0.0


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
            # Positive benefit means carbon reduction (Baseline - Policy)
            benefit = baseline_carbon_kgco2e - total

        return CarbonOutcomes(
            material_embodied_carbon_kgco2e=c_mat,
            electricity_carbon_kgco2e=c_elec,
            escape_penalty_carbon_kgco2e=c_esc,
            total_carbon_kgco2e=total,
            net_carbon_benefit_kgco2e=benefit,
        )