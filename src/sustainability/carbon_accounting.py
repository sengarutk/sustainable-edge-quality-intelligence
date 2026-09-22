"""
Carbon sustainability accounting: material embodied carbon, electricity footprint,
escape penalties, and downstream catch partitioning.
Enforces exact additive waterfall closure: Delta C = Delta C_mat + Delta C_rw + Delta C_rev + Delta C_edge + Delta C_esc + Delta C_downstream.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class CarbonOutcomes:
    material_embodied_carbon_kgco2e: float
    electricity_carbon_kgco2e: float
    escape_penalty_carbon_kgco2e: float
    total_carbon_kgco2e: float
    net_carbon_benefit_kgco2e: float = 0.0
    waterfall_components: Dict[str, float] = field(default_factory=dict)
    class_carbon_breakdown: Dict[str, float] = field(default_factory=dict)
    n_field_escape: float = 0.0
    n_downstream_catch: float = 0.0
    downstream_penalty_carbon_kgco2e: float = 0.0


class CarbonAccountingEngine:
    def __init__(
        self,
        material_carbon_factor_kgco2e_per_kg: float = 3.50,
        grid_carbon_factor_kgco2e_per_kwh: float = 0.417,
        escape_carbon_penalty_kgco2e: float = 12.0,
        downstream_catch_penalty_kgco2e: float = 0.0,
    ):
        self.ef_mat = material_carbon_factor_kgco2e_per_kg
        self.gamma = grid_carbon_factor_kgco2e_per_kwh
        self.c_escape = escape_carbon_penalty_kgco2e
        self.c_downstream = downstream_catch_penalty_kgco2e

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
        baseline_edge_kwh: float = None,
        class_counts: Dict[str, float] = None,
        n_downstream_catch: float = 0.0,
        baseline_n_downstream: float = None,
    ) -> CarbonOutcomes:
        c_mat = max(0.0, net_material_loss_kg) * self.ef_mat
        c_elec = max(0.0, total_energy_kwh) * self.gamma
        c_esc = max(0.0, n_escape) * self.c_escape
        c_down = max(0.0, n_downstream_catch) * self.c_downstream
        total = c_mat + c_elec + c_esc + c_down

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

            b_edge = baseline_edge_kwh if baseline_edge_kwh is not None else 0.0
            c_edge = current_edge_kwh if current_edge_kwh is not None else 0.0
            delta_c_edge = (b_edge - c_edge) * self.gamma

            b_esc = baseline_n_escape if baseline_n_escape is not None else 0.0
            delta_c_esc = (b_esc - n_escape) * self.c_escape

            b_down = baseline_n_downstream if baseline_n_downstream is not None else 0.0
            delta_c_down = (b_down - n_downstream_catch) * self.c_downstream

            waterfall = {
                "delta_c_mat": delta_c_mat,
                "delta_c_rw": delta_c_rw,
                "delta_c_rev": delta_c_rev,
                "c_edge": delta_c_edge,
                "delta_c_esc": delta_c_esc,
            }
            if self.c_downstream > 0.0 or abs(delta_c_down) > 1e-6:
                waterfall["delta_c_downstream"] = delta_c_down
            waterfall["delta_c_total"] = benefit

        if waterfall:
            wf_sum = sum(v for k, v in waterfall.items() if k != "delta_c_total")
            res_val = benefit - wf_sum
            assert abs(res_val) < 1e-3, f"Waterfall residual violation: sum={wf_sum}, net_benefit={benefit}"

        class_carbon_breakdown = {}
        if class_counts:
            esc_c = class_counts.get("class_c_escape", n_escape) * self.c_escape
            class_carbon_breakdown = {
                "class_c_escape_carbon": esc_c,
                "field_escape_count": n_escape,
                "downstream_catch_count": n_downstream_catch,
                "q_rw_a": class_counts.get("q_rw_a", 0.0),
                "q_rw_b": class_counts.get("q_rw_b", 0.0),
                "q_rw_c": class_counts.get("q_rw_c", 0.0),
            }

        return CarbonOutcomes(
            material_embodied_carbon_kgco2e=c_mat,
            electricity_carbon_kgco2e=c_elec,
            escape_penalty_carbon_kgco2e=c_esc,
            total_carbon_kgco2e=total,
            net_carbon_benefit_kgco2e=benefit,
            waterfall_components=waterfall,
            class_carbon_breakdown=class_carbon_breakdown,
            n_field_escape=n_escape,
            n_downstream_catch=n_downstream_catch,
            downstream_penalty_carbon_kgco2e=c_down,
        )