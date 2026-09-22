"""
Unified pipeline evaluator for edge quality inspection and sustainability accounting.
Coordinates intervention, material, energy, workload, carbon, and SQI engines.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from src.quality.intervention_model import QualityInterventionModel, InterventionOutcomes
from src.sustainability.material_accounting import MaterialAccountingEngine, MaterialOutcomes
from src.sustainability.workload_accounting import WorkloadAccountingEngine, WorkloadOutcomes
from src.sustainability.energy_accounting import EnergyAccountingEngine, EnergyOutcomes
from src.sustainability.carbon_accounting import CarbonAccountingEngine, CarbonOutcomes
from src.sustainability.sqi import SQIEngine, SQIEvaluation


@dataclass(frozen=True)
class PolicyEvaluationResult:
    scenario: str
    policy: str
    intervention: InterventionOutcomes
    material: MaterialOutcomes
    workload: WorkloadOutcomes
    energy: EnergyOutcomes
    carbon: CarbonOutcomes
    sqi: SQIEvaluation
    waterfall: Dict[str, float] = field(default_factory=dict)


class ScenarioPipelineEvaluator:
    def __init__(self, scenario_config: Dict[str, Any], sqi_config_path: Path = None):
        self.cfg = scenario_config
        self.name = scenario_config["name"]
        self.n_units = int(scenario_config.get("functional_unit_units", 1000))
        self.n_f = float(scenario_config.get("frames_per_part", 1.0))
        self.pi = float(scenario_config["defect_prevalence"])
        self.total_defects = self.n_units * self.pi

        self.m_kg = float(scenario_config["part_mass_kg"])
        self.ef_mat = float(scenario_config["material_carbon_factor_kgco2e_per_kg"])
        self.eta = float(scenario_config["material_recovery_fraction"])
        self.e_rw = float(scenario_config["rework_energy_kwh_per_unit"])
        self.c_esc = float(scenario_config["escape_carbon_penalty_kgco2e"])
        self.q0 = float(scenario_config["base_reworkability"])
        self.beta = float(scenario_config["reworkability_decay_per_sec"])
        self.fps = float(scenario_config.get("sampling_rate_fps", 30.0))
        self.q_field = float(scenario_config.get("q_field", 1.0))

        # Calibrated default parameters
        self.gamma = float(scenario_config.get("grid_carbon_factor", 0.417))
        
        # Harmonize energy scaling
        if "edge_energy_kwh_per_1k" in scenario_config:
            self.e_edge = float(scenario_config["edge_energy_kwh_per_1k"])
            self.e_frame_wh = (self.e_edge * 1000.0) / (self.n_units * self.n_f)
        elif "edge_energy_wh_per_frame" in scenario_config:
            self.e_frame_wh = float(scenario_config["edge_energy_wh_per_frame"])
            self.e_edge = (self.n_units * self.n_f * self.e_frame_wh) / 1000.0
        else:
            self.e_frame_wh = 0.171
            self.e_edge = 0.171

        self.p_station = float(scenario_config.get("workstation_power_w", 85.0))
        self.mu = float(scenario_config.get("service_rate_mu", 60.0))
        self.t_review = float(scenario_config.get("review_duration_seconds", 30.0))
        self.defect_dist = scenario_config.get("defect_distribution", {
            "class_a_reworkable": 0.70,
            "class_b_scrap_prone": 0.25,
            "class_c_escape_sensitive": 0.05,
        })
        self.class_decay_params = scenario_config.get("class_decay_params", None)

        # Initialize engines
        self.interv_engine = QualityInterventionModel(
            base_reworkability=self.q0,
            reworkability_decay_per_sec=self.beta,
            sampling_rate_fps=self.fps,
            q_field=self.q_field,
            defect_distribution=self.defect_dist,
            class_decay_params=self.class_decay_params,
        )
        self.mat_engine = MaterialAccountingEngine(
            part_mass_kg=self.m_kg, recovery_fraction=self.eta
        )
        self.work_engine = WorkloadAccountingEngine(
            service_capacity_mu_per_hr=self.mu,
            review_duration_seconds=self.t_review,
            sampling_rate_fps=self.fps,
            functional_unit_units=self.n_units,
        )
        self.energy_engine = EnergyAccountingEngine(
            edge_energy_kwh_per_1k=self.e_edge,
            review_workstation_power_w=self.p_station,
            rework_energy_kwh_per_unit=self.e_rw,
            functional_unit_units=self.n_units,
            frames_per_part=self.n_f,
            edge_energy_wh_per_frame=self.e_frame_wh,
        )
        self.carbon_engine = CarbonAccountingEngine(
            material_carbon_factor_kgco2e_per_kg=self.ef_mat,
            grid_carbon_factor_kgco2e_per_kwh=self.gamma,
            escape_carbon_penalty_kgco2e=self.c_esc,
        )
        self.sqi_engine = SQIEngine(config_path=sqi_config_path)

    @classmethod
    def from_yaml(cls, yaml_path: Path, sqi_config_path: Path = None) -> "ScenarioPipelineEvaluator":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, sqi_config_path)

    def evaluate_policy(
        self,
        policy_name: str,
        recall: float,
        alert_rate_per_hr: float,
        delay_frames: float,
        baseline_result: PolicyEvaluationResult = None,
        edge_active: bool = True,
        class_specific: bool = True,
    ) -> PolicyEvaluationResult:
        interv = self.interv_engine.evaluate(
            total_defects=self.total_defects,
            recall=recall,
            delay_frames=delay_frames,
            defect_distribution=self.defect_dist,
            q_field=self.q_field,
            class_specific=class_specific,
        )

        base_mat_loss = baseline_result.material.net_loss_kg if baseline_result else None
        mat = self.mat_engine.evaluate(interv.n_scrap, baseline_loss_kg=base_mat_loss)

        base_work_hrs = baseline_result.workload.total_review_hours if baseline_result else None
        work = self.work_engine.evaluate(alert_rate_per_hr, baseline_hours=base_work_hrs)

        base_energy_tot = baseline_result.energy.total_energy_kwh if baseline_result else None
        energy = self.energy_engine.evaluate(
            review_hours=work.total_review_hours,
            n_rework=interv.n_rework,
            baseline_total_kwh=base_energy_tot,
            edge_active=edge_active,
            frames_per_part=self.n_f,
        )

        base_carbon_tot = baseline_result.carbon.total_carbon_kgco2e if baseline_result else None
        base_rw_kwh = baseline_result.energy.rework_kwh if baseline_result else None
        base_rev_kwh = baseline_result.energy.human_review_kwh if baseline_result else None
        base_n_esc = baseline_result.intervention.n_escape if baseline_result else None
        base_edge_kwh = baseline_result.energy.edge_compute_kwh if baseline_result else None
        n_downstream = interv.class_counts.get("downstream_catch", 0.0)
        base_n_down = (
            baseline_result.carbon.n_downstream_catch if baseline_result else None
        )

        carbon = self.carbon_engine.evaluate(
            net_material_loss_kg=mat.net_loss_kg,
            total_energy_kwh=energy.total_energy_kwh,
            n_escape=interv.n_escape,
            baseline_carbon_kgco2e=base_carbon_tot,
            baseline_material_loss_kg=base_mat_loss,
            baseline_rework_kwh=base_rw_kwh,
            baseline_review_kwh=base_rev_kwh,
            current_rework_kwh=energy.rework_kwh,
            current_review_kwh=energy.human_review_kwh,
            current_edge_kwh=energy.edge_compute_kwh,
            baseline_n_escape=base_n_esc,
            baseline_edge_kwh=base_edge_kwh,
            class_counts=interv.class_counts,
            n_downstream_catch=n_downstream,
            baseline_n_downstream=base_n_down,
        )

        if baseline_result:
            sqi_res = self.sqi_engine.evaluate(
                delta_m=mat.net_savings_kg,
                m_baseline_loss=base_mat_loss,
                delta_e=energy.net_energy_diff_kwh,
                e_baseline_total=base_energy_tot,
                delta_c=carbon.net_carbon_benefit_kgco2e,
                c_baseline_total=base_carbon_tot,
                delta_h=work.avoided_hours,
                h_baseline_hours=base_work_hrs,
            )
        else:
            sqi_res = self.sqi_engine.evaluate(
                delta_m=0.0,
                m_baseline_loss=mat.net_loss_kg,
                delta_e=0.0,
                e_baseline_total=energy.total_energy_kwh,
                delta_c=0.0,
                c_baseline_total=carbon.total_carbon_kgco2e,
                delta_h=0.0,
                h_baseline_hours=work.total_review_hours,
            )

        return PolicyEvaluationResult(
            scenario=self.name,
            policy=policy_name,
            intervention=interv,
            material=mat,
            workload=work,
            energy=energy,
            carbon=carbon,
            sqi=sqi_res,
            waterfall=carbon.waterfall_components,
        )