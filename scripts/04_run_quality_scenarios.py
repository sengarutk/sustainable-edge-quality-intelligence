#!/usr/bin/env python3
"""
Step 4: Evaluate 5-Tier Policy Hierarchy (B0 to B4) across Scenarios A, B, and C.
"""

from pathlib import Path
import json
import pandas as pd

from src.models.pipeline_evaluator import ScenarioPipelineEvaluator

CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")
OUTPUT_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/processed")

SCENARIOS = ["precision_component", "machined_metal", "high_value_component"]

POLICY_TIERS = [
    {"policy": "B0_Raw", "recall": 0.88, "fa_per_hr": 180.0, "delay_frames": 0.0, "edge_active": False},
    {"policy": "B1_Quantile99", "recall": 0.92, "fa_per_hr": 90.0, "delay_frames": 0.0, "edge_active": True},
    {"policy": "B2_CCT", "recall": 0.96, "fa_per_hr": 35.0, "delay_frames": 0.0, "edge_active": True},
    {"policy": "B3_CCT_kofN", "recall": 0.98, "fa_per_hr": 15.0, "delay_frames": 3.0, "edge_active": True},
    {"policy": "B4_Full_Cascade", "recall": 0.99, "fa_per_hr": 12.0, "delay_frames": 3.0, "edge_active": True},
]


def main():
    print("=== Step 04: Running Quality Inspection & Sustainability Scenarios ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    json_records = []

    for sc_name in SCENARIOS:
        sc_path = CONFIG_DIR / f"{sc_name}.yaml"
        ev = ScenarioPipelineEvaluator.from_yaml(sc_path)

        # Baseline evaluation
        b0_spec = POLICY_TIERS[0]
        b0_res = ev.evaluate_policy(
            b0_spec["policy"],
            b0_spec["recall"],
            b0_spec["fa_per_hr"],
            b0_spec["delay_frames"],
            edge_active=b0_spec["edge_active"],
        )

        for spec in POLICY_TIERS:
            if spec["policy"] == "B0_Raw":
                res = b0_res
            else:
                res = ev.evaluate_policy(
                    spec["policy"],
                    spec["recall"],
                    spec["fa_per_hr"],
                    spec["delay_frames"],
                    baseline_result=b0_res,
                    edge_active=spec["edge_active"],
                )

            record = {
                "scenario": sc_name,
                "policy": spec["policy"],
                "recall": spec["recall"],
                "delay_frames": spec["delay_frames"],
                "alert_rate_per_hr": spec["fa_per_hr"],
                "queue_rho": res.workload.queue_utilization_rho,
                "is_overloaded": res.workload.is_overloaded,
                "n_rework": res.intervention.n_rework,
                "n_scrap": res.intervention.n_scrap,
                "n_escape": res.intervention.n_escape,
                "gross_scrap_kg": res.material.gross_scrap_kg,
                "recovered_scrap_kg": res.material.recovered_scrap_kg,
                "net_loss_kg": res.material.net_loss_kg,
                "delta_m_kg": res.material.net_savings_kg,
                "review_hours": res.workload.total_review_hours,
                "delta_h_hours": res.workload.avoided_hours,
                "edge_compute_kwh": res.energy.edge_compute_kwh,
                "review_kwh": res.energy.human_review_kwh,
                "rework_kwh": res.energy.rework_kwh,
                "total_energy_kwh": res.energy.total_energy_kwh,
                "delta_e_kwh": res.energy.net_energy_diff_kwh,
                "mat_carbon_kgco2e": res.carbon.material_embodied_carbon_kgco2e,
                "elec_carbon_kgco2e": res.carbon.electricity_carbon_kgco2e,
                "escape_carbon_kgco2e": res.carbon.escape_penalty_carbon_kgco2e,
                "total_carbon_kgco2e": res.carbon.total_carbon_kgco2e,
                "delta_c_kgco2e": res.carbon.net_carbon_benefit_kgco2e,
                "sqi_balanced": res.sqi.profile_scores.get("balanced", 0.0),
                "sqi_material_priority": res.sqi.profile_scores.get("material_priority", 0.0),
                "sqi_carbon_priority": res.sqi.profile_scores.get("carbon_priority", 0.0),
                "sqi_human_centered": res.sqi.profile_scores.get("human_centered", 0.0),
            }
            rows.append(record)
            json_records.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "scenario_evaluations.csv", index=False)
    with open(OUTPUT_DIR / "scenario_evaluations.json", "w") as f:
        json.dump(json_records, f, indent=2)

    print(f"Exported {len(df)} evaluations to {OUTPUT_DIR / 'scenario_evaluations.csv'}")
    print("Step 04 completed successfully.\n")


if __name__ == "__main__":
    main()