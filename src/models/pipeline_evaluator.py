"""
Unified evaluator: one code path for every inspection regime (no inspection,
manual end-of-line inspection, and the four edge-AI alert policies of Paper A).

All inputs come from the source registry (via src.params) and configs/policies.yaml.
Parameter values may be scalars or equally-shaped numpy arrays (Monte Carlo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import yaml

from src.paths import POLICY_CONFIG
from src.quality.intervention_model import RoutingOutcome, route_defects
from src.sustainability.carbon_accounting import COMPONENTS, CarbonOutcome, carbon_flows, return_freight_carbon
from src.sustainability.energy_accounting import EnergyOutcome, energy_flows
from src.sustainability.material_accounting import MaterialOutcome, material_flows
from src.sustainability.sqi import SQIResult, evaluate_sqi
from src.sustainability.workload_accounting import (
    WorkloadOutcome, ai_review_workload, functional_unit_hours, manual_inspection_workload, no_workload,
)

AI_TIERS = ("B0_Raw", "B1_EMA", "B2_EMA_kofN", "B3_Full_Policy")
PRIMARY_TIER = "B3_Full_Policy"
BASELINES = ("N0_NoInspection", "L0_Manual")


@dataclass(frozen=True)
class Regime:
    id: str
    label: str
    kind: str                 # none | manual | ai
    persistence: bool = False  # AI tiers with k-of-N persistence carry the recall loss delta_r


def load_regimes(path=POLICY_CONFIG) -> Dict[str, Regime]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    regimes: Dict[str, Regime] = {}
    for rid, spec in cfg["baselines"].items():
        regimes[rid] = Regime(id=rid, label=spec["label"], kind=spec["kind"])
    for rid, spec in cfg["ai_tiers"].items():
        if not isinstance(spec.get("persistence"), bool):
            raise ValueError(f"{path}: {rid} needs a boolean 'persistence'")
        regimes[rid] = Regime(id=rid, label=spec["label"], kind="ai", persistence=spec["persistence"])
    if tuple(regimes) != BASELINES + AI_TIERS:
        raise ValueError(f"{path}: expected regimes {BASELINES + AI_TIERS}, got {tuple(regimes)}")
    for r in regimes.values():
        if r.kind not in ("none", "manual", "ai"):
            raise ValueError(f"{path}: invalid kind for {r.id}")
    return regimes


@dataclass(frozen=True)
class RegimeResult:
    regime: Regime
    t_fu_h: np.ndarray
    recall: np.ndarray
    delay_s: np.ndarray
    false_alarm_rate: np.ndarray
    routing: RoutingOutcome
    material: MaterialOutcome
    workload: WorkloadOutcome
    energy: EnergyOutcome
    carbon: CarbonOutcome

    def dims(self) -> Dict[str, np.ndarray]:
        """The four SQI dimensions (lower is better for all)."""
        return {"M": self.material.net_loss_kg, "E": self.energy.total_kwh,
                "C": self.carbon.total, "H": self.workload.operator_hours}


def ai_operating_point(p: Mapping[str, float], regime: Regime):
    """Recall, interception delay [s], operator-facing false-alert rate [1/h] and alerts per
    detected defect of an AI tier (rates and delays measured in Paper A)."""
    t = regime.id
    recall = np.maximum(0.0, p["ai_recall"] - (p["persistence_recall_loss"] if regime.persistence else 0.0))
    delay_s = p[f"detection_delay@{t}"] / p["camera_fps"]
    fa = p[f"nominal_false_alarm_rate@{t}"] + p["glare_burst_rate"] * p[f"glare_alerts_per_burst@{t}"]
    return recall, delay_s, fa, p[f"alerts_per_defect@{t}"]


def evaluate_regime(p: Mapping[str, float], class_shares, regime: Regime) -> RegimeResult:
    n = p["functional_unit"]
    t_fu = functional_unit_hours(n, p["line_throughput"])
    defects = n * np.asarray(p["defect_prevalence"], dtype=float)
    zero = 0.0 * defects

    if regime.kind == "none":
        recall, delay_s, fa, per_defect = zero, zero, zero, zero
    elif regime.kind == "manual":
        recall, delay_s, fa, per_defect = p["manual_inspection_recall"] + zero, p["manual_discovery_delay"] + zero, zero, zero
    else:
        recall, delay_s, fa, per_defect = ai_operating_point(p, regime)

    routing = route_defects(defects, recall, delay_s, p["base_reworkability"], p["reworkability_time_constant"], class_shares)
    material = material_flows(routing.n_scrap, p["part_mass"], p["material_recovery_fraction"])

    if regime.kind == "none":
        work = no_workload(defects)
    elif regime.kind == "manual":
        work = manual_inspection_workload(n, p["line_throughput"], p["manual_inspection_time"])
    else:
        work = ai_review_workload(fa, routing.routed, per_defect, t_fu, p["review_time"])

    edge_active = regime.kind == "ai"
    energy = energy_flows(edge_active, p["gpu_idle_power"], p["gpu_active_power"], p["host_power"], t_fu,
                          work.operator_hours, p["review_station_power"], routing.n_rework, p["rework_energy"])
    carbon = carbon_flows(material.net_loss_kg, material.recovered_kg, p["material_carbon_factor"], p["recycling_burden"],
                          energy, p["grid_carbon_factor"], edge_active, p["edge_embodied_carbon"], t_fu,
                          p["edge_lifetime_hours"], routing.n_escape, routing.n_escape_class_c, p["part_mass"],
                          return_freight_carbon(p["part_mass"], p["return_distance"], p["freight_carbon_intensity"]),
                          p["collateral_multiplier"])
    return RegimeResult(regime=regime, t_fu_h=t_fu + zero, recall=np.asarray(recall) + zero,
                        delay_s=np.asarray(delay_s) + zero, false_alarm_rate=np.asarray(fa) + zero,
                        routing=routing, material=material, workload=work, energy=energy, carbon=carbon)


@dataclass(frozen=True)
class Comparison:
    """Savings of `policy` relative to `baseline` (positive = improvement)."""
    baseline: RegimeResult
    policy: RegimeResult
    delta: Dict[str, np.ndarray]        # M, E, C, H
    waterfall: Dict[str, np.ndarray]    # carbon components; sum == delta['C']
    sqi: SQIResult


def compare(baseline: RegimeResult, policy: RegimeResult, profiles=None) -> Comparison:
    base_dims, policy_dims = baseline.dims(), policy.dims()
    delta = {k: base_dims[k] - policy_dims[k] for k in base_dims}
    waterfall = baseline.carbon.delta_components(policy.carbon)
    closure = np.max(np.abs(sum(waterfall[c] for c in COMPONENTS) - delta["C"]))
    if closure > 1e-9 * max(1.0, float(np.max(np.abs(base_dims["C"])))):
        raise ArithmeticError(f"carbon waterfall does not close (residual {closure})")
    return Comparison(baseline=baseline, policy=policy, delta=delta, waterfall=waterfall,
                      sqi=evaluate_sqi(base_dims, policy_dims, profiles))


def evaluate_all(p: Mapping[str, float], class_shares, regimes: Optional[Dict[str, Regime]] = None) -> Dict[str, RegimeResult]:
    regimes = regimes or load_regimes()
    return {rid: evaluate_regime(p, class_shares, r) for rid, r in regimes.items()}


def delta_carbon(p: Mapping[str, float], class_shares, base: str, policy: str) -> float:
    """Scalar Delta C [kgCO2e/FU] of `policy` relative to `base` for one parameter set."""
    res = evaluate_all(p, class_shares)
    return float(compare(res[base], res[policy]).delta["C"])
