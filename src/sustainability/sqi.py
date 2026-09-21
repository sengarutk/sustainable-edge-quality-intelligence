"""
Multi-Profile Sustainability Quality Index (SQI) Engine.
Computes normalized sub-indicators (S_M, S_E, S_C, S_H) and stakeholder-weighted scalar scores.
"""

from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SQISubIndicators:
    s_m: float  # Material normalized savings
    s_e: float  # Energy normalized savings
    s_c: float  # Carbon normalized savings
    s_h: float  # Workload normalized savings


@dataclass(frozen=True)
class SQIEvaluation:
    sub_indicators: SQISubIndicators
    profile_scores: Dict[str, float]


DEFAULT_PROFILES = {
    "balanced": {"w_M": 0.25, "w_E": 0.25, "w_C": 0.25, "w_H": 0.25},
    "material_priority": {"w_M": 0.45, "w_E": 0.15, "w_C": 0.25, "w_H": 0.15},
    "carbon_priority": {"w_M": 0.15, "w_E": 0.20, "w_C": 0.50, "w_H": 0.15},
    "human_centered": {"w_M": 0.15, "w_E": 0.15, "w_C": 0.20, "w_H": 0.50},
}


class SQIEngine:
    def __init__(self, config_path: Path = None):
        self.profiles = dict(DEFAULT_PROFILES)
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "profiles" in data:
                    self.profiles = {k: v["weights"] for k, v in data["profiles"].items()}

        # Verify weights sum to 1.0 for each profile
        for prof, w in self.profiles.items():
            tot = sum(w.values())
            if abs(tot - 1.0) > 1e-4:
                raise ValueError(f"Profile {prof} weights sum to {tot}, expected 1.0")

    def compute_sub_indicators(
        self,
        delta_m: float,
        m_baseline_loss: float,
        delta_e: float,
        e_baseline_total: float,
        delta_c: float,
        c_baseline_total: float,
        delta_h: float,
        h_baseline_hours: float,
    ) -> SQISubIndicators:
        s_m = delta_m / max(m_baseline_loss, 1e-6)
        s_e = delta_e / max(e_baseline_total, 1e-6)
        s_c = delta_c / max(c_baseline_total, 1e-6)
        s_h = delta_h / max(h_baseline_hours, 1e-6)
        return SQISubIndicators(s_m=s_m, s_e=s_e, s_c=s_c, s_h=s_h)

    def evaluate(
        self,
        delta_m: float,
        m_baseline_loss: float,
        delta_e: float,
        e_baseline_total: float,
        delta_c: float,
        c_baseline_total: float,
        delta_h: float,
        h_baseline_hours: float,
    ) -> SQIEvaluation:
        subs = self.compute_sub_indicators(
            delta_m, m_baseline_loss,
            delta_e, e_baseline_total,
            delta_c, c_baseline_total,
            delta_h, h_baseline_hours,
        )
        scores = {}
        for prof, w in self.profiles.items():
            score = (
                w["w_M"] * subs.s_m
                + w["w_E"] * subs.s_e
                + w["w_C"] * subs.s_c
                + w["w_H"] * subs.s_h
            )
            scores[prof] = score

        return SQIEvaluation(sub_indicators=subs, profile_scores=scores)