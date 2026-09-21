"""
Multi-Profile Sustainability Quality Index (SQI) Engine.
Computes robust bounded normalized sub-indicators (S_M, S_E, S_C, S_H in [-1, +1])
and stakeholder-weighted scalar scores.
"""

from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SQISubIndicators:
    s_m: float  # Material normalized savings in [-1, 1]
    s_e: float  # Energy normalized savings in [-1, 1]
    s_c: float  # Carbon normalized savings in [-1, 1]
    s_h: float  # Workload normalized savings in [-1, 1]


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


def _bounded_relative_indicator(diff: float, val_base: float, val_current: float) -> float:
    denom = max(abs(val_base), abs(val_current), 1e-6)
    ratio = diff / denom
    return float(max(-1.0, min(1.0, ratio)))


class SQIEngine:
    def __init__(self, config_path: Path = None):
        self.profiles = dict(DEFAULT_PROFILES)
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "profiles" in data:
                    self.profiles = {k: v["weights"] for k, v in data["profiles"].items()}

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
        m_current_loss: float = None,
        e_current_total: float = None,
        c_current_total: float = None,
        h_current_hours: float = None,
    ) -> SQISubIndicators:
        curr_m = m_current_loss if m_current_loss is not None else (m_baseline_loss - delta_m)
        curr_e = e_current_total if e_current_total is not None else (e_baseline_total - delta_e)
        curr_c = c_current_total if c_current_total is not None else (c_baseline_total - delta_c)
        curr_h = h_current_hours if h_current_hours is not None else (h_baseline_hours - delta_h)

        s_m = _bounded_relative_indicator(delta_m, m_baseline_loss, curr_m)
        s_e = _bounded_relative_indicator(delta_e, e_baseline_total, curr_e)
        s_c = _bounded_relative_indicator(delta_c, c_baseline_total, curr_c)
        s_h = _bounded_relative_indicator(delta_h, h_baseline_hours, curr_h)

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
        m_current_loss: float = None,
        e_current_total: float = None,
        c_current_total: float = None,
        h_current_hours: float = None,
    ) -> SQIEvaluation:
        subs = self.compute_sub_indicators(
            delta_m, m_baseline_loss,
            delta_e, e_baseline_total,
            delta_c, c_baseline_total,
            delta_h, h_baseline_hours,
            m_current_loss=m_current_loss,
            e_current_total=e_current_total,
            c_current_total=c_current_total,
            h_current_hours=h_current_hours,
        )
        scores = {}
        for prof, w in self.profiles.items():
            score = (
                w["w_M"] * subs.s_m
                + w["w_E"] * subs.s_e
                + w["w_C"] * subs.s_c
                + w["w_H"] * subs.s_h
            )
            scores[prof] = round(score, 4)

        return SQIEvaluation(sub_indicators=subs, profile_scores=scores)