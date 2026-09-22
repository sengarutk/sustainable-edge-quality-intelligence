"""
Quality intervention model enforcing delay-sensitive reworkability and physical count conservation.
Incorporates class-specific decay (q_rw,j), class-specific field escape vs downstream catch (q_field,j),
and strict defect count conservation: |D - (N_rework + N_scrap + N_escape)| < 1e-6.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class InterventionOutcomes:
    total_defects: float
    routed_defects: float
    missed_defects: float
    n_rework: float
    n_scrap: float
    n_escape: float
    reworkability_effective: float
    delay_seconds: float
    q_field: float = 1.0
    class_counts: Dict[str, float] = field(default_factory=dict)
    waterfall_defect_breakdown: Dict[str, float] = field(default_factory=dict)


class QualityInterventionModel:
    """
    Evaluates quality routing, delay-sensitive degradation, and defect outcomes.
    Guarantees strict count conservation: |D - (N_rework + N_scrap + N_escape)| < 1e-6.
    Supports class-specific breakdown (Class A reworkable, Class B scrap-prone, Class C escape-sensitive)
    and partitions missed defects into field escape vs downstream catch.
    """
    def __init__(
        self,
        base_reworkability: float = 0.85,
        reworkability_decay_per_sec: float = 0.20,
        sampling_rate_fps: float = 30.0,
        q_field: float = 1.0,
        defect_distribution: Dict[str, float] = None,
        class_decay_params: Dict[str, Dict[str, float]] = None,
    ):
        self.q0 = base_reworkability
        self.beta = reworkability_decay_per_sec
        self.fps = sampling_rate_fps
        self.q_field = q_field
        self.defect_distribution = defect_distribution or {
            "class_a_reworkable": 0.70,
            "class_b_scrap_prone": 0.25,
            "class_c_escape_sensitive": 0.05,
        }
        self.class_decay_params = class_decay_params or {}

    def compute_reworkability(self, delay_frames: float) -> float:
        delay_sec = max(0.0, delay_frames) / max(1.0, self.fps)
        q_rw = max(0.0, self.q0 - self.beta * delay_sec)
        return min(1.0, q_rw)

    def evaluate(
        self,
        total_defects: float,
        recall: float,
        delay_frames: float,
        defect_distribution: Dict[str, float] = None,
        q_field: float = None,
        class_specific: bool = False,
        class_decay_params: Dict[str, Dict[str, float]] = None,
    ) -> InterventionOutcomes:
        r_p = max(0.0, min(1.0, recall))
        d_routed = total_defects * r_p
        d_missed = total_defects * (1.0 - r_p)

        delay_sec = max(0.0, delay_frames) / max(1.0, self.fps)
        q_rw = self.compute_reworkability(delay_frames)
        q_sc = 1.0 - q_rw

        effective_q_field = self.q_field if q_field is None else q_field
        dist = defect_distribution or self.defect_distribution

        f_a = float(dist.get("class_a_reworkable", 0.70))
        f_b = float(dist.get("class_b_scrap_prone", 0.25))
        f_c = float(dist.get("class_c_escape_sensitive", 0.05))
        tot_f = f_a + f_b + f_c
        if tot_f > 0:
            f_a, f_b, f_c = f_a / tot_f, f_b / tot_f, f_c / tot_f

        d_a = total_defects * f_a
        d_b = total_defects * f_b
        d_c = total_defects * f_c

        cd_params = class_decay_params or self.class_decay_params

        q_rw_a = q_rw
        q_rw_b = 0.0
        q_rw_c = 0.0

        q_field_a = effective_q_field
        q_field_b = effective_q_field
        q_field_c = 1.0

        if cd_params:
            if "class_a_reworkable" in cd_params:
                p_a = cd_params["class_a_reworkable"]
                q0_a = float(p_a.get("q0", self.q0))
                beta_a = float(p_a.get("beta", self.beta))
                q_rw_a = max(0.0, min(1.0, q0_a - beta_a * delay_sec))
                q_field_a = float(p_a.get("q_field", effective_q_field))
            if "class_b_scrap_prone" in cd_params:
                p_b = cd_params["class_b_scrap_prone"]
                q_rw_b = float(p_b.get("q_rw", 0.0))
                q_field_b = float(p_b.get("q_field", effective_q_field))
            if "class_c_escape_sensitive" in cd_params:
                p_c = cd_params["class_c_escape_sensitive"]
                q_rw_c = float(p_c.get("q_rw", 0.0))
                q_field_c = float(p_c.get("q_field", 1.0))

        q_sc_a = 1.0 - q_rw_a

        if class_specific:
            n_rework = d_a * r_p * q_rw_a + d_b * r_p * q_rw_b + d_c * r_p * q_rw_c
            n_sc_a = d_a * r_p * q_sc_a
            n_sc_b = d_b * r_p * (1.0 - q_rw_b)
            n_sc_c = d_c * r_p * (1.0 - q_rw_c)

            n_esc_a = (d_a * (1.0 - r_p)) * q_field_a
            n_down_a = (d_a * (1.0 - r_p)) * (1.0 - q_field_a)

            n_esc_b = (d_b * (1.0 - r_p)) * q_field_b
            n_down_b = (d_b * (1.0 - r_p)) * (1.0 - q_field_b)

            n_esc_c = (d_c * (1.0 - r_p)) * q_field_c
            n_down_c = (d_c * (1.0 - r_p)) * (1.0 - q_field_c)

            n_esc = n_esc_a + n_esc_b + n_esc_c
            n_sc_missed = n_down_a + n_down_b + n_down_c
            n_scrap = n_sc_a + n_sc_b + n_sc_c + n_sc_missed
            n_escape = n_esc
            n_downstream = n_sc_missed
        else:
            n_rework = d_routed * q_rw
            n_esc = d_missed * effective_q_field
            n_sc_missed = d_missed * (1.0 - effective_q_field)
            n_scrap = d_routed * q_sc + n_sc_missed
            n_escape = n_esc
            n_downstream = n_sc_missed

        total_outcomes = n_rework + n_scrap + n_escape
        assert abs(total_defects - total_outcomes) < 1e-6, (
            f"Conservation violated: D={total_defects}, outcomes={total_outcomes}"
        )

        class_counts = {
            "class_a_total": d_a,
            "class_b_total": d_b,
            "class_c_total": d_c,
            "class_a_routed": d_a * r_p,
            "class_b_routed": d_b * r_p,
            "class_c_routed": d_c * r_p,
            "class_a_missed": d_a * (1.0 - r_p),
            "class_b_missed": d_b * (1.0 - r_p),
            "class_c_missed": d_c * (1.0 - r_p),
            "class_a_rework": d_a * r_p * q_rw_a if class_specific else d_a * r_p * q_rw,
            "class_a_scrap": d_a * r_p * q_sc_a if class_specific else d_a * r_p * q_sc,
            "class_b_scrap": d_b * r_p * (1.0 - q_rw_b) if class_specific else d_b * r_p,
            "class_c_scrap": d_c * r_p * (1.0 - q_rw_c) if class_specific else d_c * r_p,
            "class_c_escape": (d_c * (1.0 - r_p)) * q_field_c if class_specific else d_c * (1.0 - r_p) * effective_q_field,
            "field_escapes": n_escape,
            "downstream_catch": n_downstream,
            "q_rw_a": q_rw_a,
            "q_rw_b": q_rw_b,
            "q_rw_c": q_rw_c,
            "q_field_a": q_field_a,
            "q_field_b": q_field_b,
            "q_field_c": q_field_c,
        }

        waterfall_breakdown = {
            "total_defects": total_defects,
            "routed": d_routed,
            "missed": d_missed,
            "salvaged_rework": n_rework,
            "condemned_scrap": n_scrap,
            "field_escapes": n_escape,
            "downstream_catch": n_downstream,
        }

        return InterventionOutcomes(
            total_defects=total_defects,
            routed_defects=d_routed,
            missed_defects=d_missed,
            n_rework=n_rework,
            n_scrap=n_scrap,
            n_escape=n_escape,
            reworkability_effective=q_rw,
            delay_seconds=delay_sec,
            q_field=effective_q_field,
            class_counts=class_counts,
            waterfall_defect_breakdown=waterfall_breakdown,
        )