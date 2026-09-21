"""
Quality intervention model enforcing delay-sensitive reworkability and physical count conservation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


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
    Incorporates class-specific breakdown (Class A reworkable, Class B scrap-prone, Class C escape-sensitive)
    and field escape fraction q_field.
    """
    def __init__(
        self,
        base_reworkability: float = 0.85,
        reworkability_decay_per_sec: float = 0.20,
        sampling_rate_fps: float = 30.0,
        q_field: float = 1.0,
        defect_distribution: Dict[str, float] = None,
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

        if class_specific:
            n_rework = d_a * r_p * q_rw
            n_sc_a = d_a * r_p * q_sc
            n_sc_b = d_b * r_p
            n_sc_c = d_c * r_p
            n_esc = d_missed * effective_q_field
            n_sc_missed = d_missed * (1.0 - effective_q_field)
            n_scrap = n_sc_a + n_sc_b + n_sc_c + n_sc_missed
            n_escape = n_esc
        else:
            n_rework = d_routed * q_rw
            n_scrap = d_routed * q_sc + d_missed * (1.0 - effective_q_field)
            n_escape = d_missed * effective_q_field

        # Strict physical count conservation
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
            "class_c_escape": d_c * (1.0 - r_p) * effective_q_field,
        }

        waterfall_breakdown = {
            "total_defects": total_defects,
            "routed": d_routed,
            "missed": d_missed,
            "salvaged_rework": n_rework,
            "condemned_scrap": n_scrap,
            "field_escapes": n_escape,
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
