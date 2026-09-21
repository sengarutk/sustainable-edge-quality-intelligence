"""
Quality intervention model enforcing delay-sensitive reworkability and physical count conservation.
"""

from dataclasses import dataclass
from typing import Dict
from src.quality.defect_taxonomy import DefectClass, DefectBatch


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


class QualityInterventionModel:
    """
    Evaluates quality routing, delay-sensitive degradation, and defect outcomes.
    Guarantees strict count conservation: |D - (N_rework + N_scrap + N_escape)| < 1e-6.
    """
    def __init__(
        self,
        base_reworkability: float = 0.85,
        reworkability_decay_per_sec: float = 0.20,
        sampling_rate_fps: float = 30.0,
    ):
        self.q0 = base_reworkability
        self.beta = reworkability_decay_per_sec
        self.fps = sampling_rate_fps

    def compute_reworkability(self, delay_frames: float) -> float:
        delay_sec = max(0.0, delay_frames) / max(1.0, self.fps)
        q_rw = max(0.0, self.q0 - self.beta * delay_sec)
        return min(1.0, q_rw)

    def evaluate(
        self,
        total_defects: float,
        recall: float,
        delay_frames: float,
    ) -> InterventionOutcomes:
        # Clamp recall to [0.0, 1.0]
        r_p = max(0.0, min(1.0, recall))
        d_routed = total_defects * r_p
        d_missed = total_defects * (1.0 - r_p)

        delay_sec = max(0.0, delay_frames) / max(1.0, self.fps)
        q_rw = self.compute_reworkability(delay_frames)
        q_sc = 1.0 - q_rw

        n_rework = d_routed * q_rw
        n_scrap = d_routed * q_sc
        n_escape = d_missed

        # Strict physical count conservation
        total_outcomes = n_rework + n_scrap + n_escape
        assert abs(total_defects - total_outcomes) < 1e-6, (
            f"Conservation violated: D={total_defects}, outcomes={total_outcomes}"
        )

        return InterventionOutcomes(
            total_defects=total_defects,
            routed_defects=d_routed,
            missed_defects=d_missed,
            n_rework=n_rework,
            n_scrap=n_scrap,
            n_escape=n_escape,
            reworkability_effective=q_rw,
            delay_seconds=delay_sec,
        )