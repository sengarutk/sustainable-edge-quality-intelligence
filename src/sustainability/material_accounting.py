"""Material accounting: gross scrap mass, recovered fraction and unrecovered loss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialOutcome:
    gross_scrap_kg: np.ndarray
    recovered_kg: np.ndarray
    net_loss_kg: np.ndarray


def material_flows(n_scrap, part_mass_kg, recovery_fraction) -> MaterialOutcome:
    if np.any(np.asarray(part_mass_kg) <= 0):
        raise ValueError("part mass must be positive")
    if np.any((np.asarray(recovery_fraction) < 0) | (np.asarray(recovery_fraction) > 1)):
        raise ValueError("recovery fraction must lie in [0, 1]")
    gross = np.asarray(n_scrap, dtype=float) * part_mass_kg
    recovered = gross * recovery_fraction
    return MaterialOutcome(gross_scrap_kg=gross, recovered_kg=recovered, net_loss_kg=gross - recovered)
