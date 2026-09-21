"""
Material sustainability accounting: gross scrap, circular recovery, and net material savings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialOutcomes:
    gross_scrap_kg: float
    recovered_scrap_kg: float
    net_loss_kg: float
    net_savings_kg: float = 0.0


class MaterialAccountingEngine:
    def __init__(self, part_mass_kg: float, recovery_fraction: float):
        if part_mass_kg <= 0:
            raise ValueError(f"Part mass must be positive, got {part_mass_kg}")
        if not (0.0 <= recovery_fraction <= 1.0):
            raise ValueError(f"Recovery fraction must be in [0, 1], got {recovery_fraction}")
        self.mass = part_mass_kg
        self.eta = recovery_fraction

    def evaluate(self, n_scrap: float, baseline_loss_kg: float = None) -> MaterialOutcomes:
        gross = max(0.0, n_scrap) * self.mass
        recovered = gross * self.eta
        loss = gross * (1.0 - self.eta)

        # Invariant check
        assert recovered <= gross + 1e-9, f"Recovery ({recovered}) exceeds gross ({gross})"
        assert loss >= -1e-9, f"Loss cannot be negative ({loss})"

        savings = 0.0
        if baseline_loss_kg is not None:
            savings = baseline_loss_kg - loss

        return MaterialOutcomes(
            gross_scrap_kg=gross,
            recovered_scrap_kg=recovered,
            net_loss_kg=loss,
            net_savings_kg=savings,
        )