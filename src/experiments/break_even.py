"""
Break-even frontiers (Delta C = 0) for the central parameter set.

Each frontier is solved with Brent's method on a physically bounded interval.
Delta C is monotone in every swept parameter (affine in pi, P_host, r_L0, gamma
and the persistence recall loss; exponential in d_L0), so the signs at the interval ends decide
whether a root exists. When they agree the result is reported explicitly as
'none' together with the sign that holds everywhere, instead of being dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

from scipy.optimize import brentq

from src.models.pipeline_evaluator import delta_carbon
from src.params import Scenario


@dataclass(frozen=True)
class Frontier:
    name: str
    parameter: str
    comparison: str
    lower: float
    upper: float
    value: Optional[float]
    status: str          # root | none_positive | none_negative
    unit: str


def _solve(f: Callable[[float], float], lo: float, hi: float):
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo, "root"
    if fhi == 0.0:
        return hi, "root"
    if flo * fhi < 0:
        return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=500)), "root"
    return None, "none_positive" if flo > 0 else "none_negative"


SPECS = [
    # name, registry key, baseline, interval, unit
    ("pi_star_vs_N0", "defect_prevalence", "N0_NoInspection", (0.0, 0.5), "fraction"),
    ("pi_star_vs_L0", "defect_prevalence", "L0_Manual", (0.0, 0.5), "fraction"),
    ("host_power_star_vs_L0", "host_power", "L0_Manual", (0.0, 1.0e6), "W"),
    ("manual_recall_star", "manual_inspection_recall", "L0_Manual", (0.0, 1.0), "fraction"),
    ("manual_delay_star", "manual_discovery_delay", "L0_Manual", (0.0, 3600.0), "s"),
    ("gamma_star_vs_L0", "grid_carbon_factor", "L0_Manual", (0.0, 1.5), "kgCO2e/kWh"),
    ("recall_loss_star_B3_vs_B0", "persistence_recall_loss", "B0_Raw", (0.0, 0.99), "fraction"),
]


def solve_frontiers(scenario: Scenario, policy: str = "B3_Full_Policy") -> Dict[str, Dict]:
    central = scenario.central()
    out = {}
    for name, key, base, (lo, hi), unit in SPECS:
        def dc(x, key=key, base=base):
            return delta_carbon({**central, key: x}, scenario.class_shares, base, policy)

        value, status = _solve(dc, lo, hi)
        out[name] = asdict(Frontier(name, key, f"{policy}_vs_{base}", lo, hi, value, status, unit))
    return out
