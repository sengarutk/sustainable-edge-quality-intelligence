"""
Carbon accounting per functional unit [kgCO2e/FU], as additive components:

  mat      = M_loss * EF_mat                         unrecovered scrap material
  recycle  = M_recovered * EF_rec                    processing burden of recovered scrap
  rework   = E_rework * gamma
  review   = E_review * gamma
  edge     = E_edge * gamma                          edge-cell electricity
  hardware = C_hw * T_FU / L_hw                      amortised embodied carbon of edge hardware
  escape   = N_esc * (m * EF_mat + c_ret) + N_esc,C * kappa * m * EF_mat
  c_ret    = 2 * (m / 1000) * d_ret * EF_fr               return + replacement road freight

An escaped part is replaced (its full embodied material is lost) and returned
(c_ret); class-C escapes add collateral system damage kappa * m * EF_mat.
Differences between regimes decompose exactly into the same components.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

COMPONENTS = ("mat", "recycle", "rework", "review", "edge", "hardware", "escape")


@dataclass(frozen=True)
class CarbonOutcome:
    mat: np.ndarray
    recycle: np.ndarray
    rework: np.ndarray
    review: np.ndarray
    edge: np.ndarray
    hardware: np.ndarray
    escape: np.ndarray

    @property
    def total(self):
        return sum(getattr(self, c) for c in COMPONENTS)

    def delta_components(self, other: "CarbonOutcome"):
        """Savings of `other` relative to `self` (baseline): positive = other emits less."""
        return {c: getattr(self, c) - getattr(other, c) for c in COMPONENTS}


def return_freight_carbon(part_mass_kg, distance_km, intensity_kg_per_tkm):
    """Road freight of the returned part and of its replacement [kgCO2e per escape] (GLEC t-km basis)."""
    return 2.0 * (np.asarray(part_mass_kg) / 1000.0) * distance_km * intensity_kg_per_tkm


def escape_unit_penalty(part_mass_kg, ef_mat, c_ret):
    return np.asarray(part_mass_kg) * ef_mat + c_ret


def carbon_flows(net_loss_kg, recovered_kg, ef_mat, ef_rec, energy, gamma, edge_active: bool,
                 c_hw, t_fu_h, l_hw, n_escape, n_escape_c, part_mass_kg, c_ret, kappa) -> CarbonOutcome:
    hardware = np.asarray(c_hw, dtype=float) * t_fu_h / l_hw
    if not edge_active:
        hardware = np.zeros_like(hardware)
    escape = n_escape * escape_unit_penalty(part_mass_kg, ef_mat, c_ret) + n_escape_c * kappa * part_mass_kg * ef_mat
    return CarbonOutcome(
        mat=net_loss_kg * ef_mat,
        recycle=recovered_kg * ef_rec,
        rework=energy.rework_kwh * gamma,
        review=energy.review_kwh * gamma,
        edge=energy.edge_kwh * gamma,
        hardware=hardware,
        escape=escape,
    )


if tuple(f.name for f in fields(CarbonOutcome)) != COMPONENTS:  # keep the component list and dataclass in sync
    raise ImportError("CarbonOutcome fields out of sync with COMPONENTS")
