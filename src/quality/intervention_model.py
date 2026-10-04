"""
Defect routing and delay-sensitive reworkability.

Defects D = N * pi are split into three classes with shares (f_A, f_B, f_C):
  A  reworkable        -- routed parts are reworked with probability q_rw(d), else scrapped
  B  scrap-prone       -- routed parts are always scrapped
  C  escape-sensitive  -- routed parts are scrapped; field escapes carry a collateral penalty
Routed fraction = recall r; missed fraction (1 - r) escapes to the field.

Reworkability decays exponentially with interception delay d [s]:
  q_rw(d) = q0 * exp(-d / tau_rw)
All functions are vectorised (scalars or numpy arrays).
Count conservation D = N_rework + N_scrap + N_escape holds exactly and is checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RoutingOutcome:
    total_defects: np.ndarray
    routed: np.ndarray
    n_rework: np.ndarray
    n_scrap: np.ndarray
    n_escape: np.ndarray
    n_escape_class_c: np.ndarray
    q_rw: np.ndarray


def reworkability(delay_s, q0, tau_rw):
    delay_s = np.maximum(np.asarray(delay_s, dtype=float), 0.0)
    if np.any(np.asarray(tau_rw) <= 0):
        raise ValueError("tau_rw must be positive")
    if np.any((np.asarray(q0) < 0) | (np.asarray(q0) > 1)):
        raise ValueError("q0 must lie in [0, 1]")
    return q0 * np.exp(-delay_s / tau_rw)


def route_defects(total_defects, recall, delay_s, q0, tau_rw, class_shares: Tuple[float, float, float]) -> RoutingOutcome:
    f_a, f_b, f_c = class_shares
    if min(class_shares) < 0 or abs(f_a + f_b + f_c - 1.0) > 1e-9:
        raise ValueError(f"class shares must be non-negative and sum to 1, got {class_shares}")
    recall = np.asarray(recall, dtype=float)
    if np.any((recall < 0) | (recall > 1)):
        raise ValueError("recall must lie in [0, 1]")
    d = np.asarray(total_defects, dtype=float)
    q = reworkability(delay_s, q0, tau_rw)

    routed = d * recall
    missed = d - routed
    n_rework = routed * f_a * q
    n_scrap = routed - n_rework
    n_escape = missed
    out = RoutingOutcome(
        total_defects=d, routed=routed, n_rework=n_rework, n_scrap=n_scrap,
        n_escape=n_escape, n_escape_class_c=missed * f_c, q_rw=q * np.ones_like(d),
    )
    residual = np.max(np.abs(d - (n_rework + n_scrap + n_escape)))
    if residual > 1e-9 * max(1.0, float(np.max(d))):
        raise ArithmeticError(f"defect count conservation violated (residual {residual})")
    return out
