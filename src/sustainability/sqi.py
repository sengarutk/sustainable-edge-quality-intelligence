"""
Multi-profile Sustainability Quality Index (SQI).

For each dimension x in {M (unrecovered material), E (electricity), C (carbon),
H (operator hours)} the bounded sub-indicator of policy p against baseline b is

    S_x = (x_b - x_p) / max(x_b, x_p)      in [-1, 1]  for x_b, x_p >= 0

(S_x = 0 when both are zero). SQI = sum_x w_x S_x with w >= 0, sum w = 1.
Because the index depends on the weights, `weight_robustness` reports the
share of the whole weight simplex (Dirichlet(1,1,1,1) samples) for which SQI > 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from src.paths import SQI_CONFIG

DIMENSIONS = ("M", "E", "C", "H")


def bounded_indicator(base, policy):
    # The [-1, 1] bound holds only for non-negative quantities (x_b=1, x_p=-1 would give 2),
    # so negative inputs are rejected rather than clipped into range.
    base = np.asarray(base, dtype=float)
    policy = np.asarray(policy, dtype=float)
    if np.any(base < 0) or np.any(policy < 0):
        raise ValueError("SQI dimensions must be non-negative")
    denom = np.maximum(base, policy)
    return np.where(denom > 0, (base - policy) / np.where(denom > 0, denom, 1.0), 0.0)


def load_profiles(path: Path = SQI_CONFIG) -> Dict[str, Dict[str, float]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = {}
    for name, spec in data["profiles"].items():
        if set(spec["weights"]) != {f"w_{k}" for k in DIMENSIONS}:
            raise ValueError(f"profile {name}: weights must be exactly w_M, w_E, w_C, w_H")
        w = {k: float(spec["weights"][f"w_{k}"]) for k in DIMENSIONS}
        if any(v < 0 for v in w.values()) or abs(sum(w.values()) - 1.0) > 1e-9:
            raise ValueError(f"profile {name}: weights must be non-negative and sum to 1")
        profiles[name] = w
    return profiles


@dataclass(frozen=True)
class SQIResult:
    sub: Dict[str, np.ndarray]
    scores: Dict[str, np.ndarray]


def evaluate_sqi(base: Dict[str, np.ndarray], policy: Dict[str, np.ndarray], profiles=None) -> SQIResult:
    profiles = profiles or load_profiles()
    sub = {k: bounded_indicator(base[k], policy[k]) for k in DIMENSIONS}
    scores = {name: sum(w[k] * sub[k] for k in DIMENSIONS) for name, w in profiles.items()}
    return SQIResult(sub=sub, scores=scores)


def weight_robustness(sub: Dict[str, float], n_samples: int = 20000, seed: int = 7) -> float:
    """Fraction of uniformly sampled weight vectors on the simplex giving SQI > 0."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(DIMENSIONS)), size=n_samples)
    s = np.array([float(sub[k]) for k in DIMENSIONS])
    return float(np.mean(w @ s > 0))
