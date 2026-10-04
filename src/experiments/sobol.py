"""
Variance-based global sensitivity (Sobol indices).

PRCC assumes a monotone input-output relation and cannot see interactions; the accounting model
has multiplicative terms (prevalence x mass x material factor, recall x reworkability decay) and
a delay-dependent exponential, so we complement it with first-order (S1) and total-order (ST)
Sobol indices of Delta C. Inputs follow the same independent triangular distributions as the
Monte Carlo analysis, sampled by inverse CDF from a scrambled Sobol sequence (radix-2 matrices A,
B of 2^m rows each). Estimators: Saltelli et al. (2010) for S1 and Jansen (1999) for ST; 95 %
intervals from a seeded bootstrap over rows.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.experiments.monte_carlo import COMPARISONS
from src.models.pipeline_evaluator import compare, evaluate_all
from src.params import Scenario


def triangular_ppf(u: np.ndarray, a: float, c: float, b: float) -> np.ndarray:
    fc = (c - a) / (b - a)
    return np.where(u < fc, a + np.sqrt(u * (b - a) * (c - a)), b - np.sqrt((1.0 - u) * (b - a) * (b - c)))


def _delta_c(scenario: Scenario, params: Dict[str, np.ndarray], comparison: str) -> np.ndarray:
    base, policy = COMPARISONS[comparison]
    res = evaluate_all(params, scenario.class_shares)
    return np.asarray(compare(res[base], res[policy]).delta["C"], dtype=float)


def sobol_indices(scenario: Scenario, comparison: str = "B3_vs_L0", m: int = 12, seed: int = 42,
                  n_boot: int = 200) -> pd.DataFrame:
    keys: List[str] = scenario.uncertain_keys()
    k, n = len(keys), 2 ** m
    u = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random_base2(m)
    ua, ub = u[:, :k], u[:, k:]
    central = scenario.central()

    def to_params(uu: np.ndarray) -> Dict[str, np.ndarray]:
        p = {key: np.full(len(uu), v) for key, v in central.items()}
        for j, key in enumerate(keys):
            r = scenario.records[key]
            p[key] = triangular_ppf(uu[:, j], r.low, r.central, r.high)
        return p

    # one vectorised evaluation of A, B and every A_B^(i)
    blocks = [ua, ub] + [np.where(np.arange(k)[None, :] == i, ub, ua) for i in range(k)]
    y = _delta_c(scenario, to_params(np.vstack(blocks)), comparison).reshape(k + 2, n)
    fa, fb, fab = y[0], y[1], y[2:]

    def est(idx):
        a, b, ab = fa[idx], fb[idx], fab[:, idx]
        var = np.var(np.concatenate([a, b]))
        return (b * (ab - a)).mean(1) / var, 0.5 * ((a - ab) ** 2).mean(1) / var

    s1, st = est(np.arange(n))
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    b1 = np.array([b[0] for b in boot])
    bt = np.array([b[1] for b in boot])
    df = pd.DataFrame({"parameter": keys, "S1": s1, "S1_lo": np.percentile(b1, 2.5, 0), "S1_hi": np.percentile(b1, 97.5, 0),
                       "ST": st, "ST_lo": np.percentile(bt, 2.5, 0), "ST_hi": np.percentile(bt, 97.5, 0)})
    df.insert(0, "scenario", scenario.name)
    df["n_base"] = n
    return df.sort_values("ST", ascending=False).reset_index(drop=True)
