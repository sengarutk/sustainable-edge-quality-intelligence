"""
General decision rule and product-population study.

Every regime's carbon per functional unit is exactly affine in the defect count D = N pi
(the routing, material, energy and escape terms are linear in D; edge, hardware, false-alert
review and manual-inspection terms do not depend on it). Hence for any two regimes

    Delta C(pi) = N pi beta - F,        pi* = F / (N beta)

with beta the net carbon saved per defect and F the fixed carbon per functional unit.
For B3 against no inspection this is the rule "inspect when pi * beta > k", where
k = F / N is the edge cell's carbon per part (electricity, amortised hardware and
false-alert review) and beta = r [m EF_mat (1 + kappa f_C) + c_ret - h] with h the carbon of
handling an intercepted defect (scrap, recycling, rework, review).

`affine_coefficients` obtains (beta, F) exactly from two model evaluations, so the rule
holds for every parameter set without re-deriving the algebra; the population study
samples products log-uniformly over an envelope wider than the three scenarios.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.models.pipeline_evaluator import PRIMARY_TIER, compare, evaluate_all
from src.params import Scenario, load_scenario
from src.paths import SCENARIOS


def affine_coefficients(p: Dict, class_shares, base: str, policy: str = PRIMARY_TIER) -> Tuple[np.ndarray, np.ndarray]:
    """(beta [kgCO2e/defect], F [kgCO2e/FU]) with Delta C = D beta - F, exact by linearity."""
    n = p["functional_unit"]
    zero = {**p, "defect_prevalence": 0.0 * np.asarray(p["defect_prevalence"], dtype=float)}
    one = {**p, "defect_prevalence": 1.0 / n + zero["defect_prevalence"]}
    r0, r1 = evaluate_all(zero, class_shares), evaluate_all(one, class_shares)
    dc0 = compare(r0[base], r0[policy]).delta["C"]
    dc1 = compare(r1[base], r1[policy]).delta["C"]
    return dc1 - dc0, -dc0


def pi_star(p: Dict, class_shares, base: str, policy: str = PRIMARY_TIER):
    beta, f = affine_coefficients(p, class_shares, base, policy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta > 0, f / (p["functional_unit"] * beta), np.inf)


def cell_carbon_per_part(p: Dict, class_shares) -> np.ndarray:
    """k = F / N for B3 vs N0: carbon per part of the edge cell incl. false-alert review."""
    _, f = affine_coefficients(p, class_shares, "N0_NoInspection")
    return f / p["functional_unit"]


# Scenario-specific parameters are sampled log-uniformly over [0.5 x lowest, 2 x highest]
# across the three scenarios (fractions are clipped to valid ranges); global and policy
# parameters keep their registry bounds.
FRACTIONS = {"defect_prevalence", "material_recovery_fraction", "base_reworkability"}


def population_bounds() -> Dict[str, Tuple[float, float]]:
    scen = [load_scenario(s) for s in SCENARIOS]
    keys = {k for s in scen for k, r in s.records.items() if r.scope in SCENARIOS}
    out = {}
    for k in sorted(keys):
        lo = 0.5 * min(s.records[k].low for s in scen)
        hi = 2.0 * max(s.records[k].high for s in scen)
        if k in FRACTIONS:
            hi = min(hi, 0.99 if k != "defect_prevalence" else 0.2)
        out[k] = (lo, hi)
    return out


def sample_population(n: int = 20000, seed: int = 7, template: Scenario | None = None):
    """Random products: scenario-specific parameters log-uniform over the population envelope,
    global/policy parameters triangular on their registry bounds; class mix Dirichlet(2,2,2)."""
    from src.experiments.monte_carlo import sample_parameters

    template = template or load_scenario(SCENARIOS[1])
    rng = np.random.Generator(np.random.PCG64(seed))
    p = sample_parameters(template, n, seed)
    for k, (lo, hi) in population_bounds().items():
        p[k] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    shares = rng.dirichlet([2.0, 2.0, 2.0], n)
    return p, shares


def evaluate_population(n: int = 20000, seed: int = 7) -> Dict[str, np.ndarray]:
    """Delta C (B3 vs N0 and vs L0), decision ratio psi = pi beta / k, and the edge share of the
    benefit for each sampled product. Class shares vary per product, so the model is evaluated
    product by product in vectorised blocks of equal class mix."""
    p, shares = sample_population(n, seed)
    out = {k: np.empty(n) for k in ("dc_N0", "dc_L0", "psi", "beta", "k", "edge_share_of_benefit_L0", "embodied_per_part")}
    for i in range(n):
        pi_ = {k: (v[i] if np.ndim(v) else v) for k, v in p.items()}
        cs = tuple(shares[i])
        res = evaluate_all(pi_, cs)
        out["dc_N0"][i] = float(compare(res["N0_NoInspection"], res[PRIMARY_TIER]).delta["C"])
        out["dc_L0"][i] = float(compare(res["L0_Manual"], res[PRIMARY_TIER]).delta["C"])
        beta, f = affine_coefficients(pi_, cs, "N0_NoInspection")
        out["beta"][i], out["k"][i] = float(beta), float(f) / pi_["functional_unit"]
        out["psi"][i] = pi_["defect_prevalence"] * out["beta"][i] / out["k"][i]
        c = res[PRIMARY_TIER].carbon
        out["edge_share_of_benefit_L0"][i] = float(c.edge + c.hardware) / out["dc_L0"][i] if out["dc_L0"][i] > 0 else np.nan
        out["embodied_per_part"][i] = pi_["part_mass"] * pi_["material_carbon_factor"]
    out.update({k: np.asarray(p[k]) for k in ("defect_prevalence", "line_throughput", "part_mass", "material_carbon_factor")})
    return out
