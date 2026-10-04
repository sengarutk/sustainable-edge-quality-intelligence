"""Monte Carlo, PRCC, break-even, tornado and alert-policy behaviour."""

import numpy as np
import pytest

from src.experiments.break_even import SPECS, _solve, solve_frontiers
from src.experiments.energy_benchmark import DualEMAPolicy
from src.experiments.monte_carlo import prcc, run_monte_carlo, sample_parameters
from src.experiments.sensitivity import run_tornado
from src.models.pipeline_evaluator import compare, delta_carbon, evaluate_all


def test_sampling_respects_registry_bounds(scenario):
    draws = sample_parameters(scenario, 5000, seed=1)
    for k, r in scenario.records.items():
        assert draws[k].min() >= r.low - 1e-12 and draws[k].max() <= r.high + 1e-12


def test_monte_carlo_deterministic(scenario):
    a, da, pa = run_monte_carlo(scenario, n_draws=500, seed=42)
    b, db, pb = run_monte_carlo(scenario, n_draws=500, seed=42)
    c, _, _ = run_monte_carlo(scenario, n_draws=500, seed=43)
    assert a == b and da.equals(db) and pa.equals(pb)
    assert a != c


def test_prcc_recovers_known_structure(rng):
    n = 4000
    x = {k: rng.uniform(0, 1, n) for k in ("a", "b", "c")}
    y = 3 * x["a"] - np.exp(x["b"]) + 0.01 * rng.normal(size=n)
    df = prcc(x, y, ["a", "b", "c"]).set_index("parameter")
    assert df.loc["a", "prcc"] > 0.9
    assert df.loc["b", "prcc"] < -0.9
    assert abs(df.loc["c", "prcc"]) < 0.1


def test_break_even_roots_are_zeros(scenario):
    fr = solve_frontiers(scenario)
    assert set(fr) == {s[0] for s in SPECS}
    for (name, key, base, _, _) in SPECS:
        f = fr[name]
        assert f["status"] in ("root", "none_positive", "none_negative")
        if f["status"] == "root":
            dc = delta_carbon({**scenario.central(), key: f["value"]}, scenario.class_shares, base, "B3_Full_Policy")
            assert abs(dc) < 1e-6
        else:
            assert f["value"] is None


def test_break_even_delta_c_monotone_on_interval(scenario):
    # _solve only inspects the interval ends; that is sound only if Delta C is monotone there.
    central = scenario.central()
    for (_, key, base, (lo, hi), _) in SPECS:
        res = evaluate_all({**central, key: np.linspace(lo, hi, 201)}, scenario.class_shares)
        dc = compare(res[base], res["B3_Full_Policy"]).delta["C"]
        steps = np.diff(dc)
        scale = max(1.0, float(np.max(np.abs(dc))))
        assert (steps >= -1e-9 * scale).all() or (steps <= 1e-9 * scale).all(), key


@pytest.mark.parametrize("f, expected", [
    (lambda x: x - 1.0, (1.0, "root")),        # root exactly at the upper bound
    (lambda x: x, (0.0, "root")),              # root exactly at the lower bound
    (lambda x: x + 1.0, (None, "none_positive")),
    (lambda x: -x - 1.0, (None, "none_negative")),
])
def test_solve_interval_edges(f, expected):
    assert _solve(f, 0.0, 1.0) == expected


def test_solve_interior_root():
    value, status = _solve(lambda x: x - 0.25, 0.0, 1.0)
    assert status == "root" and value == pytest.approx(0.25)


def test_tornado_uses_registry_bounds(scenario):
    df = run_tornado(scenario)
    for _, row in df.iterrows():
        r = scenario.records[row.parameter]
        assert (row.low, row.high) == (r.low, r.high)
    assert (df.swing.diff().dropna() <= 1e-12).all()


def test_alert_policy_persistence_and_cooldown():
    policy = DualEMAPolicy(k=3, n=5, cooldown=4)
    alerts = [policy.update(1.0, threshold=0.5) for _ in range(12)]
    # a rising score passes the EMA gate every frame: alert on the 3rd hit, then every (cooldown + 1) frames
    assert [i for i, a in enumerate(alerts) if a] == [2, 7]
    quiet = DualEMAPolicy()
    assert not any(quiet.update(0.1, threshold=0.5) for _ in range(50))
