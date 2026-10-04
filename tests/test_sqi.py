import pytest

from src.sustainability.sqi import DIMENSIONS, bounded_indicator, evaluate_sqi, load_profiles, weight_robustness


def test_profiles_are_convex_weights():
    for w in load_profiles().values():
        assert set(w) == set(DIMENSIONS)
        assert sum(w.values()) == pytest.approx(1.0)
        assert min(w.values()) >= 0


def test_malformed_profile_rejected_with_message(tmp_path):
    cfg = tmp_path / "sqi.yaml"
    cfg.write_text("profiles:\n  bad:\n    weights: {w_M: 0.5, w_E: 0.5, w_C: 0.0}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly w_M"):
        load_profiles(cfg)


def test_bounded_indicator_range_and_identity(rng):
    b, q = rng.uniform(0, 10, 10000), rng.uniform(0, 10, 10000)
    b[:100] = 0.0  # include exact zeros on either side
    q[100:200] = 0.0
    s = bounded_indicator(b, q)
    assert (s >= -1).all() and (s <= 1).all()
    with pytest.raises(ValueError, match="non-negative"):
        bounded_indicator(1.0, -1.0)
    assert float(bounded_indicator(0.0, 0.0)) == 0.0
    assert float(bounded_indicator(5.0, 5.0)) == 0.0
    assert float(bounded_indicator(4.0, 0.0)) == 1.0


def test_sqi_monotone_in_improvement():
    base = {k: 10.0 for k in DIMENSIONS}
    lo = evaluate_sqi(base, {k: 9.0 for k in DIMENSIONS})
    hi = evaluate_sqi(base, {k: 5.0 for k in DIMENSIONS})
    for prof in lo.scores:
        assert hi.scores[prof] > lo.scores[prof]


def test_weight_robustness():
    assert weight_robustness({k: 0.5 for k in DIMENSIONS}) == 1.0
    assert weight_robustness({k: -0.5 for k in DIMENSIONS}) == 0.0
    mixed = weight_robustness({"M": 0.5, "E": -0.5, "C": 0.5, "H": -0.5})
    assert 0.4 < mixed < 0.6
