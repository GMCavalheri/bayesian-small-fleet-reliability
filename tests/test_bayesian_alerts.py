import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from smallfleet.bayesian_alert_thresholds import (
    alert_count_threshold,
    bayes_window_rule,
    characteristics,
    classical_rule,
    discounted_exceedance,
    exceedance_probability,
    simulate_histories,
    window_exceedance,
    window_false_alarm_rate,
)
from smallfleet.bayesian_point_estimation import GammaRate

PRIOR = GammaRate(2.4, 10.0)  # the discounted large-fleet prior of Lesson 2


def test_exceedance_is_the_gamma_survival_function() -> None:
    p = exceedance_probability(PRIOR, 3, 2100.0, 0.375)
    assert p == pytest.approx(stats.gamma.sf(0.375, 5.4, scale=1 / 12.1), rel=1e-12)


def test_count_threshold_brackets_p_star() -> None:
    for p_star in (0.5, 0.8, 0.9, 0.99):
        c = alert_count_threshold(PRIOR, 6 * 175.0, 0.375, p_star)
        assert exceedance_probability(PRIOR, c, 6 * 175.0, 0.375) > p_star
        if c > 0:
            assert exceedance_probability(PRIOR, c - 1, 6 * 175.0, 0.375) <= p_star
    with pytest.raises(ValueError):
        alert_count_threshold(PRIOR, 1000.0, 0.375, 1.5)
    with pytest.raises(ValueError):
        alert_count_threshold(PRIOR, 1000.0, 0.375, 0.9, max_count=0)


def test_window_and_discounting_reduce_to_the_cumulative_update() -> None:
    rng = np.random.default_rng(1)
    counts = rng.poisson(0.5, 24)
    exposure = np.full(24, 175.0)
    cum = np.array(
        [PRIOR.update(int(counts[: j + 1].sum()), 175.0 * (j + 1)).sf(0.375) for j in range(24)]
    )
    np.testing.assert_allclose(
        window_exceedance(PRIOR, counts, exposure, 0.375, 24), cum, rtol=1e-12
    )
    np.testing.assert_allclose(
        discounted_exceedance(PRIOR, counts, exposure, 0.375, 1.0), cum, rtol=1e-12
    )
    w3 = window_exceedance(PRIOR, counts, exposure, 0.375, 3)
    assert w3[10] == pytest.approx(
        PRIOR.update(int(counts[8:11].sum()), 525.0).sf(0.375), rel=1e-12
    )
    with pytest.raises(ValueError):
        discounted_exceedance(PRIOR, counts, exposure, 0.375, 0.0)


def test_window_false_alarm_rate_matches_monte_carlo() -> None:
    # Exact Poisson tail against the same rule applied to seeded simulated histories, full
    # windows only (months >= window). 5 standard errors allowed.
    rate, month_fh, window, target, p_star = 0.25, 25 * 35.0, 6, 0.375, 0.8
    exact = window_false_alarm_rate(PRIOR, rate, month_fh, window, target, p_star)
    hist = simulate_histories(np.random.default_rng(2), rate, month_fh, 12, 24, 1.0, 4000)
    rule = bayes_window_rule(PRIOR, target, p_star, window)
    flags = np.array([rule(h, hist.exposure_fh)[window:] for h in hist.steady])
    se = np.sqrt(exact * (1 - exact) / flags.size)
    assert abs(flags.mean() - exact) < 5 * se + 1e-12
    assert 0 < exact < 1


@given(st.integers(0, 50), st.floats(100.0, 1e5), st.floats(0.05, 2.0))
def test_exceedance_monotone(count: int, exposure: float, target: float) -> None:
    p = exceedance_probability(PRIOR, count, exposure, target)
    assert exceedance_probability(PRIOR, count + 1, exposure, target) >= p - 1e-15
    assert exceedance_probability(PRIOR, count, exposure * 1.5, target) <= p + 1e-15
    assert exceedance_probability(PRIOR, count, exposure, target * 1.2) <= p + 1e-15


def test_characteristics_of_trivial_rules() -> None:
    hist = simulate_histories(np.random.default_rng(3), 0.25, 175.0, 12, 12, 2.0, 200)
    always = characteristics(hist, lambda c, e: np.ones(c.size, dtype=bool))
    assert always.false_alarm_rate == 1.0 and always.detection == 1.0 and always.median_delay == 1.0
    never = characteristics(hist, lambda c, e: np.zeros(c.size, dtype=bool))
    assert never.false_alarm_rate == 0.0 and never.detection == 0.0 and np.isnan(never.median_delay)


def test_classical_rule_flags_only_after_the_baseline() -> None:
    counts = np.array([0] * 11 + [1] + [0, 0, 3, 0])
    exposure = np.full(counts.size, 175.0)
    flags = classical_rule(2.0)(counts, exposure)
    assert not flags[:12].any()
    np.testing.assert_array_equal(flags[12:], [False, False, True, False])
