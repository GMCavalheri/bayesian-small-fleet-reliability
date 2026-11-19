import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from smallfleet.frequentist_baseline import (
    alert_flags,
    degenerate_baseline_probability,
    expected_relative_width,
    garwood_coverage,
    garwood_interval,
    mean_ratio_alert_level,
    mean_sd_alert_level,
    mean_sd_operating_characteristics,
    mle_rate,
    monthly_rates,
    poisson_alert_count,
    running_average,
    zero_month_probability,
)


def test_mle_is_count_over_exposure() -> None:
    assert mle_rate(3, 6000.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mle_rate(1, 0.0)


def test_garwood_closed_forms_for_zero_and_one_event() -> None:
    # χ² with 2 degrees of freedom is an exponential with mean 2, so for T = 1000 FH:
    # U(y=0) = −ln(α/2) and L(y=1) = −ln(1 − α/2) exactly.
    for conf in (0.80, 0.90, 0.95):
        a = 1 - conf
        lo0, hi0 = garwood_interval(0, 1000.0, conf)
        assert lo0 == 0.0
        assert hi0 == pytest.approx(-math.log(a / 2), rel=1e-12)
        lo1, _ = garwood_interval(1, 1000.0, conf)
        assert lo1 == pytest.approx(-math.log(1 - a / 2), rel=1e-12)
    with pytest.raises(ValueError):
        garwood_interval(1, 1000.0, 1.0)


def test_garwood_limits_invert_the_poisson_tails() -> None:
    # Definition of the exact interval: P(Y ≥ y | L) = α/2 and P(Y ≤ y | U) = α/2.
    y, t, conf = 7, 4000.0, 0.90
    lo, hi = garwood_interval(y, t, conf)
    assert stats.poisson.sf(y - 1, lo * t / 1000) == pytest.approx(0.05, rel=1e-9)
    assert stats.poisson.cdf(y, hi * t / 1000) == pytest.approx(0.05, rel=1e-9)


@given(st.floats(0.01, 5.0), st.floats(100.0, 50_000.0))
def test_garwood_coverage_is_never_below_nominal(rate: float, exposure: float) -> None:
    # The exact interval is conservative: coverage ≥ confidence for every true rate
    # (Garwood 1936). The coverage is itself computed exactly by summing the Poisson pmf.
    assert garwood_coverage(rate, exposure, 0.90) >= 0.90 - 1e-12


def test_relative_width_converges_to_the_normal_limit() -> None:
    # For large counts the interval half-width tends to z·sqrt(y)/T, so the relative width
    # tends to 2·z/sqrt(m). The exact interval is skewed, so the error is O(m^(-1/2)):
    # quadrupling m must halve the relative error (to within 10 %), and at m = 10 000 it is
    # below 0.5 %.
    z = stats.norm.ppf(0.95)

    def error(m: float) -> float:
        return expected_relative_width(0.2, m * 5000.0, 0.90) / (2 * z / math.sqrt(m)) - 1

    assert 0 < error(10_000) < 5e-3
    assert error(2_500) / error(10_000) == pytest.approx(2.0, rel=0.1)
    assert expected_relative_width(0.2, 5_000.0) > expected_relative_width(0.2, 50_000.0)


def test_monthly_rates_and_running_average() -> None:
    rates = monthly_rates(np.array([0, 1, 2]), np.array([1000.0, 1000.0, 500.0]))
    np.testing.assert_allclose(rates, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(running_average(rates, 3), [5.0 / 3])
    with pytest.raises(ValueError):
        running_average(rates, 4)


def test_mean_sd_alert_level_arithmetic_of_ama_571_101_example_2() -> None:
    # Transport Canada AMA 571.101/1 Appendix A, example 2: with mean 20.16666667 and
    # population sd 2.763853995 the 2-sigma alert level is 25.69437466. The 12 data points
    # are in a figure the page does not reproduce, so we build a baseline with the same
    # mean and population sd and check the arithmetic.
    mean, sd = 20.16666667, 2.763853995
    base = np.array([mean - sd, mean + sd] * 6)
    assert mean_sd_alert_level(base, 2.0) == pytest.approx(25.69437466, rel=1e-9)
    assert mean_ratio_alert_level(np.full(12, 20.0), 1.3) == pytest.approx(26.0)
    with pytest.raises(ValueError):
        mean_sd_alert_level(np.array([1.0]))


def test_poisson_alert_count_reproduces_ama_571_101_example_4a() -> None:
    # Transport Canada AMA 571.101/1 Appendix A, example 4(a), autopilot pitch amplifier:
    # 62 removals in 36 840 h, 5895 h in the current period: expected 9.9 (rounded to 10 on
    # the page), and the 95 % chart gives A = 15. Example 4(b) reads A = 8 for an expected
    # 5, but P(Y ≤ 8 | 5) = 0.932 < 0.95, so the chart reading there is not reproducible;
    # Lesson 1 discusses it.
    expected = 62 / 36_840 * 5_895
    assert expected == pytest.approx(9.92, abs=0.01)
    assert poisson_alert_count(expected, 0.95) == 15
    assert poisson_alert_count(10.0, 0.95) == 15
    assert poisson_alert_count(5.0, 0.95) == 9
    with pytest.raises(ValueError):
        poisson_alert_count(-1.0)


def test_alert_flags_single_and_consecutive() -> None:
    rates = np.array([1.0, 3.0, 3.0, 1.0, 3.0])
    np.testing.assert_array_equal(alert_flags(rates, 2.0), [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(alert_flags(rates, 2.0, 2), [0, 0, 1, 0, 0])
    with pytest.raises(ValueError):
        alert_flags(rates, 2.0, 0)


def test_zero_and_degenerate_probabilities_match_monte_carlo() -> None:
    # Closed forms exp(−m) and Σ_k pmf(k)^12 against seeded simulation (5 standard errors).
    rate, exposure = 0.2, 175.0  # 5 business jets: m = 0.035 events per month
    rng = np.random.default_rng(11)
    sims = rng.poisson(rate * exposure / 1000, (100_000, 12))
    p0 = zero_month_probability(rate, exposure)
    assert p0 == pytest.approx(math.exp(-0.035))
    share0 = float(np.mean(sims == 0))
    assert abs(share0 - p0) < 5 * math.sqrt(p0 * (1 - p0) / sims.size)
    pd = degenerate_baseline_probability(rate, exposure)
    share_d = float(np.mean(np.all(sims == sims[:, :1], axis=1)))
    assert abs(share_d - pd) < 5 * math.sqrt(pd * (1 - pd) / sims.shape[0])
    # For tiny m the baseline is all zeros: Σ pmf^12 → exp(−12 m).
    assert pd == pytest.approx(math.exp(-12 * 0.035), rel=1e-3)


def test_operating_characteristics_match_closed_form_when_degenerate() -> None:
    # With m → small every baseline is all zeros, the level is 0 and a monitored month
    # alerts whenever it logs an event: the false-alarm rate tends to 1 − exp(−m). A
    # baseline holding a single event gives a level of 0.64 of that event's rate, so a
    # monitored event alerts there too; the closed form holds for all but two-event baselines.
    rate, exposure = 0.05, 20.0  # m = 0.001
    oc = mean_sd_operating_characteristics(np.random.default_rng(5), rate, exposure)
    p_event = 1 - math.exp(-0.001)
    assert oc.degenerate_share > 0.98
    assert abs(oc.false_alarm_rate - p_event) < 5 * oc.false_alarm_se
    assert 0 <= oc.detection_probability <= 1
