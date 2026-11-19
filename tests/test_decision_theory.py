import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, special, stats

from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.decision_theory import (
    AlertCosts,
    bayes_threshold,
    expected_losses,
    expected_unavailability,
    interval_for_expected_unavailability,
    interval_for_known_rate,
    interval_for_risk,
    load_decision_parameters,
    unavailability,
    value_of_waiting,
)

COSTS = AlertCosts(false_alarm=1.0, missed=9.0, delay=0.5)


def test_parameters_table_has_every_value_with_a_basis() -> None:
    params = load_decision_parameters()
    assert params["cost_missed_degradation"] == 4.0
    assert 0 < params["unavailability_target"] < 1


def test_bayes_threshold_is_the_cost_ratio_and_minimises_expected_loss() -> None:
    # Brute force: on a fine grid of p the action with the lower expected loss switches
    # exactly at c_FA / (c_FA + c_miss).
    p_star = bayes_threshold(COSTS)
    assert p_star == pytest.approx(0.1)
    grid = np.linspace(0, 1, 100_001)
    investigate = np.array(
        [expected_losses(p, COSTS)[0] < expected_losses(p, COSTS)[1] for p in grid]
    )
    switch = grid[np.argmax(investigate)]
    assert switch == pytest.approx(p_star, abs=2e-5)
    with pytest.raises(ValueError):
        AlertCosts(0.0, 1.0)


def test_expected_losses_match_monte_carlo_over_the_posterior() -> None:
    post, target = GammaRate(3.0, 10.0), 0.375
    rng = np.random.default_rng(1)
    lam = rng.gamma(3.0, 1 / 10.0, 400_000)
    above = lam > target
    mc_investigate = np.mean(np.where(above, 0.0, COSTS.false_alarm))
    mc_wait = np.mean(np.where(above, COSTS.missed, 0.0))
    exact = expected_losses(post.sf(target), COSTS)
    assert exact[0] == pytest.approx(mc_investigate, abs=5 * np.sqrt(0.25 / lam.size))
    assert exact[1] == pytest.approx(mc_wait, abs=5 * 9 * np.sqrt(0.25 / lam.size))


def test_predictive_exceedance_is_a_martingale() -> None:
    # Averaging next month's updated exceedance probability over the predictive count gives
    # today's probability (law of total probability): the basis of the EVSI computation.
    post, target, s = GammaRate(2.5, 10.0), 0.375, 0.875
    pred = post.predictive(875.0)
    ys = np.arange(200)
    p_next = special.gammaincc(post.alpha + ys, target * (post.beta + s))
    total = np.sum(stats.nbinom.pmf(ys, pred.n, pred.p) * p_next)
    assert total == pytest.approx(post.sf(target), rel=1e-12)


def test_evsi_matches_monte_carlo_preposterior_analysis() -> None:
    post, target = GammaRate(2.5, 10.0), 0.375
    v = value_of_waiting(post, target, 875.0, COSTS)
    rng = np.random.default_rng(2)
    lam = rng.gamma(2.5, 1 / 10.0, 200_000)
    y = rng.poisson(lam * 0.875)
    p_next = special.gammaincc(2.5 + y, target * 10.875)
    losses = np.minimum(COSTS.false_alarm * (1 - p_next), COSTS.missed * p_next)
    assert v.loss_after == pytest.approx(losses.mean(), abs=5 * losses.std() / np.sqrt(y.size))
    assert v.delay_cost == pytest.approx(0.5 * post.sf(target))
    assert v.wait_is_better == (v.evsi > v.delay_cost)


@settings(max_examples=60, deadline=None)
@given(st.floats(0.5, 40.0), st.floats(0.5, 200.0), st.floats(50.0, 5000.0), st.floats(0.05, 2.0))
def test_evsi_is_never_negative(alpha: float, beta: float, month_fh: float, target: float) -> None:
    v = value_of_waiting(GammaRate(alpha, beta), target, month_fh, COSTS)
    assert v.evsi >= 0.0
    assert v.loss_after <= v.loss_now + 1e-12


def test_unavailability_closed_form_and_small_x_limit() -> None:
    # U = 1 − (1 − e^(−x))/x; for small x it is x/2 (the classical FFI formula).
    assert unavailability(0.02, 1000.0) == pytest.approx(
        1 - (1 - math.exp(-0.02)) / 0.02, rel=1e-12
    )
    assert unavailability(1e-6, 1.0) == pytest.approx(0.5e-9, rel=1e-6)
    tau = interval_for_known_rate(0.02, 0.01)
    assert unavailability(0.02, tau) == pytest.approx(0.01, rel=1e-9)
    # τ ≈ 2U/λ = 1000 FH; the exact answer is slightly longer since U < x/2.
    assert tau == pytest.approx(1000.0, rel=0.02) and tau > 1000.0
    with pytest.raises(ValueError):
        interval_for_known_rate(0.02, 1.5)


def test_expected_unavailability_matches_numerical_integration() -> None:
    post = GammaRate(3.2, 40.0)
    for tau in (200.0, 1000.0, 5000.0):
        num, _ = integrate.quad(
            lambda lam, tau=tau: unavailability(lam, tau) * post.pdf(lam), 0, np.inf, limit=200
        )
        assert expected_unavailability(post, tau) == pytest.approx(num, rel=1e-8)
    with pytest.raises(ValueError):
        expected_unavailability(GammaRate(0.9, 1.0), 100.0)


def test_interval_rules_are_ordered() -> None:
    # Jensen: U is concave in λ, so E[U(λ)] ≤ U(E[λ]) and the expected-unavailability interval
    # is at least the plug-in one; the risk-based (90 %) interval is shorter than both.
    post = GammaRate(4.0, 200.0)  # mean 0.02 per 1000 FH
    target = 0.01
    plug_in = interval_for_known_rate(post.mean(), target)
    expected = interval_for_expected_unavailability(post, target)
    risk = interval_for_risk(post, target, 0.9)
    assert expected_unavailability(post, expected) == pytest.approx(target, rel=1e-8)
    assert expected >= plug_in
    assert risk < plug_in
    assert risk == pytest.approx(interval_for_known_rate(post.ppf(0.9), target))
    with pytest.raises(ValueError):
        interval_for_risk(post, target, 1.0)
