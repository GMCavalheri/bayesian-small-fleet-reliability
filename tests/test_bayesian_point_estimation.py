import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import integrate, stats

from smallfleet.bayesian_point_estimation import (
    GammaRate,
    constrained_noninformative,
    gamma_from_mean_cv,
    inflate_variance,
    jeffreys,
    power_prior,
    prior_predictive_pvalue,
)
from smallfleet.frequentist_baseline import garwood_interval

# NUREG/CR-6823 (Atwood et al. 2003), Example 6.1 and Table 6.4: 1 event in 4.89
# reactor-critical-years. Our rate unit is "per 1000 FH", so an exposure of 4.89 units is
# 4890 FH; the arithmetic is unit-free. The table prints two significant figures, so the
# tolerance is half a unit in the last printed digit.
X, T_UNITS = 1, 4.89
T_FH = T_UNITS * 1000


def close(value: float, printed: float, decimals: int) -> bool:
    return abs(value - printed) <= 0.5 * 10.0**-decimals + 1e-12


def test_nureg_6823_table_6_4_jeffreys_row() -> None:
    post = jeffreys().update(X, T_FH)
    assert (post.alpha, post.beta) == (1.5, pytest.approx(4.89))
    assert close(post.mean(), 0.31, 2)
    lo, hi = post.interval(0.90)
    assert close(lo, 0.036, 3) and close(hi, 0.80, 2)


def test_nureg_6823_table_6_4_industry_prior_row() -> None:
    post = GammaRate(1.53, 10.63).update(X, T_FH)
    assert post.alpha == pytest.approx(2.53) and post.beta == pytest.approx(15.52)
    assert close(post.mean(), 0.16, 2)
    lo, hi = post.interval(0.90)
    assert close(lo, 0.038, 3) and close(hi, 0.36, 2)
    # Section 6.2.2.5.1 also prints the mean 0.163, variance 0.0105 and sd 0.102, and the
    # 5th/95th percentiles 0.5867 and 5.5817 of a Gamma(2.53, 1).
    assert close(post.mean(), 0.163, 3)
    assert close(post.var(), 0.0105, 4)
    assert close(post.sd(), 0.102, 3)
    assert close(GammaRate(2.53, 1.0).ppf(0.05), 0.5867, 4)
    assert close(GammaRate(2.53, 1.0).ppf(0.95), 5.5817, 4)


def test_nureg_6823_table_6_4_constrained_noninformative_row() -> None:
    prior = constrained_noninformative(0.144)
    assert close(prior.beta, 3.47, 2)
    post = prior.update(X, T_FH)
    assert close(post.beta, 8.36, 2)
    assert close(post.mean(), 0.18, 2)
    lo, hi = post.interval(0.90)
    assert close(lo, 0.021, 3) and close(hi, 0.47, 2)


def test_nureg_6823_table_6_4_frequentist_row() -> None:
    lo, hi = garwood_interval(X, T_FH, 0.90)
    assert close(lo, 0.010, 3) and close(hi, 0.97, 2)


def test_posterior_equals_normalised_likelihood_times_prior() -> None:
    # Independent oracle: numerical integration of Poisson likelihood × Gamma prior.
    prior, y, t_fh = GammaRate(2.0, 8.0), 3, 6000.0
    t = t_fh / 1000

    def unnorm(lam: float) -> float:
        return float(stats.poisson.pmf(y, lam * t) * prior.pdf(lam))

    z, _ = integrate.quad(unnorm, 0, np.inf)
    mean, _ = integrate.quad(lambda lam: lam * unnorm(lam) / z, 0, np.inf)
    cdf_at, _ = integrate.quad(lambda lam: unnorm(lam) / z, 0, 0.4)
    post = prior.update(y, t_fh)
    assert post.mean() == pytest.approx(mean, rel=1e-8)
    assert post.cdf(0.4) == pytest.approx(cdf_at, rel=1e-8)


def test_predictive_is_the_gamma_poisson_mixture() -> None:
    # Negative binomial against numerical integration and against seeded Monte Carlo.
    prior, s_fh = GammaRate(1.5, 5.0), 3000.0
    pred = prior.predictive(s_fh)
    for y in range(6):
        mix, _ = integrate.quad(
            lambda lam, y=y: stats.poisson.pmf(y, lam * 3.0) * prior.pdf(lam), 0, np.inf
        )
        assert pred.pmf(y) == pytest.approx(mix, rel=1e-8)
    rng = np.random.default_rng(2)
    lam = rng.gamma(1.5, 1 / 5.0, 400_000)
    ys = rng.poisson(lam * 3.0)
    assert ys.mean() == pytest.approx(pred.mean(), abs=5 * math.sqrt(pred.var() / ys.size))
    assert pred.cdf(2) + pred.sf(2) == pytest.approx(1.0)


def test_hpd_is_shortest_and_has_equal_density_at_its_ends() -> None:
    post = GammaRate(3.0, 10.0)
    lo, hi = post.hpd(0.90)
    assert post.cdf(hi) - post.cdf(lo) == pytest.approx(0.90, abs=1e-9)
    assert post.pdf(lo) == pytest.approx(post.pdf(hi), rel=1e-5)
    et_lo, et_hi = post.interval(0.90)
    assert hi - lo < et_hi - et_lo
    # alpha <= 1: decreasing density, the HPD interval starts at zero
    assert GammaRate(0.8, 2.0).hpd(0.9) == (0.0, pytest.approx(GammaRate(0.8, 2.0).ppf(0.9)))


def test_improper_prior_has_no_moments_and_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        jeffreys().mean()
    with pytest.raises(ValueError):
        GammaRate(0.0, 1.0)
    with pytest.raises(ValueError):
        jeffreys().update(-1, 10.0)
    with pytest.raises(ValueError):
        constrained_noninformative(0.0)
    with pytest.raises(ValueError):
        gamma_from_mean_cv(0.1, 0.0)


def test_moment_matching_and_variance_inflation() -> None:
    g = gamma_from_mean_cv(0.2, 0.5)
    assert g.mean() == pytest.approx(0.2) and g.sd() / g.mean() == pytest.approx(0.5)
    wide = inflate_variance(g, 4.0)
    assert wide.mean() == pytest.approx(g.mean())
    assert wide.var() == pytest.approx(4 * g.var())
    with pytest.raises(ValueError):
        inflate_variance(g, 0.5)


def test_power_prior_limits() -> None:
    # a0 = 0 is the Jeffreys prior; a0 = 1 is the full large-fleet Jeffreys posterior.
    assert power_prior(90, 450_000.0, 0.0) == jeffreys()
    assert power_prior(90, 450_000.0, 450_000.0) == jeffreys().update(90, 450_000.0)
    half = power_prior(90, 450_000.0, 225_000.0)
    assert half.alpha == pytest.approx(0.5 + 45) and half.beta == pytest.approx(225.0)
    with pytest.raises(ValueError):
        power_prior(90, 450_000.0, 500_000.0)


def test_prior_predictive_pvalue_tails() -> None:
    prior = GammaRate(20.0, 100.0)  # mean 0.2 per 1000 FH, fairly sharp
    pred = prior.predictive(10_000.0)  # expect 2 events
    assert prior_predictive_pvalue(prior, 2, 10_000.0) == 1.0
    assert prior_predictive_pvalue(prior, 9, 10_000.0) == pytest.approx(2 * pred.sf(8))
    assert prior_predictive_pvalue(prior, 0, 10_000.0) == pytest.approx(2 * pred.cdf(0))


@given(
    st.floats(0.1, 50.0),
    st.floats(0.1, 500.0),
    st.integers(0, 200),
    st.floats(10.0, 1e6),
)
def test_posterior_mean_lies_between_prior_mean_and_mle(
    alpha: float, beta: float, y: int, t_fh: float
) -> None:
    # NUREG/CR-6823 §6.2.2.5.4: "the posterior mean is between the prior mean and the MLE".
    # It is the weighted average beta/(beta+t)·prior + t/(beta+t)·MLE.
    prior = GammaRate(alpha, beta)
    post = prior.update(y, t_fh)
    mle = 1000 * y / t_fh
    lo, hi = sorted((prior.mean(), mle))
    assert lo - 1e-12 <= post.mean() <= hi + 1e-12
    t = t_fh / 1000
    w = beta / (beta + t)
    assert post.mean() == pytest.approx(w * prior.mean() + (1 - w) * mle, rel=1e-10)


def test_exact_performance_of_the_mle_matches_closed_forms() -> None:
    # MLE y/t: unbiased, variance λ/t (per 1000 FH units), so RMSE = sqrt(λ/t). Coverage
    # must equal the exact Garwood coverage of the frequentist baseline.
    from smallfleet.bayesian_point_estimation import exact_performance, frequentist_estimator
    from smallfleet.frequentist_baseline import garwood_coverage

    rate, t_fh = 0.25, 2100.0
    perf = exact_performance(frequentist_estimator(0.90), rate, t_fh)
    assert perf.bias == pytest.approx(0.0, abs=1e-12)
    assert perf.rmse == pytest.approx(math.sqrt(rate / 2.1), rel=1e-10)
    assert perf.coverage == pytest.approx(garwood_coverage(rate, t_fh, 0.90), abs=1e-12)


def test_exact_performance_of_a_bayes_estimator_matches_monte_carlo() -> None:
    # Posterior-mean bias has the closed form w·(prior mean − λ), w = β/(β + t); coverage
    # is checked against seeded simulation (5 standard errors).
    from smallfleet.bayesian_point_estimation import bayes_estimator, exact_performance

    prior, rate, t_fh = GammaRate(4.0, 20.0), 0.3, 2100.0
    est = bayes_estimator(prior, 0.90)
    perf = exact_performance(est, rate, t_fh)
    w = 20.0 / (20.0 + 2.1)
    assert perf.bias == pytest.approx(w * (0.2 - rate), rel=1e-9)
    rng = np.random.default_rng(4)
    ys = rng.poisson(rate * 2.1, 20_000)
    hits = np.mean([lo <= rate <= hi for _, lo, hi in (est(int(y), t_fh) for y in ys)])
    assert abs(hits - perf.coverage) < 5 * math.sqrt(perf.coverage * (1 - perf.coverage) / 20_000)
