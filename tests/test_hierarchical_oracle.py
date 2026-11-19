import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

from smallfleet.hierarchical_model import (
    AlphaBetaHyperprior,
    MeanCVHyperprior,
    exact_posterior,
)

from nureg import example_8_1

NAMES, X, T = example_8_1()

# NUREG/CR-6823 §8.3.4.2 and Table 8.7: WinBUGS, alpha ~ Exp(1), beta ~ Gamma(0.0625,
# 0.0625), 10 000 Gibbs draws after 1000 burn-in. Those are Monte Carlo estimates printed to
# three figures; with alpha and beta correlated at 0.89 the effective sample size is well
# below 10 000, and a tail percentile's Monte Carlo error is then a few percent. We require
# agreement within 2.5 % (relative), which every printed value meets.
REL = 0.025
TABLE_8_7 = {
    "Callaway": (4.97, 2.87, 7.51),
    "Wash. Nucl. 2": (4.39, 3.01, 5.99),
    "Salem 1": (2.94, 1.65, 4.57),
    "Diablo Canyon 1": (2.83, 1.24, 4.99),
    "Grand Gulf": (2.57, 1.27, 4.21),
    "Pt. Beach 2": (0.156, 0.013, 0.429),
    "Prairie Island 2": (0.152, 0.012, 0.410),
}


@pytest.fixture(scope="module")
def nureg_post():
    return exact_posterior(X, T, AlphaBetaHyperprior())


def test_hyperparameters_match_nureg_6823_hierarchical_bayes(nureg_post) -> None:
    m = nureg_post.hyper_moments()
    assert m["alpha_mean"] == pytest.approx(1.38, rel=REL)
    assert m["alpha_sd"] == pytest.approx(0.30, rel=0.05)  # printed to two figures
    assert m["beta_mean"] == pytest.approx(1.21, rel=REL)
    assert m["beta_sd"] == pytest.approx(0.32, rel=0.05)
    assert m["corr_alpha_beta"] == pytest.approx(0.89, rel=REL)
    assert nureg_post.hyper_quantile("alpha", 0.05) == pytest.approx(0.94, rel=REL)
    assert nureg_post.hyper_quantile("alpha", 0.95) == pytest.approx(1.93, rel=REL)
    assert nureg_post.hyper_quantile("beta", 0.05) == pytest.approx(0.76, rel=REL)
    assert nureg_post.hyper_quantile("beta", 0.95) == pytest.approx(1.78, rel=REL)


@pytest.mark.parametrize("plant", sorted(TABLE_8_7))
def test_unit_posteriors_match_nureg_6823_table_8_7(nureg_post, plant: str) -> None:
    mean, lo, hi = TABLE_8_7[plant]
    i = NAMES.index(plant)
    q_lo, q_hi = nureg_post.unit_interval(i, 0.90)
    assert nureg_post.unit_mean(i) == pytest.approx(mean, rel=REL)
    assert q_lo == pytest.approx(lo, rel=0.05 if lo < 0.02 else REL)  # 2-figure values
    assert q_hi == pytest.approx(hi, rel=REL)


def test_grid_refinement_does_not_move_the_answer(nureg_post) -> None:
    fine = exact_posterior(X, T, AlphaBetaHyperprior(), n=401)
    i = NAMES.index("Callaway")
    assert fine.unit_mean(i) == pytest.approx(nureg_post.unit_mean(i), rel=1e-9)
    assert fine.unit_quantile(i, 0.95) == pytest.approx(nureg_post.unit_quantile(i, 0.95), rel=1e-7)
    assert nureg_post.weights.sum() == pytest.approx(1.0)


def test_two_unit_posterior_matches_direct_integration_over_alpha_beta() -> None:
    # Independent oracle: integrate the joint density over (alpha, beta) with scipy.dblquad
    # in the original coordinates (no log grid, no Jacobian), for a tiny two-unit dataset.
    x, t = np.array([1, 4]), np.array([2.0, 3.0])
    prior = AlphaBetaHyperprior(alpha_mean=2.0, beta_shape=2.0, beta_rate=1.0)

    def joint(beta: float, alpha: float) -> float:
        lik = np.prod(stats.nbinom.pmf(x, alpha, beta / (beta + t)))
        return float(
            lik * stats.expon.pdf(alpha, scale=2.0) * stats.gamma.pdf(beta, 2.0, scale=1.0)
        )

    z, _ = integrate.dblquad(joint, 0, 60, 0, 60, epsabs=1e-13)
    mean0, _ = integrate.dblquad(
        lambda b, a: joint(b, a) * (a + x[0]) / (b + t[0]) / z, 0, 60, 0, 60, epsabs=1e-13
    )
    alpha_mean, _ = integrate.dblquad(lambda b, a: joint(b, a) * a / z, 0, 60, 0, 60, epsabs=1e-13)
    post = exact_posterior(x, t, prior, n=401)
    assert post.unit_mean(0) == pytest.approx(mean0, rel=1e-4)
    assert post.hyper_moments()["alpha_mean"] == pytest.approx(alpha_mean, rel=1e-4)


def test_unit_quantiles_invert_the_mixture_cdf(nureg_post) -> None:
    i = NAMES.index("Grand Gulf")
    for q in (0.05, 0.5, 0.95):
        assert nureg_post.unit_cdf(i, nureg_post.unit_quantile(i, q)) == pytest.approx(q, abs=1e-9)
    assert nureg_post.unit_sf(i, 2.0) == pytest.approx(1 - nureg_post.unit_cdf(i, 2.0))
    assert nureg_post.new_unit_cdf(1e9) == pytest.approx(1.0)
    assert nureg_post.population_mean() == pytest.approx(nureg_post.hyper_moments()["mu_mean"])


def test_mean_cv_hyperprior_jacobian_integrates_to_one() -> None:
    # The density in (log mu, log alpha) coordinates must integrate to 1 over the plane.
    prior = MeanCVHyperprior(mean_median=0.2, mean_log_sd=1.0, cv_scale=1.0)
    u = np.linspace(np.log(0.2) - 9, np.log(0.2) + 9, 1201)
    w = np.linspace(-12, 30, 2401)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    dens = np.exp(prior.log_density_uw(uu, ww))
    total = integrate.trapezoid(integrate.trapezoid(dens, w, axis=1), u)
    assert total == pytest.approx(1.0, abs=2e-3)


def test_alpha_beta_hyperprior_jacobian_integrates_to_one() -> None:
    prior = AlphaBetaHyperprior(alpha_mean=1.0, beta_shape=2.0, beta_rate=2.0)
    u = np.linspace(-15, 15, 1501)
    w = np.linspace(-15, 6, 1501)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    dens = np.exp(prior.log_density_uw(uu, ww))
    total = integrate.trapezoid(integrate.trapezoid(dens, w, axis=1), u)
    assert total == pytest.approx(1.0, abs=2e-3)


@settings(max_examples=25, deadline=None)
@given(
    st.lists(st.integers(0, 12), min_size=3, max_size=6),
    st.floats(0.5, 10.0),
)
def test_partial_pooling_keeps_order_and_shrinks_spread(counts, scale) -> None:
    # With equal exposures every grid component maps a unit's MLE r_i to w·μ + (1 − w)·r_i
    # with the same w for all units, so posterior means keep the order of the counts and
    # E_i − E_j = Σ p_g (1 − w_g)(r_i − r_j): their spread is never wider than the MLEs'.
    x = np.array(counts)
    if x.sum() == 0:
        return
    t = np.full(x.size, scale)
    post = exact_posterior(x, t, MeanCVHyperprior(mean_median=float(x.sum() / t.sum())), n=121)
    rates = x / t
    means = np.array([post.unit_mean(i) for i in range(x.size)])
    for i in range(x.size):
        for j in range(x.size):
            if x[i] < x[j]:
                assert means[i] < means[j]
            assert abs(means[i] - means[j]) <= abs(rates[i] - rates[j]) + 1e-9


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        exact_posterior([1, 2], [1.0], AlphaBetaHyperprior())
