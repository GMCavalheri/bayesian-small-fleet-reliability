import math

import numpy as np
import pytest

from smallfleet.simulation_study import (
    convergence_size,
    moment_prior,
    operator_pooling_point,
    single_fleet_point,
)


def test_single_fleet_point_reproduces_closed_forms() -> None:
    # MLE: unbiased with RMSE sqrt(λ/t); posterior mean: bias w(prior mean − λ) with
    # w = β/(β + t), as in Lesson 2.
    p = single_fleet_point(5, 0.3, 0.2, 35.0, 12, 10_000.0, 900_000.0)
    t = 5 * 35 * 12 / 1000
    mle = p.methods["MLE + Garwood"]
    assert mle.bias == pytest.approx(0.0, abs=1e-12)
    assert mle.rmse == pytest.approx(math.sqrt(0.3 / t), rel=1e-10)
    w = 10.0 / (10.0 + t)
    assert p.methods["borrowed prior"].bias == pytest.approx(w * (0.2 - 0.3), rel=1e-9)
    big = 900.0 / (900.0 + t)
    assert p.methods["full borrowing"].bias == pytest.approx(big * (0.2 - 0.3), rel=1e-9)
    assert p.methods["Jeffreys"].bias == pytest.approx(0.5 / t, rel=1e-9)


def test_moment_prior_has_the_large_fleet_mean() -> None:
    g = moment_prior(0.2, 10_000.0)
    assert g.mean() == pytest.approx(0.2) and g.beta == pytest.approx(10.0)


def test_convergence_size_requires_staying_within_tolerance() -> None:
    pts = [single_fleet_point(n, 0.2, 0.2, 35.0, 12, 10_000.0, 900_000.0) for n in (2, 10, 50, 400)]
    n = convergence_size(pts, "borrowed prior")
    # With a correct prior (mean = true rate) the posterior mean is unbiased and its RMSE is
    # (1 − w)·sqrt(λ/t) with w = β/(β + t): the ratio to the MLE is exactly t/(β + t). It is
    # within 10 % once t ≥ 9β = 90 000 FH, about 214 aircraft-years at 420 FH a year, so
    # only the 400-aircraft point (168 000 FH) qualifies.
    for p in pts:
        t = p.exposure_fh / 1000
        ratio = p.methods["borrowed prior"].rmse / p.methods["MLE + Garwood"].rmse
        assert ratio == pytest.approx(t / (10.0 + t), rel=1e-9)
    assert n == 400
    assert convergence_size(pts, "MLE + Garwood") == 2
    assert convergence_size(pts[:1], "full borrowing") is None


def test_pooling_point_standard_errors_and_determinism() -> None:
    a = operator_pooling_point(np.random.default_rng(4), 5, 6, 0.25, 0.5, 0.2, 35.0, 12, 30)
    b = operator_pooling_point(np.random.default_rng(4), 5, 6, 0.25, 0.5, 0.2, 35.0, 12, 30)
    assert a == b
    cov = a.coverage["hierarchical Bayes"]
    assert cov.se == pytest.approx(math.sqrt(cov.value * (1 - cov.value) / 180))
    assert 0 < a.rmse["MLE + Garwood"].se < a.rmse["MLE + Garwood"].value
