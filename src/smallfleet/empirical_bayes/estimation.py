"""Maximum marginal likelihood, moments, plug-in and Kass-Steffey posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special

from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.data_generation import FloatArray

ALPHA_CAP = 1e6
"""Beyond this the population is treated as having no spread (all units share one rate)."""


@dataclass(frozen=True)
class EmpiricalBayesFit:
    alpha: float
    beta: float
    loglik: float
    pooled: bool
    """True when the marginal likelihood is maximised at no between-unit spread."""

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def prior(self) -> GammaRate:
        return GammaRate(self.alpha, self.beta)


def _arrays(x: ArrayLike, t: ArrayLike) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    if xa.shape != ta.shape or xa.ndim != 1 or xa.size < 2:
        raise ValueError("need matching 1-D counts and exposures for at least two units")
    if np.any(xa < 0) or np.any(ta <= 0):
        raise ValueError("counts must be non-negative and exposures positive")
    return xa, ta


def marginal_loglik(alpha: float, beta: float, x: ArrayLike, t: ArrayLike) -> float:
    """Σ log NB(x_i; α, β/(β + t_i)), dropping nothing (the ``log x!`` term included)."""
    xa, ta = _arrays(x, t)
    p = beta / (beta + ta)
    return float(
        np.sum(
            special.gammaln(alpha + xa)
            - special.gammaln(alpha)
            - special.gammaln(xa + 1)
            + alpha * np.log(p)
            + xa * np.log1p(-p)
        )
    )


def fit_marginal_mle(x: ArrayLike, t: ArrayLike) -> EmpiricalBayesFit:
    """Maximum marginal likelihood estimate of (α, β), optimised over (log μ, log α)."""
    xa, ta = _arrays(x, t)
    mu0 = max(xa.sum() / ta.sum(), 1e-12)

    def nll(theta: FloatArray) -> float:
        mu, alpha = np.exp(theta)
        return -marginal_loglik(float(alpha), float(alpha / mu), xa, ta)

    best = None
    for log_alpha0 in (-1.0, 0.0, 2.0, 5.0):
        res = optimize.minimize(
            nll,
            np.array([np.log(mu0), log_alpha0]),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    mu, alpha = (float(v) for v in np.exp(best.x))
    pooled = alpha >= ALPHA_CAP or not np.isfinite(alpha)
    if pooled:
        alpha = ALPHA_CAP
        mu = float(xa.sum() / ta.sum())
    beta = alpha / mu
    return EmpiricalBayesFit(alpha, beta, marginal_loglik(alpha, beta, xa, ta), pooled)


def method_of_moments(x: ArrayLike, t: ArrayLike) -> EmpiricalBayesFit:
    """Moment estimate from ``E[(x_i − μ t_i)²] = μ t_i + σ² t_i²``.

    ``μ̂ = Σx/Σt``; ``σ̂² = max(0, Σ[(x_i − μ̂t_i)² − μ̂t_i] / Σt_i²)``; then α = μ²/σ²,
    β = μ/σ². A zero spread estimate returns the pooled fit.
    """
    xa, ta = _arrays(x, t)
    mu = float(xa.sum() / ta.sum())
    s2 = float(np.sum((xa - mu * ta) ** 2 - mu * ta) / np.sum(ta**2))
    if s2 <= 0 or mu <= 0:
        alpha = ALPHA_CAP
        beta = alpha / max(mu, 1e-12)
        return EmpiricalBayesFit(alpha, beta, marginal_loglik(alpha, beta, xa, ta), True)
    alpha, beta = mu**2 / s2, mu / s2
    return EmpiricalBayesFit(alpha, beta, marginal_loglik(alpha, beta, xa, ta), False)


def plugin_posteriors(fit: EmpiricalBayesFit, x: ArrayLike, t: ArrayLike) -> list[GammaRate]:
    """Gamma(α̂ + x_i, β̂ + t_i) for every unit: the plug-in empirical Bayes posteriors."""
    xa, ta = _arrays(x, t)
    return [fit.prior.update(int(xi), float(ti) * 1000) for xi, ti in zip(xa, ta, strict=True)]


def kass_steffey_posteriors(fit: EmpiricalBayesFit, x: ArrayLike, t: ArrayLike) -> list[GammaRate]:
    """Plug-in posteriors with the Kass-Steffey variance adjustment, moment-matched to Gamma.

    With μ = α/β, ``E_i = μ(α + x_i)/(α + μ t_i)`` and
    ``Var_i = (α + x_i)/(α/μ + t_i)²``. The adjusted variance adds
    ``(∂E_i/∂μ)² var(μ̂) + (∂E_i/∂α)² var(α̂)``, where (NUREG/CR-6823 eq. 8.7-8.8)

    * ``var(μ̂) = 1/J₁₁``, ``J₁₁ = (α/μ) Σ t_i/(α + μt_i)``;
    * ``var(α̂) = 1/J₂₂``, ``J₂₂ = Σ[ψ'(α) − ψ'(α + x_i)] − (μ/α) Σ t_i/(α + μt_i)``;
    * ``∂E_i/∂μ = α(α + x_i)/(α + μt_i)²`` and ``∂E_i/∂α = μ(μt_i − x_i)/(α + μt_i)²``.

    The derivatives are derived from ``E_i`` directly; the printed handbook's denominator for
    ∂E/∂α (``(α + x_i μ)²``) does not reproduce its own Table 8.3, ours does.
    """
    if fit.pooled:
        raise ValueError("no between-unit spread was estimated; the adjustment is undefined")
    xa, ta = _arrays(x, t)
    a, mu = fit.alpha, fit.mean
    d = a + mu * ta
    j11 = (a / mu) * np.sum(ta / d)
    j22 = float(np.sum(special.polygamma(1, a) - special.polygamma(1, a + xa))) - (mu / a) * float(
        np.sum(ta / d)
    )
    if j22 <= 0:
        raise ValueError("the information for alpha is not positive; the fit is degenerate")
    mean = mu * (a + xa) / d
    var = (a + xa) / (a / mu + ta) ** 2
    de_dmu = a * (a + xa) / d**2
    de_da = mu * (mu * ta - xa) / d**2
    adjusted = var + de_dmu**2 / j11 + de_da**2 / j22
    return [GammaRate(float(m**2 / v), float(m / v)) for m, v in zip(mean, adjusted, strict=True)]
