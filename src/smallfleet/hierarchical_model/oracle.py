"""The exact posterior of the hierarchical Gamma-Poisson model, by quadrature. No MCMC.

Model: ``x_i | λ_i ~ Poisson(λ_i t_i)``, ``λ_i | α, β ~ Gamma(α, β)``, ``(α, β) ~ hyperprior``.
Integrating each λ_i out gives the negative-binomial marginal likelihood of (α, β) in closed
form, so the hyperparameter posterior is two-dimensional and can be computed on a grid.
Given (α, β), each λ_i is Gamma(α + x_i, β + t_i), so every unit's posterior is an exact
mixture of Gammas weighted by the grid. The only approximation is the grid, and its
resolution is checked by refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special

from smallfleet.data_generation import FloatArray


class Hyperprior(Protocol):
    def log_density_uw(self, u: FloatArray, w: FloatArray) -> FloatArray: ...


def _loglik_grid(u: FloatArray, w: FloatArray, x: FloatArray, t: FloatArray) -> FloatArray:
    """Negative-binomial log marginal likelihood at every grid point (u = log μ, w = log α).

    Written to stay accurate up to α ~ 1e20 (the complete-pooling limit): for integer counts
    ``lnΓ(α + x) − lnΓ(α) = Σ_{k<x} ln(α + k)``, and ``α ln(β/(β + t)) = −α log1p(μt/α)``.
    """
    alpha = np.exp(w)[..., None]
    mu = np.exp(u)[..., None]
    xi = x.astype(np.int64)
    gamma_ratio = np.zeros(np.broadcast_shapes(alpha.shape, x.shape))
    for k in range(int(np.max(xi, initial=0))):
        gamma_ratio = gamma_ratio + np.where(xi > k, np.log(alpha + k), 0.0)
    mt = mu * t
    terms = (
        gamma_ratio
        - special.gammaln(x + 1)
        - alpha * np.log1p(mt / alpha)
        + np.where(x > 0, x * (np.log(mt) - np.log(alpha + mt)), 0.0)
    )
    return np.asarray(terms.sum(axis=-1), dtype=np.float64)


@dataclass(frozen=True)
class HierarchicalPosterior:
    """Grid posterior over (log μ, log α) and the exact unit posteriors it implies."""

    u: FloatArray
    """log μ of each retained grid cell (1-D)."""
    w: FloatArray
    """log α of each retained grid cell (1-D)."""
    weights: FloatArray
    """Normalised posterior mass of each retained cell (sums to 1)."""
    x: FloatArray
    t: FloatArray

    @property
    def alpha(self) -> FloatArray:
        return np.exp(self.w)

    @property
    def beta(self) -> FloatArray:
        return np.exp(self.w - self.u)

    def hyper_moments(self) -> dict[str, float]:
        """Posterior mean and sd of α, β, μ and cv, and the correlation of α and β."""
        out: dict[str, float] = {}
        values = {
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": np.exp(self.u),
            "cv": np.exp(-self.w / 2),
        }
        for name, v in values.items():
            m = float(np.sum(self.weights * v))
            out[f"{name}_mean"] = m
            out[f"{name}_sd"] = float(np.sqrt(np.sum(self.weights * (v - m) ** 2)))
        cov = np.sum(
            self.weights * (self.alpha - out["alpha_mean"]) * (self.beta - out["beta_mean"])
        )
        out["corr_alpha_beta"] = float(cov / (out["alpha_sd"] * out["beta_sd"]))
        return out

    def hyper_quantile(self, name: str, q: float) -> float:
        """Quantile of the marginal posterior of α, β, μ or cv (weighted grid quantile)."""
        v = {
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": np.exp(self.u),
            "cv": np.exp(-self.w / 2),
        }[name]
        order = np.argsort(v)
        cdf = np.cumsum(self.weights[order])
        return float(np.interp(q, cdf - self.weights[order] / 2, v[order]))

    def sample_hyper(self, rng: np.random.Generator, size: int) -> tuple[FloatArray, FloatArray]:
        """Draws of (α, β) from the grid posterior (cell centres, by weight)."""
        idx = rng.choice(self.weights.size, size=size, p=self.weights)
        return self.alpha[idx], self.beta[idx]

    def unit_mean(self, i: int) -> float:
        return float(np.sum(self.weights * (self.alpha + self.x[i]) / (self.beta + self.t[i])))

    def unit_var(self, i: int) -> float:
        a = self.alpha + self.x[i]
        b = self.beta + self.t[i]
        second = np.sum(self.weights * a * (a + 1) / b**2)
        return float(second - self.unit_mean(i) ** 2)

    def unit_cdf(self, i: int, lam: float) -> float:
        z = max(lam, 0.0) * (self.beta + self.t[i])
        return float(np.sum(self.weights * special.gammainc(self.alpha + self.x[i], z)))

    def unit_quantile(self, i: int, q: float) -> float:
        hi = self.unit_mean(i) + 50 * np.sqrt(self.unit_var(i))
        return float(optimize.brentq(lambda lam: self.unit_cdf(i, lam) - q, 0.0, hi, xtol=1e-12))

    def unit_interval(self, i: int, mass: float = 0.90) -> tuple[float, float]:
        a = (1 - mass) / 2
        return self.unit_quantile(i, a), self.unit_quantile(i, 1 - a)

    def unit_sf(self, i: int, lam: float) -> float:
        """``P(λ_i > lam | data)``."""
        return 1.0 - self.unit_cdf(i, lam)

    def population_mean(self) -> float:
        """Mean of the population distribution with (α, β) integrated out (= E[μ])."""
        return float(np.sum(self.weights * self.alpha / self.beta))

    def new_unit_cdf(self, lam: float) -> float:
        """Predictive distribution of a new unit's rate: a Gamma(α, β) mixture."""
        return float(np.sum(self.weights * special.gammainc(self.alpha, max(lam, 0.0) * self.beta)))


W_MIN, W_MAX = -12.0, 46.0
"""Grid limits on log α: cv from e^6 ≈ 400 down to 1e-10 (complete pooling to 1e-10)."""


def _log_post(
    u: FloatArray, w: FloatArray, x: FloatArray, t: FloatArray, hyperprior: Hyperprior
) -> FloatArray:
    with np.errstate(all="ignore"):
        logp = _loglik_grid(u, w, x, t) + hyperprior.log_density_uw(u, w)
    return np.where(np.isfinite(logp), logp, -np.inf)


def exact_posterior(
    x: ArrayLike,
    t: ArrayLike,
    hyperprior: Hyperprior,
    n: int = 241,
) -> HierarchicalPosterior:
    """Grid posterior over (log μ, log α), ``n × n`` cells on a box found by a wide scan.

    ``t`` is in thousands of FH (the rate's unit). A coarse scan over log μ within ±12 of the
    pooled estimate and log α in [W_MIN, W_MAX] locates every cell within e^-30 of the peak;
    the fine grid covers that region with a margin, and is widened until its edges hold
    less than 1e-9 of the peak density (or reach the scan limits, where the hyperprior or
    the pooling limit makes the remaining mass negligible).
    """
    xa = np.asarray(x, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    if xa.shape != ta.shape or xa.ndim != 1 or np.any(ta <= 0) or np.any(xa < 0):
        raise ValueError("need matching 1-D non-negative counts and positive exposures")
    centre = float(np.log(max(xa.sum(), 0.5) / ta.sum()))
    u_lim = (centre - 12.0, centre + 12.0)

    u = np.linspace(*u_lim, 241)
    w = np.linspace(W_MIN, W_MAX, 241)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    logp = _log_post(uu, ww, xa, ta, hyperprior)
    live = logp > np.max(logp) - 30.0
    du, dw = u[1] - u[0], w[1] - w[0]
    lo = np.array([np.min(uu[live]) - 2 * du, np.min(ww[live]) - 2 * dw])
    hi = np.array([np.max(uu[live]) + 2 * du, np.max(ww[live]) + 2 * dw])

    for _ in range(6):
        lo = np.maximum(lo, [u_lim[0], W_MIN])
        hi = np.minimum(hi, [u_lim[1], W_MAX])
        u = np.linspace(lo[0], hi[0], n)
        w = np.linspace(lo[1], hi[1], n)
        uu, ww = np.meshgrid(u, w, indexing="ij")
        logp = _log_post(uu, ww, xa, ta, hyperprior)
        peak = float(np.max(logp))
        cut = peak + np.log(1e-9)
        open_edges = [
            np.max(logp[0, :]) > cut and lo[0] > u_lim[0],
            np.max(logp[-1, :]) > cut and hi[0] < u_lim[1],
            np.max(logp[:, 0]) > cut and lo[1] > W_MIN,
            np.max(logp[:, -1]) > cut and hi[1] < W_MAX,
        ]
        if not any(open_edges):
            break
        width = hi - lo
        lo = lo - 0.25 * width * np.array([open_edges[0], open_edges[2]])
        hi = hi + 0.25 * width * np.array([open_edges[1], open_edges[3]])
    dens = np.exp(logp - peak)
    # Equal cells (the rectangle rule): the edges hold < 1e-9 of the peak, so it agrees with
    # the trapezoid rule to that order; refinement tests check the resolution.
    weights = (dens / dens.sum()).ravel()
    # Keep the heaviest cells holding all but 1e-13 of the mass: every later sum runs over
    # far fewer cells, and nothing is lost at the printed precision.
    order = np.argsort(weights)[::-1]
    keep = order[: int(np.searchsorted(np.cumsum(weights[order]), 1 - 1e-13)) + 1]
    w_keep = weights[keep] / weights[keep].sum()
    return HierarchicalPosterior(uu.ravel()[keep], ww.ravel()[keep], w_keep, xa, ta)
