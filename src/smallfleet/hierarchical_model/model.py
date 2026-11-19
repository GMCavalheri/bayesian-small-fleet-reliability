"""The hierarchical Gamma-Poisson model in PyMC, with diagnostics.

Two forms of the same model:

* **marginal** (default): the unit rates are integrated out, so the sampler sees only the two
  hyperparameters and a negative-binomial likelihood. Each unit's rate is then drawn exactly
  from its conditional Gamma(α + x_i, β + t_i) for every posterior draw. No funnel, no
  divergences, and the result is the same posterior;
* **centred**: λ_i ~ Gamma(α, β) and x_i ~ Poisson(λ_i t_i) as written. With sparse data this
  is the classic hierarchical funnel, which shows up as divergences.

Both are checked against the exact oracle (``oracle.py``) before MCMC is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pymc as pm
from numpy.typing import ArrayLike

from smallfleet.data_generation import FloatArray
from smallfleet.hierarchical_model.hyperpriors import AlphaBetaHyperprior, MeanCVHyperprior


def build_model(
    x: ArrayLike,
    t: ArrayLike,
    hyperprior: AlphaBetaHyperprior | MeanCVHyperprior,
    marginal: bool = True,
) -> pm.Model:
    """The PyMC model for counts ``x`` in exposures ``t`` (thousands of FH)."""
    xa = np.asarray(x, dtype=np.int64)
    ta = np.asarray(t, dtype=np.float64)
    with pm.Model() as model:
        if isinstance(hyperprior, AlphaBetaHyperprior):
            alpha = pm.Exponential("alpha", lam=1.0 / hyperprior.alpha_mean)
            beta = pm.Gamma("beta", alpha=hyperprior.beta_shape, beta=hyperprior.beta_rate)
            mu = pm.Deterministic("mu", alpha / beta)
        else:
            mu = pm.LogNormal("mu", mu=np.log(hyperprior.mean_median), sigma=hyperprior.mean_log_sd)
            cv = pm.HalfNormal("cv", sigma=hyperprior.cv_scale)
            alpha = pm.Deterministic("alpha", 1.0 / cv**2)
            beta = pm.Deterministic("beta", alpha / mu)
        if marginal:
            pm.NegativeBinomial("x", mu=mu * ta, alpha=alpha, observed=xa)
        else:
            lam = pm.Gamma("lam", alpha=alpha, beta=beta, shape=xa.size)
            pm.Poisson("x", mu=lam * ta, observed=xa)
    return model


@dataclass(frozen=True)
class McmcResult:
    alpha: FloatArray
    """Posterior draws of α, flattened over chains."""
    beta: FloatArray
    lam: FloatArray
    """Unit-rate draws, shape (draws, units)."""
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int

    def unit_mean(self, i: int) -> float:
        return float(self.lam[:, i].mean())

    def unit_interval(self, i: int, mass: float = 0.90) -> tuple[float, float]:
        a = (1 - mass) / 2
        lo, hi = np.quantile(self.lam[:, i], [a, 1 - a])
        return float(lo), float(hi)

    def unit_mcse(self, i: int, ess: float | None = None) -> float:
        """Monte Carlo standard error of a unit mean, from the smallest bulk ESS."""
        return float(self.lam[:, i].std() / np.sqrt(ess or self.ess_bulk_min))


def sample(
    x: ArrayLike,
    t: ArrayLike,
    hyperprior: AlphaBetaHyperprior | MeanCVHyperprior,
    marginal: bool = True,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    seed: int = 1,
    target_accept: float = 0.95,
) -> McmcResult:
    """Sample with NUTS (one core, fixed seed) and collect draws and diagnostics."""
    xa = np.asarray(x, dtype=np.int64)
    ta = np.asarray(t, dtype=np.float64)
    model = build_model(xa, ta, hyperprior, marginal)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    arrays = {v: np.asarray(post[v]) for v in ("alpha", "beta")}  # (chain, draw)
    if not marginal:
        arrays["lam"] = np.asarray(post["lam"])  # (chain, draw, unit)
    alpha = arrays["alpha"].reshape(-1)
    beta = arrays["beta"].reshape(-1)
    if marginal:
        rng = np.random.default_rng(seed)
        lam = rng.gamma(alpha[:, None] + xa, 1.0 / (beta[:, None] + ta))
    else:
        lam = arrays["lam"].reshape(-1, xa.size)
    rhat = [np.max(az.rhat(a)) for a in arrays.values()]
    ess_bulk = [np.min(az.ess(a, method="bulk")) for a in arrays.values()]
    ess_tail = [np.min(az.ess(a, method="tail", prob=(0.05, 0.95))) for a in arrays.values()]
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    return McmcResult(
        alpha=alpha,
        beta=beta,
        lam=lam,
        rhat_max=float(max(rhat)),
        ess_bulk_min=float(min(ess_bulk)),
        ess_tail_min=float(min(ess_tail)),
        divergences=div,
    )
