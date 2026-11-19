"""Several operators of one size: per-operator performance of partial pooling (Monte Carlo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smallfleet.data_generation import PER, gamma_population
from smallfleet.frequentist_baseline import garwood_interval
from smallfleet.hierarchical_model import MeanCVHyperprior, exact_posterior


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float


@dataclass(frozen=True)
class PoolingPoint:
    aircraft: int
    rmse: dict[str, Estimate]
    """Relative RMSE (RMSE / true rate) per method, with its delta-method standard error."""
    coverage: dict[str, Estimate]


def _rel_rmse(errors: np.ndarray) -> Estimate:
    """sqrt(mean(e²)) and its delta-method standard error sd(e²) / (2 sqrt(n) rmse)."""
    sq = errors**2
    r = float(np.sqrt(sq.mean()))
    return Estimate(r, float(sq.std(ddof=1) / (2 * np.sqrt(sq.size) * r)))


def _share(hits: np.ndarray) -> Estimate:
    p = float(hits.mean())
    return Estimate(p, float(np.sqrt(p * (1 - p) / hits.size)))


def operator_pooling_point(
    rng: np.random.Generator,
    aircraft: int,
    operators: int,
    mean_rate: float,
    cv: float,
    hyperprior_median: float,
    fh_per_aircraft_month: float,
    months: int,
    replicates: int,
    grid: int = 121,
    mass: float = 0.90,
) -> PoolingPoint:
    """``operators`` fleets of ``aircraft`` each; no pooling against exact hierarchical Bayes."""
    t = np.full(operators, aircraft * fh_per_aircraft_month * months / PER)
    prior = MeanCVHyperprior(mean_median=hyperprior_median, mean_log_sd=1.0, cv_scale=1.0)
    err: dict[str, list[float]] = {"MLE + Garwood": [], "hierarchical Bayes": []}
    hit: dict[str, list[bool]] = {"MLE + Garwood": [], "hierarchical Bayes": []}
    a = (1 - mass) / 2
    for _ in range(replicates):
        lam = gamma_population(rng, mean_rate, cv, operators)
        x = rng.poisson(lam * t)
        post = exact_posterior(x, t, prior, n=grid)
        for i in range(operators):
            lo, hi = garwood_interval(int(x[i]), t[i] * PER, mass)
            err["MLE + Garwood"].append((x[i] / t[i] - lam[i]) / lam[i])
            hit["MLE + Garwood"].append(lo <= lam[i] <= hi)
            err["hierarchical Bayes"].append((post.unit_mean(i) - lam[i]) / lam[i])
            c = post.unit_cdf(i, float(lam[i]))
            hit["hierarchical Bayes"].append(a < c < 1 - a)
    return PoolingPoint(
        aircraft,
        {k: _rel_rmse(np.array(v)) for k, v in err.items()},
        {k: _share(np.array(v)) for k, v in hit.items()},
    )
