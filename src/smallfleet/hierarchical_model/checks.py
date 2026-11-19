"""Posterior predictive checks of the population assumption.

The Pearson dispersion statistic ``X² = Σ (x_i − μ̂ t_i)² / (μ̂ t_i)``, with μ̂ the pooled
rate, measures extra-Poisson variation between units; NUREG/CR-6823 §6.2.3.1.2 uses it to
test poolability. The posterior predictive p-value replicates the whole dataset from the
fitted model (new unit rates from the population, then new counts) and reports how often the
replicated X² is at least the observed one (Gelman, Meng & Stern 1996).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from smallfleet.data_generation import FloatArray


def pearson_dispersion(x: ArrayLike, t: ArrayLike) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    mu = xa.sum() / ta.sum()
    if mu <= 0:
        raise ValueError("no events: the dispersion statistic is undefined")
    return float(np.sum((xa - mu * ta) ** 2 / (mu * ta)))


def predictive_pvalue(
    rng: np.random.Generator,
    alpha: FloatArray,
    beta: FloatArray,
    x: ArrayLike,
    t: ArrayLike,
) -> float:
    """``P(X²_rep ≥ X²_obs)`` with units redrawn from Gamma(α, β) for each posterior draw.

    Pass a huge α (and β = α/μ) to check the complete-pooling Poisson model instead.
    Replicates with no events at all are counted as not exceeding.
    """
    xa = np.asarray(x, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    observed = pearson_dispersion(xa, ta)
    lam = rng.gamma(alpha[:, None], 1.0 / beta[:, None], size=(alpha.size, xa.size))
    rep = rng.poisson(lam * ta)
    tot = rep.sum(axis=1)
    mu = np.where(tot > 0, tot / ta.sum(), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.sum((rep - mu[:, None] * ta) ** 2 / (mu[:, None] * ta), axis=1)
    return float(np.mean(np.nan_to_num(stat, nan=-np.inf) >= observed))
