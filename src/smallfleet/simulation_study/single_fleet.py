"""One fleet, one component: exact performance of each estimator by fleet size.

The prior borrowed from the large fleet is moment-matched to the large fleet's rate ``L`` and
worth ``equivalent_fh`` flight hours: Gamma(L·E, E) with E in thousands of FH. (Unlike the
power prior with a Jeffreys start, it has no extra half event, so "good prior" means a prior
whose mean equals the small fleet's true rate.)
"""

from __future__ import annotations

from dataclasses import dataclass

from smallfleet.bayesian_point_estimation import (
    GammaRate,
    Performance,
    bayes_estimator,
    exact_performance,
    frequentist_estimator,
    jeffreys,
)
from smallfleet.data_generation import PER


def moment_prior(large_rate: float, equivalent_fh: float) -> GammaRate:
    """Gamma with mean ``large_rate`` worth ``equivalent_fh`` flight hours."""
    e = equivalent_fh / PER
    return GammaRate(large_rate * e, e)


@dataclass(frozen=True)
class SingleFleetPoint:
    aircraft: int
    true_rate: float
    exposure_fh: float
    methods: dict[str, Performance]


def single_fleet_point(
    aircraft: int,
    true_rate: float,
    large_rate: float,
    fh_per_aircraft_month: float,
    months: int,
    equivalent_fh: float,
    large_exposure_fh: float,
) -> SingleFleetPoint:
    """Exact RMSE, bias, coverage and width of four estimators at one grid point."""
    t_fh = aircraft * fh_per_aircraft_month * months
    est = {
        "MLE + Garwood": frequentist_estimator(),
        "Jeffreys": bayes_estimator(jeffreys()),
        "borrowed prior": bayes_estimator(moment_prior(large_rate, equivalent_fh)),
        "full borrowing": bayes_estimator(moment_prior(large_rate, large_exposure_fh)),
    }
    return SingleFleetPoint(
        aircraft,
        true_rate,
        t_fh,
        {k: exact_performance(e, true_rate, t_fh) for k, e in est.items()},
    )


def convergence_size(
    points: list[SingleFleetPoint], method: str, reference: str = "MLE + Garwood", tol: float = 0.1
) -> int | None:
    """Smallest fleet size from which ``method``'s RMSE stays within ``tol`` of ``reference``'s.

    "Stays": every larger size in the grid also satisfies it. None if it never converges.
    """
    ok = [
        abs(p.methods[method].rmse / p.methods[reference].rmse - 1) <= tol
        for p in sorted(points, key=lambda p: p.aircraft)
    ]
    sizes = sorted(p.aircraft for p in points)
    for i in range(len(sizes)):
        if all(ok[i:]):
            return sizes[i]
    return None
