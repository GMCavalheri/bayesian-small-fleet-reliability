"""Phase 5: when does borrowing strength help, and when do the methods converge?

``single_fleet`` is exact (sums over the Poisson count); ``pooling`` is seeded Monte Carlo
with a standard error on every number.
"""

from smallfleet.simulation_study.pooling import PoolingPoint, operator_pooling_point
from smallfleet.simulation_study.single_fleet import (
    SingleFleetPoint,
    convergence_size,
    moment_prior,
    single_fleet_point,
)

__all__ = [
    "PoolingPoint",
    "SingleFleetPoint",
    "convergence_size",
    "moment_prior",
    "operator_pooling_point",
    "single_fleet_point",
]
