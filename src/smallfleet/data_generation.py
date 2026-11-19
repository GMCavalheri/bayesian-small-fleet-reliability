"""Synthetic fleet histories with known true failure rates.

Every history is a sequence of monthly periods. In month ``j`` the fleet logs exposure
``T_j`` flight hours (FH) and a failure count ``y_j ~ Poisson(λ T_j / 1000)``, with ``λ`` the
true rate per 1000 FH. Because the truth is known, estimation error can be measured directly.
The parameters live in ``data/*.csv``; every row carries its basis, and every rate is an
illustrative assumption, not field data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PER = 1000.0
"""Rates are failures per ``PER`` flight hours."""


@dataclass(frozen=True)
class ComponentSpec:
    """One component shared by the large fleet and the small fleet."""

    name: str
    ata_chapter: int
    large_fleet_rate: float
    """True unscheduled-removal rate on the large fleet, per 1000 FH."""
    small_fleet_rate: float
    """True rate on the small fleet, per 1000 FH (operations differ, so it can differ)."""
    basis: str


@dataclass(frozen=True)
class FleetSpec:
    name: str
    fh_per_aircraft_month: float
    basis: str


@dataclass(frozen=True)
class FleetHistory:
    """Monthly exposure (FH) and failure counts of one component on one fleet."""

    exposure_fh: FloatArray
    counts: IntArray

    def __post_init__(self) -> None:
        if self.exposure_fh.shape != self.counts.shape or self.exposure_fh.ndim != 1:
            raise ValueError("exposure and counts must be 1-D arrays of the same length")

    @property
    def months(self) -> int:
        return int(self.counts.size)

    @property
    def total_exposure_fh(self) -> float:
        return float(self.exposure_fh.sum())

    @property
    def total_count(self) -> int:
        return int(self.counts.sum())


def _rows(filename: str) -> list[dict[str, str]]:
    text = resources.files("smallfleet.data").joinpath(filename).read_text()
    return list(csv.DictReader(text.splitlines()))


def load_components() -> dict[str, ComponentSpec]:
    """The component table, keyed by component name."""
    return {
        r["component"]: ComponentSpec(
            name=r["component"],
            ata_chapter=int(r["ata_chapter"]),
            large_fleet_rate=float(r["large_fleet_rate_per_1000fh"]),
            small_fleet_rate=float(r["small_fleet_rate_per_1000fh"]),
            basis=r["basis"],
        )
        for r in _rows("component_parameters.csv")
    }


def load_fleets() -> dict[str, FleetSpec]:
    """The fleet utilisation table, keyed by fleet name."""
    return {
        r["fleet"]: FleetSpec(r["fleet"], float(r["fh_per_aircraft_month"]), r["basis"])
        for r in _rows("fleet_parameters.csv")
    }


def monthly_exposure(n_aircraft: int, fh_per_aircraft_month: float, months: int) -> FloatArray:
    """Fleet exposure per month: every aircraft flies the same hours each month."""
    if n_aircraft < 1 or months < 1 or fh_per_aircraft_month <= 0:
        raise ValueError("need at least one aircraft, one month and positive utilisation")
    return np.full(months, n_aircraft * fh_per_aircraft_month, dtype=np.float64)


def simulate_counts(
    rng: np.random.Generator, rate: float | FloatArray, exposure_fh: FloatArray
) -> IntArray:
    """Poisson counts per month for a rate per 1000 FH (scalar or one value per month)."""
    mean = np.asarray(rate, dtype=np.float64) * exposure_fh / PER
    if np.any(mean < 0):
        raise ValueError("rates and exposure must be non-negative")
    return rng.poisson(mean).astype(np.int64)


def simulate_history(
    rng: np.random.Generator,
    rate: float | FloatArray,
    n_aircraft: int,
    fh_per_aircraft_month: float,
    months: int,
) -> FleetHistory:
    """One component's monthly history on a fleet of ``n_aircraft``."""
    exposure = monthly_exposure(n_aircraft, fh_per_aircraft_month, months)
    return FleetHistory(exposure, simulate_counts(rng, rate, exposure))


def gamma_population(rng: np.random.Generator, mean: float, cv: float, size: int) -> FloatArray:
    """True rates drawn from Gamma(α, β) with the given mean and coefficient of variation.

    ``α = 1/cv²`` and ``β = α/mean`` (shape-rate form), the population the hierarchical model
    assumes. ``cv = 0`` returns ``size`` copies of the mean.
    """
    if mean <= 0 or cv < 0:
        raise ValueError("mean must be positive and cv non-negative")
    if cv == 0:
        return np.full(size, mean)
    alpha = 1.0 / cv**2
    return rng.gamma(alpha, mean / alpha, size)
