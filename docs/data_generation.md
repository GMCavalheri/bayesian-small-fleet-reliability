# `data_generation`: synthetic fleets with known truth

**Lessons:** [1](lessons/01-the-alert-level-that-needs-a-big-fleet.md),
[2](lessons/02-borrowing-a-prior-from-a-bigger-fleet.md) ·
**Code:** [`src/smallfleet/data_generation.py`](../src/smallfleet/data_generation.py)

## Model

A fleet history is a sequence of months. Month $j$ has exposure $T_j$ flight hours (FH) and a
count $y_j \sim \text{Poisson}(\lambda T_j / 1000)$, with $\lambda$ the true rate per 1000 FH.
Every aircraft flies the same hours each month, so $T_j = n \cdot u$ for $n$ aircraft at $u$ FH
per aircraft-month. For the hierarchical phases, unit rates are drawn from
Gamma(1/cv², 1/(cv² μ)), a population with mean μ and coefficient of variation cv.

## API

| function | purpose |
|---|---|
| `load_components()`, `load_fleets()` | parameter tables, each row with its basis |
| `monthly_exposure(n, u, months)` | fleet exposure per month |
| `simulate_counts(rng, rate, exposure)`, `simulate_history(...)` | Poisson counts |
| `gamma_population(rng, mean, cv, size)` | unit rates for the hierarchical studies |

## Assumptions

Constant rates within a period, independent months, equal utilisation across aircraft. All
rates and utilisations are illustrative assumptions
([`component_parameters.csv`](../src/smallfleet/data/component_parameters.csv),
[`fleet_parameters.csv`](../src/smallfleet/data/fleet_parameters.csv)).

## Verification

Poisson mean and variance, and the Gamma population's mean and cv, by seeded Monte Carlo;
seed determinism ([`tests/test_data_generation.py`](../tests/test_data_generation.py)).
