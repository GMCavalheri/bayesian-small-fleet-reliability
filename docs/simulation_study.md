# `simulation_study`: when does borrowing help?

**Lesson:** [7 · How big is a big fleet?](lessons/07-how-big-is-a-big-fleet.md) ·
**Code:** [`src/smallfleet/simulation_study/`](../src/smallfleet/simulation_study/)

## Model

* Single fleet: exact RMSE, bias, coverage and width of the MLE, Jeffreys, a borrowed prior
  Gamma(L·E, E) and full borrowing, by fleet size and prior quality. With a correct prior the
  RMSE ratio to the MLE is exactly $t/(\beta + t)$.
* Pooling: operators of one size with Gamma-distributed rates; per-operator RMSE and coverage
  of no pooling and exact hierarchical Bayes, with Monte Carlo standard errors.

## API

| function | purpose |
|---|---|
| `single_fleet_point`, `moment_prior`, `convergence_size` | exact grid |
| `operator_pooling_point` → `PoolingPoint` | Monte Carlo with standard errors |

## Verification

Closed-form bias and RMSE at grid points; the exact RMSE ratio $t/(\beta + t)$; standard
errors and determinism ([`tests/test_simulation_study.py`](../tests/test_simulation_study.py)).
