# `decision_theory`: decisions under rate uncertainty

**Lesson:** [6 · Deciding under uncertainty](lessons/06-deciding-under-uncertainty.md) ·
**Code:** [`src/smallfleet/decision_theory/`](../src/smallfleet/decision_theory/)

## Model

* Alert decision: investigate when $p > c_{FA}/(c_{FA} + c_{miss})$. Value of waiting a month:
  EVSI from the negative-binomial predictive (preposterior analysis), compared with
  $c_{delay} \cdot p$.
* Hidden failure, test interval τ: $U(\lambda, \tau) = 1 - (1 - e^{-\lambda\tau})/(\lambda\tau)$.
  Interval by known rate, by expected unavailability (closed form for Gamma(a, b), a > 1),
  or by risk (the known-rate interval at a posterior quantile).

## API

| function | purpose |
|---|---|
| `AlertCosts`, `bayes_threshold`, `expected_losses`, `value_of_waiting` | alerting |
| `unavailability`, `interval_for_known_rate`, `expected_unavailability`, `interval_for_expected_unavailability`, `interval_for_risk` | test intervals |
| `load_decision_parameters` | costs and targets, each with its basis |

## Verification

Threshold against brute force; expected losses and EVSI against Monte Carlo; predictive
exceedance is a martingale; EVSI ≥ 0 (Hypothesis); expected unavailability against
quadrature; interval ordering by Jensen
([`tests/test_decision_theory.py`](../tests/test_decision_theory.py)).
