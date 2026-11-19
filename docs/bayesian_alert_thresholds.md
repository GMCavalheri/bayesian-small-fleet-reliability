# `bayesian_alert_thresholds`: alerts as posterior probabilities

**Lesson:** [5 · An alert as a probability](lessons/05-an-alert-as-a-probability.md) ·
**Code:** [`src/smallfleet/bayesian_alert_thresholds/`](../src/smallfleet/bayesian_alert_thresholds/)

## Model

Alert in month $j$ when $P(\lambda > \lambda^\ast \mid \text{prior}, \text{last } w \text{ months}) > p^\ast$,
a Gamma survival function. It is equivalent to a count threshold $Y_w \ge c$, so the
false-alarm rate at a constant rate is the Poisson tail $P(Y_w \ge c)$. A geometric forgetting
factor is the smooth alternative. Operating characteristics compare any rules on the same
simulated histories (baseline, then an unchanged or shifted rate).

## API

| function | purpose |
|---|---|
| `exceedance_probability`, `alert_count_threshold`, `window_false_alarm_rate` | the rule and its exact false alarms |
| `window_exceedance`, `discounted_exceedance` | per-month probabilities |
| `simulate_histories`, `characteristics`, `classical_rule`, `bayes_window_rule` | paired comparison |

## Verification

Gamma survival function; count threshold brackets p*; window and discounting reduce to the
cumulative update; exact false-alarm rate against Monte Carlo; monotonicity (Hypothesis)
([`tests/test_bayesian_alerts.py`](../tests/test_bayesian_alerts.py)).
