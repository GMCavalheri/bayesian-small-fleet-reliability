# `frequentist_baseline`: the classical estimates and alert levels

**Lesson:** [1 · The alert level that needs a big fleet](lessons/01-the-alert-level-that-needs-a-big-fleet.md) ·
**Code:** [`src/smallfleet/frequentist_baseline/`](../src/smallfleet/frequentist_baseline/)

## Model

* MLE $\hat\lambda = 1000 y/T$ and the exact Garwood interval
  from $\chi^2_{\alpha/2}(2y)/2t$ to $\chi^2_{1-\alpha/2}(2y+2)/2t$, with $t = T/1000$.
* Alert levels from a baseline of monthly rates (AMA 571.101/1, Appendix A): $1.3\bar r$,
  $\bar r + k s$ with the population divisor, and the 95 % Poisson count limit.
* Closed forms: $P(y_j = 0) = e^{-m}$ and $P(s = 0) = \sum_k \text{pmf}(k)^{12}$ for a 12-month
  baseline.

## API

| function | purpose |
|---|---|
| `mle_rate`, `garwood_interval`, `garwood_coverage`, `expected_relative_width` | estimation |
| `monthly_rates`, `running_average` | rates |
| `mean_sd_alert_level`, `mean_ratio_alert_level`, `poisson_alert_count`, `alert_flags` | alert levels |
| `zero_month_probability`, `degenerate_baseline_probability`, `mean_sd_operating_characteristics` | instability |

## Verification

Garwood closed forms for 0 and 1 events and tail inversion; exact coverage never below
nominal (Hypothesis); AMA 571.101/1 examples 2 and 4(a); closed forms against Monte Carlo
([`tests/test_frequentist_baseline.py`](../tests/test_frequentist_baseline.py)).
