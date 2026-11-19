# Course: reliability statistics for small fleets

Nine lessons, in order. They take a reader from the classical alert level of an airline
reliability programme to Bayesian estimation, pooling, alerting and decisions for fleets too
small for it. Each lesson has:
- learning objectives;
- the theory with derivations;
- a code walkthrough and a worked example;
- exercises with solutions, and references;
- an executed notebook.

Every number comes from [`validation/results/`](../../validation/results/).

**Prerequisites:** basic probability (the Poisson and Gamma distributions, conditional
probability) and Python. Bayesian inference is introduced from scratch in Lesson 2.

| # | lesson | key ideas | the finding | notebook |
|---|---|---|---|---|
| 1 | [The alert level that needs a big fleet](01-the-alert-level-that-needs-a-big-fleet.md) | rates per 1000 FH, mean + kσ and Poisson alert levels, Garwood intervals, closed forms for zero months and degenerate baselines | on five aircraft the baseline has σ = 0 in 59 % of years and the first event alerts | [01](../../notebooks/01_the_alert_level_that_needs_a_big_fleet.ipynb) |
| 2 | [Borrowing a prior from a bigger fleet](02-borrowing-a-prior-from-a-bigger-fleet.md) | conjugacy, priors as pseudo-data, predictive distribution, power prior, prior-data conflict | a discounted prior cuts RMSE up to 5.7×; full borrowing covers 0 % when the fleets differ | [02](../../notebooks/02_borrowing_a_prior_from_a_bigger_fleet.ipynb) |
| 3 | [Letting the data choose the prior](03-letting-the-data-choose-the-prior.md) | negative-binomial marginal, empirical Bayes, shrinkage, Kass-Steffey | NUREG's 66 plants reproduced; on sparse operators empirical Bayes covers 11 to 50 % | [03](../../notebooks/03_letting_the_data_choose_the_prior.ipynb) |
| 4 | [Partial pooling](04-partial-pooling.md) | hyperpriors, exact quadrature, NUTS, R-hat and ESS, the funnel, predictive checks, calibration | hierarchical Bayes covers 0.90 to 0.99 where empirical Bayes fails | [04](../../notebooks/04_partial_pooling.ipynb) |
| 5 | [An alert as a probability](05-an-alert-as-a-probability.md) | posterior exceedance, count thresholds, exact false alarms, matched comparisons | better detection on 25 aircraft; on 5, control of false alarms but no more detection | [05](../../notebooks/05_an_alert_as_a_probability.ipynb) |
| 6 | [Deciding under uncertainty](06-deciding-under-uncertainty.md) | expected loss, cost-ratio thresholds, EVSI, failure-finding intervals | not knowing the rate costs a hidden failure's test interval 1007 → 31 FH | [06](../../notebooks/06_deciding_under_uncertainty.ipynb) |
| 7 | [How big is a big fleet?](07-how-big-is-a-big-fleet.md) | the algebra of convergence, bias against overconfidence, robust pooling | with a good prior no business-jet fleet is big; a wrong prior is overconfident | [07](../../notebooks/07_how_big_is_a_big_fleet.ipynb) |
| 8 | [A dataset with real ground truth](08-a-dataset-with-real-ground-truth.md) | renewal theory, stationary counts, under-dispersion, external data handling | on C-MAPSS, Poisson intervals over-cover and pooling wins on short windows | [08](../../notebooks/08_a_dataset_with_real_ground_truth.ipynb) |
| 9 | [What is (and isn't) new](09-what-is-and-isnt-new.md) | method, application or measurement; closest work; a novelty statement and its search plan | the methods are established; the claims are measurements, stated narrowly | [09](../../notebooks/09_what_is_and_isnt_new.ipynb) |

The report is [`docs/report.md`](../report.md); the boundaries of what is modelled are in
[`docs/scope.md`](../scope.md); each module has a theory note in [`docs/`](../).

## Suggested paths

* **Reliability engineer, short on time:** Lessons 1, 2 and 5, then the report.
* **Statistician:** Lessons 3, 4 and 7, then the related-work note.
* **Manager of a small fleet's reliability programme:** Lesson 1, the report's summary and its
  section 4.
