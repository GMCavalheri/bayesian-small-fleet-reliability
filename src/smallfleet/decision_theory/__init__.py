"""Extension: decisions under rate uncertainty (PLAN extension 2).

* ``alerts``: when to investigate, as an expected-loss decision, and the expected value of
  waiting a month for more data (preposterior analysis);
* ``intervals``: a failure-finding (inspection) interval for a hidden failure, chosen with
  the whole posterior instead of a point estimate.

Every cost and target is a labelled assumption in ``data/decision_parameters.csv``.
"""

from smallfleet.decision_theory.alerts import (
    AlertCosts,
    bayes_threshold,
    expected_losses,
    value_of_waiting,
)
from smallfleet.decision_theory.intervals import (
    expected_unavailability,
    interval_for_expected_unavailability,
    interval_for_known_rate,
    interval_for_risk,
    unavailability,
)
from smallfleet.decision_theory.parameters import load_decision_parameters

__all__ = [
    "AlertCosts",
    "bayes_threshold",
    "expected_losses",
    "expected_unavailability",
    "interval_for_expected_unavailability",
    "interval_for_known_rate",
    "interval_for_risk",
    "load_decision_parameters",
    "unavailability",
    "value_of_waiting",
]
