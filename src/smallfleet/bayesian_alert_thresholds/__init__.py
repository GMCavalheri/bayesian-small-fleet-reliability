"""Phase 4: alert thresholds as posterior probabilities of exceeding a target rate."""

from smallfleet.bayesian_alert_thresholds.operating import (
    Characteristics,
    Histories,
    bayes_window_rule,
    characteristics,
    classical_rule,
    simulate_histories,
)
from smallfleet.bayesian_alert_thresholds.rules import (
    alert_count_threshold,
    discounted_exceedance,
    exceedance_probability,
    window_exceedance,
    window_false_alarm_rate,
)

__all__ = [
    "Characteristics",
    "Histories",
    "alert_count_threshold",
    "bayes_window_rule",
    "characteristics",
    "classical_rule",
    "discounted_exceedance",
    "exceedance_probability",
    "simulate_histories",
    "window_exceedance",
    "window_false_alarm_rate",
]
