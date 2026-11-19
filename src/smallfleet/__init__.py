"""Bayesian reliability for small fleets.

The classical alert-level method of airline reliability programs (a control limit on monthly
failure rates per 1000 flight hours) assumes a fleet large enough to log many events. A small
business-jet fleet logs a handful per component per year, and the method becomes unstable.
``smallfleet`` estimates failure rates and alert thresholds from such sparse data with
Bayesian methods: an informative prior from a related large fleet, empirical Bayes and
hierarchical partial pooling across components, and alerts stated as posterior probabilities.
It compares them with the frequentist baseline on synthetic fleets of known true rates.
"""

from importlib.metadata import version

__version__ = version("smallfleet")
