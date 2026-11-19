"""Phase 3: hierarchical partial pooling of failure rates across units (fleets or operators).

``oracle`` computes the exact posterior by two-dimensional quadrature; ``model`` builds and
samples the same model in PyMC, so that MCMC can be checked against the exact answer before
it is trusted on models the oracle cannot handle.
"""

from smallfleet.hierarchical_model.checks import pearson_dispersion, predictive_pvalue
from smallfleet.hierarchical_model.hyperpriors import AlphaBetaHyperprior, MeanCVHyperprior
from smallfleet.hierarchical_model.oracle import HierarchicalPosterior, exact_posterior

__all__ = [
    "AlphaBetaHyperprior",
    "HierarchicalPosterior",
    "MeanCVHyperprior",
    "exact_posterior",
    "pearson_dispersion",
    "predictive_pvalue",
]
