"""Matplotlib style shared by every figure in the repository (same as the CPS series)."""

from __future__ import annotations

import matplotlib.pyplot as plt

FREQUENTIST_COLOR = "#c8452c"
"""The classical, large-fleet method."""
BAYES_COLOR = "#2f6db5"
"""Bayesian estimate with a prior borrowed from the large fleet."""
POOLED_COLOR = "#3a8f5c"
"""Hierarchical (partial pooling) estimate."""
NEUTRAL_COLOR = "#555555"
SERIES_COLORS = (FREQUENTIST_COLOR, BAYES_COLOR, POOLED_COLOR, "#e08a1e", "#7a4fa3")


def use_style() -> None:
    """A clean, consistent style for every figure in the repository."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "svg.hashsalt": "smallfleet",
        }
    )
