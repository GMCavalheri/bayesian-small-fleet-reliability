import matplotlib.pyplot as plt

from smallfleet import plotting


def test_style_sets_the_shared_rc_params() -> None:
    plotting.use_style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["savefig.bbox"] == "tight"
    assert len(set(plotting.SERIES_COLORS)) == len(plotting.SERIES_COLORS)
