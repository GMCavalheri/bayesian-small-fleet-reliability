"""Rerun every validation script, in phase order, and report the time each took.

    uv run python validation/run_all.py

Every result in ``validation/results/`` is exact or computed from a seeded generator, so a
rerun reproduces it byte for byte; CI checks that with ``git diff``. Two files are the
documented exceptions: ``hierarchical_mcmc.md`` (NUTS trajectories depend on floating-point
details of the compiled model; agreement with the exact oracle is enforced by the tests
instead) and ``cmapss.md`` (the C-MAPSS data are not in the repository, so CI skips that
study; ``scripts/fetch_cmapss.py`` downloads them). Figures go to ``docs/figures/`` and are
not compared (fonts differ between platforms).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("frequentist.py", "Phase 1: the classical method on small fleets"),
    ("bayes_point.py", "Phase 2: Bayesian estimation with a borrowed prior"),
    ("empirical_bayes.py", "Extra: empirical Bayes across operators"),
    ("hierarchical.py", "Phase 3: hierarchical partial pooling (exact)"),
    ("hierarchical_mcmc.py", "Phase 3: NUTS against the exact posterior"),
    ("alerts.py", "Phase 4: Bayesian alert thresholds"),
    ("decisions.py", "Extra: decisions under rate uncertainty"),
    ("simulation_study.py", "Phase 5: simulation study"),
    ("cmapss.py", "Extra: C-MAPSS (skipped without the data)"),
]


def main() -> int:
    for name, title in SCRIPTS:
        print(f"== {title}", flush=True)
        t0 = time.perf_counter()
        done = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE.parent)
        if done.returncode:
            print(f"   {name} failed (exit {done.returncode})", flush=True)
            return done.returncode
        print(f"   done in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
