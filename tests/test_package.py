import tomllib
from pathlib import Path

import smallfleet

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert smallfleet.__version__ == project["version"]
