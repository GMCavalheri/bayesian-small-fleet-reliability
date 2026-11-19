"""Download NASA C-MAPSS (turbofan run-to-failure, FD001-FD004) into data/external/cmapss/.

Source: NASA Ames Prognostics Center of Excellence data repository, PHM data bucket
("6. Turbofan Engine Degradation Simulation Data Set", 12.4 MB). Public data of the US
Government; no licence is stated. Cite A. Saxena and K. Goebel (2008), "Turbofan Engine
Degradation Simulation Data Set", NASA Ames Prognostics Data Repository, and A. Saxena,
K. Goebel, D. Simon and N. Eklund, "Damage propagation modeling for aircraft engine
run-to-failure simulation", PHM08, 2008.

Every file is checked against the SHA-256 recorded below (taken from the download of
2026-11-19); the data are never committed (data/external/ is in .gitignore).
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external" / "cmapss"
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
FILES_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    outer = DEST / "turbofan_degradation.zip"
    if not outer.exists():
        print(f"Downloading {URL}")
        urlretrieve(URL, outer)
    if sha256(outer) != OUTER_SHA256:
        print(f"Checksum mismatch for {outer.name}: {sha256(outer)}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(outer) as z:
        inner_name = next(n for n in z.namelist() if n.endswith("CMAPSSData.zip"))
        inner = DEST / "CMAPSSData.zip"
        inner.write_bytes(z.read(inner_name))
    with zipfile.ZipFile(inner) as z:
        for name in FILES_SHA256:
            member = next(n for n in z.namelist() if n.endswith(name))
            (DEST / name).write_bytes(z.read(member))
    bad = [n for n, h in FILES_SHA256.items() if sha256(DEST / n) != h]
    if bad:
        print(f"Checksum mismatch: {bad}", file=sys.stderr)
        return 1
    print(f"C-MAPSS training files verified in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
