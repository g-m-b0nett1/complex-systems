"""Download the two SNAP networks used in the network study into data/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "facebook_combined.txt.gz": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "roadNet-PA.txt.gz": "https://snap.stanford.edu/data/roadNet-PA.txt.gz",
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = DATA / name
        if target.exists() or (DATA / name.removesuffix(".gz")).exists():
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, target)


if __name__ == "__main__":
    main()
