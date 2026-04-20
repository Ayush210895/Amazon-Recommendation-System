#!/usr/bin/env python
"""Download the Amazon All Beauty dataset used by this project."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


DATA_URLS = {
    "reviews": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz",
    "metadata": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz",
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already exists: {destination}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, destination)
    print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw", type=Path)
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Only download the review file.",
    )
    args = parser.parse_args()

    download_file(DATA_URLS["reviews"], args.output_dir / "All_Beauty_5.json.gz")
    if not args.skip_metadata:
        download_file(DATA_URLS["metadata"], args.output_dir / "meta_All_Beauty.json.gz")


if __name__ == "__main__":
    main()
