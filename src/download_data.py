"""Script to fetch and verify the official UNSW-NB15 benchmark dataset splits.
Treats data/raw as immutable.
"""

import sys
import requests
from pathlib import Path
from src.config import (
    RAW_TRAIN_CSV,
    RAW_TEST_CSV,
    DATASET_SOURCE_URL_TRAIN,
    DATASET_SOURCE_URL_TEST,
    RAW_DATA_DIR,
)


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with streaming chunks if it does not already exist."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[OK] File already exists: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
        return

    print(f"Downloading {dest_path.name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    
    temp_path.replace(dest_path)
    print(f"[SUCCESS] Downloaded {dest_path.name} ({dest_path.stat().st_size:,} bytes)")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_file(DATASET_SOURCE_URL_TRAIN, RAW_TRAIN_CSV)
    download_file(DATASET_SOURCE_URL_TEST, RAW_TEST_CSV)
    print("[ALL DONE] Raw dataset files verified in data/raw/")


if __name__ == "__main__":
    main()
