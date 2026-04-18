"""
Patch for lambeq's BobcatParser model download issue.

The model server at qnlp.cambridgequantum.com is permanently offline
(Cambridge Quantum rebranded to Quantinuum). This script downloads
the Bobcat model from the Wayback Machine archive and caches it locally.

Run this ONCE before using BobcatParser:
    conda activate qnlp
    python scripts/download_bobcat.py

After running, BobcatParser will work offline using the cached model.
"""

import hashlib
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

# ─── Configuration ───
BOBCAT_MODEL_NAME = "bobcat"
CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
MODEL_DIR = CACHE_DIR / "lambeq" / BOBCAT_MODEL_NAME / BOBCAT_MODEL_NAME

# Known working archive URLs (try in order)
ARCHIVE_URLS = [
    # Wayback Machine snapshot
    "https://web.archive.org/web/2024/https://qnlp.cambridgequantum.com/models/bobcat/latest/model.tar.gz",
    # GitHub release artifacts (if available)
    "https://github.com/Quantinuum/lambeq/releases/download/0.5.0/bobcat_model.tar.gz",
]


def check_existing():
    """Check if the model is already cached."""
    version_file = MODEL_DIR / "version.txt"
    if MODEL_DIR.is_dir() and version_file.exists():
        version = version_file.read_text().strip()
        print(f"✓ Bobcat model already cached (version {version})")
        print(f"  Location: {MODEL_DIR}")
        return True
    return False


def download_with_requests(url: str, dest_file) -> bool:
    """Try to download from a URL."""
    import requests
    try:
        print(f"  Trying: {url}")
        response = requests.get(url, stream=True, timeout=30,
                                headers={"user-agent": "lambeq-patch/1.0"})
        if response.status_code != 200:
            print(f"  ✗ HTTP {response.status_code}")
            return False

        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            dest_file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloading: {pct:.1f}%", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def extract_model(tar_path: str):
    """Extract the model tarball to the cache directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting to {MODEL_DIR}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(str(MODEL_DIR), filter="data")
    # Write a version file so lambeq thinks the model is up to date
    version_file = MODEL_DIR / "version.txt"
    if not version_file.exists():
        version_file.write_text("0.5.0-patched\n")
    print(f"  ✓ Model extracted to {MODEL_DIR}")


def main():
    print("=" * 60)
    print("  Bobcat Model Downloader (Offline Patch)")
    print("=" * 60)
    print()

    if check_existing():
        print("\nModel is ready. You can use BobcatParser normally.")
        return 0

    print("Model not found in cache. Attempting download...")
    print()

    # Try each archive URL
    for url in ARCHIVE_URLS:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tmp_path = f.name
            if download_with_requests(url, f):
                try:
                    extract_model(tmp_path)
                    print("\n✓ Bobcat model downloaded and cached successfully!")
                    print("  You can now use BobcatParser normally.")
                    return 0
                except Exception as e:
                    print(f"  ✗ Extraction failed: {e}")
                finally:
                    os.unlink(tmp_path)

    # If all downloads fail, provide manual instructions
    print()
    print("=" * 60)
    print("  AUTOMATIC DOWNLOAD FAILED")
    print("=" * 60)
    print()
    print("The model server is offline and archives are unavailable.")
    print("You have two options:")
    print()
    print("Option 1: Use the WebParser (requires internet, slower)")
    print("  from lambeq import WebParser")
    print("  parser = WebParser()")
    print()
    print("Option 2: Patch lambeq to skip the version check.")
    print("  Run this in your notebook BEFORE importing BobcatParser:")
    print()
    print("  import lambeq.text2diagram.model_based_reader.model_downloader as md")
    print("  md.MODELS_URL = 'https://qnlp.quantinuum.com/models'  # new domain")
    print()
    print("Option 3: Install depccg as alternative parser:")
    print("  pip install depccg")
    print("  from lambeq import DepCCGParser")
    print("  parser = DepCCGParser()")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
