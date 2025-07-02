
import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any, List

def unzip_archives(source_dir: Path, target_dir: Path, force: bool = False) -> None:
    """Unzip all zip files from source_dir into subfolders of target_dir.

    Args:
        source_dir: Directory containing zip files
        target_dir: Directory to extract archives to
        force: If True, overwrite existing extracted folders
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in source_dir.glob("*.zip"):
        dest = target_dir / zip_path.stem
        if dest.exists() and not force:
            print(f"Skipping {zip_path.name} - already extracted at {dest}")
            continue

        with zipfile.ZipFile(zip_path, "r") as archive:
            dest.mkdir(parents=True, exist_ok=True)
            archive.extractall(dest)

def main() -> None:
    parser = argparse.ArgumentParser(description="Unzip archives")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing zip files or images",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory to extract archives and store results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted folders",
    )
    args = parser.parse_args()

    unzip_archives(args.source_dir, args.target_dir, args.force)

    print("Done")