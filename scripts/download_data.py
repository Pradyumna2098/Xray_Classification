#!/usr/bin/env python3
"""Utility for downloading and extracting the Chest X-ray dataset."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

DEFAULT_URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "8f91315f-9649-4d45-8fc4-1f9e50a87eae/file_downloaded"
)
DEFAULT_ARCHIVE_NAME = "chest_xray_pneumonia.zip"
EXPECTED_SUBFOLDERS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Chest X-ray Images (Pneumonia) dataset and prepare the"
            " local datasets/ directory with train/val/test subfolders."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "Source URL for the dataset archive. Defaults to the official "
            "Mendeley Data file download link."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory where the dataset will be extracted (default: datasets).",
    )
    parser.add_argument(
        "--archive-path",
        help=(
            "Optional custom path for the downloaded archive. Defaults to"
            " <output-dir>/<archive-name>."
        ),
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help="Filename to use when saving the downloaded archive.",
    )
    parser.add_argument(
        "--sha256",
        help=(
            "Optional SHA256 checksum for the archive. When provided, the"
            " checksum is verified before extraction."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download and re-extract the archive even if files already exist."
        ),
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url}\n -> {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        block_size = 1024 * 8
        total_size = response.length or 0
        downloaded = 0
        while True:
            block = response.read(block_size)
            if not block:
                break
            output.write(block)
            downloaded += len(block)
            if total_size:
                percent = downloaded * 100 // total_size
                print(f"\rDownloaded {downloaded} / {total_size} bytes ({percent}%)", end="")
        print()
    print("Download completed.")


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_hash: str) -> bool:
    print(f"Verifying checksum for {path} ...")
    actual = compute_sha256(path)
    if actual.lower() != expected_hash.lower():
        print(
            "Checksum mismatch!",
            f"Expected: {expected_hash.lower()}",
            f"Actual:   {actual.lower()}",
            sep="\n",
        )
        return False
    print("Checksum verified.")
    return True


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    print(f"Extracting {archive_path} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(temp_path)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                archive.extractall(temp_path)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        dataset_root = locate_dataset_root(temp_path)
        if dataset_root is None:
            raise FileNotFoundError(
                "Unable to locate train/val/test folders inside the archive."
            )
        for subfolder in EXPECTED_SUBFOLDERS:
            source = dataset_root / subfolder
            destination = target_dir / subfolder
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(source), destination)
        # Clean up to avoid leaving empty directory structures inside target
        return dataset_root


def locate_dataset_root(base_path: Path) -> Optional[Path]:
    # First, check the base directory itself.
    if all((base_path / name).is_dir() for name in EXPECTED_SUBFOLDERS):
        return base_path
    # Then check immediate subdirectories.
    for child in base_path.iterdir():
        if child.is_dir() and all((child / name).is_dir() for name in EXPECTED_SUBFOLDERS):
            return child
    return None


def ensure_structure(target_dir: Path, force: bool) -> bool:
    """Return True if extraction is needed."""
    existing = [name for name in EXPECTED_SUBFOLDERS if (target_dir / name).exists()]
    if not existing:
        return True

    if set(existing) == set(EXPECTED_SUBFOLDERS) and not force:
        print("Dataset already present. Skipping extraction.")
        return False

    if not force:
        raise RuntimeError(
            "Partial dataset detected. Rerun with --force to overwrite the existing"
            " folders or remove them manually."
        )

    for name in EXPECTED_SUBFOLDERS:
        path = target_dir / name
        if path.exists():
            shutil.rmtree(path)
    return True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    archive_path = (
        Path(args.archive_path).expanduser().resolve()
        if args.archive_path
        else (output_dir / args.archive_name)
    )

    if args.force and archive_path.exists():
        print("--force specified, removing existing archive before download.")
        archive_path.unlink()

    if archive_path.exists():
        print(f"Archive already exists at {archive_path}.")
        if args.sha256:
            if not verify_checksum(archive_path, args.sha256):
                print("Existing archive failed checksum verification; re-downloading.")
                download_file(args.url, archive_path)
                if args.sha256 and not verify_checksum(archive_path, args.sha256):
                    print("Checksum verification failed after re-download.")
                    return 1
        else:
            print("Skipping download because archive already exists. Use --force to re-download.")
    else:
        download_file(args.url, archive_path)
        if args.sha256 and not verify_checksum(archive_path, args.sha256):
            print("Checksum verification failed after download.")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        needs_extraction = ensure_structure(output_dir, args.force)
    except RuntimeError as error:
        print(error)
        return 1
    if not needs_extraction:
        return 0

    try:
        extract_archive(archive_path, output_dir)
    except Exception as error:  # noqa: BLE001
        print(f"Failed to extract archive: {error}")
        return 1

    print(
        "Dataset extraction completed. You should now have train/val/test"
        f" folders inside {output_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
