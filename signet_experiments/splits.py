"""Person-disjoint dataset discovery, splitting, and pair generation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PAIR_COLUMNS = ("image1", "image2", "label")


def discover_person_ids(data_dir: Path) -> list[str]:
    """Return signer IDs with both genuine and forged image directories."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image dataset directory does not exist: {data_dir}")

    person_ids = sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and not path.name.endswith("_forg")
    )
    if not person_ids:
        raise ValueError(f"No signer directories found in {data_dir}")

    missing_forged = [
        person_id
        for person_id in person_ids
        if not (data_dir / f"{person_id}_forg").is_dir()
    ]
    if missing_forged:
        raise ValueError(
            "Missing forged-image directories for signer(s): "
            + ", ".join(missing_forged)
        )
    return person_ids


def split_person_ids(
    person_ids: Iterable[str],
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split signer IDs into train, validation, and test sets without overlap."""
    ids = sorted(person_ids)
    if len(ids) < 3:
        raise ValueError("At least three signer IDs are required for a split.")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must each be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    random.Random(seed).shuffle(ids)
    test_count = max(1, round(len(ids) * test_fraction))
    validation_count = max(1, round(len(ids) * validation_fraction))
    if test_count + validation_count >= len(ids):
        raise ValueError(
            "The requested split fractions leave no signer IDs for training."
        )

    test_ids = ids[:test_count]
    validation_ids = ids[test_count : test_count + validation_count]
    train_ids = ids[test_count + validation_count :]
    return train_ids, validation_ids, test_ids


def create_pairs(data_dir: Path, person_ids: Iterable[str]) -> list[tuple[str, str, int]]:
    """Create genuine-genuine (0) and genuine-forged (1) pairs."""
    pairs: list[tuple[str, str, int]] = []
    for person_id in sorted(person_ids):
        genuine_dir = data_dir / person_id
        forged_dir = data_dir / f"{person_id}_forg"
        if not genuine_dir.is_dir() or not forged_dir.is_dir():
            raise ValueError(f"Missing genuine or forged directory for {person_id}.")

        genuine_images = sorted(
            path.name
            for path in genuine_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        forged_images = sorted(
            path.name
            for path in forged_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        pairs.extend(
            (f"{person_id}/{first}", f"{person_id}/{second}", 0)
            for index, first in enumerate(genuine_images)
            for second in genuine_images[index + 1 :]
        )
        pairs.extend(
            (f"{person_id}/{genuine}", f"{person_id}_forg/{forged}", 1)
            for genuine in genuine_images
            for forged in forged_images
        )
    if not pairs:
        raise ValueError("No image pairs could be generated for this split.")
    return pairs


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    """Write rows to a UTF-8 CSV file with a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
