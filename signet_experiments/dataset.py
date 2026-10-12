"""PyTorch dataset and data-loader construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SignaturePairDataset(Dataset):
    """Load image pairs whose paths are relative to the image dataset root."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        image_dir: Path,
        image_height: int,
        image_width: int,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.pairs.iloc[index]
        paths = (self.image_dir / row["image1"], self.image_dir / row["image2"])
        images = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("L")))
            except (OSError, UnidentifiedImageError) as error:
                raise RuntimeError(f"Could not read signature image: {path}") from error

        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return images[0], images[1], label


def create_data_loader(
    pairs: pd.DataFrame,
    image_dir: Path,
    image_height: int,
    image_width: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SignaturePairDataset(pairs, image_dir, image_height, image_width),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
    )
