"""Siamese convolutional neural network used in the source notebook."""

from __future__ import annotations

import torch
from torch import nn


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_once(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        features = self.pool(features).flatten(start_dim=1)
        return self.embedding(features)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        embedding1 = self.forward_once(image1)
        embedding2 = self.forward_once(image2)
        combined = torch.cat(
            [torch.abs(embedding1 - embedding2), embedding1 * embedding2],
            dim=1,
        )
        return self.classifier(combined)
