from typing import Any, cast

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig


def create_loaders(cfg: DatasetConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    dataset = cast("DatasetDict", load_dataset(cfg.name))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = dataset[cfg.train_name].with_format("torch", device=device)
    val_dataset = dataset[cfg.val_name].with_format("torch", device=device)
    train_loader = DataLoader(
        cast("Dataset[Any]", train_dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_loaders,
    )
    val_loader = DataLoader(
        cast("Dataset[Any]", val_dataset),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_loaders,
    )
    return train_loader, val_loader
