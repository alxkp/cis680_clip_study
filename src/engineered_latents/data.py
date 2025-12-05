from typing import Any, cast

from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader

from engineered_latents.config import DatasetConfig


def create_loaders(cfg: DatasetConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    # Use streaming=True for lazy loading - data is loaded on-demand
    train_ds = cast(
        "IterableDataset",
        load_dataset(cfg.name, split=cfg.train_split, streaming=True),
    )
    val_ds = cast(
        "IterableDataset",
        load_dataset(cfg.name, split=cfg.val_split, streaming=True),
    )

    # With streaming, map is applied lazily as data is iterated
    train_ds = train_ds.map(cfg.preprocess).with_format("torch")
    val_ds = val_ds.map(cfg.preprocess).with_format("torch")

    # Shuffle with buffer for training (streaming datasets use buffer-based shuffling)
    train_ds = train_ds.shuffle(seed=42, buffer_size=cfg.batch_size * 100)

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        # Note: shuffle=True not supported for IterableDataset, done via .shuffle() above
        num_workers=cfg.num_loaders,
        collate_fn=cfg.collate_fn,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        num_workers=cfg.num_loaders,
        collate_fn=cfg.collate_fn,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import lovely_tensors as lt

    from engineered_latents.config import WhatsupAllConfig

    lt.monkey_patch()
    cfg = WhatsupAllConfig()

    train_loader, val_loader = create_loaders(cfg)
    for batch in train_loader:
        print(batch)
        break
    for batch in val_loader:
        print(batch)
        break
