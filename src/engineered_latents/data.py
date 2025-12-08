from typing import Any

from datasets import load_dataset
from torch.utils.data import DataLoader

from engineered_latents.config import DatasetConfig


def create_loaders(cfg: DatasetConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    # Load datasets (non-streaming for proper image decoding)
    train_ds = load_dataset(cfg.name, split=cfg.train_split, trust_remote_code=True)
    val_ds = load_dataset(cfg.name, split=cfg.val_split, trust_remote_code=True)

    # Preprocessing happens in collate_fn (on-the-fly per batch)
    train_loader: DataLoader[Any] = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_loaders,
        collate_fn=cfg.collate_fn,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=False,
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
