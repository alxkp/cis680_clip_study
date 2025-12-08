from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import draccus
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm

from engineered_latents.benchmarking import print_benchmark_summary, run_all_benchmarks
from engineered_latents.config import (
    CKALossConfig,
    ClipConfig,
    MainConfig,
    NCutLossConfig,
    SVDLossConfig,
)
from engineered_latents.context import aim_context, clip_context, get_clip, track
from engineered_latents.data import create_loaders
from engineered_latents.losses.cluster_loss import (
    cka_alignment_loss,
    cross_covariance_svd_loss,
)
from engineered_latents.visualization import log_ncut_visualization

draccus.set_config_type("toml")


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert config dataclass to dict with serializable values."""
    d = asdict(cfg)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    return _convert(d)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(model: nn.Module, cfg: MainConfig) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer == "adam":
        return Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "adamw":
        return AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "sgd":
        return SGD(
            params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def get_scheduler(
    optimizer: torch.optim.Optimizer, cfg: MainConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create learning rate scheduler based on config."""
    if cfg.train.scheduler == "constant":
        return None

    total_epochs = cfg.train.n_epochs
    warmup_epochs = cfg.train.warmup_epochs

    if cfg.train.scheduler == "cosine":
        # Cosine annealing after warmup
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=cfg.train.min_lr,
        )
    elif cfg.train.scheduler == "linear":
        # Linear decay after warmup
        def linear_decay(epoch: int) -> float:
            if epoch < warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max(cfg.train.min_lr / cfg.train.lr, 1.0 - progress)

        return LambdaLR(optimizer, linear_decay)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")

    # Add warmup
    def warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)

    # Combine warmup with main scheduler using SequentialLR
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )


def compute_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    cfg: MainConfig,
) -> torch.Tensor:
    """Compute loss based on config."""
    loss_cfg = cfg.train.loss

    if isinstance(loss_cfg, CKALossConfig):
        return cka_alignment_loss(image_features, text_features, center=loss_cfg.center)
    elif isinstance(loss_cfg, SVDLossConfig):
        return cross_covariance_svd_loss(
            image_features,
            text_features,
            k=loss_cfg.k,
            alpha=loss_cfg.alpha,
            beta=loss_cfg.beta,
            normalize=loss_cfg.normalize,
        )
    elif isinstance(loss_cfg, NCutLossConfig):
        # NCut loss requires eigenvalues from normalized Laplacian
        # For now, we use SVD as a proxy
        return cross_covariance_svd_loss(
            image_features,
            text_features,
            k=loss_cfg.k,
            alpha=loss_cfg.alpha,
            beta=loss_cfg.beta,
        )
    else:
        raise ValueError(f"Unknown loss config: {loss_cfg}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    cfg: MainConfig,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint."""
    cfg.checkpoint.save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
    }

    # Save periodic checkpoint
    path = cfg.checkpoint.save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    # Save best checkpoint
    if is_best and cfg.checkpoint.save_best:
        best_path = cfg.checkpoint.save_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint: {best_path}")

    # Cleanup old checkpoints (keep only max_checkpoints most recent)
    checkpoints = sorted(cfg.checkpoint.save_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > cfg.checkpoint.max_checkpoints:
        for old_ckpt in checkpoints[: -cfg.checkpoint.max_checkpoints]:
            old_ckpt.unlink()


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: MainConfig,
) -> tuple[int, float]:
    """Load checkpoint if it exists. Returns (start_epoch, best_val_loss)."""
    best_path = cfg.checkpoint.save_dir / "best.pt"
    if not best_path.exists():
        return 0, float("inf")

    checkpoint = torch.load(best_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Resumed from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def extract_features(batch: dict, ctx) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract image and text features from a batch."""
    model = ctx.model
    processor = ctx.processor
    device = ctx.device

    # Handle different batch formats
    if "images" in batch:
        images = batch["images"]
        texts = batch["captions"]
    elif "image_0" in batch:
        # Winoground-style batch
        images = batch["image_0"]  # just use first image for now
        texts = batch["caption_0"]
    else:
        raise ValueError(f"Unknown batch format: {batch.keys()}")

    # Process inputs
    img_inputs = processor(images=images, return_tensors="pt", padding=True)
    txt_inputs = processor(text=texts, return_tensors="pt", padding=True)

    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

    # Get features
    image_features = model.get_image_features(**img_inputs)
    text_features = model.get_text_features(**txt_inputs)

    # Normalize
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

    return image_features, text_features


def compute_clip_score(image_features: torch.Tensor, text_features: torch.Tensor) -> float:
    """Compute mean CLIP score (cosine similarity) for paired image-text embeddings."""
    # Features are already normalized, so dot product = cosine similarity
    similarities = (image_features * text_features).sum(dim=-1)
    return similarities.mean().item()


def train_epoch(
    train_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: MainConfig,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, avg_clip_score)."""
    model.train()
    ctx = get_clip()
    total_loss = 0.0
    total_clip_score = 0.0
    n_batches = 0

    # Use fp16 on CUDA for memory efficiency (bf16 doesn't support SVD)
    use_amp = ctx.device.type == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    assert isinstance(pbar, tqdm)  # HACK: another fix - if this breaks, we have problems anyway

    for batch in pbar:
        optimizer.zero_grad()

        with autocast_ctx:
            image_features, text_features = extract_features(batch, ctx)
            loss = compute_loss(image_features, text_features, cfg)
            clip_score = compute_clip_score(image_features, text_features)

        loss.backward()

        # Gradient clipping
        if cfg.train.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip_norm
            )

        optimizer.step()

        total_loss += loss.item()
        total_clip_score += clip_score
        n_batches += 1
        pbar.set_postfix(loss=loss.item(), clip=clip_score)

    return total_loss / max(n_batches, 1), total_clip_score / max(n_batches, 1)


@torch.no_grad()
def validate(val_loader, model: nn.Module, cfg: MainConfig) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, avg_clip_score)."""
    model.eval()
    ctx = get_clip()
    total_loss = 0.0
    total_clip_score = 0.0
    n_batches = 0

    # Use fp16 on CUDA for memory efficiency (bf16 doesn't support SVD)
    use_amp = ctx.device.type == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    val_pbar = tqdm(val_loader, desc="Validation")
    assert isinstance(val_pbar, tqdm)  # HACK: another fix - if this breaks, we have problems anyway
    for batch in val_pbar:
        with autocast_ctx:
            image_features, text_features = extract_features(batch, ctx)
            loss = compute_loss(image_features, text_features, cfg)
            clip_score = compute_clip_score(image_features, text_features)

        total_loss += loss.item()
        total_clip_score += clip_score
        n_batches += 1

    return total_loss / max(n_batches, 1), total_clip_score / max(n_batches, 1)


def run_training(cfg: MainConfig) -> None:
    """Full training loop with all best practices."""
    set_seed(cfg.seed)

    # Create data loaders
    train_loader, val_loader = create_loaders(cfg.dataset)

    ctx = get_clip()
    model = ctx.model.to(ctx.device)

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    # Try to resume from checkpoint
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, cfg)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.train.n_epochs):
        # Training
        train_loss, train_clip = train_epoch(train_loader, model, optimizer, cfg, epoch)
        track(train_loss, "train/loss", step=epoch)
        track(train_clip, "train/clip_score", step=epoch)
        track(optimizer.param_groups[0]["lr"], "train/lr", step=epoch)

        if epoch % cfg.print_every_n == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, clip={train_clip:.4f}")

        # Validation
        if epoch % cfg.train.val_every_n_epochs == 0:
            val_loss, val_clip = validate(val_loader, model, cfg)
            track(val_loss, "val/loss", step=epoch)
            track(val_clip, "val/clip_score", step=epoch)

            if epoch % cfg.print_every_n == 0:
                print(f"Epoch {epoch}: val_loss={val_loss:.4f}, clip={val_clip:.4f}")

            # Check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if (
                cfg.train.early_stopping_patience is not None
                and epochs_without_improvement >= cfg.train.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {epochs_without_improvement} epochs)"
                )
                break

            # Checkpointing
            if epoch % cfg.checkpoint.save_every_n_epochs == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_loss, cfg, is_best
                )

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


def run_eval(cfg: MainConfig, step: int | None = None) -> None:
    """Run benchmarks on the model."""
    results = run_all_benchmarks(
        include_winoground=cfg.eval.include_winoground,
        include_coco=cfg.eval.include_coco,
        include_flickr=cfg.eval.include_flickr,
        include_imagenet=cfg.eval.include_imagenet,
        batch_size=cfg.eval.batch_size,
        max_samples=cfg.eval.max_samples,
        track_results=True,
    )
    print_benchmark_summary(results)

    # Log NCut visualization
    print("\n=== NCut Visualization ===")
    log_ncut_visualization(step=step, max_samples=100)


@draccus.wrap()
def main(cfg: MainConfig):
    assert isinstance(cfg.model, ClipConfig)
    set_seed(cfg.seed)

    with aim_context(repo=".", experiment=cfg.experiment) as run:
        run["config"] = config_to_dict(cfg)

        with clip_context(cfg.model):
            if cfg.mode == "eval":
                run_eval(cfg)
            elif cfg.mode == "train":
                run_training(cfg)
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
