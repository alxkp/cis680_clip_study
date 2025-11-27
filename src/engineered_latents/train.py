import torch

from .compute_clip_scores import compute_clip_scores
from .context import get_clip


def train_step(batch: tuple[torch.Tensor, torch.Tensor]):
    images, captions = batch
    ctx = get_clip()

    model = ctx.model
    processor = ctx.processor
    device = ctx.device

    clip_scores = compute_clip_scores(images, captions, model, processor, device)
    return clip_scores, None  # TODO: add gradient computation/decide if no jax


def val_step(batch: tuple[torch.Tensor, torch.Tensor]):
    images, captions = batch
    ctx = get_clip()

    model = ctx.model
    processor = ctx.processor
    device = ctx.device

    clip_scores = compute_clip_scores(images, captions, model, processor, device)
    return clip_scores, None  # TODO: add gradient computation/decide if no jax
