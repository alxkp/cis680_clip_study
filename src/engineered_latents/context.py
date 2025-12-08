from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from aim import Run  # type: ignore - this is a valid import
from transformers import CLIPModel, CLIPProcessor

from .config import ClipConfig


@dataclass
class ClipContext:
    model: CLIPModel
    processor: CLIPProcessor
    device: Any = "mps"  # type: ignore (should be DeviceLikeType, but there are import errors)


_clip_ctx: ContextVar[ClipContext | None] = ContextVar("clip_ctx", default=None)
_aim_run: ContextVar[Run | None] = ContextVar("aim_run", default=None)


@contextmanager
def clip_context(cfg: ClipConfig):
    """Stores clip for use in different modules"""

    model: CLIPModel = CLIPModel.from_pretrained(cfg.name)
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(cfg.name)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device = torch.device(device)

    token = _clip_ctx.set(ClipContext(model, processor, device))

    try:
        yield
    finally:
        _clip_ctx.reset(token)


def get_clip() -> ClipContext:
    """Clip context manager getter"""
    if (ctx := _clip_ctx.get()) is None:
        raise RuntimeError("no clip")

    return ctx


@contextmanager
def aim_context(repo: str, experiment: str, **run_kwargs):
    run = Run(repo=repo, experiment=experiment, **run_kwargs)
    assert isinstance(
        run, Run
    )  # naive way to do this, but it should always work, and if it doesnt we have problems anyway
    token = _aim_run.set(run)

    try:
        yield run
    finally:
        run.close()
        _aim_run.reset(token)


def get_run() -> Run:
    if (run := _aim_run.get()) is None:
        raise RuntimeError("run not initialized")
    return run


def track(value, name: str, step: int | None = None, context: dict | None = None):
    """
    convenience wrapper to track without getting run
    recomennd currying with name via functools partial

    example:
    track_loss = partial(track, name='train_loss')
    track_loss(loss.item(), step)
    """
    get_run().track(value, name=name, step=step, context=context)
