from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from draccus import ChoiceRegistry
from einops import rearrange
from PIL import Image

from .utils import OUTPUT_DIR


def preprocess_image(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize, normalize to [-1, 1], and rearrange to CHW."""
    img = img.convert("RGB").resize((size, size))
    tensor = torch.from_numpy(np.array(img)).float()
    tensor = (tensor / 127.5) - 1.0
    return rearrange(tensor, "h w c -> c h w")


@dataclass
class ModelConfig(ChoiceRegistry):
    name: str


@dataclass
@ModelConfig.register_subclass("clip32")
class ClipConfig(ModelConfig):
    name: str = "openai/clip-vit-base-patch32"


@dataclass
class LossConfig(ChoiceRegistry):
    """Base class for loss function configuration."""

    name: str


@dataclass
@LossConfig.register_subclass("cka")
class CKALossConfig(LossConfig):
    """CKA alignment loss."""

    name: str = "cka"
    center: bool = True


@dataclass
@LossConfig.register_subclass("svd")
class SVDLossConfig(LossConfig):
    """Cross-covariance SVD loss."""

    name: str = "svd"
    k: int = 3
    alpha: float = 1.0
    beta: float = 0.5
    normalize: bool = True


@dataclass
@LossConfig.register_subclass("ncut")
class NCutLossConfig(LossConfig):
    """Normalized cut cluster loss."""

    name: str = "ncut"
    k: int = 3
    alpha: float = 1.0
    beta: float = 0.5


@dataclass
@LossConfig.register_subclass("clip_svd")
class ClipSVDLossConfig(LossConfig):
    """Combined CLIP score + SVD loss."""

    name: str = "clip_svd"
    k: int = 3
    alpha: float = 1.0  # spectral gap weight
    beta: float = 0.5   # magnitude weight
    gamma: float = 1.0  # CLIP score weight
    normalize: bool = True


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    save_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "checkpoints")
    save_every_n_epochs: int = 10
    save_best: bool = True
    max_checkpoints: int = 3  # keep only N most recent checkpoints


@dataclass
class TrainConfig:
    lr: float = 1e-4
    n_epochs: int = 100
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    weight_decay: float = 0.01
    # Scheduler
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    # Gradient clipping
    grad_clip_norm: float | None = 1.0
    # Early stopping
    early_stopping_patience: int | None = 10  # None to disable
    # Validation
    val_every_n_epochs: int = 1
    # Loss
    loss: LossConfig = field(default_factory=lambda: CKALossConfig())


@dataclass
class DatasetConfig(ChoiceRegistry):
    name: str
    num_loaders: int = 2
    batch_size: int = 32
    train_split: str = "train"
    val_split: str = "validation"

    @abstractmethod
    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Dataset-specific batching logic."""
        ...

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """Optional per-example preprocessing. Override as needed."""
        return example


@dataclass
@DatasetConfig.register_subclass("winoground")
class WinogroundConfig(DatasetConfig):
    name: str = "facebook/winoground"
    train_split: str = "test"
    val_split: str = "test"

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image_0": [x["image_0"].convert("RGB") for x in batch],
            "image_1": [x["image_1"].convert("RGB") for x in batch],
            "caption_0": [x["caption_0"] for x in batch],
            "caption_1": [x["caption_1"] for x in batch],
        }


@dataclass
@DatasetConfig.register_subclass("whatsup_all_coco_qa_one_obj")
class WhatsupAllConfig(DatasetConfig):
    name: str = "ServiceNow/whatsup_all"
    train_split: str = "COCO_QA_one_obj"
    val_split: str = "COCO_QA_one_obj"

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        img = [x["image_options"].convert("RGB") for x in batch]
        return {
            "image_0": img,
            "image_1": img,
            "caption_0": [x["caption_options"][0] for x in batch],
            "caption_1": [x["caption_options"][1] for x in batch],
        }


@dataclass
@DatasetConfig.register_subclass("coco_captions")
class COCOCaptionsConfig(DatasetConfig):
    """COCO Captions for retrieval benchmarking."""

    name: str = "HuggingFaceM4/COCO"
    train_split: str = "train"
    val_split: str = "test"

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "images": [x["image"].convert("RGB") for x in batch],
            "captions": [x["sentences"]["raw"][0] for x in batch],  # first caption
        }


@dataclass
@DatasetConfig.register_subclass("flickr30k")
class Flickr30KConfig(DatasetConfig):
    """Flickr30K for retrieval benchmarking."""

    name: str = "nlphuji/flickr30k"
    train_split: str = "train"
    val_split: str = "test"

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "images": [x["image"].convert("RGB") for x in batch],
            "captions": [x["caption"][0] for x in batch],  # first caption
        }


@dataclass
class EvalConfig:
    """Evaluation/benchmarking configuration."""

    include_winoground: bool = True
    include_coco: bool = True
    include_flickr: bool = False
    include_imagenet: bool = False
    batch_size: int = 32
    max_samples: int | None = None  # None for full dataset


@dataclass
class MainConfig:
    # Run mode
    mode: Literal["train", "eval"] = "eval"
    experiment: str = field(
        default_factory=lambda: f"engineered_latents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Sub-configs
    model: ModelConfig = field(default_factory=lambda: ClipConfig())
    dataset: DatasetConfig = field(default_factory=lambda: WinogroundConfig())
    train: TrainConfig = field(default_factory=lambda: TrainConfig())
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    eval: EvalConfig = field(default_factory=lambda: EvalConfig())

    # Logging
    print_every_n: int = 10
    seed: int = 42
