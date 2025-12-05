from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import torch
from draccus import ChoiceRegistry
from einops import rearrange
from PIL import Image


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
class TrainConfig:
    lr: float = 1e-4
    n_epochs = 100
    optimizer: Literal["adam", "adamw"] = "adamw"


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

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        example["image_0"] = preprocess_image(example["image_0"])
        example["image_1"] = preprocess_image(example["image_1"])
        return example

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image_0": torch.stack([x["image_0"] for x in batch]),
            "image_1": torch.stack([x["image_1"] for x in batch]),
            "caption_0": [x["caption_0"] for x in batch],
            "caption_1": [x["caption_1"] for x in batch],
        }


@dataclass
@DatasetConfig.register_subclass("whatsup_all_coco_qa_one_obj")
class WhatsupAllConfig(DatasetConfig):
    name: str = "ServiceNow/whatsup_all"
    train_split: str = "COCO_QA_one_obj"
    val_split: str = "COCO_QA_one_obj"

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        example["image_options"] = preprocess_image(example["image_options"])
        return example

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image_0": torch.stack([x["image_options"] for x in batch]),
            "image_1": torch.stack([x["image_options"] for x in batch]),
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

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        example["image"] = preprocess_image(example["image"])
        return example

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "images": torch.stack([x["image"] for x in batch]),
            "captions": [x["sentences"]["raw"][0] for x in batch],  # first caption
        }


@dataclass
@DatasetConfig.register_subclass("flickr30k")
class Flickr30KConfig(DatasetConfig):
    """Flickr30K for retrieval benchmarking."""

    name: str = "nlphuji/flickr30k"
    train_split: str = "train"
    val_split: str = "test"

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        example["image"] = preprocess_image(example["image"])
        return example

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "images": torch.stack([x["image"] for x in batch]),
            "captions": [x["caption"][0] for x in batch],  # first caption
        }


@dataclass
class MainConfig:
    experiment: str = f"engineered_latents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # TODO: add timedate
    model: ModelConfig = field(default_factory=lambda: ClipConfig())
    dataset: DatasetConfig = field(default_factory=lambda: WinogroundConfig())
    train: TrainConfig = field(default_factory=lambda: TrainConfig())

    print_every_n: int = 10
