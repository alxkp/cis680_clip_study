from dataclasses import dataclass
from typing import Literal

from draccus import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    pass


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
    train_name: str = "train"
    val_name: str = "validation"


@dataclass
@DatasetConfig.register_subclass("winoground")
class WinogroundConfig(DatasetConfig):
    name: str = "facebook/winoground"
    train_name: str = "test"
    val_name: str = "test"


@dataclass
class MainConfig:
    experiment: str = "engineered_latents"  # TODO: add timedate
    model: ModelConfig = ClipConfig()
    dataset: DatasetConfig = WinogroundConfig()
    train: TrainConfig = TrainConfig()

    print_every_n: int = 10
