from dataclasses import dataclass
from typing import Literal
from draccus import ChoiceRegistry

@dataclass 
class ModelConfig(ChoiceRegistry):
    pass

@dataclass
@ModelConfig.register_subclass('clip32')
class ClipConfig(ModelConfig):
    name:str = "openai/clip-vit-base-patch32"

@dataclass
class TrainConfig:
    lr: float = 1e-4
    optimizer: Literal['adam','adamw'] = "adamw"

@dataclass
class DatasetConfig(ChoiceRegistry):
    name: str

@dataclass
@DatasetConfig.register_subclass("winoground")
class WinogroundConfig(DatasetConfig):
    name: str =  'facebook/winoground'

@dataclass
class MainConfig:
    model: ModelConfig = ClipConfig()
    dataset: DatasetConfig = WinogroundConfig()
    train: TrainConfig = TrainConfig()