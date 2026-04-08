"""Configuration management — loads YAML configs into typed dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    train_images: str = "data/images/train"
    val_images: str = "data/images/val"
    train_annotations: str = "data/annotations/train.json"
    val_annotations: str = "data/annotations/val.json"
    num_workers: int = 4


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 21
    enable_masks: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    momentum: float = 0.9
    amp: bool = True
    grad_clip: float = 0.0
    early_stopping_patience: int = 10


@dataclass
class Config:
    task: str = "detect"
    experiment_name: str = "default"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file, mapping nested dicts to dataclasses."""
        with open(path) as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        data_cfg = DataConfig(**raw.pop("data", {}))
        model_cfg = ModelConfig(**raw.pop("model", {}))
        train_cfg = TrainConfig(**raw.pop("train", {}))

        return cls(data=data_cfg, model=model_cfg, train=train_cfg, **raw)

    @property
    def exp_dir(self) -> Path:
        """Return the experiment output directory, creating it if needed."""
        p = Path(self.output_dir) / self.experiment_name
        p.mkdir(parents=True, exist_ok=True)
        return p
