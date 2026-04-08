"""Model factory — build the right model for the configured task."""

from torch import nn

from src.config import Config
from src.models.classifier import build_classifier
from src.models.detector import build_detector
from src.models.segmentor import build_segmentor


def build_model(cfg: Config) -> nn.Module:
    """Instantiate a model for *cfg.task* using the specified backbone."""
    if cfg.task == "classify":
        return build_classifier(cfg.model)
    elif cfg.task == "detect":
        return build_detector(cfg.model)
    elif cfg.task == "segment":
        return build_segmentor(cfg.model)
    else:
        raise ValueError(f"Unknown task: {cfg.task!r}")
