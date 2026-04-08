"""Image classification models built on torchvision backbones."""

from torch import nn
from torchvision import models

from src.config import ModelConfig

# Map config backbone names to torchvision weight enums
_BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
    ),
}


def build_classifier(model_cfg: ModelConfig) -> nn.Module:
    """Fine-tune a pretrained backbone by replacing its classification head."""
    if model_cfg.backbone not in _BACKBONES:
        raise ValueError(
            f"Unsupported backbone {model_cfg.backbone!r}. "
            f"Choose from {list(_BACKBONES)}"
        )

    factory, weights = _BACKBONES[model_cfg.backbone]
    model = factory(weights=weights if model_cfg.pretrained else None)

    # Replace the final fully-connected layer
    if model_cfg.backbone.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, model_cfg.num_classes)
    elif model_cfg.backbone.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, model_cfg.num_classes)

    return model
