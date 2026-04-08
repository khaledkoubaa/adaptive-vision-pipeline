"""Object detection model (Faster R-CNN) with configurable backbone."""

from torch import nn
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import ModelConfig


def build_detector(model_cfg: ModelConfig) -> nn.Module:
    """Build a Faster R-CNN detector.

    Uses a ResNet-50-FPN backbone pretrained on COCO, then replaces the
    box predictor head to match the target number of classes.
    """
    if model_cfg.backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if model_cfg.pretrained else None,
        )
    elif model_cfg.backbone == "resnet50_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if model_cfg.pretrained else None,
        )
    else:
        raise ValueError(
            f"Detection supports resnet50 / resnet50_v2 backbones, "
            f"got {model_cfg.backbone!r}"
        )

    # Replace the box predictor head for custom class count
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, model_cfg.num_classes
    )

    return model
