"""Instance segmentation model (Mask R-CNN) — extends Faster R-CNN with a
pixel-level mask head.  This is the component that can be *toggled on* when
segmentation boosts downstream performance.
"""

from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.config import ModelConfig


def build_segmentor(model_cfg: ModelConfig) -> nn.Module:
    """Build a Mask R-CNN model for instance segmentation.

    Architecture mirrors the detector (Faster R-CNN) with an additional
    mask prediction branch — keeping the two interchangeable.
    """
    if model_cfg.backbone == "resnet50":
        model = maskrcnn_resnet50_fpn(
            weights="DEFAULT" if model_cfg.pretrained else None,
        )
    elif model_cfg.backbone == "resnet50_v2":
        model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if model_cfg.pretrained else None,
        )
    else:
        raise ValueError(
            f"Segmentation supports resnet50 / resnet50_v2 backbones, "
            f"got {model_cfg.backbone!r}"
        )

    # Replace box predictor (same as detector)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box, model_cfg.num_classes
    )

    # Replace mask predictor for custom class count
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, model_cfg.num_classes
    )

    return model
