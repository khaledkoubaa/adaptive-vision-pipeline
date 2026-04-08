"""Evaluation metrics for classification, detection, and segmentation."""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

def compute_classification_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1, and the confusion matrix."""
    return {
        "val_acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ------------------------------------------------------------------
# Detection / Segmentation
# ------------------------------------------------------------------

def _iou_boxes(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format)."""
    x1 = torch.max(box_a[:, None, 0], box_b[None, :, 0])
    y1 = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    x2 = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    y2 = torch.min(box_a[:, None, 3], box_b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _iou_masks(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of binary masks."""
    a = mask_a.flatten(1).float()
    b = mask_b.flatten(1).float()
    inter = (a[:, None] * b[None, :]).sum(-1)
    union = a.sum(-1)[:, None] + b.sum(-1)[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _average_precision(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
    pred_masks: torch.Tensor = None,
    gt_masks: torch.Tensor = None,
) -> Dict[int, float]:
    """Per-class average precision at a single IoU threshold."""
    all_classes = set(gt_labels.tolist())
    ap_per_class: Dict[int, float] = {}

    for cls in all_classes:
        p_mask = pred_labels == cls
        g_mask = gt_labels == cls
        p_boxes = pred_boxes[p_mask]
        p_scores = pred_scores[p_mask]
        g_boxes = gt_boxes[g_mask]

        if g_boxes.numel() == 0:
            ap_per_class[cls] = 0.0
            continue
        if p_boxes.numel() == 0:
            ap_per_class[cls] = 0.0
            continue

        # Sort predictions by confidence
        order = p_scores.argsort(descending=True)
        p_boxes = p_boxes[order]

        # Compute IoU (use masks when available for segmentation AP)
        if pred_masks is not None and gt_masks is not None:
            p_m = pred_masks[p_mask][order]
            g_m = gt_masks[g_mask]
            ious = _iou_masks(p_m, g_m)
        else:
            ious = _iou_boxes(p_boxes, g_boxes)

        matched = torch.zeros(g_boxes.size(0), dtype=torch.bool)
        tp = torch.zeros(p_boxes.size(0))
        fp = torch.zeros(p_boxes.size(0))

        for i in range(p_boxes.size(0)):
            iou_vals = ious[i]
            best_j = iou_vals.argmax().item()
            if iou_vals[best_j] >= iou_threshold and not matched[best_j]:
                tp[i] = 1
                matched[best_j] = True
            else:
                fp[i] = 1

        tp_cumsum = tp.cumsum(0)
        fp_cumsum = fp.cumsum(0)
        recalls = tp_cumsum / g_boxes.size(0)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # AP via trapezoidal integration
        recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
        precisions = torch.cat([torch.tensor([1.0]), precisions, torch.tensor([0.0])])
        for j in range(precisions.size(0) - 2, -1, -1):
            precisions[j] = max(precisions[j], precisions[j + 1])
        ap = ((recalls[1:] - recalls[:-1]) * precisions[1:]).sum().item()
        ap_per_class[cls] = ap

    return ap_per_class


def compute_detection_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute mAP (and mask-mAP when masks are present)."""
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []
    all_pred_masks, all_gt_masks = [], []
    has_masks = False

    offset = 0
    for pred, gt in zip(predictions, targets):
        n_gt = gt["boxes"].size(0)
        n_pred = pred["boxes"].size(0)

        all_pred_boxes.append(pred["boxes"])
        all_pred_scores.append(pred["scores"])
        all_pred_labels.append(pred["labels"])
        all_gt_boxes.append(gt["boxes"])
        all_gt_labels.append(gt["labels"])

        if "masks" in pred and "masks" in gt:
            has_masks = True
            all_pred_masks.append(pred["masks"][:, 0] if pred["masks"].dim() == 4 else pred["masks"])
            all_gt_masks.append(gt["masks"])

        offset += n_gt

    pred_boxes = torch.cat(all_pred_boxes) if all_pred_boxes else torch.zeros(0, 4)
    pred_scores = torch.cat(all_pred_scores) if all_pred_scores else torch.zeros(0)
    pred_labels = torch.cat(all_pred_labels) if all_pred_labels else torch.zeros(0, dtype=torch.long)
    gt_boxes = torch.cat(all_gt_boxes) if all_gt_boxes else torch.zeros(0, 4)
    gt_labels = torch.cat(all_gt_labels) if all_gt_labels else torch.zeros(0, dtype=torch.long)

    ap_dict = _average_precision(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels, iou_threshold,
    )
    mAP = float(np.mean(list(ap_dict.values()))) if ap_dict else 0.0
    metrics: Dict[str, float] = {"mAP": mAP}

    if has_masks:
        pred_m = torch.cat(all_pred_masks)
        gt_m = torch.cat(all_gt_masks)
        mask_ap = _average_precision(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, iou_threshold,
            pred_masks=pred_m, gt_masks=gt_m,
        )
        metrics["mask_mAP"] = float(np.mean(list(mask_ap.values()))) if mask_ap else 0.0

    return metrics
