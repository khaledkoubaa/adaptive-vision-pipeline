"""Plotting utilities for training curves, confusion matrices, and predictions."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and metric curves over epochs."""
    epochs = range(1, len(next(iter(history.values()))) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    for key in history:
        if "loss" in key:
            axes[0].plot(epochs, history[key], label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics
    for key in history:
        if "loss" not in key and not key.startswith("_"):
            axes[1].plot(epochs, history[key], label=key)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Metrics Over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves -> {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix -> {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Detection / Segmentation visualisation
# ------------------------------------------------------------------

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: Optional[List[str]] = None,
    masks: Optional[torch.Tensor] = None,
    score_thr: float = 0.5,
) -> np.ndarray:
    """Overlay bounding boxes (and optionally masks) onto an image."""
    vis = image.copy()

    for i in range(boxes.size(0)):
        if scores[i] < score_thr:
            continue

        color = COLORS[int(labels[i]) % len(COLORS)]
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label_text = f"{labels[i].item()}"
        if class_names and labels[i].item() < len(class_names):
            label_text = class_names[labels[i].item()]
        label_text += f" {scores[i]:.2f}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label_text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Overlay mask
        if masks is not None:
            mask = masks[i]
            if mask.dim() == 3:
                mask = mask[0]
            binary = (mask > 0.5).numpy().astype(np.uint8)
            colored = np.zeros_like(vis)
            colored[:] = color
            vis = np.where(binary[..., None], cv2.addWeighted(vis, 0.6, colored, 0.4, 0), vis)

    return vis


def save_prediction_grid(
    images: List[np.ndarray],
    save_path: str,
    ncols: int = 4,
) -> None:
    """Arrange prediction visualisations in a grid and save."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved prediction grid -> {save_path}")
    plt.close(fig)
