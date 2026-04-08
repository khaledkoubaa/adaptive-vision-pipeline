#!/usr/bin/env python3
"""Standalone evaluation script — loads a trained checkpoint and reports metrics.

Usage:
    python evaluate.py --config configs/detect.yaml --checkpoint outputs/detection_baseline/best_model.pth
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import CocoDataset, collate_fn
from src.engine import validate_cls, validate_det
from src.models import build_model
from src.preprocessing import build_transforms
from src.visualize import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Validation set
    val_transforms = build_transforms(cfg.task, train=False)
    val_ds = CocoDataset(
        images_dir=cfg.data.val_images,
        annotation_file=cfg.data.val_annotations,
        task=cfg.task,
        transforms=val_transforms,
    )
    is_cls = cfg.task == "classify"
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=None if is_cls else collate_fn,
    )

    # Evaluate
    if is_cls:
        metrics = validate_cls(model, val_loader, device)
        # Confusion matrix
        cm = metrics.pop("_confusion_matrix", None)
        if cm is not None:
            class_names = [c["name"] for c in val_ds.categories]
            plot_confusion_matrix(
                np.array(cm), class_names,
                save_path=str(cfg.exp_dir / "confusion_matrix.png"),
            )
    else:
        metrics = validate_det(model, val_loader, device)

    # Print and save
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    results_path = cfg.exp_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved results -> {results_path}")


if __name__ == "__main__":
    main()
