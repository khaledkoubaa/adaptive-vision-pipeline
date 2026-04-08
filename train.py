#!/usr/bin/env python3
"""Main training entry point.

Usage:
    python train.py --config configs/detect.yaml
    python train.py --config configs/segment.yaml
    python train.py --config configs/classify.yaml
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import CocoDataset, collate_fn
from src.engine import run_training
from src.models import build_model
from src.preprocessing import build_transforms
from src.visualize import plot_training_curves


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train the vision pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.seed)

    # Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Task: {cfg.task} | Device: {device} | Experiment: {cfg.experiment_name}")

    # Transforms
    train_transforms = build_transforms(cfg.task, train=True)
    val_transforms = build_transforms(cfg.task, train=False)

    # Datasets
    train_ds = CocoDataset(
        images_dir=cfg.data.train_images,
        annotation_file=cfg.data.train_annotations,
        task=cfg.task,
        transforms=train_transforms,
    )
    val_ds = CocoDataset(
        images_dir=cfg.data.val_images,
        annotation_file=cfg.data.val_annotations,
        task=cfg.task,
        transforms=val_transforms,
    )
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Dataloaders
    is_cls = cfg.task == "classify"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=None if is_cls else collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=None if is_cls else collate_fn,
        pin_memory=True,
    )

    # Model
    model = build_model(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model.backbone} | Trainable params: {n_params:,}")

    # Train
    history = run_training(cfg, model, train_loader, val_loader, device)

    # Save training curves
    plot_training_curves(history, save_path=str(cfg.exp_dir / "training_curves.png"))
    print(f"\nResults saved to {cfg.exp_dir}")


if __name__ == "__main__":
    main()
