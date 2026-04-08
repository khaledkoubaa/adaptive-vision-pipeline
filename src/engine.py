"""Training and validation loops for all three tasks."""

import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.metrics import compute_classification_metrics, compute_detection_metrics


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

def train_one_epoch_cls(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = torch.stack(images).to(device) if isinstance(images, (list, tuple)) else images.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device) if isinstance(labels, (list, tuple)) else labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "train_loss": running_loss / total,
        "train_acc": correct / total,
    }


@torch.no_grad()
def validate_cls(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = torch.stack(images).to(device) if isinstance(images, (list, tuple)) else images.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device) if isinstance(labels, (list, tuple)) else labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["val_loss"] = running_loss / len(all_labels)
    return metrics


# ------------------------------------------------------------------
# Detection / Segmentation (torchvision API)
# ------------------------------------------------------------------

def train_one_epoch_det(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Dict[str, float]:
    """Train one epoch for Faster R-CNN / Mask R-CNN.

    Torchvision detection models return a loss dict in training mode.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        n_batches += 1

    return {"train_loss": running_loss / max(n_batches, 1)}


@torch.no_grad()
def validate_det(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate detection / segmentation on the validation set."""
    model.eval()
    all_preds: List[Dict] = []
    all_targets: List[Dict] = []

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            all_preds.append({k: v.cpu() for k, v in out.items()})
            all_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tgt.items()})

    return compute_detection_metrics(all_preds, all_targets, iou_threshold)


# ------------------------------------------------------------------
# Unified driver
# ------------------------------------------------------------------

def run_training(
    cfg: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Full training loop with LR scheduling, checkpointing, and early stopping."""

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )

    if cfg.train.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.lr_step_size, gamma=cfg.train.lr_gamma,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.num_epochs,
        )

    scaler = GradScaler(enabled=cfg.train.amp)
    use_amp = cfg.train.amp

    is_cls = cfg.task == "classify"
    train_fn = train_one_epoch_cls if is_cls else train_one_epoch_det
    val_fn = validate_cls if is_cls else validate_det

    history: Dict[str, List[float]] = {}
    best_metric = 0.0
    patience_counter = 0
    monitor = "val_acc" if is_cls else "mAP"

    for epoch in range(1, cfg.train.num_epochs + 1):
        t0 = time.time()

        train_metrics = train_fn(model, train_loader, optimizer, device, scaler, use_amp)
        val_metrics = val_fn(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        # Accumulate history
        for k, v in {**train_metrics, **val_metrics}.items():
            history.setdefault(k, []).append(v)

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        parts = [f"Epoch {epoch}/{cfg.train.num_epochs}"]
        for k, v in {**train_metrics, **val_metrics}.items():
            parts.append(f"{k}={v:.4f}")
        parts.append(f"lr={lr:.2e}  ({elapsed:.1f}s)")
        print(" | ".join(parts))

        # Checkpoint best model
        current = val_metrics.get(monitor, 0.0)
        if current > best_metric:
            best_metric = current
            patience_counter = 0
            ckpt_path = cfg.exp_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> Saved best model ({monitor}={best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg.train.early_stopping_patience} epochs)")
                break

    # Save final model
    torch.save(model.state_dict(), cfg.exp_dir / "final_model.pth")
    return history
