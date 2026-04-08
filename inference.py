#!/usr/bin/env python3
"""Run inference on new images.

Usage:
    # Single image
    python inference.py --config configs/detect.yaml \\
        --checkpoint outputs/detection_baseline/best_model.pth \\
        --input path/to/image.jpg

    # Directory of images
    python inference.py --config configs/segment.yaml \\
        --checkpoint outputs/segmentation_baseline/best_model.pth \\
        --input path/to/images/ --output results/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import Config
from src.models import build_model
from src.preprocessing import build_transforms
from src.visualize import draw_detections, save_prediction_grid


def load_model(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def classify_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transforms,
    device: torch.device,
    class_names: list,
) -> str:
    """Classify a single image and return the predicted label."""
    img_t, _ = transforms(image, 0)
    img_t = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
    idx = logits.argmax(1).item()
    label = class_names[idx] if idx < len(class_names) else str(idx)
    conf = torch.softmax(logits, 1)[0, idx].item()
    return f"{label} ({conf:.2%})"


def detect_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transforms,
    device: torch.device,
    class_names: list,
    score_thr: float = 0.5,
) -> np.ndarray:
    """Detect objects in a single image and return the annotated image."""
    img_t, _ = transforms(image, {})
    with torch.no_grad():
        outputs = model([img_t.to(device)])[0]

    outputs = {k: v.cpu() for k, v in outputs.items()}
    vis = draw_detections(
        image,
        boxes=outputs["boxes"],
        labels=outputs["labels"],
        scores=outputs["scores"],
        class_names=class_names,
        masks=outputs.get("masks"),
        score_thr=score_thr,
    )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--score-thr", type=float, default=0.5, help="Detection score threshold")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)

    transforms = build_transforms(cfg.task, train=False)
    class_names = ["background", "rectangle", "circle", "triangle"]

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather images
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

    print(f"Running {cfg.task} inference on {len(image_paths)} image(s)...")
    vis_images = []

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if cfg.task == "classify":
            result = classify_image(model, image, transforms, device, class_names)
            print(f"  {img_path.name}: {result}")
        else:
            vis = detect_image(
                model, image, transforms, device, class_names, args.score_thr,
            )
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            vis_images.append(vis)
            n_det = "detections"
            print(f"  {img_path.name} -> {out_path}")

    # Save a summary grid
    if vis_images:
        save_prediction_grid(
            vis_images[:16],
            save_path=str(output_dir / "prediction_grid.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
