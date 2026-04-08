# Adaptive Vision Pipeline

End-to-end computer vision system for **image classification**, **object detection**, and **instance segmentation** — controlled by a single YAML config.

## Architecture

```
Input Image
    │
    ▼
┌──────────────┐
│  ResNet-50   │  ← Shared pretrained backbone
│  + FPN       │
└──────┬───────┘
       │
  ┌────┴──────────────────────┐
  │            │               │
  ▼            ▼               ▼
Classify    Detect          Segment
(FC head)  (Faster R-CNN)  (Mask R-CNN)
```

- **Classification** — fine-tunes a pretrained ResNet-50/EfficientNet with a replaced FC head
- **Detection** — Faster R-CNN with ResNet-50-FPN backbone
- **Segmentation** — Mask R-CNN (extends the detector with a mask branch, toggled via `enable_masks: true`)

## Project Structure

```
├── configs/
│   ├── classify.yaml       # Classification config
│   ├── detect.yaml         # Detection config
│   └── segment.yaml        # Segmentation config (masks enabled)
├── src/
│   ├── config.py           # YAML → typed dataclasses
│   ├── dataset.py          # COCO-format dataset loader
│   ├── preprocessing.py    # OpenCV-based transforms
│   ├── models/
│   │   ├── classifier.py   # Classification model
│   │   ├── detector.py     # Faster R-CNN
│   │   └── segmentor.py    # Mask R-CNN
│   ├── engine.py           # Train/val loops
│   ├── metrics.py          # mAP, IoU, confusion matrix
│   └── visualize.py        # Plots and overlays
├── train.py                # Training entry point
├── evaluate.py             # Standalone evaluation
├── inference.py            # Run on new images
├── data/
│   ├── prepare_data.py     # Synthetic data generator
│   └── README.md           # Data format docs
└── report/
    ├── main.tex            # LaTeX report
    └── references.bib
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Generate a synthetic dataset for testing:

```bash
python data/prepare_data.py --output data --num-images 200
```

Or place your own COCO-format dataset under `data/` (see `data/README.md` for the expected format).

### 3. Train

```bash
# Classification
python train.py --config configs/classify.yaml

# Object detection
python train.py --config configs/detect.yaml

# Instance segmentation (detection + masks)
python train.py --config configs/segment.yaml
```

Training outputs (checkpoints, curves) are saved to `outputs/<experiment_name>/`.

### 4. Evaluate

```bash
python evaluate.py --config configs/detect.yaml \
    --checkpoint outputs/detection_baseline/best_model.pth
```

### 5. Inference

```bash
# Single image
python inference.py --config configs/detect.yaml \
    --checkpoint outputs/detection_baseline/best_model.pth \
    --input path/to/image.jpg

# Directory
python inference.py --config configs/segment.yaml \
    --checkpoint outputs/segmentation_baseline/best_model.pth \
    --input path/to/images/ --output results/
```

## Retraining on New Data

1. Organise images into `data/images/train/` and `data/images/val/`.
2. Create COCO-format annotation files at `data/annotations/train.json` and `data/annotations/val.json`.
3. Update `num_classes` in the config YAML to match your dataset (add 1 for background in detection/segmentation).
4. Run `python train.py --config configs/<task>.yaml`.

## Evaluation Outputs

| Task           | Metrics                                          |
|----------------|--------------------------------------------------|
| Classification | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Detection      | mAP@0.5, per-class AP                            |
| Segmentation   | mAP@0.5 (box), mask-mAP@0.5                      |

All plots (training curves, confusion matrices, prediction grids) are saved automatically.

## Key Design Decisions

- **Shared backbone** — ResNet-50 pretrained on ImageNet, fine-tuned per task.
- **Toggleable segmentation** — Mask R-CNN extends Faster R-CNN; flip one config flag to add/remove the mask head.
- **OpenCV preprocessing** — all image transforms use OpenCV for consistency and speed.
- **Mixed-precision training** — AMP is enabled by default for faster training on GPU.
- **Early stopping** — training halts when the monitored metric stops improving.

## Tech Stack

- **PyTorch** + **torchvision** — models and training
- **OpenCV** — image I/O and preprocessing
- **scikit-learn** — classification metrics
- **matplotlib** + **seaborn** — visualization
- **LaTeX** — publication-ready report

## License

MIT
