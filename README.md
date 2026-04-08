<div align="center">

# Adaptive Vision Pipeline

**A unified, config-driven framework for image classification, object detection, and instance segmentation.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Quick Start](#quick-start) &bull; [Architecture](#architecture) &bull; [Training](#3-train) &bull; [Inference](#5-inference) &bull; [Report](#latex-report)

</div>

---

## Highlights

- **One pipeline, three tasks** &mdash; switch between classification, detection, and segmentation by changing a single YAML file.
- **Toggleable segmentation** &mdash; Mask R-CNN adds a pixel-level mask head on top of Faster R-CNN with one flag (`enable_masks: true`), so you only pay for segmentation when it improves results.
- **Production patterns** &mdash; mixed-precision training (AMP), cosine/step LR scheduling, early stopping, deterministic seeding, and automatic checkpointing.
- **Full evaluation suite** &mdash; confusion matrices, mAP / IoU curves, mask-mAP, and per-class AP &mdash; all plotted and saved automatically.
- **Publication-ready LaTeX report** &mdash; methodology, equations, hyper-parameter tables, and bibliography included.

---

## Architecture

```
                        ┌─────────────────────┐
                        │    Input Image       │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   OpenCV Transforms  │  Resize, flip, jitter, normalize
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   ResNet-50 + FPN    │  Shared pretrained backbone
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐ ┌───────▼────────┐ ┌─────────▼─────────┐
    │   Classification  │ │   Detection    │ │   Segmentation    │
    │   FC Head         │ │  Faster R-CNN  │ │   Mask R-CNN      │
    │                   │ │  (RPN + RoI)   │ │  (+ mask branch)  │
    └───────────────────┘ └────────────────┘ └───────────────────┘
              │                    │                     │
         Class label       Bounding boxes       Boxes + pixel masks
```

| Task | Model | Output |
|------|-------|--------|
| **Classification** | ResNet-50 / EfficientNet-B0 with replaced FC head | Class label + confidence |
| **Detection** | Faster R-CNN with ResNet-50-FPN | Bounding boxes + class scores |
| **Segmentation** | Mask R-CNN (extends detector with mask branch) | Boxes + per-instance binary masks |

> The segmentation model is architecturally identical to the detector plus one extra head &mdash; making the two fully interchangeable at config level.

---

## Project Structure

```
adaptive-vision-pipeline/
│
├── configs/
│   ├── classify.yaml           # Classification hyper-parameters
│   ├── detect.yaml             # Detection hyper-parameters
│   └── segment.yaml            # Segmentation (enable_masks: true)
│
├── src/
│   ├── config.py               # YAML -> typed dataclasses
│   ├── dataset.py              # COCO-format dataset loader
│   ├── preprocessing.py        # OpenCV-based transform pipeline
│   ├── models/
│   │   ├── __init__.py         # build_model() factory
│   │   ├── classifier.py       # ResNet-50 / EfficientNet classifier
│   │   ├── detector.py         # Faster R-CNN wrapper
│   │   └── segmentor.py        # Mask R-CNN wrapper
│   ├── engine.py               # Train / validation loops (AMP, scheduling)
│   ├── metrics.py              # mAP, IoU, confusion matrix, mask-mAP
│   └── visualize.py            # Training curves, overlays, prediction grids
│
├── train.py                    # Training entry point
├── evaluate.py                 # Standalone evaluation & metric export
├── inference.py                # Run on new images with visualisation
│
├── data/
│   ├── prepare_data.py         # Synthetic COCO dataset generator
│   └── README.md               # Annotation format documentation
│
├── report/
│   ├── main.tex                # LaTeX report (methodology + equations)
│   └── references.bib          # Bibliography (ResNet, Faster/Mask R-CNN, FPN)
│
├── requirements.txt
└── outputs/                    # Checkpoints, plots, eval results (gitignored)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/khaledkoubaa/adaptive-vision-pipeline.git
cd adaptive-vision-pipeline
pip install -r requirements.txt
```

### 2. Prepare data

Generate a synthetic dataset to validate the full pipeline:

```bash
python data/prepare_data.py --output data --num-images 200
```

Or bring your own COCO-format annotations &mdash; see [`data/README.md`](data/README.md) for the expected schema.

### 3. Train

```bash
# Image classification
python train.py --config configs/classify.yaml

# Object detection (Faster R-CNN)
python train.py --config configs/detect.yaml

# Instance segmentation (Mask R-CNN)
python train.py --config configs/segment.yaml
```

Outputs (checkpoints, training curves, logs) are saved to `outputs/<experiment_name>/`.

### 4. Evaluate

```bash
python evaluate.py \
    --config configs/detect.yaml \
    --checkpoint outputs/detection_baseline/best_model.pth
```

Produces a JSON metrics file, and for classification tasks, a confusion matrix plot.

### 5. Inference

```bash
# Single image
python inference.py \
    --config configs/detect.yaml \
    --checkpoint outputs/detection_baseline/best_model.pth \
    --input path/to/image.jpg

# Batch (entire directory)
python inference.py \
    --config configs/segment.yaml \
    --checkpoint outputs/segmentation_baseline/best_model.pth \
    --input path/to/images/ \
    --output results/
```

Annotated images and a prediction summary grid are saved automatically.

---

## Evaluation Metrics

| Task | Metrics | Visualisations |
|------|---------|----------------|
| **Classification** | Accuracy, Precision, Recall, F1 (macro) | Confusion matrix heatmap |
| **Detection** | mAP@0.5, per-class AP | Training curves, prediction overlays |
| **Segmentation** | Box mAP@0.5, Mask mAP@0.5 | Training curves, mask overlays |

All plots are generated with matplotlib/seaborn and saved to the experiment directory.

---

## Retraining on Custom Data

1. Place images in `data/images/train/` and `data/images/val/`.
2. Create COCO-format annotations at `data/annotations/train.json` and `data/annotations/val.json`.
3. Set `num_classes` in the config (detection/segmentation: add 1 for background).
4. Run training:

```bash
python train.py --config configs/detect.yaml
```

The dataset loader, transforms, and evaluation all work out of the box with any COCO-compatible dataset.

---

## Training Features

| Feature | Details |
|---------|---------|
| **Mixed precision** | `torch.cuda.amp` enabled by default for ~2x throughput on GPU |
| **LR scheduling** | Step decay or cosine annealing, configurable per task |
| **Early stopping** | Monitors val metric; halts when no improvement for *N* epochs |
| **Checkpointing** | Saves `best_model.pth` (by metric) and `final_model.pth` |
| **Deterministic** | Global seed for `random`, `numpy`, `torch`, and CUDA |
| **OpenCV transforms** | Resize, horizontal flip, colour jitter &mdash; all box/mask aware |

---

## LaTeX Report

A publication-ready report is included in `report/`:

- Full methodology with equations (cross-entropy, RPN loss, mask BCE, mAP)
- Hyper-parameter table for all three tasks
- Figure placeholders for training curves, confusion matrix, prediction samples
- BibTeX references for ResNet, Faster R-CNN, Mask R-CNN, FPN, ImageNet, COCO

Compile with:

```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep Learning | PyTorch, torchvision |
| Image Processing | OpenCV |
| Metrics | scikit-learn, custom mAP/IoU |
| Visualisation | matplotlib, seaborn |
| Configuration | PyYAML, dataclasses |
| Report | LaTeX |

---

## License

MIT
