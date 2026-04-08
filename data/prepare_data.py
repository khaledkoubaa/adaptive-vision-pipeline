"""Generate a synthetic COCO-format dataset for pipeline testing.

Creates random images with simple geometric shapes (rectangles, circles,
triangles) along with bounding-box and polygon-mask annotations so that
every stage of the pipeline can be validated without external data.

Usage:
    python data/prepare_data.py --output data --num-images 200
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = [
    {"id": 1, "name": "rectangle", "supercategory": "shape"},
    {"id": 2, "name": "circle", "supercategory": "shape"},
    {"id": 3, "name": "triangle", "supercategory": "shape"},
]

IMG_SIZE = 640


def random_color():
    return [random.randint(60, 255) for _ in range(3)]


def draw_rectangle(img, min_size=40, max_size=200):
    h, w = img.shape[:2]
    rw = random.randint(min_size, max_size)
    rh = random.randint(min_size, max_size)
    x = random.randint(0, w - rw)
    y = random.randint(0, h - rh)
    color = random_color()
    cv2.rectangle(img, (x, y), (x + rw, y + rh), color, -1)
    bbox = [x, y, rw, rh]
    seg = [x, y, x + rw, y, x + rw, y + rh, x, y + rh]
    return bbox, seg, rw * rh


def draw_circle(img, min_r=20, max_r=100):
    h, w = img.shape[:2]
    r = random.randint(min_r, max_r)
    cx = random.randint(r, w - r)
    cy = random.randint(r, h - r)
    color = random_color()
    cv2.circle(img, (cx, cy), r, color, -1)
    bbox = [cx - r, cy - r, 2 * r, 2 * r]
    # Approximate circle mask as polygon
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    seg = []
    for a in angles:
        seg.extend([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
    return bbox, seg, int(np.pi * r * r)


def draw_triangle(img, min_size=40, max_size=200):
    h, w = img.shape[:2]
    s = random.randint(min_size, max_size)
    x = random.randint(0, w - s)
    y = random.randint(s, h)
    pts = np.array([
        [x + s // 2, y - s],
        [x, y],
        [x + s, y],
    ], dtype=np.int32)
    color = random_color()
    cv2.fillPoly(img, [pts], color)
    bx = int(pts[:, 0].min())
    by = int(pts[:, 1].min())
    bw = int(pts[:, 0].max()) - bx
    bh = int(pts[:, 1].max()) - by
    bbox = [bx, by, bw, bh]
    seg = pts.flatten().tolist()
    return bbox, seg, int(0.5 * bw * bh)


SHAPE_DRAWERS = {1: draw_rectangle, 2: draw_circle, 3: draw_triangle}


def generate_image(img_id: int):
    """Generate a single image with 1-5 random shapes."""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
    # Slight background noise
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    annotations = []
    num_objects = random.randint(1, 5)
    for _ in range(num_objects):
        cat_id = random.choice([1, 2, 3])
        bbox, seg, area = SHAPE_DRAWERS[cat_id](img)
        annotations.append({
            "category_id": cat_id,
            "bbox": bbox,
            "segmentation": [seg],
            "area": float(area),
            "iscrowd": 0,
        })
    return img, annotations


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic COCO dataset")
    parser.add_argument("--output", type=str, default="data", help="Output root dir")
    parser.add_argument("--num-images", type=int, default=200, help="Total images")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.output)
    num_val = int(args.num_images * args.val_split)
    num_train = args.num_images - num_val

    for split, count in [("train", num_train), ("val", num_val)]:
        img_dir = root / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        coco = {
            "images": [],
            "annotations": [],
            "categories": CATEGORIES,
        }
        ann_id = 1

        for i in range(count):
            img_id = i + 1
            fname = f"{img_id:06d}.jpg"
            img, anns = generate_image(img_id)
            cv2.imwrite(str(img_dir / fname), img)

            coco["images"].append({
                "id": img_id,
                "file_name": fname,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
            })
            for ann in anns:
                ann["id"] = ann_id
                ann["image_id"] = img_id
                coco["annotations"].append(ann)
                ann_id += 1

        ann_dir = root / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_path = ann_dir / f"{split}.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)

        print(f"[{split}] {count} images -> {img_dir}")
        print(f"[{split}] {ann_id - 1} annotations -> {ann_path}")

    print("Done.")


if __name__ == "__main__":
    main()
