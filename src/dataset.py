"""COCO-format dataset loaders for classification, detection, and segmentation."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """Unified dataset that serves all three tasks from COCO-format annotations.

    For *classification* the label is the category of the first annotation.
    For *detection* targets include bounding boxes and class labels.
    For *segmentation* targets additionally include polygon masks.
    """

    def __init__(
        self,
        images_dir: str,
        annotation_file: str,
        task: str = "detect",
        transforms: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.task = task
        self.transforms = transforms

        with open(annotation_file) as f:
            coco = json.load(f)

        self.images: List[Dict[str, Any]] = coco["images"]
        self.categories: List[Dict[str, Any]] = coco["categories"]
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(self.categories)}

        # Group annotations by image id
        self._anns_by_img: Dict[int, List[Dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self._anns_by_img[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self._anns_by_img[img_info["id"]]

        if self.task == "classify":
            target = self._build_cls_target(anns)
        else:
            target = self._build_det_target(anns, image.shape[:2])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    # ------------------------------------------------------------------
    # Target builders
    # ------------------------------------------------------------------

    def _build_cls_target(self, anns: List[Dict]) -> int:
        """Return the majority category index among annotations."""
        if not anns:
            return 0
        cats = [self.cat_id_to_idx[a["category_id"]] for a in anns]
        return max(set(cats), key=cats.count)

    def _build_det_target(
        self, anns: List[Dict], img_shape: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Build a target dict compatible with torchvision detection models."""
        boxes, labels, areas, iscrowd = [], [], [], []
        masks = [] if self.task == "segment" else None

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann["category_id"]] + 1)  # +1 for bg
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

            if masks is not None and "segmentation" in ann:
                mask = self._poly_to_mask(ann["segmentation"], img_shape)
                masks.append(mask)

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if masks is not None and masks:
            target["masks"] = torch.as_tensor(
                np.stack(masks), dtype=torch.uint8
            )

        return target

    @staticmethod
    def _poly_to_mask(
        segmentation: List[List[float]], shape: Tuple[int, int]
    ) -> np.ndarray:
        """Rasterise COCO polygon segmentation to a binary mask using OpenCV."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segmentation:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask


def collate_fn(batch):
    """Custom collate for detection / segmentation (variable-size targets)."""
    return tuple(zip(*batch))
