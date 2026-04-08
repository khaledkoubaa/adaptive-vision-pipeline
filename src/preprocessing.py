"""OpenCV-based image transforms for training and inference.

Each transform is a callable that receives ``(image, target)`` and returns
the transformed pair.  ``image`` is an HWC uint8 NumPy array (RGB);
``target`` is either an int (classification) or a dict (detection /
segmentation).
"""

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------
# Composition
# -----------------------------------------------------------------------

class Compose:
    """Chain multiple transforms sequentially."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, target: Any) -> Tuple[np.ndarray, Any]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# -----------------------------------------------------------------------
# Geometric transforms
# -----------------------------------------------------------------------

class Resize:
    """Resize image (and scale boxes) to a fixed square size using OpenCV."""

    def __init__(self, size: int = 640):
        self.size = size

    def __call__(self, image: np.ndarray, target: Any):
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        if isinstance(target, dict) and "boxes" in target:
            sx, sy = self.size / w, self.size / h
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target["boxes"] = boxes

            if "masks" in target:
                masks = target["masks"].numpy()
                resized = np.stack([
                    cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    for m in masks
                ])
                target["masks"] = torch.as_tensor(resized, dtype=torch.uint8)

        return image, target


class RandomHorizontalFlip:
    """Flip image and targets horizontally with probability *p*."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, target: Any):
        if np.random.rand() >= self.p:
            return image, target

        image = cv2.flip(image, 1)
        w = image.shape[1]

        if isinstance(target, dict) and "boxes" in target:
            boxes = target["boxes"].clone()
            x_min = w - boxes[:, 2]
            x_max = w - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            target["boxes"] = boxes

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

        return image, target


class RandomResizedCrop:
    """Random crop then resize — mainly useful for classification."""

    def __init__(self, size: int = 224, scale: Tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, image: np.ndarray, target: Any):
        h, w = image.shape[:2]
        area = h * w
        ratio = np.random.uniform(*self.scale)
        new_area = int(area * ratio)
        side = int(new_area ** 0.5)
        side = min(side, h, w)

        y = np.random.randint(0, h - side + 1)
        x = np.random.randint(0, w - side + 1)
        image = image[y : y + side, x : x + side]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image, target


# -----------------------------------------------------------------------
# Photometric transforms
# -----------------------------------------------------------------------

class ColorJitter:
    """Random brightness and contrast adjustment via OpenCV."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: np.ndarray, target: Any):
        alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        beta = np.random.uniform(-self.brightness, self.brightness) * 255
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image, target


class Normalize:
    """Normalize to [0, 1] then apply channel-wise mean/std (ImageNet defaults)."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray, target: Any):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, target


# -----------------------------------------------------------------------
# Tensor conversion
# -----------------------------------------------------------------------

class ToTensor:
    """Convert HWC NumPy image to CHW float32 torch.Tensor."""

    def __call__(self, image: np.ndarray, target: Any):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image, target


# -----------------------------------------------------------------------
# Prebuilt pipelines
# -----------------------------------------------------------------------

def build_transforms(task: str, train: bool = True, img_size: int = 640):
    """Return a ``Compose`` pipeline appropriate for *task* and split."""
    if task == "classify":
        if train:
            return Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(),
                Normalize(),
                ToTensor(),
            ])
        return Compose([
            Resize(224),
            Normalize(),
            ToTensor(),
        ])

    # Detection / segmentation
    if train:
        return Compose([
            Resize(img_size),
            RandomHorizontalFlip(),
            ColorJitter(),
            ToTensor(),
        ])
    return Compose([
        Resize(img_size),
        ToTensor(),
    ])
