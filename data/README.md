# Data Directory

Place your dataset here using the following structure:

```
data/
├── images/
│   ├── train/       # Training images
│   └── val/         # Validation images
└── annotations/
    ├── train.json   # COCO-format training annotations
    └── val.json     # COCO-format validation annotations
```

## Annotation Format

Annotations must follow the [COCO JSON format](https://cocodataset.org/#format-data):

```json
{
  "images": [
    {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.0,
      "iscrowd": 0,
      "segmentation": [[x1, y1, x2, y2, ...]]
    }
  ],
  "categories": [
    {"id": 1, "name": "class_name"}
  ]
}
```

- `bbox` is required for detection and segmentation tasks.
- `segmentation` polygons are required only when `enable_masks: true`.
- For classification-only, the `category_id` on each annotation is used as the image label.

## Quick Start with Sample Data

Run the data preparation script to generate a small synthetic dataset for testing:

```bash
python data/prepare_data.py --output data --num-images 200
```
