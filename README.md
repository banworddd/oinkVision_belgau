# oinkVision_belgau

Baseline repository for multi-label classification of hind limb condition in pigs from 4 synchronized videos: `top`, `right`, `left`, `rear`.

## Goal

For each animal, predict 4 binary labels:

- `bad_posture`
- `bumps`
- `soft_pastern`
- `x_shape`

The final prediction is made at the **animal level**, not per frame and not per camera.

## Baseline Idea

1. Read one animal as a group of 4 videos.
2. Sample annotated frames from each camera.
3. Crop the informative region using bbox annotations when available.
4. Run a frame encoder such as `EfficientNet-B0`.
5. Predict 4 logits for each sampled frame.
6. Aggregate predictions across frames and cameras.
7. Convert probabilities to binary labels using per-class thresholds.

## Data Layout

The project expects the hackathon dataset to stay in the parent workspace:

```text
/Users/banworddd/University/Hakaton/data/
```

Important folders:

- `data/train/raw/`
- `data/train/annotation/`
- `data/train/metadata/`
- `data/val/raw/`
- `data/val/metadata/`
- `data/test/`

## Annotation Format

Each file `train/annotation/pig_{id}.json` contains:

- `pig_id`: animal id
- `video`: relative paths to 4 camera videos
- `target`: 4 binary target labels for hind limbs
- `annotations`: per-frame bbox annotations for the 4 views

Example:

```json
{
  "pig_id": "12345",
  "video": {
    "top": "train/raw/pig_12345_cam_top.mp4",
    "right": "train/raw/pig_12345_cam_right.mp4",
    "left": "train/raw/pig_12345_cam_left.mp4",
    "rear": "train/raw/pig_12345_cam_rear.mp4"
  },
  "target": {
    "bad_posture": 0,
    "bumps": 1,
    "soft_pastern": 0,
    "x_shape": 0
  }
}
```

Per annotated frame:

- `frame_id`: 0-based frame index
- `top`: usually one object with label `pig`
- `right`: usually one object with label `right_back_leg`
- `left`: usually one object with label `left_back_leg`
- `rear`: usually two objects with labels `left_back_leg` and `right_back_leg`

Bounding boxes use normalized YOLO format:

```text
[x_center, y_center, width, height]
```

## Project Structure

```text
oinkVision_belgau/
├── README.md
├── configs/
│   └── baseline.yaml
├── notebooks/
├── outputs/
├── scripts/
│   ├── build_index.py
│   └── run_baseline.sh
└── src/
    └── oinkvision/
        ├── __init__.py
        ├── constants.py
        ├── dataset.py
        ├── indexing.py
        ├── metrics.py
        ├── model.py
        ├── train.py
        └── infer.py
```

## What To Do First

1. Build a train index from `train/annotation/*.json`.
2. Create an internal train/valid split on the animal level.
3. Implement a dataset that samples annotated frames from all 4 cameras.
4. Train an `EfficientNet-B0` multi-label baseline with `BCEWithLogitsLoss`.
5. Tune class thresholds on the internal validation split.
