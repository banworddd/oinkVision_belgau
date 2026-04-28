"""Conservative rear-embedding anomaly helper for x_shape."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import CAMERAS, LABELS
from .dataset import PigVideoDataset
from .model import build_model
from .train import choose_device


def prepare_raw_nometa_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        copied = deepcopy(row)
        copied["annotation_path"] = ""
        for key in list(copied.keys()):
            if key.startswith("front_"):
                copied[key] = 0
        prepared.append(copied)
    return prepared


def _build_loader(config: dict[str, Any], rows: list[dict[str, Any]]) -> DataLoader:
    dataset = PigVideoDataset(
        index_path="",
        rows=rows,
        frames_per_camera=int(config["data"]["frames_per_camera"]),
        image_size=int(config["data"]["image_size"]),
        use_bbox_crops=bool(config["data"]["use_bbox_crops"]),
        frame_cache_dir=config["data"].get("frame_cache_dir"),
        augmentation_profile=config.get("augmentation", {}),
        seed=int(config["seed"]),
        augment=False,
        raw_sample_ratio=0.0,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )


@torch.no_grad()
def _extract_single_checkpoint_rear_embeddings(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    device = choose_device()
    loader = _build_loader(config, rows)
    model = build_model(config).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    cameras = list(config.get("cameras", CAMERAS))
    frames_per_camera = int(config["data"]["frames_per_camera"])
    rear_index = cameras.index("rear")
    rear_start = rear_index * frames_per_camera
    rear_end = rear_start + frames_per_camera

    pig_ids: list[str] = []
    targets: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        batch_size, num_frames, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_frames, channels, height, width)
        features = model.extract_features(flat_images)  # type: ignore[attr-defined]
        features = features.view(batch_size, num_frames, -1)

        rear_features = features[:, rear_start:rear_end, :]
        rear_mask = frame_mask[:, rear_start:rear_end].unsqueeze(-1)
        denom = torch.clamp(rear_mask.sum(dim=1), min=1.0)
        rear_embedding = (rear_features * rear_mask).sum(dim=1) / denom

        pig_ids.extend(batch["pig_id"])
        targets.append(batch["target"][:, LABELS.index("x_shape")].cpu().numpy())
        embeddings.append(rear_embedding.cpu().numpy())

    return pig_ids, np.concatenate(embeddings, axis=0), np.concatenate(targets, axis=0)


def extract_ensemble_rear_embeddings(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoints: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    pig_ids_ref: list[str] | None = None
    targets_ref: np.ndarray | None = None
    all_embeddings: list[np.ndarray] = []

    for checkpoint in checkpoints:
        pig_ids, embeddings, targets = _extract_single_checkpoint_rear_embeddings(config, rows, checkpoint)
        if pig_ids_ref is None:
            pig_ids_ref = pig_ids
            targets_ref = targets
        else:
            if pig_ids_ref != pig_ids:
                raise ValueError("Checkpoint embeddings are misaligned by pig_id.")
        all_embeddings.append(embeddings)

    if pig_ids_ref is None or targets_ref is None:
        raise ValueError("No checkpoints were provided.")

    mean_embeddings = np.mean(np.stack(all_embeddings, axis=0), axis=0)
    return pig_ids_ref, mean_embeddings, targets_ref


def _distances_to_centroid(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embeddings - centroid[None, :], axis=1)


def fit_xshape_anomaly_guard(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoints: list[str],
) -> dict[str, Any]:
    pig_ids, embeddings, targets = extract_ensemble_rear_embeddings(
        config=config,
        rows=prepare_raw_nometa_rows(rows),
        checkpoints=checkpoints,
    )
    targets = targets.astype(np.int64)
    normal_mask = targets == 0
    positive_mask = targets == 1
    if normal_mask.sum() < 5:
        raise ValueError("Not enough normal samples to fit x_shape anomaly guard.")

    centroid = embeddings[normal_mask].mean(axis=0)
    distances = _distances_to_centroid(embeddings, centroid)
    normal_distances = distances[normal_mask]
    q90 = float(np.percentile(normal_distances, 90))
    q99 = float(np.percentile(normal_distances, 99))
    scale = max(q99 - q90, 1e-6)
    scores = np.clip((distances - q90) / scale, 0.0, 1.0)

    normal_scores = scores[normal_mask]
    score_threshold = float(max(0.92, np.percentile(normal_scores, 99.5)))
    if positive_mask.any():
        pos_scores = scores[positive_mask]
        score_threshold = float(min(score_threshold, max(0.85, float(pos_scores.min()) * 0.98)))
    else:
        pos_scores = np.array([], dtype=np.float32)

    predictions = (scores >= score_threshold).astype(np.int64)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))

    return {
        "pig_ids": pig_ids,
        "centroid": centroid.astype(np.float32),
        "q90_distance": q90,
        "q99_distance": q99,
        "score_threshold": score_threshold,
        "gate_main_prob": 0.08,
        "boost_floor": 0.45,
        "boost_cap": 0.60,
        "decision_threshold": 0.40,
        "train_summary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "num_normals": int(normal_mask.sum()),
            "num_positives": int(positive_mask.sum()),
            "positive_scores": [float(x) for x in pos_scores.tolist()],
            "normal_score_p995": float(np.percentile(normal_scores, 99.5)),
        },
    }


def compute_xshape_anomaly_scores(embeddings: np.ndarray, artifact_payload: dict[str, Any]) -> np.ndarray:
    centroid = np.asarray(artifact_payload["centroid"], dtype=np.float32)
    q90 = float(artifact_payload["q90_distance"])
    q99 = float(artifact_payload["q99_distance"])
    scale = max(q99 - q90, 1e-6)
    distances = _distances_to_centroid(embeddings, centroid)
    return np.clip((distances - q90) / scale, 0.0, 1.0)


def apply_xshape_anomaly_guard(
    probs: np.ndarray,
    rear_embeddings: np.ndarray,
    artifact_payload: dict[str, Any],
) -> tuple[np.ndarray, float]:
    scores = compute_xshape_anomaly_scores(rear_embeddings, artifact_payload)
    threshold = float(artifact_payload.get("score_threshold", 0.95))
    gate_main_prob = float(artifact_payload.get("gate_main_prob", 0.08))
    boost_floor = float(artifact_payload.get("boost_floor", 0.45))
    boost_cap = float(artifact_payload.get("boost_cap", 0.60))
    decision_threshold = float(artifact_payload.get("decision_threshold", 0.40))
    xshape_idx = LABELS.index("x_shape")
    fused = probs.copy()

    for idx, score in enumerate(scores):
        main_prob = float(fused[idx, xshape_idx])
        if score < threshold or main_prob < gate_main_prob:
            continue
        normalized = (score - threshold) / max(1.0 - threshold, 1e-6)
        candidate = boost_floor + 0.10 * normalized
        fused[idx, xshape_idx] = max(main_prob, min(boost_cap, candidate))

    return fused, decision_threshold
