"""Rear-view embedding anomaly specialist for x_shape."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import CAMERAS, LABELS
from .dataset import FRONT_META_FIELDS, PigVideoDataset, load_index
from .model import build_model
from .train import build_aggregation_spec, choose_device


def prepare_raw_nometa_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["annotation_path"] = ""
        for field in FRONT_META_FIELDS:
            item[field] = 0
        prepared.append(item)
    return prepared


def build_rows_loader(config: dict[str, Any], rows: list[dict[str, Any]]) -> DataLoader:
    dataset = PigVideoDataset(
        index_path="",
        rows=rows,
        frames_per_camera=config["data"]["frames_per_camera"],
        image_size=config["data"]["image_size"],
        use_bbox_crops=config["data"]["use_bbox_crops"],
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


def _extract_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encoder"):
        return model.encoder(images)  # type: ignore[attr-defined]

    base_model = getattr(model, "model", None)
    if base_model is None:
        raise ValueError("Model does not expose a feature extractor for x_shape specialist.")

    if hasattr(base_model, "forward_features"):
        features = base_model.forward_features(images)
        if hasattr(base_model, "forward_head"):
            try:
                return base_model.forward_head(features, pre_logits=True)
            except TypeError:
                pass
        if features.ndim == 4:
            return F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    raise ValueError("Unsupported model type for feature extraction.")


def aggregate_rear_embeddings(
    frame_features: torch.Tensor,
    frame_mask: torch.Tensor,
    cameras: list[str],
    frames_per_camera: int,
) -> torch.Tensor:
    total_expected_frames = len(cameras) * frames_per_camera
    if frame_features.shape[1] != total_expected_frames:
        mask = frame_mask.unsqueeze(-1)
        masked = frame_features * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return masked.sum(dim=1) / denom

    rear_index = cameras.index("rear")
    rear_start = rear_index * frames_per_camera
    rear_end = rear_start + frames_per_camera
    rear_features = frame_features[:, rear_start:rear_end, :]
    rear_mask = frame_mask[:, rear_start:rear_end].unsqueeze(-1)

    rear_mean = (rear_features * rear_mask).sum(dim=1) / torch.clamp(rear_mask.sum(dim=1), min=1.0)
    masked_max = rear_features.masked_fill(~rear_mask.bool(), float("-inf"))
    rear_max = masked_max.max(dim=1).values
    rear_max = torch.where(torch.isfinite(rear_max), rear_max, torch.zeros_like(rear_max))
    rear_embedding = torch.cat([rear_mean, rear_max], dim=1)
    return F.normalize(rear_embedding, dim=1)


@torch.no_grad()
def extract_rear_embeddings_from_checkpoint(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)
    cameras = list(aggregation_spec.get("cameras", CAMERAS))
    frames_per_camera = int(aggregation_spec["frames_per_camera"])

    loader = build_rows_loader(config, rows)
    model = build_model(config).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    pig_ids: list[str] = []
    rear_embeddings: list[np.ndarray] = []
    xshape_probs: list[np.ndarray] = []
    xshape_idx = LABELS.index("x_shape")

    for batch in loader:
        images = batch["images"].to(device)
        frame_mask = batch["frame_mask"].to(device)

        batch_size, num_frames, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_frames, channels, height, width)
        features = _extract_features(model, flat_images).view(batch_size, num_frames, -1)
        rear_emb = aggregate_rear_embeddings(features, frame_mask, cameras=cameras, frames_per_camera=frames_per_camera)

        model_output = model(flat_images)
        if isinstance(model_output, dict):
            frame_logits = model_output["logits"].view(batch_size, num_frames, len(LABELS))
        else:
            frame_logits = model_output.view(batch_size, num_frames, len(LABELS))
        mask = frame_mask.unsqueeze(-1)
        logits = (frame_logits * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        xshape_prob = torch.sigmoid(logits[:, xshape_idx]).cpu().numpy()

        pig_ids.extend(batch["pig_id"])
        rear_embeddings.append(rear_emb.cpu().numpy())
        xshape_probs.append(xshape_prob)

    return pig_ids, np.concatenate(rear_embeddings, axis=0), np.concatenate(xshape_probs, axis=0)


def extract_ensemble_rear_embeddings(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoints: list[Path],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    pig_ids_ref: list[str] | None = None
    embeddings_per_model: list[np.ndarray] = []
    xshape_probs_per_model: list[np.ndarray] = []

    for checkpoint in checkpoints:
        pig_ids, rear_embeddings, xshape_probs = extract_rear_embeddings_from_checkpoint(config, rows, checkpoint)
        if pig_ids_ref is None:
            pig_ids_ref = pig_ids
        elif pig_ids_ref != pig_ids:
            raise ValueError("Checkpoint embeddings are misaligned by pig_id.")
        embeddings_per_model.append(rear_embeddings)
        xshape_probs_per_model.append(xshape_probs)

    if pig_ids_ref is None:
        raise ValueError("No checkpoints were provided.")

    mean_embeddings = np.mean(np.stack(embeddings_per_model, axis=0), axis=0)
    norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
    mean_embeddings = mean_embeddings / np.clip(norms, 1e-8, None)
    mean_xshape_probs = np.mean(np.stack(xshape_probs_per_model, axis=0), axis=0)
    return pig_ids_ref, mean_embeddings.astype(np.float32), mean_xshape_probs.astype(np.float32)


@dataclass
class XShapeSpecialistArtifact:
    centroid: np.ndarray
    normal_bank: np.ndarray
    normal_p50: float
    normal_p95: float
    knn_k: int
    fusion_alpha: float
    decision_threshold: float
    train_metrics: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "centroid": self.centroid,
            "normal_bank": self.normal_bank,
            "normal_p50": float(self.normal_p50),
            "normal_p95": float(self.normal_p95),
            "knn_k": int(self.knn_k),
            "fusion_alpha": float(self.fusion_alpha),
            "decision_threshold": float(self.decision_threshold),
            "train_metrics": self.train_metrics,
        }


def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sim = np.clip(a @ b.T, -1.0, 1.0)
    return 1.0 - sim


def compute_xshape_specialist_scores(embeddings: np.ndarray, artifact_payload: dict[str, Any]) -> np.ndarray:
    centroid = np.asarray(artifact_payload["centroid"], dtype=np.float32)
    centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-8, None)
    normal_bank = np.asarray(artifact_payload["normal_bank"], dtype=np.float32)
    normal_bank = normal_bank / np.clip(np.linalg.norm(normal_bank, axis=1, keepdims=True), 1e-8, None)

    centroid_dist = 1.0 - np.clip(embeddings @ centroid, -1.0, 1.0)
    pairwise = _cosine_distance_matrix(embeddings, normal_bank)
    knn_k = max(1, int(artifact_payload.get("knn_k", 5)))
    knn_dists = np.sort(pairwise, axis=1)[:, : min(knn_k, normal_bank.shape[0])]
    knn_dist = knn_dists.mean(axis=1)
    raw_score = 0.5 * centroid_dist + 0.5 * knn_dist

    normal_p50 = float(artifact_payload.get("normal_p50", np.percentile(raw_score, 50)))
    normal_p95 = float(artifact_payload.get("normal_p95", np.percentile(raw_score, 95)))
    denom = max(normal_p95 - normal_p50, 1e-6)
    normalized = np.clip((raw_score - normal_p50) / denom, 0.0, 1.0)
    return normalized.astype(np.float32)


def fit_xshape_specialist(
    embeddings: np.ndarray,
    xshape_labels: np.ndarray,
    base_xshape_probs: np.ndarray,
    knn_k: int = 5,
) -> XShapeSpecialistArtifact:
    normal_mask = xshape_labels == 0
    if not np.any(normal_mask):
        raise ValueError("Cannot fit x_shape specialist without normal examples.")

    normal_bank = embeddings[normal_mask]
    centroid = normal_bank.mean(axis=0)
    centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-8, None)

    centroid_dist = 1.0 - np.clip(normal_bank @ centroid, -1.0, 1.0)
    pairwise = _cosine_distance_matrix(normal_bank, normal_bank)
    np.fill_diagonal(pairwise, np.inf)
    knn_k = max(1, min(int(knn_k), max(normal_bank.shape[0] - 1, 1)))
    knn_dists = np.sort(pairwise, axis=1)[:, :knn_k]
    knn_dist = knn_dists.mean(axis=1)
    normal_raw = 0.5 * centroid_dist + 0.5 * knn_dist
    normal_p50 = float(np.percentile(normal_raw, 50))
    normal_p95 = float(np.percentile(normal_raw, 95))

    artifact_payload = {
        "centroid": centroid,
        "normal_bank": normal_bank,
        "normal_p50": normal_p50,
        "normal_p95": normal_p95,
        "knn_k": knn_k,
    }
    specialist_scores = compute_xshape_specialist_scores(embeddings, artifact_payload)

    best_alpha = 0.8
    best_threshold = 0.5
    best_f1 = -1.0
    for alpha in np.arange(0.25, 1.25, 0.05, dtype=np.float32):
        fused = np.maximum(base_xshape_probs, alpha * specialist_scores)
        for threshold in np.arange(0.05, 1.0, 0.05, dtype=np.float32):
            preds = (fused >= threshold).astype(np.int64)
            tp = int(np.sum((preds == 1) & (xshape_labels == 1)))
            fp = int(np.sum((preds == 1) & (xshape_labels == 0)))
            fn = int(np.sum((preds == 0) & (xshape_labels == 1)))
            if tp == 0 and fp == 0 and fn == 0:
                f1 = 0.0
            else:
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_alpha = float(alpha)
                best_threshold = float(threshold)

    fused = np.maximum(base_xshape_probs, best_alpha * specialist_scores)
    preds = (fused >= best_threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (xshape_labels == 1)))
    fp = int(np.sum((preds == 1) & (xshape_labels == 0)))
    fn = int(np.sum((preds == 0) & (xshape_labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    train_metrics = {
        "xshape_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": int(xshape_labels.sum()),
    }
    return XShapeSpecialistArtifact(
        centroid=centroid.astype(np.float32),
        normal_bank=normal_bank.astype(np.float32),
        normal_p50=normal_p50,
        normal_p95=normal_p95,
        knn_k=knn_k,
        fusion_alpha=best_alpha,
        decision_threshold=best_threshold,
        train_metrics=train_metrics,
    )

