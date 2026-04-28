"""Prototype-based specialists over ensemble embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .constants import CAMERAS, LABELS
from .dataset import load_index
from .xshape_specialist import (
    _extract_features,
    aggregate_rear_embeddings,
    build_rows_loader,
    prepare_raw_nometa_rows,
)
from .model import build_model
from .train import build_aggregation_spec, choose_device


def aggregate_all_embeddings(
    frame_features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    mask = frame_mask.unsqueeze(-1)
    mean_emb = (frame_features * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
    masked_max = frame_features.masked_fill(~mask.bool(), float("-inf"))
    max_emb = masked_max.max(dim=1).values
    max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))
    emb = torch.cat([mean_emb, max_emb], dim=1)
    return F.normalize(emb, dim=1)


@torch.no_grad()
def extract_multiview_embeddings_from_checkpoint(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
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
    all_embeddings: list[np.ndarray] = []
    rear_embeddings: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        batch_size, num_frames, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_frames, channels, height, width)
        features = _extract_features(model, flat_images).view(batch_size, num_frames, -1)
        all_emb = aggregate_all_embeddings(features, frame_mask)
        rear_emb = aggregate_rear_embeddings(features, frame_mask, cameras=cameras, frames_per_camera=frames_per_camera)

        model_output = model(flat_images)
        if isinstance(model_output, dict):
            frame_logits = model_output["logits"].view(batch_size, num_frames, len(LABELS))
        else:
            frame_logits = model_output.view(batch_size, num_frames, len(LABELS))
        mask = frame_mask.unsqueeze(-1)
        logits = (frame_logits * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        prob = torch.sigmoid(logits).cpu().numpy()

        pig_ids.extend(batch["pig_id"])
        all_embeddings.append(all_emb.cpu().numpy())
        rear_embeddings.append(rear_emb.cpu().numpy())
        probs.append(prob)

    return (
        pig_ids,
        np.concatenate(all_embeddings, axis=0).astype(np.float32),
        np.concatenate(rear_embeddings, axis=0).astype(np.float32),
        np.concatenate(probs, axis=0).astype(np.float32),
    )


def extract_ensemble_multiview_embeddings(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoints: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    pig_ids_ref: list[str] | None = None
    all_embs_per_model: list[np.ndarray] = []
    rear_embs_per_model: list[np.ndarray] = []
    probs_per_model: list[np.ndarray] = []

    for checkpoint in checkpoints:
        pig_ids, all_emb, rear_emb, probs = extract_multiview_embeddings_from_checkpoint(config, rows, checkpoint)
        if pig_ids_ref is None:
            pig_ids_ref = pig_ids
        elif pig_ids_ref != pig_ids:
            raise ValueError("Checkpoint embeddings are misaligned by pig_id.")
        all_embs_per_model.append(all_emb)
        rear_embs_per_model.append(rear_emb)
        probs_per_model.append(probs)

    if pig_ids_ref is None:
        raise ValueError("No checkpoints were provided.")

    mean_all = np.mean(np.stack(all_embs_per_model, axis=0), axis=0)
    mean_all = mean_all / np.clip(np.linalg.norm(mean_all, axis=1, keepdims=True), 1e-8, None)
    mean_rear = np.mean(np.stack(rear_embs_per_model, axis=0), axis=0)
    mean_rear = mean_rear / np.clip(np.linalg.norm(mean_rear, axis=1, keepdims=True), 1e-8, None)
    mean_probs = np.mean(np.stack(probs_per_model, axis=0), axis=0)
    return pig_ids_ref, mean_all.astype(np.float32), mean_rear.astype(np.float32), mean_probs.astype(np.float32)


def cosine_similarity_to_prototype(embeddings: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    prototype = prototype / np.clip(np.linalg.norm(prototype), 1e-8, None)
    return np.clip(embeddings @ prototype, -1.0, 1.0)


def prototype_score(
    embeddings: np.ndarray,
    positive_proto: np.ndarray,
    negative_proto: np.ndarray,
    center: float,
    scale: float,
) -> np.ndarray:
    sim_pos = cosine_similarity_to_prototype(embeddings, positive_proto)
    sim_neg = cosine_similarity_to_prototype(embeddings, negative_proto)
    raw = sim_pos - sim_neg
    return np.clip((raw - center) / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


@dataclass
class LabelPrototypeArtifact:
    label: str
    embedding_type: str
    positive_prototype: np.ndarray
    negative_prototype: np.ndarray
    center: float
    scale: float
    fusion_alpha: float
    decision_threshold: float
    train_metrics: dict[str, Any]
    support: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "embedding_type": self.embedding_type,
            "positive_prototype": self.positive_prototype,
            "negative_prototype": self.negative_prototype,
            "center": float(self.center),
            "scale": float(self.scale),
            "fusion_alpha": float(self.fusion_alpha),
            "decision_threshold": float(self.decision_threshold),
            "train_metrics": self.train_metrics,
            "support": int(self.support),
        }


def _fit_single_label_prototype(
    label: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    base_probs: np.ndarray,
    embedding_type: str,
) -> LabelPrototypeArtifact | None:
    pos_mask = labels == 1
    neg_mask = labels == 0
    support = int(pos_mask.sum())
    if support == 0 or not np.any(neg_mask):
        return None

    positive_proto = embeddings[pos_mask].mean(axis=0)
    positive_proto = positive_proto / np.clip(np.linalg.norm(positive_proto), 1e-8, None)
    negative_proto = embeddings[neg_mask].mean(axis=0)
    negative_proto = negative_proto / np.clip(np.linalg.norm(negative_proto), 1e-8, None)

    raw_pos = cosine_similarity_to_prototype(embeddings, positive_proto) - cosine_similarity_to_prototype(embeddings, negative_proto)
    neg_raw = raw_pos[neg_mask]
    pos_raw = raw_pos[pos_mask]
    center = float(np.percentile(neg_raw, 90)) if neg_raw.size else 0.0
    scale = float(max(np.percentile(pos_raw, 50) - center, 1e-3)) if pos_raw.size else 1.0
    scores = prototype_score(embeddings, positive_proto, negative_proto, center=center, scale=scale)

    best_alpha = 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for alpha in np.arange(0.0, 1.05, 0.05, dtype=np.float32):
        fused = np.maximum(base_probs, alpha * scores) if alpha > 0 else base_probs
        for threshold in np.arange(0.05, 1.0, 0.05, dtype=np.float32):
            preds = (fused >= threshold).astype(np.int64)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_alpha = float(alpha)
                best_threshold = float(threshold)

    fused = np.maximum(base_probs, best_alpha * scores) if best_alpha > 0 else base_probs
    preds = (fused >= best_threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    train_metrics = {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    return LabelPrototypeArtifact(
        label=label,
        embedding_type=embedding_type,
        positive_prototype=positive_proto.astype(np.float32),
        negative_prototype=negative_proto.astype(np.float32),
        center=center,
        scale=scale,
        fusion_alpha=best_alpha,
        decision_threshold=best_threshold,
        train_metrics=train_metrics,
        support=support,
    )


def fit_prototype_specialist(
    all_embeddings: np.ndarray,
    rear_embeddings: np.ndarray,
    labels_matrix: np.ndarray,
    base_probs: np.ndarray,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"labels": {}}
    for idx, label in enumerate(LABELS):
        label_targets = labels_matrix[:, idx].astype(np.int64)
        label_base_probs = base_probs[:, idx].astype(np.float32)
        embedding_type = "rear" if label == "x_shape" else "all"
        embeddings = rear_embeddings if embedding_type == "rear" else all_embeddings
        artifact = _fit_single_label_prototype(
            label=label,
            embeddings=embeddings,
            labels=label_targets,
            base_probs=label_base_probs,
            embedding_type=embedding_type,
        )
        if artifact is not None:
            payload["labels"][label] = artifact.to_payload()
    return payload


def apply_prototype_specialist(
    probs: np.ndarray,
    all_embeddings: np.ndarray,
    rear_embeddings: np.ndarray,
    artifact_payload: dict[str, Any],
) -> tuple[np.ndarray, list[float]]:
    fused = probs.copy()
    thresholds = [0.5 for _ in LABELS]
    label_payloads = dict(artifact_payload.get("labels", {}))
    for idx, label in enumerate(LABELS):
        if label not in label_payloads:
            thresholds[idx] = 0.5
            continue
        label_cfg = dict(label_payloads[label])
        embeddings = rear_embeddings if label_cfg.get("embedding_type") == "rear" else all_embeddings
        scores = prototype_score(
            embeddings,
            np.asarray(label_cfg["positive_prototype"], dtype=np.float32),
            np.asarray(label_cfg["negative_prototype"], dtype=np.float32),
            center=float(label_cfg.get("center", 0.0)),
            scale=float(label_cfg.get("scale", 1.0)),
        )
        alpha = float(label_cfg.get("fusion_alpha", 0.0))
        fused[:, idx] = np.maximum(fused[:, idx], alpha * scores) if alpha > 0 else fused[:, idx]
        thresholds[idx] = float(label_cfg.get("decision_threshold", 0.5))
    return fused, thresholds


def load_rows_and_labels(index_path: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = prepare_raw_nometa_rows(load_index(index_path))
    labels_matrix = np.asarray([[int(row.get(label, 0)) for label in LABELS] for row in rows], dtype=np.int64)
    return rows, labels_matrix

