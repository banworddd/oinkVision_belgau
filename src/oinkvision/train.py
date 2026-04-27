"""Training entrypoints and loops."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oinkvision.constants import LABELS
from oinkvision.dataset import PigVideoDataset, load_index
from oinkvision.env import apply_env_overrides, get_output_root, load_local_env
from oinkvision.metrics import compute_macro_f1
from oinkvision.model import build_model


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=output_root / "index" / "train_index.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return apply_env_overrides(config)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_rows(rows: list[dict[str, Any]], seed: int, valid_size: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stratify_labels = [f"{row['bad_posture']}{row['bumps']}{row['soft_pastern']}{row['x_shape']}" for row in rows]
    try:
        train_rows, valid_rows = train_test_split(
            rows,
            test_size=valid_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        train_rows, valid_rows = train_test_split(
            rows,
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
        )
    return train_rows, valid_rows


def build_dataloaders(config: dict[str, Any], index_path: Path) -> tuple[DataLoader, DataLoader]:
    rows = load_index(index_path)
    train_rows, valid_rows = split_rows(rows, seed=config["seed"], valid_size=config["train"]["valid_size"])

    common_kwargs = {
        "index_path": index_path,
        "frames_per_camera": config["data"]["frames_per_camera"],
        "image_size": config["data"]["image_size"],
        "use_bbox_crops": config["data"]["use_bbox_crops"],
        "frame_cache_dir": config["data"].get("frame_cache_dir"),
        "seed": config["seed"],
    }
    train_dataset = PigVideoDataset(rows=train_rows, **common_kwargs)
    valid_dataset = PigVideoDataset(rows=valid_rows, **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )
    return train_loader, valid_loader


def compute_pos_weight(loader: DataLoader, device: torch.device) -> torch.Tensor:
    targets = []
    for batch in loader:
        targets.append(batch["target"])
    target_tensor = torch.cat(targets, dim=0)
    positives = target_tensor.sum(dim=0)
    negatives = target_tensor.shape[0] - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    return pos_weight.to(device)


def aggregate_frame_logits(frame_logits: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    mask = frame_mask.unsqueeze(-1)
    masked_logits = frame_logits * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return masked_logits.sum(dim=1) / denom


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    all_targets = []
    all_probs = []

    for batch in tqdm(loader, leave=False):
        images = batch["images"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        targets = batch["target"].to(device)

        batch_size, num_frames, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_frames, channels, height, width)

        with torch.set_grad_enabled(is_train):
            frame_logits = model(flat_images).view(batch_size, num_frames, len(LABELS))
            logits = aggregate_frame_logits(frame_logits, frame_mask)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    targets_np = np.concatenate(all_targets, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    return mean_loss, targets_np, probs_np


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    set_seed(int(config["seed"]))

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    train_loader, valid_loader = build_dataloaders(config, args.index_path)
    model = build_model(config).to(device)

    pos_weight = compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]))

    best_score = -1.0
    history = []

    thresholds = [float(config["inference"]["thresholds"][label]) for label in LABELS]

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_targets, valid_probs = run_epoch(model, valid_loader, None, criterion, device)
        metrics = compute_macro_f1(valid_targets, valid_probs, thresholds=thresholds)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": metrics["per_class_f1"],
        }
        history.append(epoch_result)
        print(epoch_result)

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            save_json(
                {
                    "best_epoch": epoch,
                    "best_macro_f1": best_score,
                    "thresholds": thresholds,
                    "history": history,
                },
                output_dir / "train_summary.json",
            )


if __name__ == "__main__":
    main()
