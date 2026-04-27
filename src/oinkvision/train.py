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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
        "--train-index",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--valid-index",
        type=Path,
        default=None,
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


def build_dataloaders_from_rows(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    index_path: Path,
) -> tuple[DataLoader, DataLoader]:
    common_kwargs = {
        "index_path": index_path,
        "frames_per_camera": config["data"]["frames_per_camera"],
        "image_size": config["data"]["image_size"],
        "use_bbox_crops": config["data"]["use_bbox_crops"],
        "frame_cache_dir": config["data"].get("frame_cache_dir"),
        "seed": config["seed"],
    }
    train_dataset = PigVideoDataset(
        rows=train_rows,
        augment=bool(config["train"].get("train_augmentations", False)),
        **common_kwargs,
    )
    valid_dataset = PigVideoDataset(
        rows=valid_rows,
        augment=False,
        **common_kwargs,
    )

    sampler = None
    if bool(config["train"].get("use_weighted_sampler", False)):
        sampler_power = float(config["train"].get("sampler_power", 1.0))
        sampler_max_weight = float(config["train"].get("sampler_max_weight", 10.0))
        label_counts = {
            label: max(sum(int(row[label]) for row in train_rows), 1)
            for label in LABELS
        }
        class_weights = {
            label: min((len(train_rows) / count) ** sampler_power, sampler_max_weight)
            for label, count in label_counts.items()
        }
        sample_weights = []
        for row in train_rows:
            positive_weights = [class_weights[label] for label in LABELS if int(row[label]) == 1]
            sample_weights.append(max(positive_weights) if positive_weights else 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config["train"]["num_workers"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )
    return train_loader, valid_loader


def build_dataloaders(
    config: dict[str, Any],
    index_path: Path,
    train_index_path: Path | None = None,
    valid_index_path: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    if train_index_path is not None and valid_index_path is not None:
        train_rows = load_index(train_index_path)
        valid_rows = load_index(valid_index_path)
        return build_dataloaders_from_rows(config, train_rows, valid_rows, train_index_path)

    rows = load_index(index_path)
    train_rows, valid_rows = split_rows(rows, seed=config["seed"], valid_size=config["train"]["valid_size"])
    return build_dataloaders_from_rows(config, train_rows, valid_rows, index_path)


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
    train_loader, valid_loader = build_dataloaders(
        config,
        args.index_path,
        train_index_path=args.train_index,
        valid_index_path=args.valid_index,
    )
    model = build_model(config).to(device)

    pos_weight = compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]))
    scheduler = None
    scheduler_mode = str(config["train"].get("scheduler_mode", "max"))
    if bool(config["train"].get("use_plateau_scheduler", False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=float(config["train"].get("scheduler_factor", 0.5)),
            patience=int(config["train"].get("scheduler_patience", 2)),
            min_lr=float(config["train"].get("min_lr", 1e-6)),
        )

    best_score = -1.0
    epochs_without_improvement = 0
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
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
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_result)
        print(epoch_result)

        if scheduler is not None:
            monitor_value = metrics["macro_f1"] if scheduler_mode == "max" else valid_loss
            scheduler.step(monitor_value)

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            save_json(
                {
                    "best_epoch": epoch,
                    "best_macro_f1": best_score,
                    "thresholds": thresholds,
                    "train_index": str(args.train_index) if args.train_index is not None else str(args.index_path),
                    "valid_index": str(args.valid_index) if args.valid_index is not None else "internal_split_from_index",
                    "epochs_ran": len(history),
                    "history": history,
                },
                output_dir / "train_summary.json",
            )
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                {
                    "early_stopping": True,
                    "stopped_epoch": epoch,
                    "best_macro_f1": best_score,
                }
            )
            break


if __name__ == "__main__":
    main()
