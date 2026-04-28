"""Model definitions for baseline experiments."""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


class FrameClassifier(nn.Module):
    """Simple frame-level multi-label classifier."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 4,
        use_front_meta: bool = False,
        front_meta_dim: int = 8,
        front_meta_hidden_dim: int = 16,
        front_meta_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.use_front_meta = bool(use_front_meta)
        if self.use_front_meta:
            self.front_meta_head = nn.Sequential(
                nn.Linear(front_meta_dim, front_meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(front_meta_dropout),
                nn.Linear(front_meta_hidden_dim, num_classes),
            )
        else:
            self.front_meta_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return features

    def forward_meta(self, front_meta: torch.Tensor) -> torch.Tensor | None:
        if self.front_meta_head is None:
            return None
        return self.front_meta_head(front_meta)


class FrameClassifierWithXShapeAux(nn.Module):
    """Frame classifier with an auxiliary x_shape branch."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 4,
        use_front_meta: bool = False,
        front_meta_dim: int = 8,
        front_meta_hidden_dim: int = 16,
        front_meta_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        num_features = int(getattr(self.encoder, "num_features"))
        self.main_head = nn.Linear(num_features, num_classes)
        self.xshape_aux_head = nn.Linear(num_features, 1)
        self.use_front_meta = bool(use_front_meta)
        if self.use_front_meta:
            self.front_meta_head = nn.Sequential(
                nn.Linear(front_meta_dim, front_meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(front_meta_dropout),
                nn.Linear(front_meta_hidden_dim, num_classes),
            )
        else:
            self.front_meta_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            "logits": self.main_head(features),
            "xshape_aux_logits": self.xshape_aux_head(features).squeeze(-1),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_meta(self, front_meta: torch.Tensor) -> torch.Tensor | None:
        if self.front_meta_head is None:
            return None
        return self.front_meta_head(front_meta)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    front_meta_cfg = config.get("front_metadata", {})
    use_front_meta = bool(front_meta_cfg.get("enabled", False))
    front_meta_dim = int(front_meta_cfg.get("dim", 8))
    front_meta_hidden_dim = int(front_meta_cfg.get("hidden_dim", 16))
    front_meta_dropout = float(front_meta_cfg.get("dropout", 0.1))
    if bool(model_cfg.get("use_xshape_aux_head", False)):
        return FrameClassifierWithXShapeAux(
            backbone_name=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            num_classes=int(model_cfg["num_classes"]),
            use_front_meta=use_front_meta,
            front_meta_dim=front_meta_dim,
            front_meta_hidden_dim=front_meta_hidden_dim,
            front_meta_dropout=front_meta_dropout,
        )
    return FrameClassifier(
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg["pretrained"]),
        num_classes=int(model_cfg["num_classes"]),
        use_front_meta=use_front_meta,
        front_meta_dim=front_meta_dim,
        front_meta_hidden_dim=front_meta_hidden_dim,
        front_meta_dropout=front_meta_dropout,
    )
