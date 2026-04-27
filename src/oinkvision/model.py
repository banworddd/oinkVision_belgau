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
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FrameClassifierWithXShapeAux(nn.Module):
    """Frame classifier with an auxiliary x_shape branch."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 4,
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            "logits": self.main_head(features),
            "xshape_aux_logits": self.xshape_aux_head(features).squeeze(-1),
        }


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    if bool(model_cfg.get("use_xshape_aux_head", False)):
        return FrameClassifierWithXShapeAux(
            backbone_name=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            num_classes=int(model_cfg["num_classes"]),
        )
    return FrameClassifier(
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg["pretrained"]),
        num_classes=int(model_cfg["num_classes"]),
    )
