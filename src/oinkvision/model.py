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


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    return FrameClassifier(
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg["pretrained"]),
        num_classes=int(model_cfg["num_classes"]),
    )
