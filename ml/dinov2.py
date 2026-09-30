"""Minimal official DINOv2-S/14 classifier for four radio-image products."""

from __future__ import annotations

import torch
from torch import nn


HUB_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
HUB_REPOSITORY = f"facebookresearch/dinov2:{HUB_REVISION}"


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(4, 3, 1)
        self.backbone = backbone
        self.head = nn.Linear(384, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.adapter(inputs)))


def build(num_classes: int, mode: str, *, lr_scale: float = 1.0,
          backbone: nn.Module | None = None):
    """Return the model, AdamW parameter groups, and epoch train-mode callback."""

    if mode not in {"linear", "last"}:
        raise ValueError(f"Unknown DINOv2 train mode: {mode}")
    if backbone is None:
        backbone = torch.hub.load(HUB_REPOSITORY, "dinov2_vits14", trust_repo=True)
    model = Classifier(backbone, num_classes)
    model.requires_grad_(False)
    model.adapter.requires_grad_(True)
    model.head.requires_grad_(True)
    last = None
    if mode == "last":
        last = model.backbone.blocks[-1]
        last.requires_grad_(True)

    def set_train_mode() -> None:
        model.backbone.eval()
        if last is not None:
            last.train()

    groups = [(list(model.adapter.parameters()) + list(model.head.parameters()), 1e-3)]
    if last is not None:
        groups.append((list(last.parameters()), 1e-5))
    return model, [{"params": params, "lr": lr * lr_scale} for params, lr in groups], set_train_mode


__all__ = ["HUB_REVISION", "build"]
