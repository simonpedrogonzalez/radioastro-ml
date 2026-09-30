"""Minimal ImageNet ResNet-18 construction and fine-tuning policy."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18 as _resnet18


def _replace_input(model: nn.Module, channels: int) -> None:
    old = model.conv1
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride,
                    old.padding, bias=False)
    with torch.no_grad():
        if channels == 4:
            new.weight.copy_(torch.cat((old.weight, old.weight.mean(1, keepdim=True)), 1) * 3 / 4)
        elif channels == 2:
            new.weight.copy_(old.weight[:, :2] * 3 / 2)
        else:
            raise ValueError("ResNet input must have two or four channels")
    model.conv1 = new


def build(num_classes: int, channels: int, mode: str, *, lr_scale: float = 1.0,
          weights=ResNet18_Weights.IMAGENET1K_V1):
    """Return the model, AdamW parameter groups, and epoch train-mode callback."""

    if mode not in {"head", "last", "all"}:
        raise ValueError(f"Unknown ResNet train mode: {mode}")
    model = _resnet18(weights=weights)
    _replace_input(model, channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.requires_grad_(mode == "all")
    model.fc.requires_grad_(True)
    if mode == "last":
        model.layer4[-1].requires_grad_(True)

    def set_train_mode() -> None:
        if mode != "all":
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    if mode == "head":
        groups = [(model.fc.parameters(), 1e-3)]
    elif mode == "last":
        groups = [(model.layer4[-1].parameters(), 1e-4), (model.fc.parameters(), 1e-3)]
    else:
        head = set(model.fc.parameters())
        groups = [((p for p in model.parameters() if p not in head), 1e-5),
                  (model.fc.parameters(), 1e-3)]
    return model, [{"params": list(params), "lr": lr * lr_scale}
                   for params, lr in groups], set_train_mode


__all__ = ["build"]
