"""DataLoader construction helpers."""

from __future__ import annotations

from typing import Any

from .dataset import FitsSimulationDataset, _require_torch, simulation_collate


def make_simulation_dataloader(
    dataset: FitsSimulationDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    **kwargs: Any,
):
    torch = _require_torch()
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=simulation_collate,
        **kwargs,
    )


__all__ = ["make_simulation_dataloader"]
