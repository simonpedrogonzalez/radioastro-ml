"""PyTorch dataset for compact simulation samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .fits import validate_fits_products
from .labels import AssignedLabel, assign_label
from .partitions import normalize_partition, partition_for_sample
from .schema import DatasetManifest, SampleManifest, load_dataset_manifest, load_sample_manifest

try:  # Keep schema/cleanup imports usable in lightweight Python environments.
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - depends on the caller's environment
    torch = None
    Dataset = object  # type: ignore[assignment,misc]


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required to load FITS samples as tensors")
    return torch


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metadata JSON must contain an object: {path}")
    return value


class FitsSimulationDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        root: str | Path,
        *,
        partition: str,
        index: str = "dataset.json",
        channels: Sequence[str] = ("dirty", "clean", "residual", "psf"),
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        validate: bool = True,
        load_metadata: bool = True,
        label_criterion: str | None = None,
        invalid_policy: str = "error",
        fill_value: float = 0.0,
    ) -> None:
        _require_torch()
        self.root = Path(root).expanduser().resolve()
        self.partition = normalize_partition(partition)
        self.channels = tuple(channels)
        if self.channels != ("dirty", "clean", "residual", "psf"):
            raise ValueError(
                "channels must preserve ('dirty', 'clean', 'residual', 'psf') order"
            )
        self.transform = transform
        self.target_transform = target_transform
        self.load_metadata = load_metadata
        self.label_criterion = label_criterion
        self.invalid_policy = invalid_policy
        self.fill_value = fill_value
        # Read every manifest only far enough to identify its source dataset.
        # Full file and checksum validation is deliberately limited to the
        # requested partition.
        self.index: DatasetManifest = load_dataset_manifest(
            self.root / index, require_samples=False
        )
        selected: list[SampleManifest] = []
        for path in self.index.samples:
            candidate = load_sample_manifest(
                path, require_files=False, verify_integrity=False
            )
            if self.index.labels.get(candidate.label_name) != candidate.label_id:
                raise ValueError(
                    f"Sample {candidate.sample_id!r} label does not match "
                    "dataset label map"
                )
            if partition_for_sample(candidate.sample_id) != self.partition:
                continue
            if validate:
                candidate = load_sample_manifest(
                    path, require_files=True, verify_integrity=True
                )
            selected.append(candidate)
        self.samples = tuple(selected)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch_module = _require_torch()
        manifest = self.samples[index]
        planes = validate_fits_products(
            manifest.products,
            invalid_policy=self.invalid_policy,
            fill_value=self.fill_value,
        )
        image = torch_module.stack(
            [torch_module.from_numpy(planes[name].values.copy()) for name in self.channels]
        ).to(dtype=torch_module.float32)
        simulation = _read_json(manifest.simulation)
        noise_control = simulation.get("noise_control")
        if noise_control is not None and not isinstance(noise_control, dict):
            raise ValueError(
                f"simulation.noise_control must be an object: {manifest.simulation}"
            )
        corruption_reports = (
            [_read_json(reference.corruption) for reference in manifest.corruptions]
            if self.load_metadata or self.label_criterion is not None or noise_control is not None
            else []
        )
        if self.label_criterion is None:
            if noise_control is not None:
                derived = assign_label(
                    corruption_reports,
                    "constant_antenna_type",
                    noise_control=noise_control,
                )
                if (manifest.label_id, manifest.label_name) != (0, "not_corrupted"):
                    raise ValueError(
                        f"Increased-noise sample {manifest.sample_id!r} must retain "
                        "label 0/not_corrupted"
                    )
                assigned = AssignedLabel(
                    manifest.label_id,
                    manifest.label_name,
                    derived.corruption_snr,
                    derived.corruption_snr_target,
                    derived.sample_kind,
                )
            else:
                assigned = AssignedLabel(
                    manifest.label_id,
                    manifest.label_name,
                    None,
                    None,
                    "gain" if manifest.corruptions else "baseline",
                )
        else:
            assigned = assign_label(
                corruption_reports,
                self.label_criterion,
                noise_control=noise_control,
            )
        target: Any = assigned.id
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        metadata: dict[str, Any] | None = None
        if self.load_metadata:
            metadata = {
                "simulation": simulation,
                "imaging": _read_json(manifest.imaging_qa),
                "corruptions": corruption_reports,
            }
        return {
            "image": image,
            "label": target,
            "label_name": assigned.name,
            "label_metadata": {
                "corruption_snr": assigned.corruption_snr,
                "corruption_snr_target": assigned.corruption_snr_target,
                "sample_kind": assigned.sample_kind,
            },
            "sample_id": manifest.sample_id,
            "partition": self.partition,
            "qa": None if metadata is None else metadata["imaging"],
            "metadata": metadata,
            "paths": {
                "manifest": manifest.path,
                "products": dict(manifest.products),
                "simulation": manifest.simulation,
                "imaging": manifest.imaging_qa,
                "corruptions": [
                    {
                        "corruption": reference.corruption,
                        "corruption_text": reference.corruption_text,
                    }
                    for reference in manifest.corruptions
                ],
            },
        }


def simulation_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    torch_module = _require_torch()
    return {
        "image": torch_module.stack([item["image"] for item in batch]),
        "label": torch_module.as_tensor([item["label"] for item in batch]),
        "label_name": [item["label_name"] for item in batch],
        "label_metadata": [item["label_metadata"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "partition": [item["partition"] for item in batch],
        "qa": [item["qa"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "paths": [item["paths"] for item in batch],
    }


__all__ = ["FitsSimulationDataset", "simulation_collate"]
