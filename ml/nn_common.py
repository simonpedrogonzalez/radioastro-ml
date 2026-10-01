"""Shared data, training, evaluation, and artifacts for the two NN backbones."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ml.task_evaluation import evaluate_noise_robustness, evaluate_task
from scripts.preprocessing import FitsSimulationDataset, source_dataset_id


RAW_CHANNELS = ("dirty", "clean", "residual", "psf")
PARITY_CHANNELS = ("even", "odd")
RESIDUAL_RRR_CHANNELS = ("residual_rrr",)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LoadedSplit:
    raw: torch.Tensor | None
    parity: torch.Tensor | None
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]
    residual_rrr: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedSplit:
    images: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]


def parity_channels(residual: torch.Tensor) -> torch.Tensor:
    """Return native-grid even/odd residual components about pixel (128, 128)."""

    if residual.ndim != 2 or residual.shape[0] != residual.shape[1] or residual.shape[0] % 2:
        raise ValueError("Parity requires a square, even-sized residual plane")
    output = residual.new_zeros((2, *residual.shape))
    paired = residual[1:, 1:]
    opposite = paired.flip((-2, -1))
    output[0, 1:, 1:] = (paired + opposite) / 2
    output[1, 1:, 1:] = (paired - opposite) / 2
    return output


def residual_rrr_input(residual: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Scale one native residual by annulus MAD and form ImageNet-ready RRR input."""

    if residual.shape != (256, 256) or not torch.isfinite(residual).all():
        raise ValueError("RRR input requires one finite 256x256 residual plane")
    residual = residual.float()
    y, x = torch.meshgrid(torch.arange(256, dtype=residual.dtype),
                          torch.arange(256, dtype=residual.dtype), indexing="ij")
    radius = torch.hypot(x - 128, y - 128)
    values = residual[(radius >= 32) & (radius < 72)]
    median = values.median()
    scale = 1.4826 * (values - median).abs().median()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("residual annulus has nonpositive or nonfinite MAD scale")
    transformed = torch.asinh((residual / scale).clamp(-10, 10)) / np.arcsinh(10)
    rgb = ((transformed + 1) / 2).repeat(3, 1, 1)
    rgb = F.interpolate(rgb[None], (224, 224), mode="bilinear",
                        align_corners=False, antialias=True)[0]
    mean = rgb.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = rgb.new_tensor(IMAGENET_STD)[:, None, None]
    return (rgb - mean) / std, float(scale)


def _load_split(dataset_path: Path, partition: str,
                representations: frozenset[str]) -> LoadedSplit:
    dataset = FitsSimulationDataset(dataset_path.parent, index=dataset_path.name,
                                    partition=partition,
                                    label_criterion="constant_antenna_type")
    raw = [] if "raw" in representations else None
    parity = [] if "parity" in representations else None
    residual_rrr = [] if "residual_rrr" in representations else None
    labels, ids, sources, metadata = [], [], [], []
    print(f"Loading {partition}: {len(dataset)} samples", flush=True)
    for index, sample in enumerate(dataset, 1):
        image = sample["image"]
        if image.shape != (4, 256, 256) or not torch.isfinite(image).all():
            raise ValueError(f'{partition}/{sample["sample_id"]}: expected finite (4,256,256) planes')
        if raw is not None:
            raw.append(F.interpolate(image[None], (224, 224), mode="bilinear",
                                     align_corners=False, antialias=True)[0])
        if parity is not None:
            components = parity_channels(image[2])
            parity.append(F.interpolate(components[None], (224, 224), mode="bilinear",
                                        align_corners=False, antialias=True)[0])
        if residual_rrr is not None:
            try:
                prepared, _ = residual_rrr_input(image[2])
            except ValueError as exc:
                raise ValueError(f'{partition}/{sample["sample_id"]}: {exc}') from exc
            residual_rrr.append(prepared)
        labels.append(int(sample["label"]))
        ids.append(sample["sample_id"])
        sources.append(source_dataset_id(sample["sample_id"]))
        metadata.append(dict(sample["label_metadata"]))
        if index % 100 == 0 or index == len(dataset):
            print(f"Loading {partition}: {index}/{len(dataset)}", flush=True)
    if not labels:
        raise ValueError(f"{partition} split is empty")
    return LoadedSplit(torch.stack(raw) if raw is not None else None,
                       torch.stack(parity) if parity is not None else None,
                       torch.tensor(labels), tuple(ids), tuple(sources), tuple(metadata),
                       torch.stack(residual_rrr) if residual_rrr is not None else None)


def load_data(dataset_path: str | Path, representations: Sequence[str] = ("raw", "parity"),
              ) -> tuple[dict[str, LoadedSplit], tuple[int, ...]]:
    path = Path(dataset_path).expanduser().resolve()
    requested = frozenset(representations)
    if not requested or not requested <= {"raw", "parity", "residual_rrr"}:
        raise ValueError(f"Unknown input representations: {sorted(requested)}")
    splits = {
        "train": _load_split(path, "train", requested),
        "validation": _load_split(path, "val", requested),
    }
    control_path = path.with_name("dataset_noise_controls.json")
    if not control_path.is_file():
        raise FileNotFoundError(f"Noise-control index not found: {control_path}")
    splits["noise_controls"] = _load_split(control_path, "val", requested)
    labels = tuple(sorted(set(splits["train"].labels.tolist())))
    if len(labels) < 3 or set(splits["validation"].labels.tolist()) != set(labels):
        raise ValueError("Train and validation must contain the same three or more classes")
    if set(splits["noise_controls"].labels.tolist()) != {0}:
        raise ValueError("Noise controls must all use clean label 0")
    return splits, labels


def prepare(splits: dict[str, LoadedSplit], channels: Sequence[str],
            labels: Sequence[int]) -> tuple[dict[str, PreparedSplit], dict[str, Any]]:
    """Normalize included channels from training only; leave other raw slots zero."""

    names = tuple(channels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    if names == RESIDUAL_RRR_CHANNELS:
        prepared = {}
        for split_name, split in splits.items():
            if split.residual_rrr is None:
                raise ValueError("Residual RRR representation was not loaded")
            encoded = torch.tensor([label_to_index[int(label)] for label in split.labels])
            prepared[split_name] = PreparedSplit(split.residual_rrr, encoded, split.sample_ids,
                                                 split.source_ids, split.metadata)
        settings = {
            "source": "residual_rrr",
            "slots": ["red", "green", "blue"],
            "included": ["residual", "residual", "residual"],
            "recipe_version": 1,
            "per_image_scale": {"kind": "annulus_mad", "center": [128, 128],
                                "inner_radius": 32, "outer_radius": 72,
                                "consistency_factor": 1.4826},
            "transform": {"kind": "signed_asinh", "clip": 10.0,
                          "mapped_range": [0.0, 1.0]},
            "resize": {"shape": [224, 224], "mode": "bilinear", "antialias": True},
            "imagenet_mean": list(IMAGENET_MEAN), "imagenet_std": list(IMAGENET_STD),
        }
        return prepared, settings

    parity = names == PARITY_CHANNELS
    allowed = PARITY_CHANNELS if parity else RAW_CHANNELS
    if not names or len(set(names)) != len(names) or not set(names) <= set(allowed):
        raise ValueError(f"Invalid channel set: {names}")
    source = "parity" if parity else "raw"
    indices = [allowed.index(name) for name in names]
    train = getattr(splits["train"], source)
    if train is None:
        raise ValueError(f"{source} representation was not loaded")
    stats: dict[str, dict[str, float]] = {}
    for name, index in zip(names, indices, strict=True):
        mean = float(train[:, index].mean())
        std = float(train[:, index].std(unbiased=False))
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError(f"Training channel {name!r} has invalid normalization")
        stats[name] = {"mean": mean, "std": std}
    prepared = {}
    for split_name, split in splits.items():
        values = getattr(split, source)
        if values is None:
            raise ValueError(f"{source} representation was not loaded")
        output = torch.zeros((len(values), len(allowed), 224, 224), dtype=torch.float32)
        for name, index in zip(names, indices, strict=True):
            output[:, index] = (values[:, index] - stats[name]["mean"]) / stats[name]["std"]
        encoded = torch.tensor([label_to_index[int(label)] for label in split.labels])
        prepared[split_name] = PreparedSplit(output, encoded, split.sample_ids,
                                             split.source_ids, split.metadata)
    return prepared, {"source": source, "slots": list(allowed),
                      "included": list(names), "statistics": stats}


class _Rotations(Dataset):
    def __init__(self, split: PreparedSplit) -> None:
        self.split = split

    def __len__(self) -> int:
        return len(self.split.labels)

    def __getitem__(self, index: int):
        return rotate(self.split.images[index]), self.split.labels[index]


def rotate(image: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Apply one joint right-angle rotation to every channel."""

    return torch.rot90(image, int(torch.randint(4, (1,))) if k is None else k, (-2, -1))


@torch.no_grad()
def predict(model: nn.Module, split: PreparedSplit, device: torch.device,
            batch_size: int,
            class_weights: torch.Tensor | None = None) -> tuple[np.ndarray, float]:
    model.eval()
    logits, total_loss, loss_weight = [], 0.0, 0.0
    loader = DataLoader(TensorDataset(split.images, split.labels), batch_size=batch_size)
    for images, target in loader:
        values = model(images.to(device))
        logits.append(values.cpu())
        target = target.to(device)
        losses = F.cross_entropy(values, target, weight=class_weights, reduction="none")
        total_loss += float(losses.sum())
        loss_weight += float(class_weights[target].sum()) if class_weights is not None else len(target)
    scores = torch.cat(logits).softmax(1).numpy()
    return scores, total_loss / loss_weight


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction_rows(split_name: str, split: PreparedSplit, probabilities: np.ndarray,
                     labels: Sequence[int]) -> list[dict[str, Any]]:
    predicted = np.argmax(probabilities, axis=1)
    rows = []
    for index, metadata in enumerate(split.metadata):
        row = {"sample_id": split.sample_ids[index], "source_id": split.source_ids[index],
               "split": split_name, "true_label": labels[int(split.labels[index])],
               "predicted_label": labels[int(predicted[index])], **metadata}
        row.update({f"probability_{label}": float(probabilities[index, column])
                    for column, label in enumerate(labels)})
        rows.append(row)
    return rows


def train_job(model: nn.Module, groups: list[dict[str, Any]], set_train_mode,
              splits: dict[str, PreparedSplit], labels: Sequence[int], clean_label: int,
              config: dict[str, Any], run_dir: str | Path, device: str | torch.device) -> dict[str, Any]:
    """Train one seeded job, select on validation macro F1/recall, and save it."""

    run_dir, device = Path(run_dir), torch.device(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed, batch_size = int(config["seed"]), int(config.get("batch_size", 16))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model.to(device)
    counts = torch.bincount(splits["train"].labels, minlength=len(labels)).float()
    class_weights = len(splits["train"].labels) / (len(labels) * counts)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 1e-4)))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    loader = DataLoader(_Rotations(splits["train"]), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history, best_state, best_loss, best_epoch, stale = [], None, float("inf"), None, 0
    started = time.monotonic()
    for epoch in range(int(config["epochs"])):
        warmup = int(config.get("warmup_epochs", 0))
        factor = min(1.0, (epoch + 1) / warmup) if warmup else 1.0
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * factor
        model.train(); set_train_mode()
        loss_sum = loss_weight = 0.0
        for images, target in loader:
            optimizer.zero_grad(set_to_none=True)
            target = target.to(device)
            losses = F.cross_entropy(model(images.to(device)), target,
                                     weight=class_weights, reduction="none")
            loss = losses.sum() / class_weights[target].sum()
            loss.backward(); optimizer.step()
            loss_sum += float(losses.detach().sum())
            loss_weight += float(class_weights[target].sum())
        probabilities, validation_loss = predict(
            model, splits["validation"], device, batch_size, class_weights
        )
        truth = np.asarray([labels[int(index)] for index in splits["validation"].labels])
        metrics = evaluate_task(truth, probabilities, labels, splits["validation"].metadata,
                                clean_label=clean_label)["overall"]["main"]
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / loss_weight,
                        "validation_loss": validation_loss, "validation_f1": metrics["f1"],
                        "validation_recall": metrics["recall"], "lr": [g["lr"] for g in optimizer.param_groups]})
        print(f'{config["job_id"]}: epoch {epoch + 1}/{config["epochs"]} '
              f'loss={history[-1]["train_loss"]:.5g} val_f1={metrics["f1"]:.4f}', flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        elif epoch + 1 > warmup:
            stale += 1
            patience = config.get("patience", 5)
            if patience is not None and stale >= int(patience):
                break
    assert best_state is not None and best_epoch is not None
    model.load_state_dict(best_state)
    results: dict[str, Any] = {"best_epoch": best_epoch,
                               "runtime_seconds": time.monotonic() - started,
                               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    rows, probabilities_by_split = [], {}
    for split_name, split in splits.items():
        probabilities, loss = predict(model, split, device, batch_size, class_weights)
        probabilities_by_split[split_name] = probabilities
        truth = np.asarray([labels[int(index)] for index in split.labels])
        if split_name != "noise_controls":
            results[split_name] = evaluate_task(truth, probabilities, labels, split.metadata,
                                                clean_label=clean_label)
            results[f"{split_name}_loss"] = loss
        rows.extend(_prediction_rows(split_name, split, probabilities, labels))
    controls, validation = splits["noise_controls"], splits["validation"]
    results["noise_controls"] = evaluate_noise_robustness(
        np.asarray([labels[int(index)] for index in controls.labels]),
        probabilities_by_split["noise_controls"], labels, controls.metadata,
        controls.source_ids,
        np.asarray([labels[int(index)] for index in validation.labels]),
        probabilities_by_split["validation"], validation.metadata,
        validation.source_ids, clean_label=clean_label,
    )
    checkpoint = run_dir / "checkpoint.pt.tmp"
    torch.save({"model": best_state, "labels": list(labels), "normalization": config["normalization"],
                "best_epoch": results["best_epoch"]}, checkpoint)
    checkpoint.replace(run_dir / "checkpoint.pt")
    _atomic_json(run_dir / "config.json", config)
    _atomic_json(run_dir / "history.json", history)
    _atomic_json(run_dir / "results.json", results)
    temporary = run_dir / "predictions.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temporary.replace(run_dir / "predictions.csv")
    return results


def choose_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "PARITY_CHANNELS", "RAW_CHANNELS",
           "RESIDUAL_RRR_CHANNELS",
           "LoadedSplit", "PreparedSplit",
           "choose_device", "load_data", "parity_channels", "predict", "prepare",
           "residual_rrr_input", "rotate", "train_job"]
