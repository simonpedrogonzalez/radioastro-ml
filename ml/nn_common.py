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

from ml.task_evaluation import evaluate_task
from scripts.preprocessing import FitsSimulationDataset, source_dataset_id


RAW_CHANNELS = ("dirty", "clean", "residual", "psf")
PARITY_CHANNELS = ("even", "odd")


@dataclass(frozen=True)
class LoadedSplit:
    raw: torch.Tensor
    parity: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]


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


def _load_split(dataset_path: Path, partition: str) -> LoadedSplit:
    dataset = FitsSimulationDataset(dataset_path.parent, index=dataset_path.name,
                                    partition=partition,
                                    label_criterion="constant_antenna_type")
    raw, parity, labels, ids, sources, metadata = [], [], [], [], [], []
    print(f"Loading {partition}: {len(dataset)} samples", flush=True)
    for index, sample in enumerate(dataset, 1):
        image = sample["image"]
        if image.shape != (4, 256, 256) or not torch.isfinite(image).all():
            raise ValueError(f'{partition}/{sample["sample_id"]}: expected finite (4,256,256) planes')
        raw.append(F.interpolate(image[None], (224, 224), mode="bilinear",
                                 align_corners=False, antialias=True)[0])
        components = parity_channels(image[2])
        parity.append(F.interpolate(components[None], (224, 224), mode="bilinear",
                                    align_corners=False, antialias=True)[0])
        labels.append(int(sample["label"]))
        ids.append(sample["sample_id"])
        sources.append(source_dataset_id(sample["sample_id"]))
        metadata.append(dict(sample["label_metadata"]))
        if index % 100 == 0 or index == len(dataset):
            print(f"Loading {partition}: {index}/{len(dataset)}", flush=True)
    if not raw:
        raise ValueError(f"{partition} split is empty")
    return LoadedSplit(torch.stack(raw), torch.stack(parity), torch.tensor(labels),
                       tuple(ids), tuple(sources), tuple(metadata))


def load_data(dataset_path: str | Path) -> tuple[dict[str, LoadedSplit], tuple[int, ...]]:
    path = Path(dataset_path).expanduser().resolve()
    splits = {name: _load_split(path, name) for name in ("train", "validation")}
    labels = tuple(sorted(set(splits["train"].labels.tolist())))
    if len(labels) < 3 or set(splits["validation"].labels.tolist()) != set(labels):
        raise ValueError("Train and validation must contain the same three or more classes")
    return splits, labels


def prepare(splits: dict[str, LoadedSplit], channels: Sequence[str],
            labels: Sequence[int]) -> tuple[dict[str, PreparedSplit], dict[str, Any]]:
    """Normalize included channels from training only; leave other raw slots zero."""

    names = tuple(channels)
    parity = names == PARITY_CHANNELS
    allowed = PARITY_CHANNELS if parity else RAW_CHANNELS
    if not names or len(set(names)) != len(names) or not set(names) <= set(allowed):
        raise ValueError(f"Invalid channel set: {names}")
    source = "parity" if parity else "raw"
    indices = [allowed.index(name) for name in names]
    train = getattr(splits["train"], source)
    stats: dict[str, dict[str, float]] = {}
    for name, index in zip(names, indices, strict=True):
        mean = float(train[:, index].mean())
        std = float(train[:, index].std(unbiased=False))
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError(f"Training channel {name!r} has invalid normalization")
        stats[name] = {"mean": mean, "std": std}
    label_to_index = {label: index for index, label in enumerate(labels)}
    prepared = {}
    for split_name, split in splits.items():
        values = getattr(split, source)
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
            batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    logits, total_loss = [], 0.0
    loader = DataLoader(TensorDataset(split.images, split.labels), batch_size=batch_size)
    for images, target in loader:
        values = model(images.to(device))
        logits.append(values.cpu())
        total_loss += float(F.cross_entropy(values, target.to(device), reduction="sum"))
    scores = torch.cat(logits).softmax(1).numpy()
    return scores, total_loss / len(split.labels)


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
    history, best_state, best_key, stale = [], None, None, 0
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
        probabilities, validation_loss = predict(model, splits["validation"], device, batch_size)
        truth = np.asarray([labels[int(index)] for index in splits["validation"].labels])
        metrics = evaluate_task(truth, probabilities, labels, splits["validation"].metadata,
                                clean_label=clean_label)["overall"]["main"]
        key = (metrics["f1"] if metrics["f1"] is not None else -1,
               metrics["recall"] if metrics["recall"] is not None else -1, -epoch)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / loss_weight,
                        "validation_loss": validation_loss, "validation_f1": metrics["f1"],
                        "validation_recall": metrics["recall"], "lr": [g["lr"] for g in optimizer.param_groups]})
        print(f'{config["job_id"]}: epoch {epoch + 1}/{config["epochs"]} '
              f'loss={history[-1]["train_loss"]:.5g} val_f1={metrics["f1"]:.4f}', flush=True)
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config.get("patience", 5)):
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    results: dict[str, Any] = {"best_epoch": -best_key[2] + 1,
                               "runtime_seconds": time.monotonic() - started,
                               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    rows = []
    for split_name, split in splits.items():
        probabilities, loss = predict(model, split, device, batch_size)
        truth = np.asarray([labels[int(index)] for index in split.labels])
        results[split_name] = evaluate_task(truth, probabilities, labels, split.metadata,
                                            clean_label=clean_label)
        results[f"{split_name}_loss"] = loss
        rows.extend(_prediction_rows(split_name, split, probabilities, labels))
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


__all__ = ["PARITY_CHANNELS", "RAW_CHANNELS", "LoadedSplit", "PreparedSplit",
           "choose_device", "load_data", "parity_channels", "predict", "prepare", "rotate", "train_job"]
