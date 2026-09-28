"""Square-support ResNet-18 experiment; no CASA or dataset mutations."""

import argparse
import hashlib
import platform
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from scipy.ndimage import binary_propagation
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

from ml.evaluate import _summary, evaluate
from ml.logreg import FEATURE_NAMES, features_from_dataloader, train as train_logreg
from scripts.preprocessing import FitsSimulationDataset, source_dataset_id


PREPROCESSING = {
    "support_policy": "square_only", "native_shape": [256, 256],
    "centre": [128, 128], "parity_domain": "[1:,1:]; unmatched E/O edges zero",
    "scale": "1.4826 * median absolute deviation; 32 <= radius < 72",
    "clip": 10, "transform": "(asinh(clipped X/scale)/asinh(10)+1)/2",
    "resize": [224, 224], "interpolation": "bilinear; antialias=True; align_corners=False",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "augmentation": "joint random D4, training only; no crop",
}
FULL_FEATURE_NAMES = (
    "metrics.clean_peak_jy_per_beam", "metrics.dynamic_range_rms",
    "metrics.dynamic_range_scaled_mad",
    *("metrics.residual." + name for name in (
        "n_pixels", "area_synthesized_beams", "rms_jy_per_beam",
        "scaled_mad_jy_per_beam", "residual_abs_peak_jy_per_beam",
        "residual_min_jy_per_beam", "residual_max_jy_per_beam",
        "peak_over_scaled_mad", "p99_over_scaled_mad", "p99_5_over_scaled_mad",
        "rms_over_scaled_mad",
    )),
)
PARITY_FEATURE_NAMES = ("log1p_mean_even_energy_over_mad2", "log1p_mean_odd_energy_over_mad2")


def support_mask(image):
    if image.shape != (4, 256, 256) or not torch.isfinite(image).all():
        raise ValueError("Expected finite (4,256,256) FITS planes")
    filled = (image[:3] == 0).all(dim=0).numpy()
    if filled.any():
        boundary = filled.copy()
        boundary[1:-1, 1:-1] = False
        if not np.array_equal(binary_propagation(boundary, mask=filled), filled):
            raise ValueError("Unexpected interior shared-zero holes")
        if filled[128, 128]:
            raise ValueError("Invalid support: phase centre is zero-filled")
    return filled


def residual_channels(residual, mode):
    if mode == "residual":
        return residual.repeat(3, 1, 1)
    if mode != "parity":
        raise ValueError(f"Unknown input mode: {mode}")
    even, odd = torch.zeros_like(residual), torch.zeros_like(residual)
    paired = residual[1:, 1:]
    opposite = paired.flip((-2, -1))
    even[1:, 1:] = (paired + opposite) / 2
    odd[1:, 1:] = (paired - opposite) / 2
    return torch.stack((residual, even, odd))


def residual_scale(residual):
    y, x = np.indices((256, 256))
    radius = np.hypot(x - 128, y - 128)
    background = residual.numpy()[(radius >= 32) & (radius < 72)]
    scale = float(1.4826 * np.median(np.abs(background - np.median(background))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Residual annulus has nonpositive/nonfinite MAD scale")
    return scale


def parity_features(image):
    """Two raw parity energies on the paired native grid, before display transforms."""
    components = residual_channels(image[2], "parity")[1:, 1:, 1:].double()
    energies = (components / residual_scale(image[2])).square().mean(dim=(-2, -1))
    return torch.log1p(energies).numpy()


def preprocess(image, mode):
    residual = image[2]
    scale = residual_scale(residual)
    channels = residual_channels(residual, mode) / scale
    clipped = float((channels.abs() > 10).float().mean())
    channels = (torch.asinh(channels.clamp(-10, 10)) / np.arcsinh(10) + 1) / 2
    channels = F.interpolate(channels[None], (224, 224), mode="bilinear",
                             align_corners=False, antialias=True)[0]
    mean = channels.new_tensor(PREPROCESSING["mean"])[:, None, None]
    std = channels.new_tensor(PREPROCESSING["std"])[:, None, None]
    return (channels - mean) / std, clipped


@dataclass
class Split:
    images: torch.Tensor
    batch: dict
    audit: dict
    parity_features: np.ndarray | None = None

    def loader(self, shuffle=False):
        return DataLoader(TensorDataset(self.images, self.batch["label"]),
                          batch_size=16, shuffle=shuffle, num_workers=0)


def preload(dataset_path, partition, mode):
    dataset = FitsSimulationDataset(dataset_path.parent, index=dataset_path.name,
                                    partition=partition, label_criterion="constant_antenna_type")
    groups = {}
    for index, manifest in enumerate(dataset.samples):
        groups.setdefault(source_dataset_id(manifest.sample_id), []).append(index)
    batch = {key: [] for key in ("sample_id", "label", "label_metadata", "qa")}
    images, exclusions, clipping, features = [], [], [], []
    print(f"Loading {partition}: checking {len(groups)} sources", flush=True)
    for group_index, (source, indices) in enumerate(groups.items(), 1):
        source_mask = None
        for index in indices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                sample = dataset[index]  # Validates finite pixels and matching WCS/beam.
            sample_id = sample["sample_id"]
            try:
                header = fits.getheader(sample["paths"]["products"]["clean"])
                if (header.get("CRPIX1"), header.get("CRPIX2")) != (129, 129):
                    raise ValueError("Unsupported reference pixel; expected FITS (129,129)")
                mask = support_mask(sample["image"])
                if source_mask is not None and not np.array_equal(mask, source_mask):
                    raise ValueError("Mixed support across source variants")
                source_mask = mask
                if mask.any():
                    exclusions.append({"sample_id": sample_id, "source_id": source,
                                       "label": sample["label"], "reason": "exterior shared-zero support"})
                    continue
                tensor, fraction = preprocess(sample["image"], mode)
            except ValueError as exc:
                raise ValueError(f"{partition}/{sample_id}: {exc}") from exc
            images.append(tensor)
            clipping.append(fraction)
            features.append(parity_features(sample["image"]))
            for key in batch:
                batch[key].append(sample[key])
        if group_index % 20 == 0:
            print(f"Loading {partition}: {group_index}/{len(groups)} sources checked", flush=True)
    labels = torch.tensor(batch["label"], dtype=torch.long)
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError(f"{partition}: eligible split is empty or missing a class")
    batch["label"] = labels
    audit = {
        "retained_samples": len(images),
        "retained_sources": len({source_dataset_id(s) for s in batch["sample_id"]}),
        "retained_classes": torch.bincount(labels, minlength=3).tolist(),
        "excluded_samples": len(exclusions),
        "excluded_sources": len({row["source_id"] for row in exclusions}),
        "excluded_classes": np.bincount([row["label"] for row in exclusions], minlength=3).tolist(),
        "excluded": exclusions,
        "clipping_fraction_mean": float(np.mean(clipping)),
        "clipping_fraction_max": max(clipping),
    }
    print(f"{partition}: {len(images)} samples / {audit['retained_sources']} sources retained; "
          f"{len(exclusions)} samples excluded", flush=True)
    return Split(torch.stack(images), batch, audit, np.asarray(features))


def configure_stage(model, finetune):
    model.requires_grad_(False)
    model.fc.requires_grad_(True)
    if finetune:
        model.layer4[1].requires_grad_(True)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(False)
            module.eval()


def augment(images):
    transformed = []
    for image in images:
        image = torch.rot90(image, random.randrange(4), (-2, -1))
        transformed.append(image.flip(-1) if random.randrange(2) else image)
    return torch.stack(transformed)


@torch.no_grad()
def predict(model, split, device):
    model.eval()
    return torch.cat([model(x.to(device)).softmax(1).cpu()
                      for x, _ in split.loader()]).numpy()


def score(split, probabilities):
    return evaluate([split.batch], dict(zip(split.batch["sample_id"],
                                           probabilities.argmax(1), strict=True)))


def train(model, training, validation, device, *, warmup_epochs=5, finetune_epochs=15, head_only=False):
    weights = len(training.images) / (3 * torch.bincount(training.batch["label"], minlength=3))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    best_key, best_state, best_epoch = (-1., -1.), None, None
    history = []
    for stage, epochs in (("head", warmup_epochs), ("finetune", finetune_epochs)):
        if best_state is not None:
            model.load_state_dict(best_state)
        unfreeze = stage == "finetune" and not head_only
        configure_stage(model, unfreeze)
        groups = [{"params": model.fc.parameters(), "lr": 1e-3 if stage == "head" else 3e-4}]
        if unfreeze:
            groups.append({"params": [p for p in model.layer4[1].parameters() if p.requires_grad], "lr": 1e-4})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        stale = 0
        for _ in range(epochs):
            start = time.perf_counter()
            configure_stage(model, unfreeze)
            truths, guesses, loss_sum, weight_sum = [], [], 0., 0.
            for x, y in training.loader(shuffle=True):
                x, y = augment(x).to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                mass = float(criterion.weight[y].sum())
                loss_sum += float(loss.detach()) * mass
                weight_sum += mass
                truths.extend(y.cpu().tolist())
                guesses.extend(logits.detach().argmax(1).cpu().tolist())
            probabilities = predict(model, validation, device)
            metrics = score(validation, probabilities)
            key = (metrics["balanced_accuracy"], metrics["macro_f1"])
            val_y = validation.batch["label"].numpy()
            val_weights = weights.numpy()[val_y]
            val_loss = float(np.average(-np.log(np.maximum(probabilities[np.arange(len(val_y)), val_y], 1e-30)), weights=val_weights))
            stage_name = "head_continued" if stage == "finetune" and head_only else stage
            row = {"epoch": len(history) + 1, "stage": stage_name, "train_loss": loss_sum / weight_sum,
                   "val_loss": val_loss, "train_balanced_accuracy": _summary(truths, guesses)["balanced_accuracy"],
                   "val_balanced_accuracy": key[0], "val_macro_f1": key[1],
                   "seconds": time.perf_counter() - start,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            history.append(row)
            if key > best_key:
                best_key, best_epoch = key, row["epoch"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            print(f"Epoch {row['epoch']}/{warmup_epochs + finetune_epochs} ({stage_name}): "
                  f"train loss={row['train_loss']:.4f}, BA={row['train_balanced_accuracy']:.4f}; "
                  f"val loss={val_loss:.4f}, BA={key[0]:.4f}, F1={key[1]:.4f}; "
                  f"best={best_epoch}, patience={stale}/5, {row['seconds']:.1f}s", flush=True)
            if stage == "finetune" and stale >= 5:
                break
    model.load_state_dict(best_state)
    return history, {"class_weights": weights.tolist(), "best_epoch": best_epoch, "head_only": head_only,
                     "warmup_epochs": warmup_epochs, "finetune_max_epochs": finetune_epochs,
                     "optimizer": "AdamW: head warmup 1e-3; second-stage head 3e-4; wd=1e-4"
                                  + ("; backbone frozen" if head_only else "; second-stage block 1e-4"),
                     "batch_size": 16, "patience": 5, "selection": "val balanced accuracy, then macro F1, then earlier epoch"}


def matched_baselines(training, validation):
    results = {}
    for name, features in (("logreg_4", FEATURE_NAMES), ("logreg_14", FULL_FEATURE_NAMES)):
        tr = features_from_dataloader([training.batch], features)
        va = features_from_dataloader([validation.batch], features)
        model = train_logreg(tr.X, tr.y)
        results[name] = {"features": list(features), "validation": score(validation, model.predict_proba(va.X))}
    return results


def main(argv=None, *, splits=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--input-mode", choices=("parity", "residual"), default="parity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--head-only", action="store_true", help="Keep the backbone frozen in both matched training stages")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup_epochs < 1 or args.finetune_epochs < 0:
        parser.error("Require warmup-epochs >= 1 and finetune-epochs >= 0")
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.hub.set_dir(str(Path(__file__).parent / ".cache" / "torch" / "hub"))
    dataset_path = args.dataset.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output / f"{stamp}_resnet18_{args.input_mode}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}; outputs: {output}", flush=True)
    if splits is None:
        splits = {name: preload(dataset_path, name, args.input_mode) for name in ("train", "val")}
    print(f"Training clipping mean/max: {splits['train'].audit['clipping_fraction_mean']:.6f}/"
          f"{splits['train'].audit['clipping_fraction_max']:.6f}", flush=True)
    baselines = matched_baselines(splits["train"], splits["val"])
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(device)
    history, settings = train(model, splits["train"], splits["val"], device,
                              warmup_epochs=args.warmup_epochs, finetune_epochs=args.finetune_epochs,
                              head_only=args.head_only)
    config = {"dataset": str(dataset_path), "dataset_index_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "weights": "ResNet18_Weights.IMAGENET1K_V1", "preprocessing": PREPROCESSING,
              "input_mode": args.input_mode, "seed": args.seed, "device": device,
              "classes": {0: "none", 1: "amp", 2: "phase"}, "label_criterion": "constant_antenna_type", "training": settings,
              "versions": {"python": platform.python_version(), **{name: version(name) for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "astropy")}}}
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": config}, output / "checkpoint.pt")
    if args.evaluate_test:
        splits["test"] = preload(dataset_path, "test", args.input_mode)
    results, rows = {}, []
    for name, split in splits.items():
        probabilities = predict(model, split, device)
        results[name] = score(split, probabilities)
        for i, sample_id in enumerate(split.batch["sample_id"]):
            rows.append({"sample_id": sample_id, "source_id": source_dataset_id(sample_id), "split": name,
                         "true_class": int(split.batch["label"][i]), "predicted_class": int(probabilities[i].argmax()),
                         **{f"probability_{label}": float(probabilities[i, j]) for j, label in enumerate(("none", "amp", "phase"))},
                         **split.batch["label_metadata"][i]})
    config["runtime_seconds"] = time.perf_counter() - start
    from ml.cnn_report import write_run

    write_run(output, config, history, results, rows, {name: split.audit for name, split in splits.items()}, baselines)
    print(output / "report.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
