"""Versioned manifests for compact, ML-ready simulation samples."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAMPLE_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
CORRUPTION_SCHEMA_VERSION = 1
SUPPORTED_QA_SCHEMAS = (2, 3)
SUPPORTED_SIMULATION_SCHEMAS = (1, 2)
CHANNEL_ORDER = ("dirty", "clean", "residual")


@dataclass(frozen=True)
class CorruptionReference:
    corruption: Path
    corruption_text: Path


@dataclass(frozen=True)
class SampleManifest:
    path: Path
    schema_version: int
    sample_id: str
    label_id: int
    label_name: str
    channel_order: tuple[str, ...]
    products: dict[str, Path]
    imaging_qa: Path
    imaging_text: Path
    simulation: Path
    simulation_text: Path
    corruptions: tuple[CorruptionReference, ...]
    integrity: dict[str, dict[str, Any]]
    raw: dict[str, Any]


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    schema_version: int
    labels: dict[str, int]
    samples: tuple[Path, ...]
    raw: dict[str, Any]


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read {description} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_reference(root: str | Path, value: Any, *, name: str) -> Path:
    """Resolve one manifest path while rejecting absolute and escaping paths."""
    raw_root = Path(root).expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"Manifest root cannot be a symlink: {raw_root}")
    root_path = raw_root.resolve()
    relative = Path(_string(value, name=name))
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative to the manifest: {relative}")
    if relative == Path("."):
        raise ValueError(f"{name} must name a file")
    unresolved = root_path / relative
    current = root_path
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{name} traverses a symlink: {current}")
    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the sample directory: {relative}") from exc
    return candidate


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Retained path is outside the sample directory: {path}") from exc


def _input_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_files(manifest: SampleManifest) -> tuple[Path, ...]:
    paths = [
        *(manifest.products[name] for name in manifest.channel_order),
        manifest.imaging_qa,
        manifest.imaging_text,
        manifest.simulation,
        manifest.simulation_text,
    ]
    for reference in manifest.corruptions:
        paths.extend((reference.corruption, reference.corruption_text))
    return tuple(paths)


def build_integrity(paths: Sequence[str | Path], *, root: str | Path) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    files: dict[str, dict[str, Any]] = {}
    for value in paths:
        path = _input_path(value, root_path)
        if not path.is_file():
            raise ValueError(f"Cannot checksum missing retained file: {path}")
        relative = _relative(path, root_path)
        if relative in files:
            raise ValueError(f"Duplicate retained path: {relative}")
        files[relative] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {"algorithm": "sha256", "files": files}


def _parse_integrity(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or value.get("algorithm") != "sha256":
        raise ValueError("integrity.algorithm must be 'sha256'")
    raw_files = value.get("files")
    if not isinstance(raw_files, Mapping):
        raise ValueError("integrity.files must be an object")
    result: dict[str, dict[str, Any]] = {}
    for relative, record in raw_files.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("integrity.files entries must map relative paths to objects")
        size = _integer(record.get("size"), name=f"integrity.files[{relative}].size")
        checksum = record.get("sha256")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError(f"integrity.files[{relative}].sha256 is not lowercase SHA-256")
        result[relative] = {"size": size, "sha256": checksum}
    return result


def load_sample_manifest(
    path: str | Path,
    *,
    require_files: bool = True,
    verify_integrity: bool = True,
) -> SampleManifest:
    raw_manifest_path = Path(path).expanduser()
    if raw_manifest_path.is_symlink():
        raise ValueError(f"Sample manifest cannot be a symlink: {raw_manifest_path}")
    if raw_manifest_path.parent.is_symlink():
        raise ValueError(f"Sample root cannot be a symlink: {raw_manifest_path.parent}")
    manifest_path = raw_manifest_path.resolve()
    root = manifest_path.parent
    payload = _load_json(manifest_path, description="sample manifest")
    version = _integer(payload.get("schema_version"), name="schema_version", minimum=1)
    if version != SAMPLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported sample schema_version {version}")
    sample_id = _string(payload.get("sample_id"), name="sample_id")

    label = payload.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("label must be an object")
    label_id = _integer(label.get("id"), name="label.id")
    label_name = _string(label.get("name"), name="label.name")

    products_value = payload.get("products")
    if not isinstance(products_value, Mapping):
        raise ValueError("products must be an object")
    order_value = products_value.get("channel_order")
    if not isinstance(order_value, list) or tuple(order_value) != CHANNEL_ORDER:
        raise ValueError(f"products.channel_order must be {list(CHANNEL_ORDER)!r}")
    products = {
        name: resolve_reference(root, products_value.get(name), name=f"products.{name}")
        for name in CHANNEL_ORDER
    }
    for name, product_path in products.items():
        if not product_path.name.endswith(".fits.gz"):
            raise ValueError(f"products.{name} must reference a .fits.gz file")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    metadata_paths = {
        name: resolve_reference(root, metadata.get(name), name=f"metadata.{name}")
        for name in ("imaging_qa", "imaging_text", "simulation", "simulation_text")
    }
    for name in ("imaging_qa", "simulation"):
        if metadata_paths[name].suffix.casefold() != ".json":
            raise ValueError(f"metadata.{name} must reference a JSON file")
    for name in ("imaging_text", "simulation_text"):
        if metadata_paths[name].suffix.casefold() != ".txt":
            raise ValueError(f"metadata.{name} must reference a TXT file")
    corruption_values = metadata.get("corruptions")
    if not isinstance(corruption_values, list):
        raise ValueError("metadata.corruptions must be an ordered list (use [] for none)")
    corruptions: list[CorruptionReference] = []
    seen: set[Path] = set()
    for index, value in enumerate(corruption_values):
        if not isinstance(value, Mapping):
            raise ValueError(f"metadata.corruptions[{index}] must be an object")
        json_path = resolve_reference(
            root, value.get("corruption"), name=f"metadata.corruptions[{index}].corruption"
        )
        text_path = resolve_reference(
            root,
            value.get("corruption_text"),
            name=f"metadata.corruptions[{index}].corruption_text",
        )
        if json_path.suffix.casefold() != ".json" or text_path.suffix.casefold() != ".txt":
            raise ValueError(
                f"metadata.corruptions[{index}] must reference one JSON and one TXT file"
            )
        if json_path in seen or text_path in seen or json_path == text_path:
            raise ValueError(f"metadata.corruptions[{index}] contains a duplicate path")
        seen.update((json_path, text_path))
        corruptions.append(CorruptionReference(json_path, text_path))

    integrity = _parse_integrity(payload.get("integrity"))
    manifest = SampleManifest(
        path=manifest_path,
        schema_version=version,
        sample_id=sample_id,
        label_id=label_id,
        label_name=label_name,
        channel_order=CHANNEL_ORDER,
        products=products,
        imaging_qa=metadata_paths["imaging_qa"],
        imaging_text=metadata_paths["imaging_text"],
        simulation=metadata_paths["simulation"],
        simulation_text=metadata_paths["simulation_text"],
        corruptions=tuple(corruptions),
        integrity=integrity,
        raw=payload,
    )
    paths = referenced_files(manifest)
    if len(set(paths)) != len(paths):
        raise ValueError("Sample manifest contains duplicate retained-file references")
    if require_files:
        missing = [file_path for file_path in paths if not file_path.is_file()]
        if missing:
            raise ValueError("Missing retained files:\n" + "\n".join(f"  - {p}" for p in missing))
        _validate_json_metadata(manifest)
    if verify_integrity:
        _verify_integrity(manifest)
    return manifest


def _validate_json_metadata(manifest: SampleManifest) -> None:
    for path, description, supported in (
        (manifest.imaging_qa, "imaging QA", SUPPORTED_QA_SCHEMAS),
        (manifest.simulation, "simulation report", SUPPORTED_SIMULATION_SCHEMAS),
    ):
        payload = _load_json(path, description=description)
        version = _integer(
            payload.get("schema_version"), name=f"{description}.schema_version", minimum=1
        )
        if version not in supported:
            raise ValueError(f"Unsupported {description} schema_version {version}")
    for index, reference in enumerate(manifest.corruptions):
        report = _load_json(reference.corruption, description=f"corruption report {index}")
        version = _integer(
            report.get("schema_version"),
            name=f"corruption report {index}.schema_version",
            minimum=1,
        )
        if version != CORRUPTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported corruption report schema_version {version}")
        if not isinstance(report.get("context"), Mapping):
            raise ValueError(f"corruption report {index}.context must be an object")
        if not isinstance(report.get("configuration"), Mapping):
            raise ValueError(f"corruption report {index}.configuration must be an object")
        context = report["context"]
        application_index = context.get("application_index")
        if application_index is not None and application_index != index:
            raise ValueError(
                f"corruption report {index} has application_index {application_index!r}"
            )
        retained_sample_id = context.get("retained_sample_id")
        if retained_sample_id is not None and retained_sample_id != manifest.sample_id:
            raise ValueError(
                f"corruption report {index} belongs to sample {retained_sample_id!r}, "
                f"not {manifest.sample_id!r}"
            )


def _verify_integrity(manifest: SampleManifest) -> None:
    root = manifest.path.parent
    expected_paths = {_relative(path, root) for path in referenced_files(manifest)}
    actual_paths = set(manifest.integrity)
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            "integrity.files does not match retained references; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, record in manifest.integrity.items():
        path = resolve_reference(root, relative, name=f"integrity.files[{relative}]")
        if not path.is_file():
            raise ValueError(f"Missing checksummed file: {path}")
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Size mismatch for retained file: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for retained file: {path}")


def _strict_json(value: Any, *, location: str = "value") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string key")
            result[key] = _strict_json(item, location=f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _strict_json(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{location} contains unsupported {type(value).__name__}")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        _strict_json(payload), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
    ) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_sample_manifest(path: str | Path, payload: Mapping[str, Any]) -> SampleManifest:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite sample manifest: {destination}")
    atomic_write_json(destination, payload)
    try:
        return load_sample_manifest(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def create_sample_manifest(
    path: str | Path,
    *,
    sample_id: str,
    label_id: int,
    label_name: str,
    products: Mapping[str, str | Path],
    imaging_qa: str | Path,
    imaging_text: str | Path,
    simulation: str | Path,
    simulation_text: str | Path,
    corruptions: Sequence[tuple[str | Path, str | Path]] = (),
) -> SampleManifest:
    destination = Path(path).expanduser().resolve()
    root = destination.parent
    if set(products) != set(CHANNEL_ORDER):
        raise ValueError("products must contain exactly dirty, clean, and residual")
    product_paths = {
        name: _input_path(products[name], root) for name in CHANNEL_ORDER
    }
    metadata_paths = {
        "imaging_qa": _input_path(imaging_qa, root),
        "imaging_text": _input_path(imaging_text, root),
        "simulation": _input_path(simulation, root),
        "simulation_text": _input_path(simulation_text, root),
    }
    corruption_paths = [
        (_input_path(json_path, root), _input_path(text_path, root))
        for json_path, text_path in corruptions
    ]
    retained = [*product_paths.values(), *metadata_paths.values()]
    for json_path, text_path in corruption_paths:
        retained.extend((json_path, text_path))
    payload = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_id": _string(sample_id, name="sample_id"),
        "label": {
            "id": _integer(label_id, name="label_id"),
            "name": _string(label_name, name="label_name"),
        },
        "products": {
            "channel_order": list(CHANNEL_ORDER),
            **{name: _relative(product_paths[name], root) for name in CHANNEL_ORDER},
        },
        "metadata": {
            **{name: _relative(value, root) for name, value in metadata_paths.items()},
            "corruptions": [
                {
                    "corruption": _relative(json_path, root),
                    "corruption_text": _relative(text_path, root),
                }
                for json_path, text_path in corruption_paths
            ],
        },
        "integrity": build_integrity(retained, root=root),
    }
    return write_sample_manifest(destination, payload)


def load_dataset_manifest(path: str | Path, *, require_samples: bool = True) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = _load_json(manifest_path, description="dataset manifest")
    version = _integer(payload.get("schema_version"), name="schema_version", minimum=1)
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema_version {version}")
    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, Mapping):
        raise ValueError("dataset labels must be an object mapping names to IDs")
    labels = {
        _string(name, name="label name"): _integer(value, name=f"labels.{name}")
        for name, value in raw_labels.items()
    }
    if len(set(labels.values())) != len(labels):
        raise ValueError("dataset label IDs must be unique")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("dataset samples must be an ordered list")
    samples = tuple(
        resolve_reference(manifest_path.parent, value, name=f"samples[{index}]")
        for index, value in enumerate(raw_samples)
    )
    if len(set(samples)) != len(samples):
        raise ValueError("dataset samples contains duplicate manifests")
    if require_samples:
        for sample_path in samples:
            sample = load_sample_manifest(sample_path)
            if labels.get(sample.label_name) != sample.label_id:
                raise ValueError(
                    f"Sample {sample.sample_id!r} label does not match dataset label map"
                )
    return DatasetManifest(manifest_path, version, labels, samples, payload)


def write_dataset_manifest(
    path: str | Path,
    *,
    labels: Mapping[str, int],
    samples: Sequence[str | Path],
) -> DatasetManifest:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset manifest: {destination}")
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "labels": dict(labels),
        "samples": [
            _relative(_input_path(sample, destination.parent), destination.parent)
            for sample in samples
        ],
    }
    atomic_write_json(destination, payload)
    try:
        return load_dataset_manifest(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def add_sample_to_dataset(index_path: str | Path, sample_path: str | Path) -> DatasetManifest:
    """Validate and atomically append one unique sample to an existing index."""
    destination = Path(index_path).expanduser().resolve()
    sample = load_sample_manifest(sample_path)
    if destination.exists():
        index = load_dataset_manifest(destination)
        labels = dict(index.labels)
        samples = list(index.samples)
    else:
        labels = {}
        samples = []
    existing_id = labels.get(sample.label_name)
    if existing_id is not None and existing_id != sample.label_id:
        raise ValueError(f"Label {sample.label_name!r} already maps to ID {existing_id}")
    existing_name = next(
        (name for name, value in labels.items() if value == sample.label_id), None
    )
    if existing_name is not None and existing_name != sample.label_name:
        raise ValueError(
            f"Label ID {sample.label_id} already maps to {existing_name!r}"
        )
    if any(load_sample_manifest(path).sample_id == sample.sample_id for path in samples):
        raise ValueError(f"Dataset already contains sample_id {sample.sample_id!r}")
    labels[sample.label_name] = sample.label_id
    samples.append(sample.path)
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "labels": labels,
        "samples": [_relative(path, destination.parent) for path in samples],
    }
    atomic_write_json(destination, payload)
    return load_dataset_manifest(destination)


__all__ = [
    "CHANNEL_ORDER",
    "CORRUPTION_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "SAMPLE_SCHEMA_VERSION",
    "SUPPORTED_QA_SCHEMAS",
    "SUPPORTED_SIMULATION_SCHEMAS",
    "CorruptionReference",
    "DatasetManifest",
    "SampleManifest",
    "atomic_write_json",
    "add_sample_to_dataset",
    "build_integrity",
    "create_sample_manifest",
    "load_dataset_manifest",
    "load_sample_manifest",
    "referenced_files",
    "resolve_reference",
    "sha256_file",
    "write_sample_manifest",
    "write_dataset_manifest",
]
