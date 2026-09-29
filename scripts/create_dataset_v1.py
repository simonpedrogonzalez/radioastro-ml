#!/usr/bin/env python3
"""Build the full constant-gain calibration dataset.

Run from the repository root with CASA::

    '/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa' \
        --nogui --nologger -c scripts/create_dataset_v1.py

The driver processes every completed thermal source in the fixed train, test,
and validation partitions. For each source, the baseline and every corruption
variant start from the same noiseless sky visibility ``V`` and receive the same
seeded thermal-noise realization after corruption is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import traceback
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS_ROOT = ROOT / "collect" / "experiments"
DATASET_DIR = EXPERIMENTS_ROOT / "dataset_v1_20260921_all_psf"
THERMAL_DIR = (
    EXPERIMENTS_ROOT / "extracted_thermal_simulation_comparison_20260909T005739"
)
DATASET_PBLIMIT = -0.1
NOISE_CONTROL_INDEX_NAME = "dataset_noise_controls.json"
NOISE_CONTROL_LABELS = {"not_corrupted": 0}
NOISE_CONTROL_REL_TOL = 1e-5
NOISE_CONTROL_DEFAULT_PARTITION = "val"
REPORT_SCHEMA_VERSION = 2
REPORT_TEMPLATE = ROOT / "scripts" / "reporting" / "create_dataset_v1.qmd"
LEGACY_SNR_CORR_TARGETS = (10.0, 30.0, 50.0, 100.0)
SNR_CORR_TARGETS = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 100.0)
CORRUPTION_SOLINT = "10m"
SNR_CORR_REL_TOL = 0.01
BASE_SEED = 20260914
# None processes every available source in the fixed partitions. To restrict a
# run, set a tuple such as ("0012-399", "0846-261").
SOURCE_IDS_TO_PROCESS: tuple[str, ...] | None = None
# Set either override to None for the default independent random draw per
# source variant. Integer values are shared globally by all sources/variants.
FIXED_ANTENNA_ID: int | None = None
FIXED_ERROR_SIGN: Literal[-1, 1] | None = None
if FIXED_ANTENNA_ID is not None and (
    isinstance(FIXED_ANTENNA_ID, bool)
    or not isinstance(FIXED_ANTENNA_ID, int)
    or FIXED_ANTENNA_ID < 0
):
    raise ValueError("FIXED_ANTENNA_ID must be a nonnegative integer or None")
if FIXED_ERROR_SIGN not in (None, -1, 1) or isinstance(FIXED_ERROR_SIGN, bool):
    raise ValueError("FIXED_ERROR_SIGN must be -1, +1, or None")
ANTENNA_SELECTION_POLICY = (
    f"fixed antenna ID {FIXED_ANTENNA_ID} shared by all sources and variants"
    if FIXED_ANTENNA_ID is not None
    else "independent_uniform_with_replacement_from_unflagged_antennas_per_variant"
)
CONSTANT_ERROR_SIGN_POLICY = (
    f"fixed sign {FIXED_ERROR_SIGN:+d} shared by all sources and variants"
    if FIXED_ERROR_SIGN is not None
    else "uniform random choice from {-1, +1} using the per-variant corruption seed"
)
REPORT_EVERY_SOURCES = 5
MIN_FREE_HEADROOM_BYTES = 2 * 1024**3
MATCHED_IMAGING_KEYS = (
    "imsize",
    "mask_nbeams",
    "gridder",
    "weighting",
    "robust",
    "stokes",
    "deconvolver",
    "nterms",
)


@dataclass(frozen=True)
class ThermalSource:
    source_id: str
    simulation_ms: Path
    simulation_metadata: Path
    simulation_result_dir: Path
    imsize: tuple[int, int]
    metric_min_radius_beams: float | None
    metric_max_radius_beams: float | None
    source_snr: float
    predicted_image_rms_jy_per_beam: float


@dataclass(frozen=True)
class Variant:
    family: str
    SNR_corr_target: float
    label_name: str


@dataclass(frozen=True)
class NoiseControl:
    SNR_corr_target: float
    label_name: str


@dataclass(frozen=True)
class IndexedSamples:
    manifests: dict[str, Any]
    pb_repairs: frozenset[str]


def _target_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, "g")


VARIANTS = tuple(
    Variant(family, target, f"{family}_snr_{_target_text(target)}")
    for family in ("amp", "phase")
    for target in SNR_CORR_TARGETS
)
NOISE_CONTROLS = tuple(
    NoiseControl(target, f"noise_snr_{_target_text(target)}")
    for target in SNR_CORR_TARGETS
)
LEGACY_LABELS = {
    "not_corrupted": 0,
    "amp_snr_10": 1,
    "amp_snr_30": 2,
    "amp_snr_50": 3,
    "amp_snr_100": 4,
    "phase_snr_10": 5,
    "phase_snr_30": 6,
    "phase_snr_50": 7,
    "phase_snr_100": 8,
}
LABELS = {
    **LEGACY_LABELS,
    "amp_snr_5": 9,
    "amp_snr_15": 10,
    "amp_snr_20": 11,
    "amp_snr_40": 12,
    "phase_snr_5": 13,
    "phase_snr_15": 14,
    "phase_snr_20": 15,
    "phase_snr_40": 16,
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(experiment_dir: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Thermal manifest is missing {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = experiment_dir / path
    return path.resolve()


def _load_thermal_run(path: Path, required_ids: Sequence[str]) -> dict[str, Any]:
    manifest_path = path / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = {
        item.get("id"): item
        for item in manifest.get("samples", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    missing_ids = sorted(set(required_ids) - set(samples))
    if missing_ids:
        raise ValueError(f"missing completed source IDs: {missing_ids}")
    configuration = manifest.get("configuration") or {}
    imsize = configuration.get("imsize")
    if (
        not isinstance(imsize, list)
        or len(imsize) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in imsize)
    ):
        raise ValueError(f"invalid imaging size: {imsize!r}")
    for source_id in required_ids:
        entry = samples[source_id]
        required = (
            _resolve(path, entry.get("simulation_ms"), name="simulation_ms"),
            _resolve(path, entry.get("simulation_metadata"), name="simulation_metadata"),
            _resolve(path, entry.get("simulation_result_dir"), name="simulation_result_dir"),
        )
        missing = [candidate for candidate in required if not candidate.exists()]
        if missing:
            raise ValueError(
                f"{source_id} has missing reusable products: "
                + ", ".join(str(item) for item in missing)
            )
        for filename in ("qa.json", "qa.txt"):
            if not (required[2] / filename).is_file():
                raise ValueError(f"{source_id} is missing {required[2] / filename}")
    return manifest


def _configured_source_ids() -> tuple[str, ...]:
    from scripts.preprocessing import ALL_IDS

    if SOURCE_IDS_TO_PROCESS is None:
        return tuple(ALL_IDS)
    source_ids = tuple(SOURCE_IDS_TO_PROCESS)
    if not source_ids:
        raise ValueError("SOURCE_IDS_TO_PROCESS must not be empty")
    if any(not isinstance(source_id, str) for source_id in source_ids):
        raise ValueError("SOURCE_IDS_TO_PROCESS must contain only source ID strings")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("SOURCE_IDS_TO_PROCESS contains duplicate source IDs")
    unknown = [source_id for source_id in source_ids if source_id not in ALL_IDS]
    if unknown:
        raise ValueError(f"SOURCE_IDS_TO_PROCESS contains unknown IDs: {unknown}")
    return source_ids


def find_thermal_run(
    source_run: str | Path | None = None,
) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    requested_ids = _configured_source_ids()
    candidates = [
        Path(source_run).expanduser().resolve()
        if source_run is not None
        else THERMAL_DIR.resolve()
    ]
    failures = []
    usable = []
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "report.json").is_file():
            failures.append(f"{candidate}: no report.json")
            continue
        try:
            manifest = json.loads((candidate / "report.json").read_text(encoding="utf-8"))
            completed = {
                item.get("id")
                for item in manifest.get("samples", [])
                if isinstance(item, dict)
            }
            source_ids = (
                requested_ids
                if SOURCE_IDS_TO_PROCESS is not None
                else tuple(
                    source_id for source_id in requested_ids if source_id in completed
                )
            )
            if not source_ids:
                raise ValueError("no completed partitioned sources")
            usable.append(
                (candidate.resolve(), _load_thermal_run(candidate, source_ids), source_ids)
            )
        except Exception as exc:
            failures.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
    if usable:
        return max(usable, key=lambda item: (len(item[2]), item[0].name))
    raise RuntimeError(
        "No reusable thermal experiment contains completed partition sources:\n"
        + "\n".join(f"  - {failure}" for failure in failures)
    )


def _ensure_thermal_entries(thermal_dir: Path) -> None:
    """Run the established precursor only when canonical partition inputs are new."""
    from scripts import compare_extracted_thermal_simulations as precursor

    requested = set(_configured_source_ids())
    canonical = [
        path for path in precursor.find_samples() if path.stem in requested
    ]
    manifest_path = thermal_dir / "report.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    recorded = {
        item.get("id")
        for collection in (manifest.get("samples", []), manifest.get("failures", []))
        for item in collection
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    pending = [path for path in canonical if path.stem not in recorded]
    if not pending:
        return
    print(
        f"Preparing {len(pending)} new thermal precursor entr"
        f"{'y' if len(pending) == 1 else 'ies'} in {thermal_dir}"
    )
    precursor.main(
        thermal_dir,
        sample_ids=[path.stem for path in canonical],
    )


def _thermal_source(
    experiment_dir: Path, manifest: dict[str, Any], source_id: str
) -> ThermalSource:
    entry = next(item for item in manifest["samples"] if item.get("id") == source_id)
    imsize = manifest["configuration"]["imsize"]
    region = entry.get("metric_region") or {}
    return ThermalSource(
        source_id=source_id,
        simulation_ms=_resolve(
            experiment_dir, entry.get("simulation_ms"), name="simulation_ms"
        ),
        simulation_metadata=_resolve(
            experiment_dir,
            entry.get("simulation_metadata"),
            name="simulation_metadata",
        ),
        simulation_result_dir=_resolve(
            experiment_dir,
            entry.get("simulation_result_dir"),
            name="simulation_result_dir",
        ),
        imsize=(int(imsize[0]), int(imsize[1])),
        metric_min_radius_beams=region.get("min_radius_beams"),
        metric_max_radius_beams=region.get("max_radius_beams"),
        source_snr=float(entry["source_snr"]),
        predicted_image_rms_jy_per_beam=float(
            entry["predicted_image_rms_jy_per_beam"]
        ),
    )


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _check_disk_space(sources: Sequence[ThermalSource], destination: Path) -> None:
    ms_size = max(_tree_size(source.simulation_ms) for source in sources)
    required = 4 * ms_size + MIN_FREE_HEADROOM_BYTES
    free = shutil.disk_usage(destination).free
    print(
        f"Disk preflight: largest source MS={ms_size / 1024**3:.2f} GiB, "
        f"required={required / 1024**3:.2f} GiB, free={free / 1024**3:.2f} GiB"
    )
    if free < required:
        raise RuntimeError(
            f"Insufficient disk space: require {required / 1024**3:.2f} GiB, "
            f"found {free / 1024**3:.2f} GiB"
        )


def _stable_seed(source_id: str, suffix: str) -> int:
    digest = hashlib.sha256(
        f"{BASE_SEED}:{source_id}:{suffix}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1


def _antenna_seed(source_id: str, variant: Variant) -> int | None:
    if FIXED_ANTENNA_ID is not None:
        return None
    return _stable_seed(source_id, f"{variant.label_name}:antenna")


def _error_direction_seed(source_id: str, variant: Variant) -> int | None:
    if FIXED_ERROR_SIGN is not None:
        return None
    return _stable_seed(source_id, variant.label_name)


def _sample_id(source_id: str, suffix: str) -> str:
    return f"{source_id}_{suffix}"


def _sample_dir(output: Path, sample_id: str) -> Path:
    return output / "samples" / sample_id


def _corruption_plot_path(output: Path, sample_id: str) -> Path:
    return output / "report_assets" / "corruption" / f"{sample_id}.png"


def _safe_remove_partial(path: Path, samples_root: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir() or path.resolve().parent != samples_root.resolve():
        raise RuntimeError(f"Refusing to remove unsafe partial sample path: {path}")
    print(f"Removing interrupted uncommitted sample: {path.name}")
    shutil.rmtree(path)


def _simulation_payload(source: ThermalSource) -> dict[str, Any]:
    return json.loads(source.simulation_metadata.read_text(encoding="utf-8"))


def _noise_request(source: ThermalSource) -> tuple[str, dict[str, Any], float]:
    noise = _simulation_payload(source).get("noise") or {}
    model = noise.get("noise_model")
    parameters = noise.get("requested_parameters")
    sigma = noise.get("simplenoise_jy")
    if not isinstance(model, str) or not isinstance(parameters, dict):
        raise ValueError(f"{source.source_id} simulation report has no reusable noise request")
    try:
        sigma_value = float(sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source.source_id} simulation report has invalid simplenoise_jy"
        ) from exc
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        raise ValueError(
            f"{source.source_id} simulation report has invalid simplenoise_jy"
        )
    return model, dict(parameters), sigma_value


def _prepare_V_ms(source: ThermalSource, output: Path) -> Path:
    from scripts.simulation import simulate_ms

    work = output / "work" / source.source_id
    V_ms = work / f"{source.source_id}_V.ms"
    V_json = work / f"{source.source_id}_V.simulation.json"
    V_text = work / f"{source.source_id}_V.simulation.txt"
    if V_ms.is_dir() and V_json.is_file() and V_text.is_file():
        return V_ms
    if work.exists():
        if work.is_symlink() or work.resolve().parent != (output / "work").resolve():
            raise RuntimeError(f"Refusing to replace unsafe work directory: {work}")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    payload = _simulation_payload(source)
    input_ms = payload.get("input_ms")
    components = payload.get("components")
    if not isinstance(input_ms, str) or not isinstance(components, list) or not components:
        raise ValueError(
            f"{source.source_id} simulation report cannot reconstruct noiseless V"
        )
    print(f"[{source.source_id}] predicting source-only V")
    return simulate_ms(input_ms, components, V_ms, noise_model=None).ms_path


def _cleanup_V_work(source: ThermalSource, output: Path) -> None:
    work_root = (output / "work").resolve()
    work = work_root / source.source_id
    if not work.exists():
        return
    if work.is_symlink() or not work.is_dir() or work.resolve().parent != work_root:
        raise RuntimeError(f"Refusing to remove unsafe work directory: {work}")
    shutil.rmtree(work)


def _write_branch_simulation_reports(
    source: ThermalSource,
    *,
    sample_id: str,
    sample_dir: Path,
    V_ms: Path,
    observed_ms: Path,
    provenance_observed_ms: Path | None,
    noise: dict[str, object],
    corruption_seed: int | None,
    thermal_noise_seed: int,
    noise_control: dict[str, Any] | None = None,
):
    from scripts.simulation import (
        SIMULATION_REPORT_SCHEMA_VERSION,
        SimulationResult,
        write_simulation_reports,
    )

    names = ["copy_noiseless_V"]
    if noise_control is not None:
        names += [
            "draw_extra_noise",
            "normalize_extra_noise",
            "measure_extra_noise",
        ]
    elif corruption_seed is not None:
        names += ["corrupt_V", "measure_Delta_V"]
    names += ["shared_baseline_noise", "weights"]
    stages = [
        {"name": name, "order": index}
        for index, name in enumerate(names, start=1)
    ]
    stages[-2].update(
        seed=thermal_noise_seed,
        shared_across_source_variants=True,
    )
    if noise_control is not None:
        stages[1]["seed"] = noise_control["extra_noise_seed"]
        stages[-1]["sigma_jy"] = noise_control["final_total_sigma_jy"]
    elif corruption_seed is not None:
        stages[1]["seed"] = corruption_seed
    payload = _simulation_payload(source)
    recorded_observed_ms = (
        observed_ms if provenance_observed_ms is None else provenance_observed_ms
    )
    for obsolete in ("component_list", "operations", "created_paths"):
        payload.pop(obsolete, None)
    payload.update(
        {
            "schema_version": SIMULATION_REPORT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_ms": str(V_ms),
            "output_ms": str(recorded_observed_ms),
            "input_visibility": {
                "role": "V",
                "description": "noiseless uncorrupted predicted sky visibility",
                "local_path": str(V_ms),
            },
            "noise": noise,
            "weight_initialization": noise.get("weight_initialization"),
            "stages": stages,
            "shared_noise": {
                "seed": thermal_noise_seed,
                "policy": "same CASA simplenoise request and seed for every source variant",
            },
            **({"noise_control": noise_control} if noise_control is not None else {}),
            "generated_artifacts": [
                {
                    "role": "observed_ms",
                    "path": str(recorded_observed_ms),
                    "lifecycle": "temporary",
                }
            ],
        }
    )
    json_path = sample_dir / "simulation.json"
    text_path = sample_dir / "simulation.txt"
    write_simulation_reports(payload, json_path, text_path)
    return SimulationResult(
        observed_ms,
        None,
        json_path,
        float(noise["simplenoise_jy"]),
        thermal_noise_seed,
        text_path,
    )


def _has_border_connected_common_zeros(sample: Any) -> bool:
    """Recognize the old filled CASA PB mask and reject ambiguous zero holes."""
    import numpy as np

    from scripts.preprocessing import validate_fits_products

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        planes = validate_fits_products(sample.products)
    common_zero = np.logical_and.reduce(
        [planes[name].values == 0.0 for name in ("dirty", "clean", "residual")]
    )
    height, width = common_zero.shape
    centre_rows = {(height - 1) // 2, height // 2}
    centre_columns = {(width - 1) // 2, width // 2}
    if any(common_zero[row, column] for row in centre_rows for column in centre_columns):
        raise ValueError(f"{sample.sample_id} has shared exact-zero pixels at image centre")

    connected = np.zeros_like(common_zero, dtype=bool)
    border = np.zeros_like(common_zero, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    pending = deque(
        (int(row), int(column))
        for row, column in np.argwhere(common_zero & border)
    )
    for row, column in pending:
        connected[row, column] = True
    while pending:
        row, column = pending.popleft()
        for adjacent_row, adjacent_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= adjacent_row < height
                and 0 <= adjacent_column < width
                and common_zero[adjacent_row, adjacent_column]
                and not connected[adjacent_row, adjacent_column]
            ):
                connected[adjacent_row, adjacent_column] = True
                pending.append((adjacent_row, adjacent_column))
    interior_holes = common_zero & ~connected
    if bool(np.any(interior_holes)):
        raise ValueError(
            f"{sample.sample_id} has {int(np.count_nonzero(interior_holes))} "
            "ambiguous shared exact-zero interior pixels"
        )
    return bool(np.any(connected))


def _assert_square_pb_support(sample: Any) -> None:
    if _has_border_connected_common_zeros(sample):
        raise ValueError(f"{sample.sample_id} still has a border-connected PB mask")


def _assert_sample_pb_policy(sample: Any) -> None:
    qa = json.loads(sample.imaging_qa.read_text(encoding="utf-8"))
    actual = (qa.get("effective_imaging_parameters") or {}).get("pblimit")
    if actual != DATASET_PBLIMIT:
        raise ValueError(
            f"{sample.sample_id} records pblimit={actual!r}, expected {DATASET_PBLIMIT}"
        )


def _audit_indexed_samples(dataset_index: Path) -> IndexedSamples:
    if not dataset_index.exists():
        return IndexedSamples({}, frozenset())
    from scripts.preprocessing import load_dataset_manifest, load_sample_manifest

    index = load_dataset_manifest(dataset_index, require_samples=False)
    unknown_labels = set(index.labels) - set(LABELS)
    incompatible = {
        name: (label_id, LABELS.get(name))
        for name, label_id in index.labels.items()
        if name in LABELS and label_id != LABELS[name]
    }
    if unknown_labels or incompatible:
        raise RuntimeError(
            "Existing dataset has incompatible labels: "
            f"unknown={sorted(unknown_labels)}, changed={incompatible}"
        )
    manifests: dict[str, Any] = {}
    repairs: set[str] = set()
    for path in index.samples:
        sample = load_sample_manifest(path)
        if sample.sample_id in manifests:
            raise RuntimeError(f"Duplicate indexed sample ID: {sample.sample_id}")
        expected_label = LABELS.get(sample.label_name)
        if expected_label != sample.label_id:
            raise RuntimeError(
                f"{sample.sample_id} label {sample.label_name}={sample.label_id} "
                f"does not match {expected_label}"
            )
        manifests[sample.sample_id] = sample
        if _has_border_connected_common_zeros(sample):
            repairs.add(sample.sample_id)
    return IndexedSamples(manifests, frozenset(repairs))


def _noise_control_payload(sample: Any) -> dict[str, Any]:
    payload = json.loads(sample.simulation.read_text(encoding="utf-8"))
    control = payload.get("noise_control")
    if not isinstance(control, dict) or control.get("kind") != "increased_noise":
        raise ValueError(
            f"{sample.sample_id} has no valid increased-noise control metadata"
        )
    for key in ("SNR_corr_target", "SNR_corr_measured"):
        value = control.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{sample.sample_id} has invalid noise_control.{key}")
    return control


def _assert_noise_control_sample(
    sample: Any, expected_target: float | None = None
) -> None:
    if (sample.label_id, sample.label_name) != (0, "not_corrupted"):
        raise ValueError(
            f"{sample.sample_id} must retain label 0/not_corrupted"
        )
    if sample.corruptions:
        raise ValueError(f"{sample.sample_id} must not contain a gain report")
    control = _noise_control_payload(sample)
    target = float(control["SNR_corr_target"])
    measured = float(control["SNR_corr_measured"])
    if expected_target is not None and target != float(expected_target):
        raise ValueError(
            f"{sample.sample_id} target {target:g} does not match {expected_target:g}"
        )
    if not math.isclose(measured, target, rel_tol=NOISE_CONTROL_REL_TOL):
        raise ValueError(
            f"{sample.sample_id} measured SNR_corr={measured:.12g} does not match "
            f"target {target:.12g}"
        )
    suffix = f"_noise_snr_{_target_text(target)}"
    if not sample.sample_id.endswith(suffix):
        raise ValueError(
            f"{sample.sample_id} does not match its noise-control target {target:g}"
        )
    _assert_square_pb_support(sample)
    _assert_sample_pb_policy(sample)


def _audit_noise_controls(dataset_index: Path) -> dict[str, Any]:
    if not dataset_index.exists():
        return {}
    from scripts.preprocessing import load_dataset_manifest, load_sample_manifest

    index = load_dataset_manifest(dataset_index, require_samples=False)
    if index.labels != NOISE_CONTROL_LABELS:
        raise RuntimeError(
            f"Noise-control index has incompatible labels: {index.labels}"
        )
    manifests: dict[str, Any] = {}
    for path in index.samples:
        sample = load_sample_manifest(path)
        if sample.sample_id in manifests:
            raise RuntimeError(
                f"Duplicate indexed noise-control sample ID: {sample.sample_id}"
            )
        _assert_noise_control_sample(sample)
        manifests[sample.sample_id] = sample
    return manifests


def _migrate_dataset_labels(dataset_index: Path) -> None:
    if not dataset_index.exists():
        return
    from scripts.preprocessing import atomic_write_json, load_dataset_manifest

    index = load_dataset_manifest(dataset_index, require_samples=False)
    incompatible = {
        name: (label_id, LABELS.get(name))
        for name, label_id in index.labels.items()
        if LABELS.get(name) != label_id
    }
    if incompatible:
        raise RuntimeError(f"Existing dataset labels are incompatible: {incompatible}")
    if index.labels == LABELS:
        return
    atomic_write_json(
        dataset_index,
        {
            "schema_version": index.schema_version,
            "labels": LABELS,
            "samples": [
                path.relative_to(dataset_index.parent).as_posix()
                for path in index.samples
            ],
        },
    )


def _append_sample_to_dataset(
    dataset_index: Path,
    sample: Any,
    completed: set[str],
    *,
    labels: dict[str, int] = LABELS,
) -> None:
    """Append after the run's full audit without re-hashing every old sample."""
    from scripts.preprocessing import atomic_write_json, load_dataset_manifest

    if sample.sample_id in completed:
        raise ValueError(f"Dataset already contains sample_id {sample.sample_id!r}")
    if labels.get(sample.label_name) != sample.label_id:
        raise ValueError(
            f"Sample {sample.sample_id!r} has incompatible label "
            f"{sample.label_name}={sample.label_id}"
        )
    if dataset_index.exists():
        index = load_dataset_manifest(dataset_index, require_samples=False)
        if index.labels != labels:
            raise RuntimeError("Dataset label map changed after the initial audit")
        samples = list(index.samples)
    else:
        samples = []
    if sample.path in samples:
        raise ValueError(f"Dataset already references {sample.path}")
    try:
        relative = sample.path.relative_to(dataset_index.parent).as_posix()
    except ValueError as exc:
        raise ValueError(f"Sample manifest is outside the dataset: {sample.path}") from exc
    atomic_write_json(
        dataset_index,
        {
            "schema_version": 1,
            "labels": labels,
            "samples": [
                path.relative_to(dataset_index.parent).as_posix() for path in samples
            ]
            + [relative],
        },
    )


def _recover_or_remove_sample(
    sample_dir: Path,
    dataset_index: Path,
    completed: set[str],
    *,
    labels: dict[str, int] = LABELS,
    validator: Any | None = None,
) -> bool:
    if not sample_dir.exists():
        return False
    manifest_path = sample_dir / "sample.json"
    try:
        from scripts.preprocessing import (
            cleanup_simulation_sample,
            load_sample_manifest,
        )

        sample = load_sample_manifest(manifest_path)
        if sample.sample_id in completed:
            return True
        if validator is not None:
            validator(sample)
        cleanup_simulation_sample(sample_dir, dry_run=False)
        _append_sample_to_dataset(
            dataset_index, sample, completed, labels=labels
        )
        completed.add(sample.sample_id)
        print(f"Recovered completed sample: {sample.sample_id}")
        return True
    except Exception:
        _safe_remove_partial(sample_dir, sample_dir.parent)
        return False


def _unflagged_antenna_choices(ms: Path) -> tuple[tuple[int, str], ...]:
    from scripts.corruption import get_unflagged_antennas

    _, id_to_name = get_unflagged_antennas(str(ms))
    if not id_to_name:
        raise RuntimeError(f"No antenna occurs in an unflagged row of {ms}")
    return tuple(sorted(id_to_name.items()))


def _select_antenna(
    choices: Sequence[tuple[int, str]], *, seed: int
) -> tuple[int, str]:
    if not choices:
        raise ValueError("choices must contain at least one unflagged antenna")
    return random.Random(seed).choice(choices)


def _select_fixed_antenna(
    choices: Sequence[tuple[int, str]], antenna_id: int
) -> tuple[int, str]:
    try:
        return next(choice for choice in choices if choice[0] == antenna_id)
    except StopIteration as exc:
        raise RuntimeError(
            f"Fixed antenna ID {antenna_id} is not available in unflagged rows"
        ) from exc


def _assert_imaging_match(source_qa: dict[str, Any], result: Any) -> None:
    expected = source_qa.get("effective_imaging_parameters") or {}
    actual = result.effective_imaging_parameters
    differences = {
        key: (expected.get(key), actual.get(key))
        for key in MATCHED_IMAGING_KEYS
        if expected.get(key) != actual.get(key)
    }
    if differences:
        raise RuntimeError(f"Corrupted imaging settings differ from baseline: {differences}")


def _rescale_visibility_delta(
    V_ms: Path,
    perturbed_ms: Path,
    scale: float,
    *,
    chunk_rows: int = 4096,
) -> None:
    """Replace DATA with ``V + scale * (DATA - V)`` in bounded chunks."""
    import numpy as np

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Noise-control scale must be finite and positive: {scale!r}")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    try:
        from casatools import table
    except ImportError as exc:  # pragma: no cover - CASA supplies this module
        raise RuntimeError("CASA casatools is required to normalize noise controls") from exc

    reference = table()
    perturbed = table()
    reference_open = False
    perturbed_open = False
    try:
        reference.open(str(V_ms), nomodify=True)
        reference_open = True
        perturbed.open(str(perturbed_ms), nomodify=False)
        perturbed_open = True
        if "DATA" not in reference.colnames() or "DATA" not in perturbed.colnames():
            raise RuntimeError("Noise-control Measurement Sets must contain DATA")
        total_rows = int(reference.nrows())
        if int(perturbed.nrows()) != total_rows:
            raise RuntimeError("Noise-control Measurement Sets have different row counts")
        for start in range(0, total_rows, chunk_rows):
            count = min(chunk_rows, total_rows - start)
            V = np.asarray(
                reference.getcol("DATA", startrow=start, nrow=count)
            )
            values = np.asarray(
                perturbed.getcol("DATA", startrow=start, nrow=count)
            )
            if values.shape != V.shape:
                raise RuntimeError(
                    f"Noise-control DATA shapes differ: {V.shape} vs {values.shape}"
                )
            normalized = V + scale * (values - V)
            perturbed.putcol(
                "DATA",
                normalized.astype(values.dtype, copy=False),
                startrow=start,
                nrow=count,
            )
    finally:
        if perturbed_open:
            perturbed.close()
        if reference_open:
            reference.close()


def _apply_noise_control(
    source: ThermalSource,
    control: NoiseControl,
    V_ms: Path,
    observed_ms: Path,
    sigma: float,
    noise_model: str,
    noise_parameters: dict[str, Any],
    thermal_noise_seed: int,
) -> dict[str, Any]:
    """Add and normalize an independent Gaussian-derived visibility perturbation."""
    from scripts.corruption import measure_corruption_metrics
    from scripts.simulation import add_thermal_noise_inplace

    extra_seed = _stable_seed(source.source_id, control.label_name)
    if extra_seed == thermal_noise_seed:
        raise RuntimeError("Extra-noise and baseline-noise seeds must be independent")
    raw_noise = add_thermal_noise_inplace(
        observed_ms,
        noise_model=noise_model,
        noise_parameters=noise_parameters,
        seed=extra_seed,
    )
    raw_sigma = float(raw_noise["simplenoise_jy"])
    if not math.isclose(raw_sigma, sigma, rel_tol=1e-10):
        raise RuntimeError(
            f"Extra-noise draw resolved sigma={raw_sigma:.12g}, expected {sigma:.12g}"
        )
    raw_metrics = measure_corruption_metrics(V_ms, observed_ms, sigma)
    if not math.isfinite(raw_metrics.SNR_corr) or raw_metrics.SNR_corr <= 0.0:
        raise RuntimeError("Extra-noise draw has no finite positive perturbation norm")

    applied_scale = control.SNR_corr_target / raw_metrics.SNR_corr
    _rescale_visibility_delta(V_ms, observed_ms, applied_scale)
    measured = measure_corruption_metrics(V_ms, observed_ms, sigma)
    if not math.isclose(
        measured.SNR_corr,
        control.SNR_corr_target,
        rel_tol=NOISE_CONTROL_REL_TOL,
    ):
        correction = control.SNR_corr_target / measured.SNR_corr
        if not math.isfinite(correction) or abs(correction - 1.0) > 1e-3:
            raise RuntimeError(
                "Stored extra-noise norm requires an unbounded corrective rescale: "
                f"factor={correction:.12g}"
            )
        _rescale_visibility_delta(V_ms, observed_ms, correction)
        applied_scale *= correction
        measured = measure_corruption_metrics(V_ms, observed_ms, sigma)
    relative_error = abs(measured.SNR_corr / control.SNR_corr_target - 1.0)
    if relative_error > NOISE_CONTROL_REL_TOL:
        raise RuntimeError(
            f"Stored extra-noise SNR_corr={measured.SNR_corr:.12g} misses target "
            f"{control.SNR_corr_target:.12g} by {relative_error:.3g} relative"
        )

    sigma_extra = (
        sigma
        * control.SNR_corr_target
        / math.sqrt(2.0 * measured.valid_sample_count)
    )
    sigma_total = math.hypot(sigma, sigma_extra)
    return {
        "kind": "increased_noise",
        "definition": "Y_D=V+n0+eta; SNR_corr=||eta/sigma0||_2,I",
        "draw_policy": "independent Gaussian-derived draw normalized to fixed realized norm",
        "SNR_corr_target": control.SNR_corr_target,
        "SNR_corr_measured": measured.SNR_corr,
        "SNR_corr_relative_error": relative_error,
        "SNR_corr_relative_tolerance": NOISE_CONTROL_REL_TOL,
        "reference_sigma_jy": sigma,
        "valid_sample_count": measured.valid_sample_count,
        "extra_noise_seed": extra_seed,
        "baseline_noise_seed": thermal_noise_seed,
        "raw_draw_SNR_corr": raw_metrics.SNR_corr,
        "applied_scale": applied_scale,
        "equivalent_extra_sigma_jy": sigma_extra,
        "final_total_sigma_jy": sigma_total,
        "extra_noise_request": raw_noise,
    }


def _finalize_sample(
    source: ThermalSource,
    variant: Variant | None,
    antenna_id: int | None,
    antenna_name: str | None,
    antenna_selection_seed: int | None,
    error_direction_seed: int | None,
    V_ms: Path,
    sigma: float,
    norms: Any | None,
    noise_model: str,
    noise_parameters: dict[str, Any],
    thermal_noise_seed: int,
    output: Path,
    dataset_index: Path | None,
    *,
    sample_root: Path | None = None,
    provenance_root: Path | None = None,
    corruption_plot: Path | None = None,
    noise_control: NoiseControl | None = None,
) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    from scripts.corruption import (
        AntennaGainCorruption,
        ConstantGainSpec,
        TimeGrid,
        measure_corruption_metrics,
        solve_constant_gain,
        write_corruption_reports,
    )
    from scripts.imaging import BeamRegion, DefaultImagingConfig, image_ms
    from scripts.preprocessing import finalize_simulation_sample
    from scripts.simulation import add_thermal_noise_inplace, initialize_weights

    if variant is not None and noise_control is not None:
        raise ValueError("A sample cannot be both a gain and increased-noise variant")
    suffix = (
        noise_control.label_name
        if noise_control is not None
        else "not_corrupted" if variant is None else variant.label_name
    )
    sample_id = _sample_id(source.source_id, suffix)
    final_root = _sample_dir(output, sample_id) if provenance_root is None else provenance_root
    root = final_root if sample_root is None else sample_root
    root.mkdir(parents=True)
    observed_ms = root / f"{source.source_id}.ms"
    recorded_observed_ms = final_root / f"{source.source_id}.ms"
    print(f"[{sample_id}] copying noiseless V")
    shutil.copytree(V_ms, observed_ms)

    corruption_seed = None
    corruption_reports: tuple[Any, ...] = ()
    if variant is not None:
        if (
            antenna_id is None
            or antenna_name is None
            or norms is None
        ):
            raise RuntimeError("Corrupted variants require a selected antenna and norms")
        corruption_seed = _stable_seed(source.source_id, variant.label_name)
        solution = solve_constant_gain(
            ConstantGainSpec(
                V_ms,
                antenna_id,
                variant.family,
                variant.SNR_corr_target,
                sigma,
                seed=error_direction_seed,
                sign=FIXED_ERROR_SIGN,
            ),
            norms,
        )
        corruption = AntennaGainCorruption.from_constant_gain_solution(
            TimeGrid(CORRUPTION_SOLINT), solution
        )
        gain_table = root / f"{source.source_id}_{variant.label_name}.G"
        corruption.build_corrtable(
            str(observed_ms),
            str(gain_table),
            seed=corruption_seed,
            diagnostic_plot=(
                _corruption_plot_path(output, sample_id)
                if corruption_plot is None
                else corruption_plot
            ),
        )
        corruption.apply_corrtable(
            str(observed_ms),
            str(gain_table),
            seed=corruption_seed,
        )
        metrics = measure_corruption_metrics(V_ms, observed_ms, sigma)
        for name, actual, expected in (
            ("SNR_corr", metrics.SNR_corr, variant.SNR_corr_target),
            ("eps_vis", metrics.eps_vis, solution.eps_vis_expected),
        ):
            if not math.isclose(actual, expected, rel_tol=SNR_CORR_REL_TOL):
                raise RuntimeError(
                    f"Measured {name}={actual:.12g} does not match "
                    f"expected {expected:.12g}"
                )
        corruption_reports = (
            write_corruption_reports(
                corruption,
                json_path=root / "corruption.json",
                text_path=root / "corruption.txt",
                solution=solution,
                metrics=metrics,
                context={
                    "retained_sample_id": sample_id,
                    "application_index": 0,
                    "antenna_id": antenna_id,
                    "antenna_name": antenna_name,
                    "antenna_selection_seed": antenna_selection_seed,
                    "error_direction_seed": error_direction_seed,
                    "seed": corruption_seed,
                },
            ),
        )
    noise_control_metadata = None
    if noise_control is not None:
        noise_control_metadata = _apply_noise_control(
            source,
            noise_control,
            V_ms,
            observed_ms,
            sigma,
            noise_model,
            noise_parameters,
            thermal_noise_seed,
        )
    noise = add_thermal_noise_inplace(
        observed_ms,
        noise_model=noise_model,
        noise_parameters=noise_parameters,
        seed=thermal_noise_seed,
    )
    if noise_control_metadata is not None:
        noise["weight_initialization"] = initialize_weights(
            observed_ms, noise_control_metadata["final_total_sigma_jy"]
        )
    simulation_result = _write_branch_simulation_reports(
        source,
        sample_id=sample_id,
        sample_dir=root,
        V_ms=V_ms,
        observed_ms=observed_ms,
        provenance_observed_ms=recorded_observed_ms,
        noise=noise,
        corruption_seed=corruption_seed,
        thermal_noise_seed=thermal_noise_seed,
        noise_control=noise_control_metadata,
    )
    region = BeamRegion(
        min_radius_beams=source.metric_min_radius_beams,
        max_radius_beams=source.metric_max_radius_beams,
    )
    imaging_result = image_ms(
        observed_ms,
        DefaultImagingConfig,
        root / "imaging",
        imsize=source.imsize,
        metric_region=region,
        keep_intermediate_products=False,
        fits_invalid_policy="fill",
        fits_fill_value=0.0,
        pblimit=DATASET_PBLIMIT,
    )
    source_qa = json.loads(
        (source.simulation_result_dir / "qa.json").read_text(encoding="utf-8")
    )
    _assert_imaging_match(source_qa, imaging_result)
    if imaging_result.qa.metrics.region != region:
        raise RuntimeError("Image used a different metric region")
    finalized = finalize_simulation_sample(
        root,
        sample_id=sample_id,
        label_id=0 if noise_control is not None else LABELS[suffix],
        label_name="not_corrupted" if noise_control is not None else suffix,
        imaging_result=imaging_result,
        simulation_result=simulation_result,
        corruption_reports=corruption_reports,
        dataset_index=dataset_index,
    )
    if noise_control_metadata is not None:
        print(
            f"[{sample_id}] completed: extra-noise SNR_corr="
            f"{noise_control_metadata['SNR_corr_measured']:.6g}, "
            f"sigma_total={noise_control_metadata['final_total_sigma_jy']:.6g} Jy"
        )
    elif variant is not None:
        print(
            f"[{sample_id}] completed: eps_g={solution.eps_g:.6g}, "
            f"eps_vis={metrics.eps_vis:.6g}, SNR_corr={metrics.SNR_corr:.6g}"
        )
    return finalized.manifest.path


def _repair_root(output: Path) -> Path:
    return output / "repair_transactions"


def _repair_paths(output: Path, sample_id: str) -> dict[str, Path]:
    if sample_id in {"", ".", ".."} or Path(sample_id).name != sample_id:
        raise ValueError(f"Unsafe repair sample ID: {sample_id!r}")
    transaction = _repair_root(output) / sample_id
    return {
        "transaction": transaction,
        "state": transaction / "state.json",
        "staged": transaction / "staged",
        "backup": transaction / "backup",
        "failed": transaction / "failed_replacement",
        "plot": transaction / "corruption.png",
        "live": _sample_dir(output, sample_id),
        "final_plot": _corruption_plot_path(output, sample_id),
    }


def _write_repair_state(paths: dict[str, Path], sample_id: str, status: str) -> None:
    _atomic_write_json(
        paths["state"],
        {
            "schema_version": 1,
            "sample_id": sample_id,
            "status": status,
            "live_sample": str(paths["live"]),
            "backup_sample": str(paths["backup"]),
        },
    )


def _load_valid_replacement(path: Path, sample_id: str) -> Any:
    from scripts.preprocessing import load_sample_manifest

    sample = load_sample_manifest(path / "sample.json")
    if sample.sample_id != sample_id:
        raise RuntimeError(
            f"Repair replacement has sample ID {sample.sample_id!r}, expected {sample_id!r}"
        )
    _assert_square_pb_support(sample)
    _assert_sample_pb_policy(sample)
    return sample


def _restore_repair_backup(
    paths: dict[str, Path], sample_id: str, cause: Exception
) -> None:
    live = paths["live"]
    backup = paths["backup"]
    failed = paths["failed"]
    if live.exists():
        if failed.exists():
            raise RuntimeError(
                f"Cannot preserve failed replacement for {sample_id}; {failed} exists"
            ) from cause
        os.replace(live, failed)
    if backup.exists():
        os.replace(backup, live)
    _write_repair_state(paths, sample_id, "rolled_back")


def _finish_repair_transaction(output: Path, sample_id: str) -> Path:
    """Complete or safely roll back one prepared same-path sample replacement."""
    paths = _repair_paths(output, sample_id)
    live = paths["live"]
    staged = paths["staged"]
    backup = paths["backup"]
    try:
        if staged.exists() and not backup.exists():
            _load_valid_replacement(staged, sample_id)
            if not live.is_dir():
                raise RuntimeError(f"Cannot back up missing live sample: {live}")
            os.replace(live, backup)
            _write_repair_state(paths, sample_id, "old_moved")
        if staged.exists() and not live.exists():
            os.replace(staged, live)
        replacement = _load_valid_replacement(live, sample_id)
        if not backup.is_dir():
            raise RuntimeError(f"Repair transaction has no recoverable backup: {backup}")
        if paths["plot"].is_file():
            paths["final_plot"].parent.mkdir(parents=True, exist_ok=True)
            os.replace(paths["plot"], paths["final_plot"])
        _write_repair_state(paths, sample_id, "committed")
        return replacement.path
    except Exception as exc:
        if backup.exists():
            _restore_repair_backup(paths, sample_id, exc)
        raise


def _recover_repair_transactions(output: Path) -> None:
    root = _repair_root(output)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Invalid repair transaction root: {root}")
    for transaction in sorted(path for path in root.iterdir() if path.is_dir()):
        sample_id = transaction.name
        paths = _repair_paths(output, sample_id)
        state = (
            json.loads(paths["state"].read_text(encoding="utf-8"))
            if paths["state"].is_file()
            else {}
        )
        status = state.get("status")
        if status == "committed":
            _finish_repair_transaction(output, sample_id)
            continue
        if status == "rolled_back":
            if not paths["live"].is_dir():
                raise RuntimeError(f"Rolled-back repair has no live sample: {sample_id}")
            continue
        if paths["backup"].exists() or status in {"prepared", "old_moved"}:
            print(f"Recovering interrupted PB repair: {sample_id}")
            _finish_repair_transaction(output, sample_id)
            continue
        if not paths["live"].is_dir():
            raise RuntimeError(
                f"Uncommitted repair staging exists while live sample is missing: {sample_id}"
            )
        print(f"Discarding uncommitted PB-repair staging: {sample_id}")
        shutil.rmtree(transaction)


def _same_value(old: Any, new: Any) -> bool:
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        return isinstance(new, (int, float)) and math.isclose(
            float(old), float(new), rel_tol=1e-10, abs_tol=0.0
        )
    return old == new


def _assert_repair_equivalent(old: Any, new: Any) -> None:
    if (old.sample_id, old.label_id, old.label_name) != (
        new.sample_id,
        new.label_id,
        new.label_name,
    ):
        raise RuntimeError("PB repair changed the sample identity or label")
    old_simulation = json.loads(old.simulation.read_text(encoding="utf-8"))
    new_simulation = json.loads(new.simulation.read_text(encoding="utf-8"))
    for location, old_value, new_value in (
        (
            "noise.seed",
            (old_simulation.get("noise") or {}).get("seed"),
            (new_simulation.get("noise") or {}).get("seed"),
        ),
        (
            "shared_noise.seed",
            (old_simulation.get("shared_noise") or {}).get("seed"),
            (new_simulation.get("shared_noise") or {}).get("seed"),
        ),
    ):
        if old_value != new_value:
            raise RuntimeError(f"PB repair changed {location}: {old_value!r} != {new_value!r}")
    if len(old.corruptions) != len(new.corruptions):
        raise RuntimeError("PB repair changed the corruption-report count")
    for old_reference, new_reference in zip(old.corruptions, new.corruptions):
        old_report = json.loads(old_reference.corruption.read_text(encoding="utf-8"))
        new_report = json.loads(new_reference.corruption.read_text(encoding="utf-8"))
        comparisons = []
        for section, keys in (
            (
                "context",
                (
                    "antenna_id",
                    "antenna_name",
                    "antenna_selection_seed",
                    "error_direction_seed",
                    "seed",
                ),
            ),
            (
                "solution",
                (
                    "SNR_corr_target",
                    "SNR_corr_expected",
                    "antenna_id",
                    "corruption_type",
                    "sign",
                    "g_amp",
                    "phi_deg",
                    "eps_g",
                    "eps_vis_expected",
                ),
            ),
            ("metrics", ("SNR_corr", "eps_vis")),
        ):
            old_section = old_report.get(section) or {}
            new_section = new_report.get(section) or {}
            comparisons.extend(
                (f"{section}.{key}", old_section.get(key), new_section.get(key))
                for key in keys
            )
        changed = {
            name: (old_value, new_value)
            for name, old_value, new_value in comparisons
            if not _same_value(old_value, new_value)
        }
        if changed:
            raise RuntimeError(f"PB repair changed corruption provenance: {changed}")


def _repair_sample(
    source: ThermalSource,
    variant: Variant | None,
    antenna_id: int | None,
    antenna_name: str | None,
    antenna_selection_seed: int | None,
    error_direction_seed: int | None,
    V_ms: Path,
    sigma: float,
    norms: Any | None,
    noise_model: str,
    noise_parameters: dict[str, Any],
    thermal_noise_seed: int,
    output: Path,
    old: Any,
) -> Path:
    sample_id = old.sample_id
    paths = _repair_paths(output, sample_id)
    if paths["transaction"].exists():
        state = (
            json.loads(paths["state"].read_text(encoding="utf-8"))
            if paths["state"].is_file()
            else {}
        )
        if state.get("status") == "rolled_back" and paths["live"].is_dir():
            shutil.rmtree(paths["transaction"])
        else:
            _recover_repair_transactions(output)
    paths["transaction"].mkdir(parents=True, exist_ok=False)
    try:
        staged_manifest = _finalize_sample(
            source,
            variant,
            antenna_id,
            antenna_name,
            antenna_selection_seed,
            error_direction_seed,
            V_ms,
            sigma,
            norms,
            noise_model,
            noise_parameters,
            thermal_noise_seed,
            output,
            None,
            sample_root=paths["staged"],
            provenance_root=paths["live"],
            corruption_plot=paths["plot"],
        )
        from scripts.preprocessing import load_sample_manifest

        replacement = load_sample_manifest(staged_manifest)
        _assert_square_pb_support(replacement)
        _assert_sample_pb_policy(replacement)
        _assert_repair_equivalent(old, replacement)
        _write_repair_state(paths, sample_id, "prepared")
        return _finish_repair_transaction(output, sample_id)
    except Exception:
        if paths["transaction"].exists() and not paths["backup"].exists():
            state = (
                json.loads(paths["state"].read_text(encoding="utf-8"))
                if paths["state"].is_file()
                else {}
            )
            if state.get("status") != "rolled_back":
                shutil.rmtree(paths["transaction"])
        raise


def _source_manifests(
    manifests: dict[str, Any], source_id: str, *, noise_controls: bool = False
) -> dict[str, Any]:
    requested = (
        set(_requested_noise_controls(source_id))
        if noise_controls
        else set(_requested_samples(source_id))
    )
    return {
        sample_id: sample
        for sample_id, sample in manifests.items()
        if sample_id in requested
    }


def _write_comparison_plots(
    source_id: str,
    family: str,
    manifests: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    from scripts.imaging import BeamRegion, write_fits_comparison_plots

    baseline_id = _sample_id(source_id, "not_corrupted")
    sample_ids = [baseline_id] if baseline_id in manifests else []
    suffix_family = "noise" if family == "noise" else family
    sample_ids.extend(
        sample_id
        for target in SNR_CORR_TARGETS
        if (
            sample_id := _sample_id(
                source_id, f"{suffix_family}_snr_{_target_text(target)}"
            )
        )
        in manifests
    )
    if not sample_ids:
        return {"panels": [], "recipes": {}}
    selected = [manifests[sample_id] for sample_id in sample_ids]
    reference_qa = json.loads(
        selected[0].imaging_qa.read_text(encoding="utf-8")
    )
    region_payload = (reference_qa.get("metrics") or {}).get("region") or {}
    region = BeamRegion(
        min_radius_beams=region_payload.get("min_radius_beams"),
        max_radius_beams=region_payload.get("max_radius_beams"),
    )
    plot_inputs = []
    for sample_id, manifest in zip(sample_ids, selected):
        is_baseline = sample_id == baseline_id
        target = None if is_baseline else float(sample_id.rsplit("_", 1)[1])
        title = (
            "baseline: SNR_corr=0"
            if target is None
            else f"SNR_corr={_target_text(target)}"
        )
        plot_inputs.append((sample_id, title, manifest.products))
    rows, recipes = write_fits_comparison_plots(
        plot_inputs,
        output / "report_assets" / "images" / family,
        metric_region=region,
    )
    for row in rows:
        for channel in ("dirty", "clean", "residual", "psf"):
            row[channel] = str(Path(row[channel]).relative_to(output))
    return {"panels": rows, "recipes": recipes}


def _status_rows(
    source_id: str,
    completed: set[str],
    failures: Sequence[dict[str, Any]],
    *,
    family: str,
    requested: bool = True,
) -> list[dict[str, Any]]:
    failed = {
        item.get("sample_id")
        for item in failures
        if isinstance(item, dict) and item.get("source_id") == source_id
    }
    rows = []
    levels: Sequence[float | None] = (
        (None, *SNR_CORR_TARGETS) if family != "noise" else SNR_CORR_TARGETS
    )
    for target in levels:
        suffix = (
            "not_corrupted"
            if target is None
            else f"{family}_snr_{_target_text(target)}"
        )
        sample_id = _sample_id(source_id, suffix)
        status = (
            "not_requested"
            if not requested
            else "available"
            if sample_id in completed
            else "failed"
            if sample_id in failed
            else "missing"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "SNR_corr_target": 0.0 if target is None else target,
                "status": status,
            }
        )
    return rows


def _report_assets_exist(entry: dict[str, Any], output: Path) -> bool:
    for key in ("amplitude_plots", "phase_plots", "noise_control_plots"):
        for row in entry.get(key, []) or []:
            for channel in ("dirty", "clean", "residual", "psf"):
                value = row.get(channel)
                if not isinstance(value, str) or not (output / value).is_file():
                    return False
    for rows in (entry.get("corruption_plots") or {}).values():
        for row in rows:
            value = row.get("path")
            if not isinstance(value, str) or not (output / value).is_file():
                return False
    return True


def _report_entry_is_current(
    entry: dict[str, Any] | None,
    output: Path,
    main_status: dict[str, list[dict[str, Any]]],
    noise_status: list[dict[str, Any]],
) -> bool:
    return bool(
        entry
        and entry.get("main_variant_status") == main_status
        and entry.get("noise_control_status") == noise_status
        and _report_assets_exist(entry, output)
    )


def _iterate_and_validate_dataset(
    dataset_index: Path, source_ids: Sequence[str]
) -> dict[str, Any]:
    from scripts.preprocessing import (
        PARTITION_NAMES,
        load_dataset_manifest,
        load_sample_manifest,
        partition_for_sample,
        source_dataset_id,
    )

    index = load_dataset_manifest(dataset_index, require_samples=False)
    if index.labels != LABELS:
        raise RuntimeError(f"Dataset label map is incomplete or incompatible: {index.labels}")
    label_counts = {name: 0 for name in LABELS}
    partition_counts = {name: 0 for name in PARTITION_NAMES}
    shapes: set[tuple[int, int]] = set()
    expected_sources = set(source_ids)
    seen_ids: set[str] = set()
    for path in index.samples:
        sample = load_sample_manifest(path)
        if sample.sample_id in seen_ids:
            raise RuntimeError(f"Duplicate indexed sample ID: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        source_id = source_dataset_id(sample.sample_id)
        if source_id not in expected_sources:
            raise RuntimeError(f"Unexpected source entered dataset v1: {source_id}")
        if LABELS.get(sample.label_name) != sample.label_id:
            raise RuntimeError(f"Incorrect stable label for {sample.sample_id}")
        _assert_square_pb_support(sample)
        from scripts.preprocessing import load_fits_plane

        shapes.add(load_fits_plane(sample.products["dirty"]).shape)
        label_counts[sample.label_name] += 1
        partition_counts[partition_for_sample(sample.sample_id)] += 1
    expected_count = len(source_ids) * len(LABELS)
    if len(index.samples) != expected_count:
        raise RuntimeError(
            f"Dataset has {len(index.samples)} samples; expected {expected_count}"
        )
    if any(count != len(source_ids) for count in label_counts.values()):
        raise RuntimeError(f"Dataset label counts are incomplete: {label_counts}")
    return {
        "mode": "dataset manifest iteration plus retained FITS loading",
        "sample_count": len(index.samples),
        "label_counts": label_counts,
        "partition_counts": partition_counts,
        "image_shapes": [list(shape) for shape in sorted(shapes)],
    }


def _iterate_and_validate_noise_controls(
    dataset_index: Path,
    requested_source_ids: Sequence[str],
) -> dict[str, Any]:
    from scripts.preprocessing import (
        load_dataset_manifest,
        load_sample_manifest,
        partition_for_sample,
    )

    index = load_dataset_manifest(dataset_index, require_samples=False)
    if index.labels != NOISE_CONTROL_LABELS:
        raise RuntimeError("Noise-control label map is incomplete or incompatible")
    expected = {
        sample_id
        for source_id in requested_source_ids
        for sample_id in _requested_noise_controls(source_id)
    }
    seen: set[str] = set()
    counts = {"test": 0, "val": 0}
    for path in index.samples:
        sample = load_sample_manifest(path)
        if sample.sample_id in seen:
            raise RuntimeError(
                f"Duplicate indexed noise-control sample ID: {sample.sample_id}"
            )
        seen.add(sample.sample_id)
        _assert_noise_control_sample(sample)
        partition = partition_for_sample(sample.sample_id)
        if partition == "train":
            raise RuntimeError(
                f"Training sample entered noise-control index: {sample.sample_id}"
            )
        counts[partition] += 1
    missing = expected - seen
    if missing:
        raise RuntimeError(
            f"Noise-control index is missing {len(missing)} requested samples"
        )
    return {
        "mode": "separate held-out noise-control index audit",
        "sample_count": len(seen),
        "requested_sample_count": len(expected),
        "partition_counts": counts,
    }


def _thermal_exclusions(
    manifest: dict[str, Any],
    source_ids: Sequence[str],
    requested_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    from scripts.preprocessing import ALL_IDS, partition_for_sample

    failures = {
        item.get("id"): item
        for item in manifest.get("failures", [])
        if isinstance(item, dict)
    }
    completed = set(source_ids)
    return [
        {
            "id": source_id,
            "partition": partition_for_sample(source_id),
            "stage": failures.get(source_id, {}).get("stage"),
            "error": failures.get(source_id, {}).get(
                "error", "No completed thermal simulation entry"
            ),
        }
        for source_id in (ALL_IDS if requested_ids is None else requested_ids)
        if source_id not in completed
    ]


def _source_partition_counts(source_ids: Sequence[str]) -> dict[str, int]:
    from scripts.preprocessing import PARTITION_NAMES, partition_for_sample

    return {
        partition: sum(
            partition_for_sample(source_id) == partition for source_id in source_ids
        )
        for partition in PARTITION_NAMES
    }


def _new_report_manifest(
    source_run: Path,
    source_ids: Sequence[str],
    excluded_sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    from scripts.corruption import corruption_metric_definitions
    from scripts.imaging import imaging_metric_definitions

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "title": "Constant-gain corruption dataset",
        "description": (
            "Every completed source in the fixed train, test, and validation partitions "
            "has one uncorrupted baseline and constant one-antenna amplitude and phase "
            "variants. Antenna and error-direction draws follow the configured selection "
            "modes. SNR_corr=||Delta_V/sigma||_2 is measured against noiseless V before "
            "the shared thermal noise is added."
        ),
        "partition": "all",
        "source_ids": list(source_ids),
        "source_partition_counts": _source_partition_counts(source_ids),
        "excluded_sources": list(excluded_sources),
        "source_run": str(source_run),
        "dataset_index": "dataset.json",
        "configuration": {
            "SNR_corr_targets": list(SNR_CORR_TARGETS),
            "target_metric": "SNR_corr",
            "labels": LABELS,
            "corruption_solint": CORRUPTION_SOLINT,
            "SNR_corr_relative_tolerance": SNR_CORR_REL_TOL,
            "noise_control_SNR_corr_relative_tolerance": NOISE_CONTROL_REL_TOL,
            "noise_control_levels": list(SNR_CORR_TARGETS),
            "noise_control_default_partition": NOISE_CONTROL_DEFAULT_PARTITION,
            "noise_control_index": NOISE_CONTROL_INDEX_NAME,
            "noise_control_label": NOISE_CONTROL_LABELS,
            "noise_control_policy": (
                "held-out increased-noise controls only; label 0/not_corrupted; "
                "fixed realized ||eta/sigma0||_2,I; never included in dataset.json"
            ),
            "base_seed": BASE_SEED,
            "fixed_antenna_id": FIXED_ANTENNA_ID,
            "antenna_selection_policy": ANTENNA_SELECTION_POLICY,
            "antenna_selection_seed_scheme": (
                None
                if FIXED_ANTENNA_ID is not None
                else "stable SHA-256 seed of base_seed, source_id, and "
                "'<variant_label>:antenna'"
            ),
            "fixed_error_sign": FIXED_ERROR_SIGN,
            "constant_error_sign_policy": CONSTANT_ERROR_SIGN_POLICY,
            "imaging_configuration": "DefaultImagingConfig",
            "keep_intermediate_products": False,
            "fits_invalid_policy": "fill",
            "fits_fill_value": 0.0,
            "dataset_pblimit": DATASET_PBLIMIT,
            "primary_beam_mask_policy": (
                "final retained dataset imaging uses negative pblimit to suppress the "
                "output T/F PB mask; precursor flux calibration remains unchanged"
            ),
            "simulation_policy": (
                "reconstruct noiseless V; corrupt and measure Delta_V; then add the "
                "same requested thermal-noise realization to every branch"
            ),
            "image_metric_definitions": list(imaging_metric_definitions()),
            "corruption_metric_definitions": list(
                corruption_metric_definitions()
            ),
            "plotting": (
                "scripts.imaging.casa_image_to_png; one shared color scale per "
                "dirty/clean/residual/PSF channel within each comparison grid"
            ),
        },
        "sources": [],
        "failures": [],
        "dataset_iteration": None,
        "noise_controls": {
            "dataset_index": NOISE_CONTROL_INDEX_NAME,
            "requested_source_ids": [],
            "failures": [],
            "dataset_iteration": None,
        },
    }


def _resume_report_manifest(
    report: dict[str, Any],
    thermal_dir: Path,
    source_ids: Sequence[str],
    excluded_sources: Sequence[dict[str, Any]],
) -> bool:
    """Validate a known dataset policy and migrate only its four-level form."""
    if Path(report.get("source_run", "")).resolve() != thermal_dir:
        raise RuntimeError("Existing dataset run uses a different thermal source run")
    old_source_ids = report.get("source_ids")
    if not isinstance(old_source_ids, list) or len(old_source_ids) != len(set(old_source_ids)):
        raise RuntimeError("Existing dataset report has invalid source_ids")
    if not set(old_source_ids) <= set(source_ids):
        raise RuntimeError("Existing dataset source set is not a subset of the thermal run")

    expected_configuration = _new_report_manifest(
        thermal_dir, source_ids, excluded_sources
    )["configuration"]
    configuration = report.get("configuration") or {}
    levels = tuple(configuration.get("SNR_corr_targets") or ())
    labels = configuration.get("labels")
    legacy = levels == LEGACY_SNR_CORR_TARGETS and labels == LEGACY_LABELS
    current = levels == SNR_CORR_TARGETS and labels == LABELS
    if not (legacy or current):
        raise RuntimeError(
            "Existing dataset levels/labels are not the known four-level policy or "
            "the current eight-level policy"
        )
    migration_keys = {
        "SNR_corr_targets",
        "labels",
        "dataset_pblimit",
        "primary_beam_mask_policy",
        "noise_control_SNR_corr_relative_tolerance",
        "noise_control_levels",
        "noise_control_default_partition",
        "noise_control_index",
        "noise_control_label",
        "noise_control_policy",
        "plotting",
    }
    incompatible = {
        key: (configuration.get(key), expected)
        for key, expected in expected_configuration.items()
        if key not in migration_keys and configuration.get(key) != expected
    }
    if incompatible:
        raise RuntimeError(f"Existing dataset configuration is incompatible: {incompatible}")

    from scripts.preprocessing import partition_for_sample

    changed_partitions = {
        item.get("id"): (item.get("partition"), partition_for_sample(item.get("id")))
        for item in report.get("sources", [])
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item.get("partition") != partition_for_sample(item.get("id"))
    }
    if changed_partitions:
        raise RuntimeError(
            f"Existing source partitions changed unexpectedly: {changed_partitions}"
        )

    configuration.update(expected_configuration)
    report["report_schema_version"] = REPORT_SCHEMA_VERSION
    source_policy_changed = (
        legacy
        or old_source_ids != list(source_ids)
        or report.get("source_partition_counts")
        != _source_partition_counts(source_ids)
        or report.get("excluded_sources") != list(excluded_sources)
    )
    report["source_ids"] = list(source_ids)
    report["source_partition_counts"] = _source_partition_counts(source_ids)
    report["excluded_sources"] = list(excluded_sources)
    if source_policy_changed:
        report["dataset_iteration"] = None
    if legacy:
        report["sources"] = []
    controls = report.setdefault("noise_controls", {})
    if controls.get("dataset_index", NOISE_CONTROL_INDEX_NAME) != NOISE_CONTROL_INDEX_NAME:
        raise RuntimeError("Existing report uses a different noise-control index")
    controls.setdefault("dataset_index", NOISE_CONTROL_INDEX_NAME)
    controls.setdefault("requested_source_ids", [])
    controls.setdefault("failures", [])
    controls.setdefault("dataset_iteration", None)
    return legacy


def _requested_samples(source_id: str) -> dict[str, Variant | None]:
    return {
        _sample_id(
            source_id,
            "not_corrupted" if variant is None else variant.label_name,
        ): variant
        for variant in (None, *VARIANTS)
    }


def _pending_samples(
    requested: dict[str, Variant | None],
    completed: set[str],
    pb_repairs: set[str],
) -> dict[str, Variant | None]:
    return {
        sample_id: variant
        for sample_id, variant in requested.items()
        if sample_id not in completed or sample_id in pb_repairs
    }


def _requested_noise_controls(source_id: str) -> dict[str, NoiseControl]:
    return {
        _sample_id(source_id, control.label_name): control
        for control in NOISE_CONTROLS
    }


def _select_noise_control_sources(
    source_ids: Sequence[str],
    *,
    enabled: bool,
    explicit_source_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    from scripts.preprocessing import partition_for_sample

    available = set(source_ids)
    if explicit_source_ids:
        selected = tuple(explicit_source_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("Noise-control source selection contains duplicate IDs")
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(
                f"Noise-control sources have no completed thermal entry: {missing}"
            )
        training = [
            source_id
            for source_id in selected
            if partition_for_sample(source_id) == "train"
        ]
        if training:
            raise ValueError(
                f"Training sources cannot be noise controls: {training}"
            )
        return selected
    if not enabled:
        return ()
    return tuple(
        source_id
        for source_id in source_ids
        if partition_for_sample(source_id) == NOISE_CONTROL_DEFAULT_PARTITION
    )


def _noise_control_source_id(sample_id: str) -> str:
    marker = "_noise_snr_"
    if marker not in sample_id:
        raise ValueError(f"Invalid noise-control sample ID: {sample_id!r}")
    return sample_id.split(marker, 1)[0]


def run_experiment(
    *,
    source_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    generate_noise_controls: bool = False,
    noise_control_source_ids: Sequence[str] | None = None,
) -> Path:
    from scripts.preprocessing import partition_for_sample
    from scripts.reporting import QuartoReporter

    selected_thermal_dir = (
        Path(source_run).expanduser().resolve()
        if source_run is not None
        else THERMAL_DIR.resolve()
    )
    _ensure_thermal_entries(selected_thermal_dir)
    thermal_dir, thermal_manifest, source_ids = find_thermal_run(source_run)
    sources = [
        _thermal_source(thermal_dir, thermal_manifest, source_id)
        for source_id in source_ids
    ]
    active_noise_sources = _select_noise_control_sources(
        source_ids,
        enabled=generate_noise_controls,
        explicit_source_ids=noise_control_source_ids,
    )
    excluded_sources = _thermal_exclusions(
        thermal_manifest,
        source_ids,
        requested_ids=_configured_source_ids(),
    )
    if output_dir is None:
        output = DATASET_DIR.resolve()
    else:
        output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)

    report_qmd = output / "report.qmd"
    if not report_qmd.exists():
        shutil.copy2(REPORT_TEMPLATE, report_qmd)

    dataset_index = output / "dataset.json"
    noise_control_index = output / NOISE_CONTROL_INDEX_NAME
    _recover_repair_transactions(output)
    indexed = _audit_indexed_samples(dataset_index)
    main_manifests = dict(indexed.manifests)
    completed = set(indexed.manifests)
    pb_repairs = set(indexed.pb_repairs)
    noise_manifests = _audit_noise_controls(noise_control_index)
    completed_noise = set(noise_manifests)
    if pb_repairs:
        print(f"Detected {len(pb_repairs)} indexed samples requiring PB repair")

    requested_by_source = {
        source.source_id: _requested_samples(source.source_id)
        for source in sources
    }
    noise_requested_by_source = {
        source_id: _requested_noise_controls(source_id)
        for source_id in active_noise_sources
    }
    recovered_main = False
    for source_id, requested in requested_by_source.items():
        for sample_id in requested:
            if sample_id not in completed and _recover_or_remove_sample(
                    _sample_dir(output, sample_id), dataset_index, completed
            ):
                recovered_main = True
    if recovered_main:
        indexed = _audit_indexed_samples(dataset_index)
        main_manifests = dict(indexed.manifests)
        completed = set(main_manifests)
        pb_repairs = set(indexed.pb_repairs)
    recovered_noise = False
    for source_id, requested in noise_requested_by_source.items():
        for sample_id, control in requested.items():
            if sample_id not in completed_noise and _recover_or_remove_sample(
                _sample_dir(output, sample_id),
                noise_control_index,
                completed_noise,
                labels=NOISE_CONTROL_LABELS,
                validator=lambda sample, target=control.SNR_corr_target: (
                    _assert_noise_control_sample(sample, target)
                ),
            ):
                recovered_noise = True
    if recovered_noise:
        noise_manifests = _audit_noise_controls(noise_control_index)
        completed_noise = set(noise_manifests)

    report_json = output / "report.json"
    previous_report_text = None
    if report_json.exists():
        previous_report_text = report_json.read_text(encoding="utf-8")
        report = json.loads(previous_report_text)
        _resume_report_manifest(
            report, thermal_dir, source_ids, excluded_sources
        )
    else:
        report = _new_report_manifest(thermal_dir, source_ids, excluded_sources)
    controls_report = report["noise_controls"]
    stored_requested = controls_report.get("requested_source_ids")
    if not isinstance(stored_requested, list) or len(stored_requested) != len(
        set(stored_requested)
    ):
        raise RuntimeError("Report has invalid requested noise-control sources")
    indexed_noise_sources = {
        _noise_control_source_id(sample_id) for sample_id in completed_noise
    }
    requested_noise_sources = tuple(
        source_id
        for source_id in source_ids
        if source_id
        in (set(stored_requested) | set(active_noise_sources) | indexed_noise_sources)
    )
    if stored_requested != list(requested_noise_sources):
        controls_report["dataset_iteration"] = None
    controls_report["requested_source_ids"] = list(requested_noise_sources)
    _migrate_dataset_labels(dataset_index)
    serialized_report = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_changed = serialized_report != previous_report_text
    if report_changed:
        _atomic_write_json(report_json, report)
    reporter: QuartoReporter | None = None

    def report_sample_completed() -> None:
        nonlocal reporter, report_changed
        if reporter is None:
            reporter = QuartoReporter(report_qmd, every=REPORT_EVERY_SOURCES)
        reporter.sample_completed()
        report_changed = True

    pending_sources = [
        source
        for source in sources
        if (
            _pending_samples(
                requested_by_source[source.source_id], completed, pb_repairs
            )
            or {
                sample_id: control
                for sample_id, control in noise_requested_by_source.get(
                    source.source_id, {}
                ).items()
                if sample_id not in completed_noise
            }
        )
    ]
    if pending_sources:
        _check_disk_space(pending_sources, output)
    expected_samples = {
        sample_id
        for requested in requested_by_source.values()
        for sample_id in requested
    }
    source_order = {source_id: index for index, source_id in enumerate(source_ids)}
    try:
        for source_number, source in enumerate(sources, start=1):
            print(f"[{source_number}/{len(sources)}] source {source.source_id}")
            requested = requested_by_source[source.source_id]
            report["failures"] = [
                item
                for item in report.get("failures", [])
                if item.get("sample_id") not in completed
                or item.get("sample_id") in pb_repairs
            ]

            pending = _pending_samples(requested, completed, pb_repairs)
            requested_noise = noise_requested_by_source.get(source.source_id, {})
            pending_noise = {
                sample_id: control
                for sample_id, control in requested_noise.items()
                if sample_id not in completed_noise
            }
            source_changed = False
            if pending or pending_noise:
                if pending:
                    report["dataset_iteration"] = None
                if pending_noise:
                    controls_report["dataset_iteration"] = None
                _atomic_write_json(report_json, report)
                V_ms = _prepare_V_ms(source, output)
                noise_model, noise_parameters, sigma = _noise_request(source)
                thermal_noise_seed = _stable_seed(source.source_id, "thermal_noise")
                antenna_choices = (
                    _unflagged_antenna_choices(V_ms)
                    if any(variant is not None for variant in pending.values())
                    else ()
                )
                from scripts.corruption import (
                    ConstantGainSpec,
                    measure_constant_gain_norms,
                )

                norms_by_antenna: dict[int, Any] = {}
                for sample_id, variant in pending.items():
                    path = _sample_dir(output, sample_id)
                    is_repair = sample_id in pb_repairs
                    report["failures"] = [
                        item
                        for item in report.get("failures", [])
                        if item.get("sample_id") != sample_id
                    ]
                    try:
                        antenna_id = antenna_name = antenna_selection_seed = None
                        error_direction_seed = None
                        norms = None
                        if variant is not None:
                            antenna_selection_seed = _antenna_seed(
                                source.source_id, variant
                            )
                            error_direction_seed = _error_direction_seed(
                                source.source_id, variant
                            )
                            if FIXED_ANTENNA_ID is None:
                                assert antenna_selection_seed is not None
                                antenna_id, antenna_name = _select_antenna(
                                    antenna_choices,
                                    seed=antenna_selection_seed,
                                )
                            else:
                                antenna_id, antenna_name = _select_fixed_antenna(
                                    antenna_choices, FIXED_ANTENNA_ID
                                )
                            print(
                                f"[{sample_id}] selected unflagged antenna "
                                f"{antenna_name} (ID {antenna_id})"
                            )
                            if antenna_id not in norms_by_antenna:
                                norms_by_antenna[antenna_id] = (
                                    measure_constant_gain_norms(
                                        ConstantGainSpec(
                                            V_ms,
                                            antenna_id,
                                            variant.family,
                                            variant.SNR_corr_target,
                                            sigma,
                                            seed=error_direction_seed,
                                            sign=FIXED_ERROR_SIGN,
                                        )
                                    )
                                )
                            norms = norms_by_antenna[antenna_id]
                        if is_repair:
                            repaired_path = _repair_sample(
                                source,
                                variant,
                                antenna_id,
                                antenna_name,
                                antenna_selection_seed,
                                error_direction_seed,
                                V_ms,
                                sigma,
                                norms,
                                noise_model,
                                noise_parameters,
                                thermal_noise_seed,
                                output,
                                indexed.manifests[sample_id],
                            )
                            from scripts.preprocessing import load_sample_manifest

                            main_manifests[sample_id] = load_sample_manifest(
                                repaired_path
                            )
                            pb_repairs.discard(sample_id)
                        else:
                            finalized_path = _finalize_sample(
                                source,
                                variant,
                                antenna_id,
                                antenna_name,
                                antenna_selection_seed,
                                error_direction_seed,
                                V_ms,
                                sigma,
                                norms,
                                noise_model,
                                noise_parameters,
                                thermal_noise_seed,
                                output,
                                None,
                            )
                            from scripts.preprocessing import load_sample_manifest

                            created = load_sample_manifest(finalized_path)
                            _assert_square_pb_support(created)
                            _assert_sample_pb_policy(created)
                            _append_sample_to_dataset(
                                dataset_index, created, completed
                            )
                            main_manifests[sample_id] = created
                        completed.add(sample_id)
                        source_changed = True
                    except Exception as exc:
                        traceback.print_exc()
                        report["failures"].append(
                            {
                                "source_id": source.source_id,
                                "sample_id": sample_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if not is_repair and path.exists():
                            _safe_remove_partial(path, output / "samples")
                        _atomic_write_json(report_json, report)
                        print(f"[{sample_id}] FAILED: {type(exc).__name__}: {exc}")
                for sample_id, control in pending_noise.items():
                    path = _sample_dir(output, sample_id)
                    controls_report["failures"] = [
                        item
                        for item in controls_report.get("failures", [])
                        if item.get("sample_id") != sample_id
                    ]
                    try:
                        finalized_path = _finalize_sample(
                            source,
                            None,
                            None,
                            None,
                            None,
                            None,
                            V_ms,
                            sigma,
                            None,
                            noise_model,
                            noise_parameters,
                            thermal_noise_seed,
                            output,
                            None,
                            noise_control=control,
                        )
                        from scripts.preprocessing import load_sample_manifest

                        created = load_sample_manifest(finalized_path)
                        _assert_noise_control_sample(
                            created, control.SNR_corr_target
                        )
                        _append_sample_to_dataset(
                            noise_control_index,
                            created,
                            completed_noise,
                            labels=NOISE_CONTROL_LABELS,
                        )
                        completed_noise.add(sample_id)
                        noise_manifests[sample_id] = created
                        source_changed = True
                    except Exception as exc:
                        traceback.print_exc()
                        controls_report.setdefault("failures", []).append(
                            {
                                "source_id": source.source_id,
                                "sample_id": sample_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if path.exists():
                            _safe_remove_partial(path, output / "samples")
                        _atomic_write_json(report_json, report)
                        print(
                            f"[{sample_id}] FAILED: {type(exc).__name__}: {exc}"
                        )

            main_status = {
                family: _status_rows(
                    source.source_id,
                    completed,
                    report.get("failures", []),
                    family=family,
                )
                for family in ("amp", "phase")
            }
            noise_was_requested = source.source_id in requested_noise_sources
            noise_status = _status_rows(
                source.source_id,
                completed_noise,
                controls_report.get("failures", []),
                family="noise",
                requested=noise_was_requested,
            )
            existing_entry = next(
                (
                    item
                    for item in report.get("sources", [])
                    if item.get("id") == source.source_id
                ),
                None,
            )
            if source_changed or not _report_entry_is_current(
                existing_entry, output, main_status, noise_status
            ):
                manifests = _source_manifests(
                    main_manifests, source.source_id
                )
                source_noise_manifests = _source_manifests(
                    noise_manifests,
                    source.source_id,
                    noise_controls=True,
                )
                plots = {
                    family: _write_comparison_plots(
                        source.source_id, family, manifests, output
                    )
                    for family in ("amp", "phase")
                }
                if noise_was_requested or source_noise_manifests:
                    noise_plot_manifests = dict(source_noise_manifests)
                    baseline_id = _sample_id(source.source_id, "not_corrupted")
                    if baseline_id in manifests:
                        noise_plot_manifests[baseline_id] = manifests[baseline_id]
                    noise_plots = _write_comparison_plots(
                        source.source_id,
                        "noise",
                        noise_plot_manifests,
                        output,
                    )
                else:
                    noise_plots = {"panels": [], "recipes": {}}
                corruption_plots = {}
                for family in ("amp", "phase"):
                    rows = []
                    for target in SNR_CORR_TARGETS:
                        sample_id = _sample_id(
                            source.source_id,
                            f"{family}_snr_{_target_text(target)}",
                        )
                        path = _corruption_plot_path(output, sample_id)
                        if sample_id in completed and path.is_file():
                            rows.append(
                                {
                                    "SNR_corr_target": target,
                                    "path": str(path.relative_to(output)),
                                }
                            )
                    corruption_plots[family] = rows
                source_entry = {
                    "id": source.source_id,
                    "partition": partition_for_sample(source.source_id),
                    "source_snr": source.source_snr,
                    "predicted_image_rms_jy_per_beam": (
                        source.predicted_image_rms_jy_per_beam
                    ),
                    "simulation_metadata": str(source.simulation_metadata),
                    "amplitude_plots": plots["amp"]["panels"],
                    "phase_plots": plots["phase"]["panels"],
                    "amplitude_plot_recipe": plots["amp"]["recipes"],
                    "phase_plot_recipe": plots["phase"]["recipes"],
                    "noise_control_requested": noise_was_requested,
                    "noise_control_plots": noise_plots["panels"],
                    "noise_control_plot_recipe": noise_plots["recipes"],
                    "main_variant_status": main_status,
                    "noise_control_status": noise_status,
                    "corruption_plots": corruption_plots,
                }
                report["sources"] = [
                    item
                    for item in report.get("sources", [])
                    if item.get("id") != source.source_id
                ]
                report["sources"].append(source_entry)
                report["sources"].sort(key=lambda item: source_order[item["id"]])
                _atomic_write_json(report_json, report)
                report_sample_completed()
            if pending or pending_noise:
                _cleanup_V_work(source, output)
    finally:
        if (
            dataset_index.exists()
            and expected_samples <= completed
            and not pb_repairs
            and not report.get("failures")
        ):
            iteration = _iterate_and_validate_dataset(dataset_index, source_ids)
            if report.get("dataset_iteration") != iteration:
                report["dataset_iteration"] = iteration
                _atomic_write_json(report_json, report)
                report_changed = True
        requested_noise_samples = {
            sample_id
            for source_id in requested_noise_sources
            for sample_id in _requested_noise_controls(source_id)
        }
        if (
            requested_noise_sources
            and noise_control_index.exists()
            and requested_noise_samples <= completed_noise
            and not controls_report.get("failures")
        ):
            iteration = _iterate_and_validate_noise_controls(
                noise_control_index, requested_noise_sources
            )
            if controls_report.get("dataset_iteration") != iteration:
                controls_report["dataset_iteration"] = iteration
                _atomic_write_json(report_json, report)
                report_changed = True
        if report_changed:
            if reporter is None:
                reporter = QuartoReporter(report_qmd, every=REPORT_EVERY_SOURCES)
                reporter.sample_completed()
            reporter.finish()

    if report.get("failures"):
        raise RuntimeError(
            f"Dataset v1 has {len(report['failures'])} failed samples; rerun "
            f"with --output-dir {output} after inspecting report.json"
        )
    missing_samples = expected_samples - completed
    if missing_samples:
        raise RuntimeError(f"Dataset is missing {len(missing_samples)} expected samples")
    if pb_repairs:
        raise RuntimeError(f"Dataset still has {len(pb_repairs)} samples requiring PB repair")
    active_noise_samples = {
        sample_id
        for source_id in active_noise_sources
        for sample_id in _requested_noise_controls(source_id)
    }
    active_noise_failures = [
        item
        for item in controls_report.get("failures", [])
        if item.get("source_id") in set(active_noise_sources)
    ]
    if active_noise_failures:
        raise RuntimeError(
            f"Dataset v1 has {len(active_noise_failures)} failed active noise controls"
        )
    missing_noise = active_noise_samples - completed_noise
    if missing_noise:
        raise RuntimeError(
            f"Noise-control index is missing {len(missing_noise)} active samples"
        )
    print(f"Dataset: {dataset_index}")
    if noise_control_index.exists():
        print(f"Noise controls: {noise_control_index} ({len(completed_noise)} samples)")
    print(f"Report: {output / 'report.html'}")
    print(f"Samples: {len(completed)} across {len(source_ids)} sources")
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        help="Reusable extracted-thermal-simulation experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        help="New or resumable dataset-v1 experiment directory",
    )
    parser.add_argument(
        "--noise-controls",
        action="store_true",
        help="Generate increased-noise controls for all completed validation sources",
    )
    parser.add_argument(
        "--noise-control-source",
        action="append",
        dest="noise_control_source_ids",
        help=(
            "Generate controls for one completed validation/test source; repeat for "
            "multiple sources (implies --noise-controls)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    arguments = _parse_args(argv)
    return run_experiment(
        source_run=arguments.source_run,
        output_dir=arguments.output_dir,
        generate_noise_controls=(
            arguments.noise_controls or bool(arguments.noise_control_source_ids)
        ),
        noise_control_source_ids=arguments.noise_control_source_ids,
    )


if __name__ in {"__main__", "<run_path>"}:
    main()
