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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS_ROOT = ROOT / "collect" / "experiments"
THERMAL_RUN_GLOB = "extracted_thermal_simulation_comparison_*"
REPORT_TEMPLATE = ROOT / "scripts" / "reporting" / "create_dataset_v1.qmd"
SNR_CORR_TARGETS = (10.0, 30.0, 50.0, 100.0)
CORRUPTION_SOLINT = "10m"
SNR_CORR_REL_TOL = 0.01
BASE_SEED = 20260914
ANTENNA_SELECTION_POLICY = (
    "independent_uniform_with_replacement_from_unflagged_antennas_per_variant"
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


def _target_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, "g")


VARIANTS = tuple(
    Variant(family, target, f"{family}_snr_{_target_text(target)}")
    for family in ("amp", "phase")
    for target in SNR_CORR_TARGETS
)
LABELS = {
    "not_corrupted": 0,
    **{variant.label_name: index for index, variant in enumerate(VARIANTS, start=1)},
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


def find_thermal_run(
    source_run: str | Path | None = None,
) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    from scripts.preprocessing import ALL_IDS

    if source_run is not None:
        candidates = [Path(source_run).expanduser().resolve()]
    else:
        candidates = sorted(EXPERIMENTS_ROOT.glob(THERMAL_RUN_GLOB), reverse=True)
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
            source_ids = tuple(source_id for source_id in ALL_IDS if source_id in completed)
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
    noise: dict[str, object],
    corruption_seed: int | None,
    thermal_noise_seed: int,
):
    from scripts.simulation import (
        SIMULATION_REPORT_SCHEMA_VERSION,
        SimulationResult,
        write_simulation_reports,
    )

    names = ["copy_noiseless_V"]
    if corruption_seed is not None:
        names += ["corrupt_V", "measure_Delta_V"]
    names += ["thermal_noise", "weights"]
    stages = [
        {"name": name, "order": index}
        for index, name in enumerate(names, start=1)
    ]
    stages[-2].update(
        seed=thermal_noise_seed,
        shared_across_source_variants=True,
    )
    if corruption_seed is not None:
        stages[1]["seed"] = corruption_seed
    payload = _simulation_payload(source)
    for obsolete in ("component_list", "operations", "created_paths"):
        payload.pop(obsolete, None)
    payload.update(
        {
            "schema_version": SIMULATION_REPORT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_ms": str(V_ms),
            "output_ms": str(observed_ms),
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
            "generated_artifacts": [
                {
                    "role": "observed_ms",
                    "path": str(observed_ms),
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


def _existing_sample_ids(dataset_index: Path) -> set[str]:
    if not dataset_index.exists():
        return set()
    from scripts.preprocessing import load_dataset_manifest, load_sample_manifest

    index = load_dataset_manifest(dataset_index)
    return {load_sample_manifest(path).sample_id for path in index.samples}


def _recover_or_remove_sample(
    sample_dir: Path,
    dataset_index: Path,
    completed: set[str],
) -> bool:
    if not sample_dir.exists():
        return False
    manifest_path = sample_dir / "sample.json"
    try:
        from scripts.preprocessing import (
            add_sample_to_dataset,
            cleanup_simulation_sample,
            load_sample_manifest,
        )

        sample = load_sample_manifest(manifest_path)
        if sample.sample_id in completed:
            return True
        cleanup_simulation_sample(sample_dir, dry_run=False)
        add_sample_to_dataset(dataset_index, manifest_path)
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


def _finalize_sample(
    source: ThermalSource,
    variant: Variant | None,
    antenna_id: int | None,
    antenna_name: str | None,
    antenna_selection_seed: int | None,
    V_ms: Path,
    sigma: float,
    norms: Any | None,
    noise_model: str,
    noise_parameters: dict[str, Any],
    thermal_noise_seed: int,
    output: Path,
    dataset_index: Path,
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
    from scripts.simulation import add_thermal_noise_inplace

    suffix = "not_corrupted" if variant is None else variant.label_name
    sample_id = _sample_id(source.source_id, suffix)
    root = _sample_dir(output, sample_id)
    root.mkdir(parents=True)
    observed_ms = root / f"{source.source_id}.ms"
    print(f"[{sample_id}] copying noiseless V")
    shutil.copytree(V_ms, observed_ms)

    corruption_seed = None
    corruption_reports: tuple[Any, ...] = ()
    if variant is not None:
        if (
            antenna_id is None
            or antenna_name is None
            or antenna_selection_seed is None
            or norms is None
        ):
            raise RuntimeError("Corrupted variants require a selected antenna and norms")
        solution = solve_constant_gain(
            ConstantGainSpec(
                V_ms,
                antenna_id,
                variant.family,
                variant.SNR_corr_target,
                sigma,
            ),
            norms,
        )
        corruption = AntennaGainCorruption.from_constant_gain_solution(
            TimeGrid(CORRUPTION_SOLINT), solution
        )
        corruption_seed = _stable_seed(source.source_id, variant.label_name)
        gain_table = root / f"{source.source_id}_{variant.label_name}.G"
        corruption.build_corrtable(
            str(observed_ms),
            str(gain_table),
            seed=corruption_seed,
            diagnostic_plot=_corruption_plot_path(output, sample_id),
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
                    "seed": corruption_seed,
                },
            ),
        )
    noise = add_thermal_noise_inplace(
        observed_ms,
        noise_model=noise_model,
        noise_parameters=noise_parameters,
        seed=thermal_noise_seed,
    )
    simulation_result = _write_branch_simulation_reports(
        source,
        sample_id=sample_id,
        sample_dir=root,
        V_ms=V_ms,
        observed_ms=observed_ms,
        noise=noise,
        corruption_seed=corruption_seed,
        thermal_noise_seed=thermal_noise_seed,
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
        label_id=LABELS[suffix],
        label_name=suffix,
        imaging_result=imaging_result,
        simulation_result=simulation_result,
        corruption_reports=corruption_reports,
        dataset_index=dataset_index,
    )
    if variant is not None:
        print(
            f"[{sample_id}] completed: eps_g={solution.eps_g:.6g}, "
            f"eps_vis={metrics.eps_vis:.6g}, SNR_corr={metrics.SNR_corr:.6g}"
        )
    return finalized.manifest.path


def _source_manifests(dataset_index: Path, source_id: str) -> dict[str, Any]:
    from scripts.preprocessing import load_dataset_manifest, load_sample_manifest

    index = load_dataset_manifest(dataset_index, require_samples=False)
    requested = {
        _sample_id(source_id, "not_corrupted"),
        *(_sample_id(source_id, variant.label_name) for variant in VARIANTS),
    }
    manifests = {
        sample.sample_id: sample
        for path in index.samples
        if (sample := load_sample_manifest(path)).sample_id in requested
    }
    missing = requested - set(manifests)
    if missing:
        raise KeyError(f"Missing source manifests: {sorted(missing)}")
    return manifests


def _write_comparison_plots(
    source_id: str,
    family: str,
    manifests: dict[str, Any],
    output: Path,
    shared_limits: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    from scripts.imaging import BeamRegion, write_fits_comparison_plots

    sample_ids = [_sample_id(source_id, "not_corrupted")]
    sample_ids.extend(
        _sample_id(source_id, f"{family}_snr_{_target_text(target)}")
        for target in SNR_CORR_TARGETS
    )
    selected = [manifests[sample_id] for sample_id in sample_ids]
    baseline_qa = json.loads(
        selected[0].imaging_qa.read_text(encoding="utf-8")
    )
    region_payload = (baseline_qa.get("metrics") or {}).get("region") or {}
    region = BeamRegion(
        min_radius_beams=region_payload.get("min_radius_beams"),
        max_radius_beams=region_payload.get("max_radius_beams"),
    )
    plot_inputs = []
    for index, (sample_id, manifest) in enumerate(zip(sample_ids, selected)):
        target = None if index == 0 else SNR_CORR_TARGETS[index - 1]
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
        display_limits_mjy_per_beam=shared_limits,
    )
    for row in rows:
        for channel in ("dirty", "clean", "residual"):
            row[channel] = str(Path(row[channel]).relative_to(output))
    return {"panels": rows, "recipes": recipes}


def _shared_source_display_limits(
    source_id: str,
    manifests: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    from scripts.imaging import shared_fits_display_limits

    sample_ids = [_sample_id(source_id, "not_corrupted")]
    sample_ids.extend(
        _sample_id(source_id, variant.label_name) for variant in VARIANTS
    )
    return {
        channel: shared_fits_display_limits(
            [manifests[sample_id].products[channel] for sample_id in sample_ids]
        )
        for channel in ("dirty", "clean", "residual")
    }


def _iterate_and_validate_dataset(
    dataset_index: Path, source_ids: Sequence[str]
) -> dict[str, Any]:
    from scripts.preprocessing import (
        PARTITION_NAMES,
        load_dataset_manifest,
        load_sample_manifest,
        partition_for_sample,
        source_dataset_id,
        validate_fits_triplet,
    )

    index = load_dataset_manifest(dataset_index)
    label_counts = {name: 0 for name in LABELS}
    partition_counts = {name: 0 for name in PARTITION_NAMES}
    shapes: set[tuple[int, int]] = set()
    expected_sources = set(source_ids)
    for path in index.samples:
        sample = load_sample_manifest(path)
        source_id = source_dataset_id(sample.sample_id)
        if source_id not in expected_sources:
            raise RuntimeError(f"Unexpected source entered dataset v1: {source_id}")
        planes = validate_fits_triplet(sample.products)
        shapes.add(planes["dirty"].shape)
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


def _thermal_exclusions(
    manifest: dict[str, Any], source_ids: Sequence[str]
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
        for source_id in ALL_IDS
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
        "title": "Constant-gain corruption dataset",
        "description": (
            "Every completed source in the fixed train, test, and validation partitions "
            "has one uncorrupted baseline and constant one-antenna amplitude and phase "
            "variants. Each corrupted variant independently draws an unflagged antenna. "
            "SNR_corr=||Delta_V/sigma||_2 is measured against noiseless V before the "
            "shared thermal noise is added."
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
            "base_seed": BASE_SEED,
            "antenna_selection_policy": ANTENNA_SELECTION_POLICY,
            "antenna_selection_seed_scheme": (
                "stable SHA-256 seed of base_seed, source_id, and "
                "'<variant_label>:antenna'"
            ),
            "imaging_configuration": "DefaultImagingConfig",
            "keep_intermediate_products": False,
            "fits_invalid_policy": "fill",
            "fits_fill_value": 0.0,
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
                "dirty/clean/residual channel across all variants"
            ),
        },
        "sources": [],
        "failures": [],
        "dataset_iteration": None,
    }


def run_experiment(
    *,
    source_run: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    from scripts.preprocessing import partition_for_sample
    from scripts.reporting import QuartoReporter

    thermal_dir, thermal_manifest, source_ids = find_thermal_run(source_run)
    sources = [
        _thermal_source(thermal_dir, thermal_manifest, source_id)
        for source_id in source_ids
    ]
    excluded_sources = _thermal_exclusions(thermal_manifest, source_ids)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output = EXPERIMENTS_ROOT / f"dataset_v1_{timestamp}"
    else:
        output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    _check_disk_space(sources, output)

    report_qmd = output / "report.qmd"
    if not report_qmd.exists():
        shutil.copy2(REPORT_TEMPLATE, report_qmd)
    report_json = output / "report.json"
    if report_json.exists():
        report = json.loads(report_json.read_text(encoding="utf-8"))
        if Path(report.get("source_run", "")).resolve() != thermal_dir:
            raise RuntimeError("Existing dataset run uses a different thermal source run")
        if report.get("source_ids") != list(source_ids):
            raise RuntimeError("Existing dataset run uses a different source set")
        existing_policy = (report.get("configuration") or {}).get(
            "antenna_selection_policy"
        )
        if existing_policy != ANTENNA_SELECTION_POLICY:
            raise RuntimeError(
                "Existing dataset run uses a different antenna-selection policy; "
                "use a new --output-dir"
            )
    else:
        report = _new_report_manifest(thermal_dir, source_ids, excluded_sources)
        _atomic_write_json(report_json, report)
    report.setdefault("configuration", {}).update(
        antenna_selection_policy=ANTENNA_SELECTION_POLICY,
        fits_invalid_policy="fill",
        fits_fill_value=0.0,
    )
    _atomic_write_json(report_json, report)

    dataset_index = output / "dataset.json"
    completed = _existing_sample_ids(dataset_index)
    reporter = QuartoReporter(report_qmd, every=REPORT_EVERY_SOURCES)
    variants = (None, *VARIANTS)
    requested_by_source = {
        source.source_id: {
            _sample_id(
                source.source_id,
                "not_corrupted" if variant is None else variant.label_name,
            ): variant
            for variant in variants
        }
        for source in sources
    }
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
            for sample_id in requested:
                if sample_id not in completed:
                    _recover_or_remove_sample(
                        _sample_dir(output, sample_id), dataset_index, completed
                    )
            report["failures"] = [
                item
                for item in report.get("failures", [])
                if item.get("sample_id") not in completed
            ]

            pending = {
                sample_id: variant
                for sample_id, variant in requested.items()
                if sample_id not in completed
            }
            if pending:
                V_ms = _prepare_V_ms(source, output)
                antenna_choices = _unflagged_antenna_choices(V_ms)
                noise_model, noise_parameters, sigma = _noise_request(source)
                thermal_noise_seed = _stable_seed(source.source_id, "thermal_noise")
                from scripts.corruption import (
                    ConstantGainSpec,
                    measure_constant_gain_norms,
                )

                norms_by_antenna: dict[int, Any] = {}
                for sample_id, variant in pending.items():
                    path = _sample_dir(output, sample_id)
                    report["failures"] = [
                        item
                        for item in report.get("failures", [])
                        if item.get("sample_id") != sample_id
                    ]
                    try:
                        antenna_id = antenna_name = antenna_selection_seed = None
                        norms = None
                        if variant is not None:
                            antenna_selection_seed = _stable_seed(
                                source.source_id,
                                f"{variant.label_name}:antenna",
                            )
                            antenna_id, antenna_name = _select_antenna(
                                antenna_choices,
                                seed=antenna_selection_seed,
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
                                        )
                                    )
                                )
                            norms = norms_by_antenna[antenna_id]
                        _finalize_sample(
                            source,
                            variant,
                            antenna_id,
                            antenna_name,
                            antenna_selection_seed,
                            V_ms,
                            sigma,
                            norms,
                            noise_model,
                            noise_parameters,
                            thermal_noise_seed,
                            output,
                            dataset_index,
                        )
                        completed.add(sample_id)
                    except Exception as exc:
                        traceback.print_exc()
                        report["failures"].append(
                            {
                                "source_id": source.source_id,
                                "sample_id": sample_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if path.exists():
                            _safe_remove_partial(path, output / "samples")
                        _atomic_write_json(report_json, report)
                        print(f"[{sample_id}] FAILED: {type(exc).__name__}: {exc}")

            if set(requested) <= completed:
                manifests = _source_manifests(dataset_index, source.source_id)
                shared_limits = _shared_source_display_limits(
                    source.source_id, manifests
                )
                plots = {
                    family: _write_comparison_plots(
                        source.source_id, family, manifests, output, shared_limits
                    )
                    for family in ("amp", "phase")
                }
                corruption_plots = {
                    family: [
                        {
                            "SNR_corr_target": target,
                            "path": str(
                                _corruption_plot_path(
                                    output,
                                    _sample_id(
                                        source.source_id,
                                        f"{family}_snr_{_target_text(target)}",
                                    ),
                                ).relative_to(output)
                            ),
                        }
                        for target in SNR_CORR_TARGETS
                    ]
                    for family in plots
                }
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
                reporter.sample_completed()
            _cleanup_V_work(source, output)
    finally:
        if (
            dataset_index.exists()
            and expected_samples <= completed
            and not report.get("failures")
        ):
            report["dataset_iteration"] = _iterate_and_validate_dataset(
                dataset_index, source_ids
            )
            _atomic_write_json(report_json, report)
        reporter.finish()

    if report.get("failures"):
        raise RuntimeError(
            f"Dataset v1 has {len(report['failures'])} failed samples; rerun "
            f"with --output-dir {output} after inspecting report.json"
        )
    missing_samples = expected_samples - completed
    if missing_samples:
        raise RuntimeError(f"Dataset is missing {len(missing_samples)} expected samples")
    print(f"Dataset: {dataset_index}")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    arguments = _parse_args(argv)
    return run_experiment(
        source_run=arguments.source_run,
        output_dir=arguments.output_dir,
    )


if __name__ in {"__main__", "<run_path>"}:
    main()
