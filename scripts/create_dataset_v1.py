#!/usr/bin/env python3
"""Build dataset v1 from reused matched-S/N thermal simulations.

Run from the repository root with CASA::

    '/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa' \
        --nogui --nologger -c scripts/create_dataset_v1.py

The development driver processes exactly one source from the hard-coded
preprocessing test partition. It reuses the newest complete matched-S/N thermal
simulation run, creates one uncorrupted sample and eight one-antenna
constant-gain variants, images each corrupted copy, compacts it immediately,
validates every retained FITS triplet, and renders an HTML calibration report.
Only one transient corrupted MS is retained at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS_ROOT = ROOT / "collect" / "experiments"
THERMAL_RUN_GLOB = "extracted_thermal_simulation_comparison_*"
REPORT_TEMPLATE = ROOT / "scripts" / "reporting" / "create_dataset_v1.qmd"
TARGET_DETECTABILITIES = (10.0, 30.0, 50.0, 100.0)
CORRUPTION_SOLINT = "10m"
BASE_SEED = 20260914
REPORT_EVERY_SOURCES = 1
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
LABELS = {
    "not_corrupted": 0,
    "amp_rho_10": 1,
    "amp_rho_30": 2,
    "amp_rho_50": 3,
    "amp_rho_100": 4,
    "phase_rho_10": 5,
    "phase_rho_30": 6,
    "phase_rho_50": 7,
    "phase_rho_100": 8,
}


@dataclass(frozen=True)
class ThermalSource:
    source_id: str
    experiment_dir: Path
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
    target_rho_corr: float
    label_name: str
    label_id: int

    @property
    def suffix(self) -> str:
        return self.label_name


def _target_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, "g")


def _variants() -> tuple[Variant, ...]:
    result = []
    for family in ("amp", "phase"):
        for target in TARGET_DETECTABILITIES:
            label = f"{family}_rho_{_target_text(target)}"
            result.append(Variant(family, target, label, LABELS[label]))
    return tuple(result)


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
        raise ValueError(f"missing completed test IDs: {missing_ids}")
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
    required_ids: Sequence[str], source_run: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    if source_run is not None:
        candidates = [Path(source_run).expanduser().resolve()]
    else:
        candidates = sorted(EXPERIMENTS_ROOT.glob(THERMAL_RUN_GLOB), reverse=True)
    failures = []
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "report.json").is_file():
            failures.append(f"{candidate}: no report.json")
            continue
        try:
            return candidate.resolve(), _load_thermal_run(candidate, required_ids)
        except Exception as exc:
            failures.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No reusable thermal experiment contains the complete test partition:\n"
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
        experiment_dir=experiment_dir,
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
    largest = max(_tree_size(source.simulation_ms) for source in sources)
    required = 4 * largest + MIN_FREE_HEADROOM_BYTES
    free = shutil.disk_usage(destination).free
    print(
        f"Disk preflight: largest source MS={largest / 1024**3:.2f} GiB, "
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


def _copy_simulation_reports(source: ThermalSource, sample_dir: Path) -> SimpleNamespace:
    from scripts.simulation import render_simulation_text

    payload = json.loads(source.simulation_metadata.read_text(encoding="utf-8"))
    json_path = sample_dir / "simulation.json"
    text_path = sample_dir / "simulation.txt"
    shutil.copy2(source.simulation_metadata, json_path)
    text_path.write_text(render_simulation_text(payload), encoding="utf-8")
    return SimpleNamespace(metadata_json=json_path, metadata_text=text_path)


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


def _finalize_baseline(
    source: ThermalSource,
    output: Path,
    dataset_index: Path,
) -> Path:
    from scripts.imaging import export_fits_triplet
    from scripts.imaging.config import read_beam
    from scripts.preprocessing import finalize_simulation_sample

    sample_id = _sample_id(source.source_id, "not_corrupted")
    root = _sample_dir(output, sample_id)
    root.mkdir(parents=True)
    imaging_dir = root / "imaging"
    imaging_dir.mkdir()
    source_qa = json.loads(
        (source.simulation_result_dir / "qa.json").read_text(encoding="utf-8")
    )
    products = source_qa.get("products") or {}
    dirty = Path(products["dirty_image"]).resolve()
    clean = Path(products["clean_image"]).resolve()
    residual = Path(products["residual_image"]).resolve()
    fits = export_fits_triplet(
        dirty,
        clean,
        residual,
        imaging_dir,
        fallback_beam=read_beam(clean),
        invalid_policy="fill",
        fill_value=0.0,
    )
    qa_json = imaging_dir / "qa.json"
    qa_text = imaging_dir / "qa.txt"
    shutil.copy2(source.simulation_result_dir / "qa.json", qa_json)
    shutil.copy2(source.simulation_result_dir / "qa.txt", qa_text)
    imaging_result = SimpleNamespace(
        dirty_fits=fits["dirty"],
        clean_fits=fits["clean"],
        residual_fits=fits["residual"],
        qa_json=qa_json,
        qa_text=qa_text,
    )
    simulation_result = _copy_simulation_reports(source, root)
    finalized = finalize_simulation_sample(
        root,
        sample_id=sample_id,
        label_id=LABELS["not_corrupted"],
        label_name="not_corrupted",
        imaging_result=imaging_result,
        simulation_result=simulation_result,
        dataset_index=dataset_index,
    )
    return finalized.manifest.path


def _select_antenna(ms: Path) -> tuple[int, str]:
    from scripts.corruption import get_unflagged_antennas

    _, id_to_name = get_unflagged_antennas(str(ms))
    if not id_to_name:
        raise RuntimeError(f"No antenna occurs in an unflagged row of {ms}")
    antenna_id = min(id_to_name)
    return antenna_id, id_to_name[antenna_id]


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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


def _finalize_variant(
    source: ThermalSource,
    variant: Variant,
    antenna_id: int,
    antenna_name: str,
    output: Path,
    dataset_index: Path,
) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    from scripts.corruption import (
        AntennaGainCorruption,
        ConstantGainParameters,
        TimeGrid,
        write_corruption_reports,
    )
    from scripts.imaging import BeamRegion, DefaultImagingConfig, image_ms
    from scripts.preprocessing import finalize_simulation_sample

    sample_id = _sample_id(source.source_id, variant.suffix)
    root = _sample_dir(output, sample_id)
    root.mkdir(parents=True)
    (root / "images").mkdir()
    copied_ms = root / f"{source.source_id}.ms"
    gain_table = root / f"{source.source_id}_{variant.suffix}.G"
    print(f"[{sample_id}] copying reusable thermal MS")
    shutil.copytree(source.simulation_ms, copied_ms)

    simulation_payload = json.loads(
        source.simulation_metadata.read_text(encoding="utf-8")
    )
    noise = simulation_payload.get("noise") or {}
    parameters = ConstantGainParameters(
        ms=copied_ms,
        antenna_id=antenna_id,
        corruption_type=variant.family,
        target_rho_corr=variant.target_rho_corr,
        thermal_noise_jy=noise.get("simplenoise_jy"),
    )
    corruption = AntennaGainCorruption.from_detectability(
        TimeGrid(solint=CORRUPTION_SOLINT, interp="linear"), parameters
    )
    seed = _stable_seed(source.source_id, variant.suffix)
    diagnostic_plot = _corruption_plot_path(output, sample_id)
    with _working_directory(root):
        corruption.build_corrtable(
            str(copied_ms.resolve()),
            str(gain_table.resolve()),
            seed=seed,
            diagnostic_plot=diagnostic_plot,
        ).apply_corrtable(
            str(copied_ms.resolve()), str(gain_table.resolve()), seed=seed
        )
    corruption_reports = write_corruption_reports(
        corruption,
        json_path=root / "corruption.json",
        text_path=root / "corruption.txt",
        context={
            "name": "create_dataset_v1",
            "source_dataset_id": source.source_id,
            "retained_sample_id": sample_id,
            "application_index": 0,
            "antenna_id": antenna_id,
            "antenna_name": antenna_name,
            "seed": seed,
        },
    )
    region = BeamRegion(
        min_radius_beams=source.metric_min_radius_beams,
        max_radius_beams=source.metric_max_radius_beams,
    )
    imaging_result = image_ms(
        copied_ms,
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
        raise RuntimeError("Corrupted image used a different metric region from baseline")
    simulation_result = _copy_simulation_reports(source, root)
    finalized = finalize_simulation_sample(
        root,
        sample_id=sample_id,
        label_id=variant.label_id,
        label_name=variant.label_name,
        imaging_result=imaging_result,
        simulation_result=simulation_result,
        corruption_reports=(corruption_reports,),
        dataset_index=dataset_index,
    )
    print(
        f"[{sample_id}] completed: gain_error="
        f"{corruption.metrics.gain_error_magnitude:.6g}, "
        f"epsilon_vis={corruption.metrics.epsilon_vis:.6g}, "
        f"rho_corr={corruption.metrics.rho_corr:.6g}"
    )
    return finalized.manifest.path


def _manifest_for_sample(dataset_index: Path, sample_id: str):
    from scripts.preprocessing import load_dataset_manifest, load_sample_manifest

    index = load_dataset_manifest(dataset_index, require_samples=False)
    for path in index.samples:
        sample = load_sample_manifest(path)
        if sample.sample_id == sample_id:
            return sample
    raise KeyError(sample_id)


def _write_comparison_plots(
    source_id: str,
    family: str,
    dataset_index: Path,
    output: Path,
    shared_limits: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    from scripts.imaging import BeamRegion, write_fits_comparison_plots

    sample_ids = [_sample_id(source_id, "not_corrupted")]
    sample_ids.extend(
        _sample_id(source_id, f"{family}_rho_{_target_text(target)}")
        for target in TARGET_DETECTABILITIES
    )
    manifests = [
        _manifest_for_sample(dataset_index, sample_id) for sample_id in sample_ids
    ]
    baseline_qa = json.loads(
        manifests[0].imaging_qa.read_text(encoding="utf-8")
    )
    region_payload = (baseline_qa.get("metrics") or {}).get("region") or {}
    region = BeamRegion(
        min_radius_beams=region_payload.get("min_radius_beams"),
        max_radius_beams=region_payload.get("max_radius_beams"),
    )
    plot_inputs = []
    for index, (sample_id, manifest) in enumerate(zip(sample_ids, manifests)):
        target = None if index == 0 else TARGET_DETECTABILITIES[index - 1]
        title = (
            "baseline: rho_corr=0"
            if target is None
            else f"rho_corr={_target_text(target)}"
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
    dataset_index: Path,
) -> dict[str, tuple[float, float]]:
    from scripts.imaging import shared_fits_display_limits

    sample_ids = [_sample_id(source_id, "not_corrupted")]
    sample_ids.extend(
        _sample_id(source_id, variant.suffix) for variant in _variants()
    )
    manifests = [
        _manifest_for_sample(dataset_index, sample_id) for sample_id in sample_ids
    ]
    return {
        channel: shared_fits_display_limits(
            [manifest.products[channel] for manifest in manifests]
        )
        for channel in ("dirty", "clean", "residual")
    }


def _iterate_and_validate_dataset(dataset_index: Path, expected_count: int) -> dict[str, Any]:
    from scripts.preprocessing import (
        load_dataset_manifest,
        load_sample_manifest,
        partition_for_sample,
        validate_fits_triplet,
    )

    index = load_dataset_manifest(dataset_index)
    label_counts = {name: 0 for name in LABELS}
    shapes: set[tuple[int, int]] = set()
    for path in index.samples:
        sample = load_sample_manifest(path)
        if partition_for_sample(sample.sample_id) != "test":
            raise RuntimeError(f"Non-test sample entered dataset v1: {sample.sample_id}")
        planes = validate_fits_triplet(sample.products)
        shapes.add(planes["dirty"].shape)
        label_counts[sample.label_name] += 1
    if len(index.samples) != expected_count:
        raise RuntimeError(
            f"Dataset has {len(index.samples)} samples; expected {expected_count}"
        )
    return {
        "mode": "dataset manifest iteration plus retained FITS loading",
        "sample_count": len(index.samples),
        "label_counts": label_counts,
        "image_shapes": [list(shape) for shape in sorted(shapes)],
        "partition": "test",
    }


def _new_report_manifest(
    source_run: Path, source_ids: Sequence[str]
) -> dict[str, Any]:
    from scripts.corruption import detectability_metric_definitions
    from scripts.imaging import imaging_metric_definitions

    return {
        "title": "Single-source constant-gain corruption calibration",
        "description": (
            "One test-partition source from the reused matched-S/N thermal simulation "
            "run. The source has one uncorrupted baseline and constant one-antenna "
            "amplitude and phase variants. rho_corr is an aggregate visibility-space "
            "matched-filter S/N, not an image-domain artifact S/N."
        ),
        "partition": "test",
        "source_ids": list(source_ids),
        "source_run": str(source_run),
        "dataset_index": "dataset.json",
        "configuration": {
            "targets": list(TARGET_DETECTABILITIES),
            "target_metric": "rho_corr",
            "labels": LABELS,
            "corruption_solint": CORRUPTION_SOLINT,
            "base_seed": BASE_SEED,
            "imaging_configuration": "DefaultImagingConfig",
            "keep_intermediate_products": False,
            "fits_invalid_policy": "fill",
            "fits_fill_value": 0.0,
            "simulation_policy": "reuse matched-S/N source+thermal-noise simulations",
            "image_metric_definitions": list(imaging_metric_definitions()),
            "corruption_metric_definitions": list(
                detectability_metric_definitions()
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
    source_id: str | None = None,
) -> Path:
    from scripts.preprocessing import TEST_IDS
    from scripts.reporting import QuartoReporter

    selected_source_id = TEST_IDS[0] if source_id is None else str(source_id).strip()
    if selected_source_id not in TEST_IDS:
        raise ValueError(
            f"source_id must be in the preprocessing test partition: {selected_source_id!r}"
        )
    source_ids = (selected_source_id,)
    thermal_dir, thermal_manifest = find_thermal_run(source_ids, source_run)
    sources = [
        _thermal_source(thermal_dir, thermal_manifest, source_id)
        for source_id in source_ids
    ]
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
            raise RuntimeError("Existing dataset run uses a different source-ID set")
    else:
        report = _new_report_manifest(thermal_dir, source_ids)
        _atomic_write_json(report_json, report)
    report.setdefault("configuration", {}).update(
        fits_invalid_policy="fill",
        fits_fill_value=0.0,
    )
    _atomic_write_json(report_json, report)

    dataset_index = output / "dataset.json"
    completed = _existing_sample_ids(dataset_index)
    completed_sources = {item["id"] for item in report.get("sources", [])}
    reporter = QuartoReporter(report_qmd, every=REPORT_EVERY_SOURCES)
    try:
        for source_number, source in enumerate(sources, start=1):
            print(f"[{source_number}/{len(sources)}] source {source.source_id}")
            antenna_id, antenna_name = _select_antenna(source.simulation_ms)
            requested = [
                ("not_corrupted", LABELS["not_corrupted"]),
                *((variant.suffix, variant.label_id) for variant in _variants()),
            ]
            source_failed = False
            for suffix, _ in requested:
                sample_id = _sample_id(source.source_id, suffix)
                path = _sample_dir(output, sample_id)
                if sample_id in completed or _recover_or_remove_sample(
                    path, dataset_index, completed
                ):
                    completed.add(sample_id)
                    print(f"[{sample_id}] skipping completed sample")
                    continue
                report["failures"] = [
                    item
                    for item in report.get("failures", [])
                    if item.get("sample_id") != sample_id
                ]
                try:
                    if suffix == "not_corrupted":
                        _finalize_baseline(source, output, dataset_index)
                    else:
                        variant = next(item for item in _variants() if item.suffix == suffix)
                        _finalize_variant(
                            source,
                            variant,
                            antenna_id,
                            antenna_name,
                            output,
                            dataset_index,
                        )
                    completed.add(sample_id)
                except Exception as exc:
                    source_failed = True
                    traceback.print_exc()
                    report["failures"].append(
                        {
                            "source_id": source.source_id,
                            "sample_id": sample_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if path.exists() and sample_id not in completed:
                        _safe_remove_partial(path, output / "samples")
                    _atomic_write_json(report_json, report)
                    print(f"[{sample_id}] FAILED: {type(exc).__name__}: {exc}")
            expected_source_samples = {
                _sample_id(source.source_id, suffix) for suffix, _ in requested
            }
            if expected_source_samples <= completed:
                shared_limits = _shared_source_display_limits(
                    source.source_id, dataset_index
                )
                amp_plots = _write_comparison_plots(
                    source.source_id,
                    "amp",
                    dataset_index,
                    output,
                    shared_limits,
                )
                phase_plots = _write_comparison_plots(
                    source.source_id,
                    "phase",
                    dataset_index,
                    output,
                    shared_limits,
                )
                corruption_plots = {
                    family: [
                        {
                            "target_rho_corr": target,
                            "path": str(
                                _corruption_plot_path(
                                    output,
                                    _sample_id(
                                        source.source_id,
                                        f"{family}_rho_{_target_text(target)}",
                                    ),
                                ).relative_to(output)
                            ),
                        }
                        for target in TARGET_DETECTABILITIES
                    ]
                    for family in ("amp", "phase")
                }
                source_entry = {
                    "id": source.source_id,
                    "antenna_id": antenna_id,
                    "antenna_name": antenna_name,
                    "source_snr": source.source_snr,
                    "predicted_image_rms_jy_per_beam": (
                        source.predicted_image_rms_jy_per_beam
                    ),
                    "simulation_metadata": str(source.simulation_metadata),
                    "amplitude_plots": amp_plots["panels"],
                    "phase_plots": phase_plots["panels"],
                    "amplitude_plot_recipe": amp_plots["recipes"],
                    "phase_plot_recipe": phase_plots["recipes"],
                    "corruption_plots": corruption_plots,
                }
                report["sources"] = [
                    item
                    for item in report.get("sources", [])
                    if item.get("id") != source.source_id
                ]
                report["sources"].append(source_entry)
                report["sources"].sort(key=lambda item: source_ids.index(item["id"]))
                completed_sources.add(source.source_id)
                _atomic_write_json(report_json, report)
                reporter.sample_completed()
            elif not source_failed:
                raise RuntimeError(f"Source {source.source_id} finished incompletely")
    finally:
        expected_count = len(source_ids) * (1 + len(_variants()))
        if dataset_index.exists() and not report.get("failures"):
            report["dataset_iteration"] = _iterate_and_validate_dataset(
                dataset_index, expected_count
            )
            _atomic_write_json(report_json, report)
        reporter.finish()

    if report.get("failures"):
        raise RuntimeError(
            f"Dataset v1 has {len(report['failures'])} failed samples; rerun "
            f"with --output-dir {output} after inspecting report.json"
        )
    if len(completed_sources) != len(source_ids):
        raise RuntimeError(
            f"Only {len(completed_sources)}/{len(source_ids)} sources completed"
        )
    print(f"Dataset: {dataset_index}")
    print(f"Report: {output / 'report.html'}")
    print(f"Samples: {len(completed)} across {len(completed_sources)} test sources")
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
        "--source-id",
        help="Exactly one source ID from the preprocessing test partition",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    arguments = _parse_args(argv)
    return run_experiment(
        source_run=arguments.source_run,
        output_dir=arguments.output_dir,
        source_id=arguments.source_id,
    )


if __name__ in {"__main__", "<run_path>"}:
    main()
