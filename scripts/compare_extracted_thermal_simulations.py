#!/usr/bin/env python3
"""Compare default direct images with matched-dynamic-range thermal simulations.

Run from the repository root with the standard CASA installation::

    '/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa' \
        --nogui --nologger \
        -c scripts/compare_extracted_thermal_simulations.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_local_packages_if_loaded() -> None:
    """Refresh local code when CASA execfile reuses its Python process."""
    for package in ("scripts.simulation", "scripts.imaging"):
        if package not in sys.modules:
            continue
        children = sorted(
            (name for name in sys.modules if name == package or name.startswith(package + ".")),
            key=lambda name: name.count("."),
            reverse=True,
        )
        for module_name in children:
            module = sys.modules.get(module_name)
            if module is not None:
                importlib.reload(module)
        print(f"Reloaded {package} for this CASA session")


_reload_local_packages_if_loaded()

from scripts.imaging import BeamRegion, DefaultImagingConfig, image_ms
from scripts.imaging.metadata import DEFAULT_EXTRACTED_MS_ROOT, resolve_ms_band
from scripts.nrao_flux import FluxUnavailable, nrao_flux_decision
from scripts.reporting import QuartoReporter
from scripts.simulation import (
    get_reference_frequency,
    phase_center_point_source,
    natural_image_rms_from_simplenoise,
    phase_center_point_source_from_snr,
    simulate_ms,
    theoretical_vla_simplenoise,
)
from scripts.simulation.noise import VLA_OSS_2026A_SEFD_JY


TITLE = "Extracted VLA data and matched robust-dynamic-range thermal-noise simulations"
DESCRIPTION = (
    "Direct CASA images of each extracted MS and a phase-centre, "
    "constant-spectrum, unpolarized point-source simulation with theoretical VLA "
    "thermal noise on the same sampling and flags. Each simulated source targets "
    "the original image's robust beam-region dynamic range (global clean peak / "
    "selected residual scaled MAD). Residual metrics use a circular annulus from "
    "three synthesized-beam major FWHM to one synthesized-beam major FWHM "
    "inside the nearest image "
    "border. The simulations assume 8-bit sampling and eta_c=0.93."
)
SAMPLE_IDS: list[str] | None = None
REPORT_EVERY = 1
SAMPLER = "8bit"
ETA_C = 0.93
BASE_SEED = 20260907
DISK_SPACE_FACTOR = 4.0
MIN_HEADROOM_GIB = 5.0
EXPERIMENT_DIR: str | Path | None = None
IMAGING_IMSIZE = (256, 256)
METRIC_INNER_RADIUS_BEAMS = 3.0
METRIC_BORDER_MARGIN_BEAMS = 1.0
FITS_INVALID_POLICY = "fill"
SOURCE_SNR_BASIS = (
    "original_global_clean_peak_over_three_beams_to_border_minus_one_beam_"
    "residual_scaled_mad"
)
# Opt-in extension policies (absent = the legacy matched-dynamic-range simulation above).
FLUX_POLICIES = ("nrao_then_clean",)
SIMULATION_FLAG_POLICIES = ("unflag",)
EXTENSION_SNR_BASIS = "decided_flux_over_predicted_natural_rms_of_simulation_samples"
_REUSED_QA_SETTINGS = ("datacolumn", "specmode", "stokes", "gridder", "deconvolver", "nterms",
                       "weighting", "robust", "gain", "mask_nbeams", "uvrange", "savemodel")
_SEED_MAX = 2_147_483_646
_MATCHED_IMAGING_KEYS = (
    "imsize",
    "mask_nbeams",
    "gridder",
    "weighting",
    "robust",
    "stokes",
    "deconvolver",
    "nterms",
)


class SampleProcessingError(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


def find_samples(
    root: str | Path = DEFAULT_EXTRACTED_MS_ROOT,
    sample_ids: list[str] | None = None,
) -> list[Path]:
    """Return only canonical ``<id>/<id>/<id>.ms`` inputs."""
    extracted = Path(root).expanduser().resolve()
    wanted = None if sample_ids is None else set(sample_ids)
    samples: list[Path] = []
    for sample_dir in sorted(extracted.iterdir()):
        if not sample_dir.is_dir() or (wanted is not None and sample_dir.name not in wanted):
            continue
        candidate = sample_dir / sample_dir.name / f"{sample_dir.name}.ms"
        if candidate.is_dir():
            samples.append(candidate.resolve())
    if wanted is not None:
        found = {path.stem for path in samples}
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(f"Canonical extracted samples not found: {missing}")
    return samples


def stable_sample_seed(sample_id: str, base_seed: int = BASE_SEED) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MAX + 1


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _allocated_bytes(path: Path) -> int:
    total = 0
    pending = [path]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    total += int(getattr(stat, "st_blocks", 0)) * 512 or int(stat.st_size)
    return total


def _required_free_bytes(samples: list[Path]) -> tuple[int, int]:
    input_bytes = sum(_allocated_bytes(path) for path in samples)
    required = int(DISK_SPACE_FACTOR * input_bytes + MIN_HEADROOM_GIB * 1024**3)
    return input_bytes, required


def _check_disk_space(samples: list[Path], destination_parent: Path) -> None:
    input_bytes, required = _required_free_bytes(samples)
    free = shutil.disk_usage(destination_parent).free
    print(
        "Disk preflight: "
        f"inputs={input_bytes / 1024**3:.2f} GiB, "
        f"required={required / 1024**3:.2f} GiB, free={free / 1024**3:.2f} GiB"
    )
    if free < required:
        raise RuntimeError(
            "Insufficient free space for the selected samples: "
            f"need {required / 1024**3:.2f} GiB, found {free / 1024**3:.2f} GiB"
        )


def _assert_paths_exist(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Expected products are missing:\n" + "\n".join(map(str, missing)))


def _assert_matched_imaging_settings(original: Any, simulated: Any) -> None:
    differences = {
        key: (
            original.effective_imaging_parameters.get(key),
            simulated.effective_imaging_parameters.get(key),
        )
        for key in _MATCHED_IMAGING_KEYS
        if original.effective_imaging_parameters.get(key)
        != simulated.effective_imaging_parameters.get(key)
    }
    if differences:
        raise RuntimeError(f"Original and simulation imaging settings differ: {differences}")


def metric_region_for_resolved_grid(resolved: Any) -> BeamRegion:
    """Resolve the 3-beam to border-minus-1-beam annulus for one image grid."""
    grid = resolved.grid.image_grid
    beam_major = float(resolved.grid.beam.major_arcsec)
    nx, ny = grid.imsize
    cell_x, cell_y = grid.cell_arcsec
    nearest_x = min(nx // 2, nx - 1 - nx // 2) * float(cell_x)
    nearest_y = min(ny // 2, ny - 1 - ny // 2) * float(cell_y)
    nearest_border_radius_beams = min(nearest_x, nearest_y) / beam_major
    maximum = nearest_border_radius_beams - METRIC_BORDER_MARGIN_BEAMS
    if maximum <= METRIC_INNER_RADIUS_BEAMS:
        raise ValueError(
            "Image is too narrow for the requested metric annulus: "
            f"inner radius={METRIC_INNER_RADIUS_BEAMS:.6g} beams, "
            f"nearest border={nearest_border_radius_beams:.6g} beams, "
            f"outer radius after border inset={maximum:.6g} beams, "
            f"border margin={METRIC_BORDER_MARGIN_BEAMS:.6g} beams"
        )
    return BeamRegion(
        min_radius_beams=METRIC_INNER_RADIUS_BEAMS,
        max_radius_beams=maximum,
    )


def metric_region_policy() -> dict[str, Any]:
    return {
        "shape": "circular annulus",
        "radius_unit": "synthesized beam major FWHM",
        "min_radius_beams": METRIC_INNER_RADIUS_BEAMS,
        "outer_border_margin_beams": METRIC_BORDER_MARGIN_BEAMS,
    }


def _verify_completed_entry(entry: dict[str, Any], experiment_dir: Path) -> None:
    if "flux_decision" in entry:
        _verify_extension_entry(entry, experiment_dir)
        return
    paths = [
        Path(entry["original_ms"]),
        experiment_dir / entry["simulation_ms"],
        experiment_dir / entry["simulation_component_list"],
        experiment_dir / entry["simulation_metadata"],
    ]
    for key in ("original_result_dir", "simulation_result_dir"):
        result_dir = experiment_dir / entry[key]
        paths.extend(
            result_dir / filename
            for filename in ("qa.json", "qa.txt", "dirty.png", "clean.png", "residual.png")
        )
    _assert_paths_exist(paths)
    qa_reports = [
        json.loads(
            (experiment_dir / entry[key] / "qa.json").read_text(encoding="utf-8")
        )
        for key in ("original_result_dir", "simulation_result_dir")
    ]
    if any(int(qa.get("schema_version", 1)) not in {2, 3} for qa in qa_reports):
        raise RuntimeError(
            f"Completed sample {entry.get('id')} does not use a supported QA schema"
        )
    regions = [(qa.get("metrics", {}) or {}).get("region") for qa in qa_reports]
    if regions[0] != regions[1] or regions[0] != entry.get("metric_region"):
        raise RuntimeError(
            f"Completed sample {entry.get('id')} has inconsistent metric regions: "
            f"{regions!r} vs {entry.get('metric_region')!r}"
        )


def _verify_extension_entry(entry: dict[str, Any], experiment_dir: Path) -> None:
    """Extension entries: original-image QA is required only on the CLEAN-fallback route."""
    result_keys = ["simulation_result_dir"] + (["original_result_dir"] if "original_result_dir" in entry else [])
    paths = [
        Path(entry["original_ms"]),
        experiment_dir / entry["simulation_ms"],
        experiment_dir / entry["simulation_component_list"],
        experiment_dir / entry["simulation_metadata"],
    ]
    for key in result_keys:
        paths.extend(
            experiment_dir / entry[key] / filename
            for filename in ("qa.json", "qa.txt", "dirty.png", "clean.png", "residual.png")
        )
    decision = entry["flux_decision"]
    if decision.get("method") == "original_clean":
        paths.append(_resolve_in(experiment_dir, decision["provenance"]["qa_json"]))
    elif decision.get("method") != "nrao_pipeline":
        raise RuntimeError(f"Completed sample {entry.get('id')} has unknown flux method {decision.get('method')!r}")
    _assert_paths_exist(paths)
    flux = decision.get("flux_jy")
    if not isinstance(flux, (int, float)) or not math.isfinite(flux) or flux <= 0:
        raise RuntimeError(f"Completed sample {entry.get('id')} has an invalid decided flux {flux!r}")
    simulation = json.loads((experiment_dir / entry["simulation_metadata"]).read_text(encoding="utf-8"))
    if [c["flux"][0] for c in simulation.get("components", [])] != [flux]:
        raise RuntimeError(f"Completed sample {entry.get('id')} simulated a different flux than decided")
    if ("simulation_flags" in simulation) != (entry.get("simulation_flags") == "unflag"):
        raise RuntimeError(f"Completed sample {entry.get('id')} simulation flag policy differs from its entry")
    qa = json.loads((experiment_dir / entry["simulation_result_dir"] / "qa.json").read_text(encoding="utf-8"))
    if int(qa.get("schema_version", 1)) not in {2, 3}:
        raise RuntimeError(f"Completed sample {entry.get('id')} does not use a supported QA schema")
    if (qa.get("metrics", {}) or {}).get("region") != entry.get("metric_region"):
        raise RuntimeError(f"Completed sample {entry.get('id')} has inconsistent metric regions")


def _resolve_in(experiment_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else experiment_dir / path


def _relative_or_absolute(path: Path, experiment_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(experiment_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _frequency_hz(ms_path: Path) -> float:
    """Frequency of the simulated constant-spectrum component (SPW 0 REF_FREQUENCY)."""
    text = get_reference_frequency(ms_path, spw_id=0)
    value = float(text.split()[-1].removesuffix("Hz"))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid simulation frequency {text!r}")
    return value


def _clean_flux(qa: dict[str, Any]) -> float:
    flux = float((qa.get("tclean_summary") or {}).get("final_model_flux_jy", float("nan")))
    if not math.isfinite(flux) or flux <= 0:
        raise FluxUnavailable(f"original CLEAN model flux is {flux!r}")
    return flux


def reusable_original_qa(sample_id: str, ms_path: Path, reuse_dirs: Sequence[Path]) -> tuple[Path | None, list[str]]:
    """First earlier original-image QA for this exact input and imaging settings (read-only)."""
    from dataclasses import asdict

    expected = json.loads(json.dumps(asdict(DefaultImagingConfig), default=str))
    reasons = []
    for root in reuse_dirs:
        path = Path(root) / sample_id / "original" / "default_imaging" / "qa.json"
        if not path.is_file():
            continue
        try:
            qa = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            reasons.append(f"{path}: unreadable ({exc})")
            continue
        params = qa.get("effective_imaging_parameters") or {}
        problems = []
        if qa.get("engine") != "direct" or int(qa.get("schema_version", 1)) not in {2, 3}:
            problems.append("engine/schema")
        if qa.get("visibility_id") != sample_id or Path(str(qa.get("ms_path"))).resolve() != ms_path.resolve():
            problems.append("different input MS")
        if params.get("imsize") != list(IMAGING_IMSIZE):
            problems.append("imsize")
        problems += [key for key in _REUSED_QA_SETTINGS if params.get(key) != expected.get(key)]
        if params.get("requested_clean") != expected.get("clean"):
            problems.append("clean settings")
        if problems:
            reasons.append(f"{path}: not reusable ({', '.join(problems)})")
            continue
        return path, reasons
    return None, reasons


def decide_flux(
    ms_path: Path, experiment_dir: Path, frequency_hz: float, reuse_dirs: Sequence[Path]
) -> tuple[dict[str, Any], Any]:
    """NRAO pipeline flux first; else original-observation CLEAN flux (reused QA or new imaging)."""
    sample_id = ms_path.stem
    try:
        return nrao_flux_decision(ms_path, frequency_hz), None
    except FluxUnavailable as exc:
        nrao_reason = str(exc)
    original = None
    qa_path, reuse_notes = reusable_original_qa(sample_id, ms_path, reuse_dirs)
    reused = qa_path is not None
    try:
        if not reused:
            original = image_ms(
                ms_path,
                DefaultImagingConfig,
                experiment_dir / sample_id / "original" / "default_imaging",
                imsize=IMAGING_IMSIZE,
                metric_region_resolver=metric_region_for_resolved_grid,
                keep_intermediate_products=True,
                fits_invalid_policy=FITS_INVALID_POLICY,
            )
            qa_path = Path(original.qa_json)
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        flux = _clean_flux(qa)
    except Exception as exc:  # imaging of fully flagged/unusable originals fails here
        notes = "; ".join(reuse_notes)
        raise FluxUnavailable(
            f"no NRAO flux ({nrao_reason}); no original CLEAN flux ({type(exc).__name__}: {exc})"
            + (f"; reuse: {notes}" if notes else "")
        ) from exc
    return dict(
        method="original_clean", flux_jy=flux, frequency_hz=float(frequency_hz),
        nrao_unavailable=nrao_reason,
        provenance=dict(qa_json=_relative_or_absolute(qa_path, experiment_dir),
                        qa_sha256=hashlib.sha256(qa_path.read_bytes()).hexdigest(),
                        reused=reused, flux_field="tclean_summary.final_model_flux_jy"),
    ), original


def _remove_interrupted_sample_output(sample_dir: Path, experiment_dir: Path) -> None:
    """Remove a narrowly scoped, uncommitted per-sample work tree before retrying."""
    if sample_dir.is_symlink():
        raise RuntimeError(f"Interrupted sample output must not be a symlink: {sample_dir}")
    sample_dir = sample_dir.resolve()
    experiment_dir = experiment_dir.resolve()
    if sample_dir.parent != experiment_dir:
        raise RuntimeError(
            f"Refusing to remove interrupted output outside {experiment_dir}: {sample_dir}"
        )
    if not sample_dir.exists():
        return
    if not sample_dir.is_dir():
        raise RuntimeError(f"Interrupted sample output is not a directory: {sample_dir}")
    unexpected = sorted(
        child.name
        for child in sample_dir.iterdir()
        if child.name not in {"original", "simulation"}
    )
    if unexpected:
        raise RuntimeError(
            f"Refusing to remove {sample_dir}; unexpected entries: {unexpected}"
        )
    print(f"Removing interrupted, uncommitted sample output before retry: {sample_dir}")
    shutil.rmtree(sample_dir)


def process_sample(
    ms_path: Path,
    experiment_dir: Path,
    policy: dict[str, str] | None = None,
    reuse_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    if policy:
        return process_extension_sample(ms_path, experiment_dir, policy, reuse_dirs)
    sample_id = ms_path.stem
    sample_dir = experiment_dir / sample_id
    stage = "original_imaging"
    try:
        original = image_ms(
            ms_path,
            DefaultImagingConfig,
            sample_dir / "original" / "default_imaging",
            imsize=IMAGING_IMSIZE,
            metric_region_resolver=metric_region_for_resolved_grid,
            keep_intermediate_products=True,
            fits_invalid_policy=FITS_INVALID_POLICY,
        )
        metric_region = original.qa.metrics.region

        stage = "noise_calculation"
        if original.resolved_config is None:
            raise RuntimeError("Default direct imaging did not return a resolved configuration")
        band = str(original.resolved_config.ms_band_meta.selected_band)
        sefd_jy = VLA_OSS_2026A_SEFD_JY[band]
        simplenoise_jy = theoretical_vla_simplenoise(
            ms_path,
            sefd_jy=sefd_jy,
            eta_c=ETA_C,
        )
        predicted_rms = natural_image_rms_from_simplenoise(ms_path, simplenoise_jy)
        target_dynamic_range = float(original.qa.metrics.dynamic_range_scaled_mad)
        if not math.isfinite(target_dynamic_range) or target_dynamic_range <= 0:
            raise RuntimeError(
                "Original image dynamic range must be finite and positive, found "
                f"{target_dynamic_range!r}"
            )
        component = phase_center_point_source_from_snr(
            ms_path,
            target_dynamic_range,
            predicted_rms,
        )

        stage = "simulation"
        sample_seed = stable_sample_seed(sample_id)
        simulation = simulate_ms(
            ms_path,
            [component],
            sample_dir / "simulation" / f"{sample_id}_matched_snr.ms",
            noise_model="vla-thermal",
            noise_parameters={"band": band, "sampler": SAMPLER},
            seed=sample_seed,
        )
        if simulation.simplenoise_jy is None or not math.isclose(
            simulation.simplenoise_jy, simplenoise_jy, rel_tol=1e-12, abs_tol=0.0
        ):
            raise RuntimeError(
                "Simulation simplenoise does not match the source-flux calculation: "
                f"{simulation.simplenoise_jy!r} != {simplenoise_jy!r}"
            )

        stage = "simulation_imaging"
        simulated = image_ms(
            simulation.ms_path,
            DefaultImagingConfig,
            sample_dir / "simulation" / "default_imaging",
            imsize=IMAGING_IMSIZE,
            metric_region=metric_region,
            keep_intermediate_products=True,
            fits_invalid_policy=FITS_INVALID_POLICY,
        )
        _assert_matched_imaging_settings(original, simulated)
        if original.qa.metrics.region != simulated.qa.metrics.region:
            raise RuntimeError(
                "Original and simulation metric regions differ: "
                f"{original.qa.metrics.region!r} != {simulated.qa.metrics.region!r}"
            )
        required = [
            simulation.ms_path,
            simulation.metadata_json,
            simulation.component_list,
        ]
        for result in (original, simulated):
            required.extend(
                [
                    result.dirty_image,
                    result.clean_image,
                    result.residual_image,
                    result.dirty_png,
                    result.clean_png,
                    result.residual_png,
                    result.qa_text,
                    result.qa_json,
                ]
            )
        _assert_paths_exist([path for path in required if path is not None])

        entry = {
            "id": sample_id,
            "original_ms": str(ms_path.resolve()),
            "original_result_dir": str(original.output_dir.relative_to(experiment_dir)),
            "simulation_ms": str(simulation.ms_path.relative_to(experiment_dir)),
            "simulation_component_list": str(
                simulation.component_list.relative_to(experiment_dir)
            ),
            "simulation_metadata": str(simulation.metadata_json.relative_to(experiment_dir)),
            "simulation_result_dir": str(simulated.output_dir.relative_to(experiment_dir)),
            "source_snr": target_dynamic_range,
            "source_snr_basis": SOURCE_SNR_BASIS,
            "metric_region": {
                "min_radius_beams": metric_region.min_radius_beams,
                "max_radius_beams": metric_region.max_radius_beams,
            },
            "target_dynamic_range": target_dynamic_range,
            "predicted_image_rms_jy_per_beam": predicted_rms,
        }
        _verify_completed_entry(entry, experiment_dir)
        return entry
    except Exception as exc:
        raise SampleProcessingError(stage, exc) from exc


def process_extension_sample(
    ms_path: Path, experiment_dir: Path, policy: dict[str, str], reuse_dirs: Sequence[Path] = ()
) -> dict[str, Any]:
    """Thermal simulation at a decided physical flux (no original dynamic range needed)."""
    sample_id = ms_path.stem
    sample_dir = experiment_dir / sample_id
    stage = "flux_decision"
    try:
        frequency_hz = _frequency_hz(ms_path)
        decision, original = decide_flux(ms_path, experiment_dir, frequency_hz, reuse_dirs)

        stage = "noise_calculation"
        band = str(resolve_ms_band(ms_path, None).selected_band)
        simplenoise_jy = theoretical_vla_simplenoise(
            ms_path, sefd_jy=VLA_OSS_2026A_SEFD_JY[band], eta_c=ETA_C
        )
        component = phase_center_point_source(ms_path, decision["flux_jy"])

        stage = "simulation"
        simulation = simulate_ms(
            ms_path,
            [component],
            sample_dir / "simulation" / f"{sample_id}_decided_flux.ms",
            noise_model="vla-thermal",
            noise_parameters={"band": band, "sampler": SAMPLER},
            seed=stable_sample_seed(sample_id),
            simulation_flags=policy.get("simulation_flags"),
        )
        if simulation.simplenoise_jy is None or not math.isclose(
            simulation.simplenoise_jy, simplenoise_jy, rel_tol=1e-12, abs_tol=0.0
        ):
            raise RuntimeError(
                f"Simulation simplenoise {simulation.simplenoise_jy!r} != {simplenoise_jy!r}"
            )
        # RMS from the simulation copy's own sample mask (all samples after unflagging).
        predicted_rms = natural_image_rms_from_simplenoise(simulation.ms_path, simplenoise_jy)

        stage = "simulation_imaging"
        simulated = image_ms(
            simulation.ms_path,
            DefaultImagingConfig,
            sample_dir / "simulation" / "default_imaging",
            imsize=IMAGING_IMSIZE,
            metric_region_resolver=metric_region_for_resolved_grid,
            keep_intermediate_products=True,
            fits_invalid_policy=FITS_INVALID_POLICY,
        )
        if original is not None:
            _assert_matched_imaging_settings(original, simulated)
        required = [simulation.ms_path, simulation.metadata_json, simulation.component_list]
        for result in (original, simulated):
            if result is None:
                continue
            required.extend([
                result.dirty_image, result.clean_image, result.residual_image,
                result.dirty_png, result.clean_png, result.residual_png,
                result.qa_text, result.qa_json,
            ])
        _assert_paths_exist([path for path in required if path is not None])
        region = simulated.qa.metrics.region
        entry = {
            "id": sample_id,
            "original_ms": str(ms_path.resolve()),
            **({"original_result_dir": str(original.output_dir.relative_to(experiment_dir))}
               if original is not None else {}),
            "simulation_ms": str(simulation.ms_path.relative_to(experiment_dir)),
            "simulation_component_list": str(simulation.component_list.relative_to(experiment_dir)),
            "simulation_metadata": str(simulation.metadata_json.relative_to(experiment_dir)),
            "simulation_result_dir": str(simulated.output_dir.relative_to(experiment_dir)),
            "source_snr": decision["flux_jy"] / predicted_rms,
            "source_snr_basis": EXTENSION_SNR_BASIS,
            "metric_region": {
                "min_radius_beams": region.min_radius_beams,
                "max_radius_beams": region.max_radius_beams,
            },
            "predicted_image_rms_jy_per_beam": predicted_rms,
            **policy,
            "flux_decision": decision,
        }
        _verify_completed_entry(entry, experiment_dir)
        return entry
    except Exception as exc:
        raise SampleProcessingError(stage, exc) from exc


def extension_policy(flux_policy: str | None, simulation_flags: str | None) -> dict[str, str]:
    """Validated opt-in policy ({} = legacy behaviour)."""
    if flux_policy is None and simulation_flags is None:
        return {}
    if flux_policy not in FLUX_POLICIES:
        raise ValueError(f"flux_policy must be one of {FLUX_POLICIES} (required with simulation_flags)")
    if simulation_flags is not None and simulation_flags not in SIMULATION_FLAG_POLICIES:
        raise ValueError(f"simulation_flags must be one of {SIMULATION_FLAG_POLICIES}")
    policy = {"flux_policy": flux_policy}
    if simulation_flags is not None:
        policy["simulation_flags"] = simulation_flags
    return policy


def _new_manifest(total_samples: int, policy: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "total_samples": total_samples,
        "configuration": {
            "imaging_engine": "direct",
            "imaging_configuration": "DefaultImagingConfig",
            "imsize": list(IMAGING_IMSIZE),
            "source_snr_basis": SOURCE_SNR_BASIS,
            "metric_region_policy": metric_region_policy(),
            "fits_invalid_policy": FITS_INVALID_POLICY,
            "sampler": SAMPLER,
            "eta_c": ETA_C,
            "base_seed": BASE_SEED,
            "sefd_table": "VLA OSS 2026A fiducial values",
            **(policy or {}),
            **({"source_snr_basis": EXTENSION_SNR_BASIS} if policy else {}),
        },
        "samples": [],
        "failures": [],
    }


def _pending_samples(
    samples: Sequence[Path],
    completed: dict[str, dict[str, Any]],
    failed: dict[str, dict[str, Any]],
    output: Path,
) -> list[Path]:
    pending = []
    for ms_path in samples:
        sample_id = ms_path.stem
        if sample_id in completed:
            continue
        partial = output / sample_id
        if sample_id in failed and partial.exists() and any(partial.iterdir()):
            continue
        pending.append(ms_path)
    return pending


def main(
    experiment_dir: str | Path | None = None,
    *,
    sample_ids: Sequence[str] | None = None,
    flux_policy: str | None = None,
    simulation_flags: str | None = None,
    reuse_qa_from: Sequence[str | Path] = (),
    extracted_root: str | Path | None = None,
) -> Path:
    policy = extension_policy(flux_policy, simulation_flags)
    if reuse_qa_from and not policy:
        raise ValueError("reuse_qa_from applies only with flux_policy")
    reuse_dirs = [Path(d).expanduser().resolve() for d in reuse_qa_from]
    selected_ids = SAMPLE_IDS if sample_ids is None else list(sample_ids)
    samples = find_samples(extracted_root or DEFAULT_EXTRACTED_MS_ROOT, sample_ids=selected_ids)
    if not samples:
        raise RuntimeError("No canonical extracted Measurement Sets were selected")

    requested_dir = experiment_dir if experiment_dir is not None else EXPERIMENT_DIR
    if requested_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        requested_dir = (
            ROOT
            / "collect"
            / "experiments"
            / f"extracted_thermal_simulation_comparison_{timestamp}"
        )
    output = Path(requested_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "report.qmd"
    if not report_path.exists():
        shutil.copyfile(
            ROOT / "scripts" / "reporting" / "vla_original_simulation_comparison.qmd",
            report_path,
        )
    manifest_path = output / "report.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_policy = (manifest.get("configuration", {}) or {}).get(
            "metric_region_policy"
        )
        if actual_policy != metric_region_policy():
            raise RuntimeError(
                "Existing experiment uses a different or legacy metric-region "
                "policy; choose a new experiment directory instead of mixing results"
            )
        actual_imsize = (manifest.get("configuration", {}) or {}).get("imsize")
        if actual_imsize != list(IMAGING_IMSIZE):
            raise RuntimeError(
                "Existing experiment uses a different or legacy image size; "
                "choose a new experiment directory instead of mixing results"
            )
        configuration = manifest.get("configuration", {}) or {}
        actual_policy = {k: configuration[k] for k in ("flux_policy", "simulation_flags") if k in configuration}
        if actual_policy != policy:
            raise RuntimeError(
                f"Existing experiment uses flux/flag policy {actual_policy or 'legacy'}, not "
                f"{policy or 'legacy'}; choose a new experiment directory instead of mixing results"
            )
        manifest["total_samples"] = len(samples)
        manifest.setdefault("samples", [])
        manifest.setdefault("failures", [])
    else:
        manifest = _new_manifest(len(samples), policy)

    completed = {entry["id"]: entry for entry in manifest["samples"]}
    for entry in completed.values():
        _verify_completed_entry(entry, output)
    failed = {entry["id"]: entry for entry in manifest["failures"]}
    pending = _pending_samples(samples, completed, failed, output)
    if pending:
        _check_disk_space(pending, output)
    write_manifest(manifest_path, manifest)
    reporter = QuartoReporter(report_path, every=REPORT_EVERY)

    try:
        for index, ms_path in enumerate(samples, start=1):
            sample_id = ms_path.stem
            if sample_id in completed:
                print(f"[{index}/{len(samples)}] Skipping verified completed pair: {sample_id}")
                continue
            if sample_id in failed:
                partial = output / sample_id
                if partial.exists() and any(partial.iterdir()):
                    print(f"[{index}/{len(samples)}] Skipping failed pair with partial output: {sample_id}")
                    continue
                manifest["failures"] = [
                    item for item in manifest["failures"] if item["id"] != sample_id
                ]
                failed.pop(sample_id)

            partial = output / sample_id
            if partial.exists():
                _remove_interrupted_sample_output(partial, output)

            _, one_sample_required = _required_free_bytes([ms_path])
            free = shutil.disk_usage(output).free
            if free < one_sample_required:
                raise RuntimeError(
                    f"Stopping before {sample_id}: {free / 1024**3:.2f} GiB free, "
                    f"but {one_sample_required / 1024**3:.2f} GiB is required"
                )

            print(f"[{index}/{len(samples)}] Processing {sample_id}")
            completed_this_sample = False
            try:
                entry = process_sample(ms_path, output, policy, reuse_dirs)
            except SampleProcessingError as exc:
                print(
                    f"[{index}/{len(samples)}] Failed during {exc.stage}: "
                    f"{sample_id}: {exc}"
                )
                traceback.print_exc()
                failure = {
                    "id": sample_id,
                    "ms_path": str(ms_path),
                    "stage": exc.stage,
                    "error": str(exc),
                }
                manifest["failures"].append(failure)
                failed[sample_id] = failure
            else:
                manifest["samples"].append(entry)
                completed[sample_id] = entry
                completed_this_sample = True
            write_manifest(manifest_path, manifest)
            if completed_this_sample:
                reporter.sample_completed()
    finally:
        reporter.finish()

    report_html = report_path.with_suffix(".html")
    print(f"Experiment directory: {output}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_html}")
    print(
        f"Completed pairs: {len(manifest['samples'])}; "
        f"failed samples: {len(manifest['failures'])}"
    )
    return output


if __name__ in {"__main__", "<run_path>"}:
    main()
