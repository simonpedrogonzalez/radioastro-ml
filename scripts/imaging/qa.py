"""QA metric calculation and report serialization."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .metrics import IMAGING_METRIC_DEFINITIONS
from .models import QAReport, TcleanRunSummary


QA_SCHEMA_VERSION = 3


def json_safe(value: Any) -> Any:
    """Convert CASA, NumPy, dataclass, and path values to strict JSON values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _case_get(record: Mapping[str, Any], *names: str) -> Any:
    wanted = {name.casefold() for name in names}
    for key, value in record.items():
        if str(key).casefold() in wanted:
            return value
    return None


def _numbers(value: Any) -> List[float]:
    converted = json_safe(value)
    result: List[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            number = float(item)
            if math.isfinite(number):
                result.append(number)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(converted)
    return result


def _last_recursive_value(value: Any, names: Iterable[str]) -> Optional[float]:
    wanted = {name.casefold() for name in names}
    found: List[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in wanted:
                    found.extend(_numbers(child))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return found[-1] if found else None


def _scalar_int(value: Any) -> Optional[int]:
    values = _numbers(value)
    return int(values[-1]) if values else None


def _scalar_text(value: Any) -> Optional[str]:
    converted = json_safe(value)
    if converted is None:
        return None
    if isinstance(converted, list):
        return str(converted[-1]) if converted else None
    return str(converted)


def normalize_tclean_summary(
    record: Any,
    *,
    source: str = "return_record",
) -> Tuple[TcleanRunSummary, Tuple[str, ...]]:
    """Normalize version-varying tclean summaries without losing raw fields."""
    warnings: List[str] = []
    if isinstance(record, Mapping):
        mapping = dict(record)
    else:
        mapping = {}
        warnings.append(
            f"The final tclean summary was {type(record).__name__}, not a mapping; "
            "execution-summary fields are unavailable."
        )
    raw = json_safe(mapping)
    if not isinstance(raw, dict):
        raw = {"value": raw}
    summary_major = _case_get(mapping, "summarymajor", "summary_major")
    major_counts = tuple(int(number) for number in _numbers(summary_major))
    summary_minor = _case_get(mapping, "summaryminor", "summary_minor")
    minor_json = json_safe(summary_minor)
    if minor_json is not None and not isinstance(minor_json, dict):
        minor_json = {"value": minor_json}
    peak_residual = _last_recursive_value(summary_minor, ("peakRes", "peak_residual"))
    if peak_residual is None:
        peak_residual = _last_recursive_value(mapping, ("peakres",))
    model_flux = _last_recursive_value(summary_minor, ("modelFlux", "model_flux"))
    if model_flux is None:
        model_flux = _last_recursive_value(mapping, ("modflux", "modelFlux"))
    summary = TcleanRunSummary(
        source=source,  # type: ignore[arg-type]
        iterdone=_scalar_int(_case_get(mapping, "iterdone", "iterDone")),
        nmajordone=_scalar_int(_case_get(mapping, "nmajordone", "nMajorDone")),
        stopcode=_scalar_int(_case_get(mapping, "stopcode", "stopCode")),
        stop_description=_scalar_text(
            _case_get(mapping, "stopDescription", "stop_description", "stopdescription")
        ),
        major_cycle_iteration_counts=major_counts,
        final_peak_residual_jy_per_beam=peak_residual,
        final_model_flux_jy=model_flux,
        final_cycle_threshold_jy_per_beam=_last_recursive_value(
            summary_minor, ("cycleThresh", "cycle_threshold")
        ),
        minor_cycle_history=minor_json,
        raw=raw,
    )
    if mapping and summary.iterdone is None and summary.stopcode is None:
        warnings.append(
            "The final tclean return mapping did not expose recognizable iteration or stop fields; "
            f"raw keys were retained: {', '.join(str(key) for key in mapping)}."
        )
    return summary, tuple(warnings)


def write_qa_reports(report: QAReport, text_path: Path, json_path: Path) -> None:
    payload = json_safe(report)
    for field in ("products", "temporary_products"):
        for name, value in payload.get(field, {}).items():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.is_absolute():
                try:
                    payload[field][name] = path.relative_to(json_path.parent).as_posix()
                except ValueError:
                    pass
    from scripts.preprocessing.schema import atomic_write_json

    atomic_write_json(json_path, payload)
    destination = text_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_qa_text(report)
    import os
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False
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


def _display(value: Any, precision: int = 6) -> str:
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{precision}g}" if math.isfinite(value) else "invalid"
    if isinstance(value, (list, tuple)):
        return " x ".join(_display(item, precision) for item in value)
    return str(value)


def _parameter_lines(report: QAReport) -> List[str]:
    effective = report.effective_imaging_parameters
    resolved = report.resolved_config
    lines: List[str] = []
    if resolved is not None:
        validation = resolved.data_column_validation
        csv_meta = resolved.csv_meta
        band = resolved.ms_band_meta
        grid = resolved.grid.image_grid
        beam = resolved.grid.beam
        controls = resolved.clean
        lines.extend(
            [
                "data: "
                f"datacolumn={resolved.datacolumn} ({validation.table_column}) | "
                f"rows checked={validation.rows_examined} | "
                f"finite, unflagged, nonzero samples={validation.nonzero_finite_unflagged_samples}",
                "metadata: "
                f"array={_display(None if csv_meta is None else csv_meta.array_configuration)} | "
                f"band={band.selected_band} | frequency={_display(band.representative_frequency_ghz)} GHz | "
                f"reference SPW={band.frequency_reference_spw}",
                "grid: "
                f"imsize={_display(grid.imsize)} px | cell={_display(grid.cell_arcsec)} arcsec | "
                f"FoV={_display(grid.field_of_view_arcsec)} arcsec",
                "beam: "
                f"major={_display(beam.major_arcsec)} arcsec | minor={_display(beam.minor_arcsec)} arcsec | "
                f"PA={_display(beam.position_angle_deg)} deg",
            ]
        )
    else:
        validation = effective.get("data_column_validation", {}) or {}
        metadata = effective.get("ms_band_meta", {}) or {}
        csv_meta = effective.get("csv_meta", {}) or {}
        geometry = effective.get("measured_geometry", {}) or {}
        lines.extend(
            [
                "data: "
                f"source datacolumn={_display(effective.get('pipeline_source_datacolumn'))} "
                f"({_display(validation.get('table_column'))}) | "
                f"rows checked={_display(validation.get('rows_examined'))} | "
                "finite, unflagged, nonzero samples="
                f"{_display(validation.get('nonzero_finite_unflagged_samples'))}",
                "metadata: "
                f"array={_display(csv_meta.get('array_configuration'))} | "
                f"band={_display(metadata.get('selected_band'))} | "
                f"frequency={_display(metadata.get('representative_frequency_ghz'))} GHz | "
                f"reference SPW={_display(metadata.get('frequency_reference_spw'))}",
                "selection: "
                f"field={_display(effective.get('field'))} | SPW={_display(effective.get('spw'))} | "
                f"uvrange={_display(effective.get('uvrange'))}",
                "grid: "
                f"imsize={_display(geometry.get('imsize', effective.get('imsize')))} px | "
                f"cell={_display(geometry.get('cell_arcsec', effective.get('cell')))} arcsec | "
                f"FoV={_display(geometry.get('field_of_view_arcsec'))} arcsec",
                "beam: "
                f"major={_display((geometry.get('beam') or {}).get('major_arcsec'))} arcsec | "
                f"minor={_display((geometry.get('beam') or {}).get('minor_arcsec'))} arcsec | "
                f"PA={_display((geometry.get('beam') or {}).get('position_angle_deg'))} deg",
            ]
        )

    lines.extend(
        [
            "imaging: "
            f"specmode={_display(effective.get('specmode'))} | gridder={_display(effective.get('gridder'))} | "
            f"stokes={_display(effective.get('stokes'))}",
            "deconvolution: "
            f"deconvolver={_display(effective.get('deconvolver'))} | "
            f"nterms={_display(effective.get('nterms', 1))} | gain={_display(effective.get('gain'))}",
            "weighting: "
            f"weighting={_display(effective.get('weighting'))} | robust={_display(effective.get('robust'))}",
        ]
    )
    if effective.get("mask_nbeams") is not None:
        mask_description = (
            "central circle | "
            f"diameter={_display(effective.get('mask_nbeams'))} synthesized beams"
        )
        lines.append(f"mask: {mask_description}")
    elif resolved is not None:
        lines.append("mask: none")
    else:
        lines.append(
            "mask: "
            f"mode={_display(effective.get('usemask'))} | sidelobe threshold={_display(effective.get('sidelobethreshold'))} | "
            f"noise thresholds={_display(effective.get('noisethreshold'))}/{_display(effective.get('lownoisethreshold'))} | "
            f"minimum beam fraction={_display(effective.get('minbeamfrac'))}"
        )

    if resolved is not None:
        if controls is not None:
            lines.append(
                "CLEAN request: "
                f"niter ceiling={controls.niter} | threshold={_display(controls.threshold)} | "
                f"nsigma={_display(controls.nsigma)} | cycleniter={_display(controls.cycleniter)}"
            )
    else:
        lines.append(
            "CLEAN request: "
            f"niter ceiling={_display(effective.get('niter'))} | threshold={_display(effective.get('threshold'))} | "
            f"nsigma={_display(effective.get('nsigma'))} | nmajor ceiling={_display(effective.get('nmajor'))}"
        )
    summary = report.tclean_summary
    if summary is not None:
        lines.append(
            "CLEAN result: "
            f"iterations={_display(summary.iterdone)} | major cycles={_display(summary.nmajordone)} | "
            f"stop code={_display(summary.stopcode)} ({_display(summary.stop_description)})"
        )
    return lines


def _metric_lines(report: QAReport) -> List[str]:
    metrics = report.metrics
    residual = metrics.residual
    region = metrics.region
    minimum = "center" if region.min_radius_beams is None else f"{_display(region.min_radius_beams)} beams"
    maximum = "edge" if region.max_radius_beams is None else f"{_display(region.max_radius_beams)} beams"
    definitions = {definition.key: definition for definition in IMAGING_METRIC_DEFINITIONS}
    lines = [
        f"metric region = radial {minimum} to {maximum} (lower inclusive, upper exclusive)",
        f"selected area = {residual.n_pixels} pixels = {_display(residual.area_synthesized_beams)} synthesized beams",
        f"clean_peak=max(I_clean)={_display(metrics.clean_peak_jy_per_beam)} Jy/beam",
        definitions["rms"].format_value(_display(residual.rms_jy_per_beam)),
        definitions["sigma"].format_value(_display(residual.scaled_mad_jy_per_beam)),
        "residual absolute peak = max(|residual|) = "
        f"{_display(residual.residual_abs_peak_jy_per_beam)} Jy/beam",
        f"residual minimum = min(residual) = {_display(residual.residual_min_jy_per_beam)} Jy/beam",
        f"residual maximum = max(residual) = {_display(residual.residual_max_jy_per_beam)} Jy/beam",
        definitions["max"].format_value(_display(residual.peak_over_scaled_mad)),
        definitions["p99"].format_value(_display(residual.p99_over_scaled_mad)),
        definitions["p995"].format_value(_display(residual.p99_5_over_scaled_mad)),
        f"RMS / scaled MAD = {_display(residual.rms_over_scaled_mad)}",
        "dynamic range (RMS) = global max(clean image) / selected residual RMS = "
        f"{_display(metrics.dynamic_range_rms)}",
        definitions["DR"].format_value(_display(metrics.dynamic_range_scaled_mad)),
    ]
    if report.pipeline_background is not None:
        lines.append(
            "VLA Pipeline background RMS = pipeline non-PB-corrected noise-annulus RMS = "
            f"{_display(report.pipeline_background.rms_jy_per_beam)} Jy/beam "
            f"[{report.pipeline_background.region}; {report.pipeline_background.algorithm}]"
        )
    return lines


def render_qa_text(report: QAReport) -> str:
    lines = [
        "Imaging QA Report",
        "=" * 80,
        f"Engine: {report.engine}",
        f"Input MS: {report.ms_path}",
        f"Visibility ID: {report.visibility_id or 'unknown'}",
        "",
        "Parameters",
        "-" * 80,
        *_parameter_lines(report),
        "",
        "Metrics",
        "-" * 80,
        *_metric_lines(report),
    ]
    if report.warnings:
        lines.extend(["", "Warnings", "-" * 80, *(f"- {warning}" for warning in report.warnings)])
    return "\n".join(lines) + "\n"


__all__ = [
    "QA_SCHEMA_VERSION",
    "json_safe",
    "normalize_tclean_summary",
    "render_qa_text",
    "write_qa_reports",
]
