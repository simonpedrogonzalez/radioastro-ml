"""Direct imaging entrypoint and concrete shared finalization helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

from .config import (
    DEFAULT_IMSIZE,
    ImagingConfig,
    _assert_prefix_available,
    _image_product,
    central_circle_mask,
    read_beam,
)
from .metrics import METRIC_UNITS, _metric_validity, _metric_warnings, measure_image_metrics
from .models import (
    BeamRegion,
    ImagingResult,
    PipelineBackground,
    QAReport,
    ResolvedImagingConfig,
    TcleanRunSummary,
)
from .plot_utils import write_individual_plots
from .qa import QA_SCHEMA_VERSION, normalize_tclean_summary, write_qa_reports


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Imaging output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mask_region(resolved: ResolvedImagingConfig, nbeams: float) -> str:
    grid = resolved.grid.image_grid
    beam = resolved.grid.beam
    return central_circle_mask(grid.imsize, beam.major_arcsec, nbeams)


def _direct_base_parameters(
    config: ImagingConfig,
    resolved: ResolvedImagingConfig,
    base: Path,
) -> Dict[str, Any]:
    grid = resolved.grid.image_grid
    parameters = config._base_parameters()
    parameters.update(
        vis=str(resolved.ms_path),
        imagename=str(base),
        imsize=list(grid.imsize),
        cell=[f"{grid.cell_arcsec[0]:.12g}arcsec", f"{grid.cell_arcsec[1]:.12g}arcsec"],
    )
    return parameters


def _require_products(paths: Dict[str, Path], label: str) -> None:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{label} did not create expected CASA products:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def _optional_existing(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _finalize_result(
    *,
    engine: Literal["direct", "vla_pipeline"],
    input_value: str,
    ms_path: Path,
    visibility_id: Optional[str],
    output_dir: Path,
    resolved_config: Optional[ResolvedImagingConfig],
    effective_parameters: Dict[str, Any],
    dirty_image: Path,
    clean_image: Path,
    residual_image: Path,
    model_image: Optional[Path],
    mask_image: Optional[Path],
    psf_image: Optional[Path],
    tclean_summary: Optional[TcleanRunSummary],
    warnings: Tuple[str, ...],
    metric_region: BeamRegion,
    pipeline_background: Optional[PipelineBackground] = None,
) -> ImagingResult:
    _require_products(
        {"dirty": dirty_image, "clean": clean_image, "residual": residual_image},
        engine,
    )
    dirty_png, clean_png, residual_png = write_individual_plots(
        dirty_image,
        clean_image,
        residual_image,
        output_dir,
        visibility_id=visibility_id,
        mask_image=mask_image,
        metric_region=metric_region,
        fallback_beam=read_beam(clean_image),
    )
    metrics = measure_image_metrics(clean_image, residual_image, region=metric_region)
    validity = _metric_validity(metrics)
    metric_warnings = _metric_warnings(metrics)
    qa_text = output_dir / "qa.txt"
    qa_json = output_dir / "qa.json"
    products: Dict[str, Optional[Path]] = {
        "dirty_image": dirty_image,
        "clean_image": clean_image,
        "residual_image": residual_image,
        "model_image": model_image,
        "mask_image": mask_image,
        "psf_image": psf_image,
        "dirty_png": dirty_png,
        "clean_png": clean_png,
        "residual_png": residual_png,
        "qa_text": qa_text,
        "qa_json": qa_json,
    }
    report = QAReport(
        schema_version=QA_SCHEMA_VERSION,
        engine=engine,
        input_value=input_value,
        ms_path=ms_path,
        visibility_id=visibility_id,
        resolved_config=resolved_config,
        effective_imaging_parameters=effective_parameters,
        products=products,
        metrics=metrics,
        metric_units=dict(METRIC_UNITS),
        metric_validity=validity,
        tclean_summary=tclean_summary,
        pipeline_background=pipeline_background,
        warnings=tuple(warnings) + tuple(metric_warnings),
    )
    write_qa_reports(report, qa_text, qa_json)
    return ImagingResult(
        engine=engine,
        ms_path=ms_path,
        visibility_id=visibility_id,
        output_dir=output_dir,
        resolved_config=resolved_config,
        effective_imaging_parameters=effective_parameters,
        dirty_image=dirty_image,
        clean_image=clean_image,
        residual_image=residual_image,
        model_image=model_image,
        mask_image=mask_image,
        psf_image=psf_image,
        dirty_png=dirty_png,
        clean_png=clean_png,
        residual_png=residual_png,
        qa_text=qa_text,
        qa_json=qa_json,
        tclean_summary=tclean_summary,
        qa=report,
        pipeline_background=pipeline_background,
    )


def image_ms(
    ms: str | Path,
    config: ImagingConfig,
    output_dir: str | Path,
    *,
    imsize: int | Sequence[int] = DEFAULT_IMSIZE,
    metric_region: BeamRegion = BeamRegion(),
    metric_region_resolver: Optional[
        Callable[[ResolvedImagingConfig], BeamRegion]
    ] = None,
) -> ImagingResult:
    """Image one MS directly, resolving any grid-dependent metric region once."""
    if not isinstance(config, ImagingConfig):
        raise TypeError(f"config must be ImagingConfig, got {type(config).__name__}")
    if not isinstance(metric_region, BeamRegion):
        raise TypeError(
            f"metric_region must be BeamRegion, got {type(metric_region).__name__}"
        )
    if metric_region_resolver is not None and not callable(metric_region_resolver):
        raise TypeError("metric_region_resolver must be callable or None")
    if metric_region_resolver is not None and metric_region != BeamRegion():
        raise ValueError(
            "metric_region and metric_region_resolver cannot both specify a region"
        )
    output = _prepare_output_dir(output_dir)
    resolved = config.resolve(ms, output, imsize=imsize)
    if metric_region_resolver is not None:
        metric_region = metric_region_resolver(resolved)
        if not isinstance(metric_region, BeamRegion):
            raise TypeError(
                "metric_region_resolver must return BeamRegion, got "
                f"{type(metric_region).__name__}"
            )
    try:
        from casatasks import tclean
    except ImportError as exc:
        raise RuntimeError("CASA casatasks is required for direct imaging") from exc

    dirty_base = output / "dirty"
    _assert_prefix_available(dirty_base)
    dirty_parameters = _direct_base_parameters(config, resolved, dirty_base)
    dirty_parameters["niter"] = 0
    tclean(**dirty_parameters)
    dirty_image = _image_product(dirty_base, "image", config.deconvolver)
    dirty_residual = _image_product(dirty_base, "residual", config.deconvolver)
    _require_products({"dirty image": dirty_image, "dirty residual": dirty_residual}, "Dirty tclean")

    controls = config.clean.resolve(dirty_residual)
    clean_base = output / "clean"
    _assert_prefix_available(clean_base)
    clean_parameters = _direct_base_parameters(config, resolved, clean_base)
    clean_parameters.update(niter=controls.niter, fullsummary=True)
    if controls.threshold is not None:
        clean_parameters["threshold"] = controls.threshold
    if controls.nsigma is not None:
        clean_parameters["nsigma"] = controls.nsigma
    if controls.cycleniter is not None:
        clean_parameters["cycleniter"] = controls.cycleniter
    if config.mask_nbeams is not None:
        clean_parameters["usemask"] = "user"
        clean_parameters["mask"] = _mask_region(resolved, config.mask_nbeams)

    return_record = tclean(**clean_parameters)
    summary, summary_warnings = normalize_tclean_summary(return_record)
    effective = dict(resolved.effective_imaging_parameters)
    effective.update(
        niter=controls.niter,
        threshold=controls.threshold,
        nsigma=controls.nsigma,
        cycleniter=controls.cycleniter,
        usemask=clean_parameters.get("usemask", "none"),
        mask=clean_parameters.get("mask"),
    )
    resolved = replace(
        resolved,
        clean=controls,
        effective_imaging_parameters=effective,
    )
    clean_image = _image_product(clean_base, "image", config.deconvolver)
    residual_image = _image_product(clean_base, "residual", config.deconvolver)
    model = _optional_existing(_image_product(clean_base, "model", config.deconvolver))
    mask = _optional_existing(_image_product(clean_base, "mask", config.deconvolver))
    psf = _optional_existing(_image_product(clean_base, "psf", config.deconvolver))
    return _finalize_result(
        engine="direct",
        input_value=str(ms),
        ms_path=resolved.ms_path,
        visibility_id=resolved.visibility_id,
        output_dir=output,
        resolved_config=resolved,
        effective_parameters=effective,
        dirty_image=dirty_image,
        clean_image=clean_image,
        residual_image=residual_image,
        model_image=model,
        mask_image=mask,
        psf_image=psf,
        tclean_summary=summary,
        warnings=resolved.warnings + summary_warnings,
        metric_region=metric_region,
    )


__all__ = ["image_ms"]
