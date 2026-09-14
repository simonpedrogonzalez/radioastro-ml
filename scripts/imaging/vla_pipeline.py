"""CASA VLA Imaging Pipeline entrypoint."""

from __future__ import annotations

import ast
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import (
    DEFAULT_IMSIZE,
    DEFAULT_MASK_NBEAMS,
    ImagingConfig,
    central_circle_mask,
    normalize_imsize,
    read_beam,
)
from .imaging import _finalize_result, _prepare_output_dir
from .metadata import (
    DEFAULT_CALIBRATOR_META_CSV,
    DEFAULT_EXTRACTED_MS_ROOT,
    ms_column_names,
    resolve_csv_meta,
    resolve_ms_band,
    resolve_path,
    validate_data_column,
)
from .models import BeamRegion, ImagingResult, PipelineBackground
from .qa import normalize_tclean_summary


_PIPELINE_TASKS = (
    "h_init",
    "h_save",
    "hifv_importdata",
    "hifv_flagtargetsdata",
    "hif_checkproductsize",
    "hif_makeimlist",
    "hif_makeimages",
)


def _load_pipeline_tasks() -> Dict[str, Any]:
    try:
        import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The CASA VLA Pipeline is unavailable in this CASA installation. Start the "
            "pipeline-enabled installation, for example: "
            "'/Users/u1528314/Applications/CASA 2.app/Contents/MacOS/casa' --pipeline"
        ) from exc
    if hasattr(pipeline, "initcli"):
        pipeline.initcli()
    main_scope = vars(sys.modules.get("__main__")) if sys.modules.get("__main__") else {}
    scopes = (globals(), main_scope)
    tasks = {
        name: next((scope[name] for scope in scopes if callable(scope.get(name))), None)
        for name in _PIPELINE_TASKS
    }
    missing = [name for name, task in tasks.items() if task is None]
    if missing:
        raise RuntimeError(
            "The VLA Pipeline initialized without required tasks: " + ", ".join(missing)
        )
    return tasks


def _choose_pipeline_source_column(ms_path: Path):
    warnings: List[str] = []
    columns = ms_column_names(ms_path)
    if "CORRECTED_DATA" in columns:
        try:
            return "corrected", validate_data_column(ms_path, "corrected"), tuple(warnings)
        except RuntimeError as exc:
            warnings.append(f"CORRECTED_DATA was not usable for pipeline input: {exc}")
    if "DATA" in columns:
        return "data", validate_data_column(ms_path, "data"), tuple(warnings)
    raise RuntimeError(
        f"Pipeline input {ms_path} has neither CORRECTED_DATA nor DATA; available columns: "
        + ", ".join(columns)
    )


def _prepare_pipeline_input(source: Path, destination: Path, datacolumn: str) -> str:
    try:
        import numpy as np
        from casatasks import split
        from casatools import table
    except ImportError as exc:
        raise RuntimeError("CASA casatasks and casatools are required for pipeline setup") from exc
    split(vis=str(source), outputvis=str(destination), datacolumn=datacolumn)
    tb = table()
    tb.open(str(destination / "FIELD"))
    try:
        names = [str(name) for name in tb.getcol("NAME")]
    finally:
        tb.close()
    if not names:
        raise RuntimeError(f"Pipeline input {destination} contains no FIELD names")
    tb.open(str(destination / "STATE"), nomodify=False)
    try:
        if "OBS_MODE" not in tb.colnames():
            raise RuntimeError(f"Pipeline input STATE table has no OBS_MODE column: {destination}")
        tb.putcol("OBS_MODE", np.asarray(["OBSERVE_TARGET#ON_SOURCE"] * tb.nrows()))
    finally:
        tb.close()
    return names[0]


def _write_central_mask(target: Mapping[str, Any], path: Path, radius_arcsec: float) -> None:
    """Create a circular CASA mask on one Pipeline clean target's exact grid."""
    import numpy as np
    from casatools import image, quanta

    nx, ny = (int(value) for value in target["imsize"][:2])
    cell = target["cell"][0] if isinstance(target["cell"], (list, tuple)) else target["cell"]
    qa = quanta()
    cell_arcsec = abs(float(qa.convert(qa.quantity(cell), "arcsec")["value"]))
    radius_pixels = radius_arcsec / cell_arcsec

    values = np.zeros((nx, ny, 1, 1), dtype=np.float32)
    x, y = np.ogrid[:nx, :ny]
    values[((x - nx // 2) ** 2 + (y - ny // 2) ** 2) <= radius_pixels**2, 0, 0] = 1

    mask = image()
    mask.fromshape(outfile=str(path), shape=list(values.shape), overwrite=False)
    coordinates = mask.coordsys()
    try:
        _, right_ascension, declination = str(target["phasecenter"]).split(maxsplit=2)
        cell_radians = qa.convert(qa.quantity(cell), "rad")["value"]
        coordinates.setunits(["rad", "rad", "", "Hz"])
        coordinates.setincrement([-cell_radians, cell_radians], "direction")
        coordinates.setreferencepixel([nx // 2, ny // 2], "direction")
        coordinates.setreferencevalue(
            [
                qa.convert(right_ascension, "rad")["value"],
                qa.convert(declination, "rad")["value"],
            ],
            "direction",
        )
        coordinates.setreferencevalue(target["reffreq"], "spectral")
        coordinates.setincrement("1GHz", "spectral")
        coordinates.setrestfrequency(target["reffreq"])
        coordinates.settelescope("EVLA")
        mask.setcoordsys(coordinates.torecord())
        mask.putchunk(values)
    finally:
        coordinates.done()
        mask.done()


def _run_pipeline(
    work: Path,
    input_ms: Path,
    field: str,
    imsize: Tuple[int, int],
    tasks: Mapping[str, Any],
    *,
    mask_radius_arcsec: Optional[float] = None,
):
    previous = Path.cwd()
    context = None
    makeimages_result = None
    os.chdir(work)
    try:
        context = tasks["h_init"]()
        context.set_state("ProjectSummary", "observatory", "Karl G. Jansky Very Large Array")
        context.set_state("ProjectSummary", "telescope", "EVLA")
        try:
            tasks["hifv_importdata"](
                vis=[input_ms.name],
                datacolumns={"data": "regcal_contline_science"},
                specline_spws="none",
            )
            tasks["hifv_flagtargetsdata"]()
            tasks["hif_checkproductsize"](maximsize=16384)
            tasks["hif_makeimlist"](
                specmode="cont",
                datatype="regcal",
                field=field,
                hm_imsize=list(imsize),
            )
            for index, target in enumerate(context.clean_list_pending):
                if mask_radius_arcsec is None:
                    target["mask"] = None
                    continue
                mask_path = work / f"central_clean_{index}.mask"
                _write_central_mask(target, mask_path, mask_radius_arcsec)
                target["mask"] = str(mask_path)
            makeimages_result = tasks["hif_makeimages"](
                hm_masking="manual" if mask_radius_arcsec is not None else "none",
                hm_cyclefactor=3.0,
            )
        finally:
            tasks["h_save"]()
    finally:
        os.chdir(previous)
    directories = sorted(path for path in work.glob("pipeline-*") if path.is_dir())
    if len(directories) != 1:
        raise RuntimeError(
            f"Expected exactly one pipeline weblog directory in {work}, found "
            f"{len(directories)}: {directories}"
        )
    return context, makeimages_result, directories[0]


def parse_tclean_calls(log_path: Path) -> List[Dict[str, Any]]:
    """Parse the exact heuristic-generated tclean calls from one pipeline log."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    calls: List[Dict[str, Any]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("tclean("):
            index += 1
            continue
        chunk = [lines[index].strip()]
        depth = lines[index].count("(") - lines[index].count(")")
        index += 1
        while depth > 0 and index < len(lines):
            chunk.append(lines[index].strip())
            depth += lines[index].count("(") - lines[index].count(")")
            index += 1
        source = " ".join(chunk)
        try:
            expression = ast.parse(source).body[0].value
            parameters = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in expression.keywords
                if keyword.arg is not None
            }
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f"Could not parse pipeline tclean call in {log_path}: {source}") from exc
        calls.append({"command": source, "parameters": parameters})
    if not calls:
        raise RuntimeError(f"No tclean calls were recorded in pipeline log {log_path}")
    return calls


def _pipeline_products(
    work: Path, calls: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Path], Dict[str, Any], Dict[str, Any]]:
    imaging = [call for call in calls if "imagename" in call["parameters"]]
    clean_calls = [call for call in imaging if int(call["parameters"].get("niter", 0)) > 0]
    if not clean_calls:
        raise RuntimeError("The VLA Pipeline log contains no positive-niter imaging call")
    clean_call = clean_calls[-1]
    clean_index = imaging.index(clean_call)
    dirty_calls = [
        call for call in imaging[:clean_index] if int(call["parameters"].get("niter", -1)) == 0
    ]
    if not dirty_calls:
        raise RuntimeError("The VLA Pipeline log contains no dirty call before final imaging")
    dirty_call = dirty_calls[-1]
    clean_parameters = dict(clean_call["parameters"])
    dirty_parameters = dict(dirty_call["parameters"])

    def base(parameters: Mapping[str, Any]) -> Path:
        value = Path(str(parameters["imagename"]))
        return value if value.is_absolute() else work / value

    multiterm = str(clean_parameters.get("deconvolver", "")) == "mtmfs"
    clean_base, dirty_base = base(clean_parameters), base(dirty_parameters)
    products = {
        "dirty": Path(f"{dirty_base}.residual.tt0" if multiterm else f"{dirty_base}.residual"),
        "clean": Path(f"{clean_base}.image.tt0" if multiterm else f"{clean_base}.image"),
        "residual": Path(f"{clean_base}.residual.tt0" if multiterm else f"{clean_base}.residual"),
        "model": Path(f"{clean_base}.model.tt0" if multiterm else f"{clean_base}.model"),
        "mask": Path(f"{clean_base}.mask"),
        "psf": Path(f"{clean_base}.psf.tt0" if multiterm else f"{clean_base}.psf"),
    }
    required = [products[name] for name in ("dirty", "clean", "residual")]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Pipeline log resolved products that do not exist:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return products, dirty_parameters, clean_parameters


def _walk_values(root: Any, *, max_depth: int = 8, max_nodes: int = 50000):
    seen = set()
    stack = [(root, 0)]
    nodes = 0
    while stack and nodes < max_nodes:
        value, depth = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        nodes += 1
        yield value
        if depth >= max_depth:
            continue
        if isinstance(value, Mapping):
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend((child, depth + 1) for child in value)
        else:
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                stack.append((attributes, depth + 1))


def _mapping_score(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    keys = {str(key).casefold() for key in value}
    return sum(
        weight
        for key, weight in (
            ("summaryminor", 5),
            ("iterdone", 4),
            ("stopcode", 4),
            ("nmajordone", 3),
            ("summarymajor", 2),
        )
        if key in keys
    )


def _pipeline_summary(result: Any, context: Any, weblog: Path):
    # Pipeline's TcleanResult retains the final inner tclean fields under
    # stable private storage names even though its public properties have
    # varied between releases.
    result_objects = []
    for root in (result, context):
        root_objects = []
        for value in _walk_values(root):
            attributes = getattr(value, "__dict__", None)
            if (
                isinstance(attributes, dict)
                and isinstance(attributes.get("iterations"), Mapping)
                and "_tclean_stopcode" in attributes
            ):
                root_objects.append(value)
        if root_objects:
            result_objects = root_objects
            break
    if result_objects:
        pipeline_result = max(
            result_objects,
            key=lambda item: max(item.__dict__["iterations"], default=-1),
        )
        attributes = pipeline_result.__dict__
        iterations = attributes["iterations"]
        final_iteration = iterations[max(iterations)] if iterations else {}
        retained_record = dict(final_iteration) if isinstance(final_iteration, Mapping) else {}
        retained_record.update(
            iterdone=attributes.get("_tclean_iterdone"),
            stopcode=attributes.get("_tclean_stopcode"),
            stopDescription=attributes.get("_tclean_stopreason"),
            pipeline_iterations=iterations,
        )
        if "nmajordone" not in retained_record:
            retained_record["nmajordone"] = getattr(pipeline_result, "nmajordone", None)
        if "summarymajor" not in retained_record and "nminordone_array" in retained_record:
            retained_record["summarymajor"] = retained_record["nminordone_array"]
        return normalize_tclean_summary(retained_record, source="pipeline_context")

    candidates = [value for root in (result, context) for value in _walk_values(root)]
    best = max(candidates, key=_mapping_score, default=None)
    if best is not None and _mapping_score(best) > 0:
        return normalize_tclean_summary(best, source="pipeline_context")

    raw: Dict[str, Any] = {"weblog": str(weblog)}
    stage_text = _final_imaging_stage_text(weblog)
    match = re.search(
        r"total number of major cycles done</th>\s*<td[^>]*>\s*(\d+)",
        stage_text,
        flags=re.IGNORECASE,
    )
    if match:
        raw["nmajordone"] = int(match.group(1))
    summary, warnings = normalize_tclean_summary(raw, source="pipeline_log")
    return summary, warnings + (
        "The Pipeline did not expose the inner tclean return record; only explicit weblog "
        "execution values were retained, and unavailable summary fields remain null.",
    )


def _final_imaging_stage_text(weblog: Path) -> str:
    matches = []
    for page in sorted((weblog / "html").glob("stage*/t2-4m_details.html")):
        text = page.read_text(encoding="utf-8", errors="replace")
        if "non-pbcor image RMS" in text or "total number of major cycles done" in text:
            matches.append(text)
    return matches[-1] if matches else ""


def _weblog_background_rms(weblog: Path) -> Optional[float]:
    """Read only the explicit non-PB-corrected RMS field from the final weblog."""
    text = _final_imaging_stage_text(weblog)
    match = re.search(
        r"non-pbcor image RMS</th>\s*<td[^>]*>\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([^<]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = re.sub(r"\s+", "", match.group(2)).casefold().replace("µ", "u")
    factors = {
        "jy/beam": 1.0,
        "jy/bm": 1.0,
        "mjy/beam": 1e-3,
        "mjy/bm": 1e-3,
        "ujy/beam": 1e-6,
        "ujy/bm": 1e-6,
        "njy/beam": 1e-9,
        "njy/bm": 1e-9,
    }
    factor = factors.get(unit)
    if factor is None or not math.isfinite(value):
        return None
    return value * factor


def _image_geometry(path: Path) -> Dict[str, Any]:
    try:
        from casatasks import imhead
    except ImportError as exc:
        raise RuntimeError("CASA casatasks is required to inspect pipeline image geometry") from exc
    info = imhead(imagename=str(path), mode="summary")
    beam = read_beam(path, imhead_task=imhead)

    def arcsec(value: float, unit: str) -> float:
        unit = unit.casefold()
        if unit.startswith("rad"):
            return abs(math.degrees(float(value)) * 3600.0)
        if unit.startswith("deg"):
            return abs(float(value) * 3600.0)
        if unit.startswith("arcmin"):
            return abs(float(value) * 60.0)
        return abs(float(value))

    cell = [arcsec(info["incr"][index], info["axisunits"][index]) for index in (0, 1)]
    shape = [int(info["shape"][index]) for index in (0, 1)]
    return {
        "beam": asdict(beam),
        "imsize": shape,
        "cell_arcsec": cell,
        "field_of_view_arcsec": [shape[index] * cell[index] for index in (0, 1)],
    }


def image_ms_VLA_pipe(
    ms: str | Path,
    output_dir: str | Path,
    *,
    imsize: int | Sequence[int] = DEFAULT_IMSIZE,
    mask_nbeams: Optional[float] = DEFAULT_MASK_NBEAMS,
    metric_region: BeamRegion = BeamRegion(),
) -> ImagingResult:
    """Image one MS with a beam-scaled central mask on an explicitly sized grid."""
    requested_imsize = normalize_imsize(imsize)
    if not isinstance(metric_region, BeamRegion):
        raise TypeError(
            f"metric_region must be BeamRegion, got {type(metric_region).__name__}"
        )
    output = _prepare_output_dir(output_dir)
    pipeline_root = output / "pipeline"
    if pipeline_root.exists() and any(pipeline_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty pipeline directory {pipeline_root}")
    pipeline_root.mkdir(parents=True, exist_ok=True)
    resolved_ms = resolve_path(ms, DEFAULT_EXTRACTED_MS_ROOT)
    csv_meta, preferred_spw = resolve_csv_meta(
        resolved_ms.visibility_id, DEFAULT_CALIBRATOR_META_CSV
    )
    ms_band = resolve_ms_band(resolved_ms.path, csv_meta, preferred_spw=preferred_spw)
    datacolumn, validation, column_warnings = _choose_pipeline_source_column(resolved_ms.path)
    warnings = list(column_warnings)
    if csv_meta is None:
        warnings.append("No calibrator metadata row was found for the pipeline input.")
    elif ms_band.catalog_band_matches is False:
        warnings.append(
            f"MS-derived band {ms_band.selected_band} does not match catalog bands "
            f"{', '.join(csv_meta.catalog_band_codes)}."
        )

    mask_region = None
    mask_radius_arcsec = None
    mask_probe = None
    if mask_nbeams is not None:
        if mask_nbeams <= 0:
            raise ValueError("mask_nbeams must be positive or None")
        probe_config = ImagingConfig(datacolumn=datacolumn, nterms=1, mask_nbeams=None)
        mask_probe = probe_config.resolve(
            resolved_ms.path,
            output / "mask_probe",
            imsize=requested_imsize,
        )
        mask_region = central_circle_mask(
            requested_imsize,
            mask_probe.grid.beam.major_arcsec,
            mask_nbeams,
        )
        mask_radius_arcsec = 0.5 * mask_nbeams * mask_probe.grid.beam.major_arcsec

    tasks = _load_pipeline_tasks()
    input_ms = pipeline_root / f"{resolved_ms.path.stem}_pipeline_input.ms"
    field = _prepare_pipeline_input(resolved_ms.path, input_ms, datacolumn)
    context, task_result, weblog = _run_pipeline(
        pipeline_root,
        input_ms,
        field,
        requested_imsize,
        tasks,
        mask_radius_arcsec=mask_radius_arcsec,
    )
    log = weblog / "html/casa_commands.log"
    products, dirty_parameters, clean_parameters = _pipeline_products(
        pipeline_root, parse_tclean_calls(log)
    )
    summary, summary_warnings = _pipeline_summary(task_result, context, weblog)
    warnings.extend(summary_warnings)
    background_rms = _weblog_background_rms(weblog)
    rms_source = "pipeline_weblog" if background_rms is not None else "unavailable"
    pipeline_background = None
    if background_rms is not None:
        pipeline_background = PipelineBackground(
            rms_jy_per_beam=background_rms,
            region="adaptive primary-beam annulus outside CLEAN mask",
            algorithm="VLA Pipeline Chauvenet RMS, median across spectral planes",
        )
    if background_rms is None:
        warnings.append(
            "The final VLA Pipeline imaging weblog did not contain a readable "
            "'non-pbcor image RMS'; background RMS is unavailable."
        )

    geometry = _image_geometry(products["clean"])
    actual_imsize = tuple(geometry["imsize"])
    if actual_imsize != requested_imsize:
        raise RuntimeError(
            "The VLA Pipeline did not honor the requested image size: "
            f"requested {requested_imsize[0]}x{requested_imsize[1]}, "
            f"produced {actual_imsize[0]}x{actual_imsize[1]}"
        )

    effective: Dict[str, Any] = dict(clean_parameters)
    effective.update(
        pipeline_dirty_tclean_parameters=dirty_parameters,
        pipeline_final_tclean_parameters=clean_parameters,
        pipeline_weblog=str(weblog),
        pipeline_commands_log=str(log),
        pipeline_input_ms=str(input_ms),
        pipeline_source_datacolumn=datacolumn,
        data_column_validation=asdict(validation),
        csv_meta=None if csv_meta is None else asdict(csv_meta),
        ms_band_meta=asdict(ms_band),
        calibrator_meta_csv=str(DEFAULT_CALIBRATOR_META_CSV),
        extracted_ms_root=str(DEFAULT_EXTRACTED_MS_ROOT),
        requested_imsize=list(requested_imsize),
        pipeline_masking="manual" if mask_region else "none",
        mask_nbeams=mask_nbeams,
        mask_region=mask_region,
        mask_probe_beam=None if mask_probe is None else asdict(mask_probe.grid.beam),
        measured_geometry=geometry,
        vla_background_rms_jy_per_beam=background_rms,
        vla_background_rms_source=rms_source,
    )
    optional = {
        name: path if path.exists() else None for name, path in products.items()
    }
    return _finalize_result(
        engine="vla_pipeline",
        input_value=str(ms),
        ms_path=resolved_ms.path,
        visibility_id=resolved_ms.visibility_id,
        output_dir=output,
        resolved_config=None,
        effective_parameters=effective,
        dirty_image=products["dirty"],
        clean_image=products["clean"],
        residual_image=products["residual"],
        model_image=optional["model"],
        mask_image=optional["mask"],
        psf_image=optional["psf"],
        tclean_summary=summary,
        warnings=tuple(warnings),
        metric_region=metric_region,
        pipeline_background=pipeline_background,
    )


__all__ = ["image_ms_VLA_pipe", "parse_tclean_calls"]
