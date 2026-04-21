from __future__ import annotations

import shutil
import os
import sys
from pathlib import Path

import numpy as np
from casatasks import split
from casatools import table

from scripts.io_utils import copy_ms


DEFAULT_ORIGINAL_MS = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted/0205+322/0205+322/0205+322.ms")

PIPELINE_TASK_NAMES = (
    "h_init",
    "h_save",
    "hifv_importdata",
    "hifv_flagtargetsdata",
    "hif_checkproductsize",
    "hif_makeimlist",
    "hif_makeimages",
    "hif_selfcal",
    "hifv_pbcor",
)


def load_pipeline_tasks() -> dict[str, object]:
    """Return CASA pipeline task callables visible to this module."""
    tasks = {name: globals()[name] for name in PIPELINE_TASK_NAMES if name in globals()}
    if len(tasks) == len(PIPELINE_TASK_NAMES):
        return tasks

    try:
        import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "CASA pipeline tasks are not available. Restart CASA with "
            "`casa_run --pipeline` or `casa --pipeline`. If startup prints "
            "`WARN: could not import pipeline`, install/update the pipeline with "
            "`/Users/u1528314/Applications/CASA.app/Contents/MacOS/update-pipeline`."
        ) from exc

    if hasattr(pipeline, "initcli"):
        pipeline.initcli()

    main_globals = getattr(sys.modules.get("__main__"), "__dict__", {})
    candidates = (globals(), main_globals)
    tasks = {}
    missing = []
    for name in PIPELINE_TASK_NAMES:
        task = next((namespace[name] for namespace in candidates if name in namespace), None)
        if task is None:
            missing.append(name)
        else:
            tasks[name] = task

    if missing:
        raise RuntimeError(
            "CASA pipeline initialized, but these tasks are still missing from "
            f"the Python namespace: {', '.join(missing)}"
        )

    return tasks


def main(
    original_ms: str | Path = DEFAULT_ORIGINAL_MS,
    *,
    workdir: str | Path | None = None,
    output_ms: str | Path | None = None,
    field_name: str = "",
    promote_calibrator_to_target: bool = True,
    overwrite: bool = True,
) -> dict:
    # ---------------------------------------------------------------------
    # User settings
    # ---------------------------------------------------------------------
    ORIGINAL_MS = Path(original_ms).expanduser()
    if not ORIGINAL_MS.is_absolute():
        ORIGINAL_MS = (Path.cwd() / ORIGINAL_MS).resolve()
    FIELD_NAME = field_name  # empty means use the first FIELD table name
    PROMOTE_CALIBRATOR_TO_TARGET = promote_calibrator_to_target

    # The NRAO docs recommend starting CASA in the same directory as the data.
    # So this script writes the working copies into the current working directory.
    WORKDIR = Path.cwd() if workdir is None else Path(workdir).expanduser()
    if not WORKDIR.is_absolute():
        WORKDIR = (Path.cwd() / WORKDIR).resolve()
    WORKDIR.mkdir(parents=True, exist_ok=True)
    SOURCE_COPY_MS = WORKDIR / f"{ORIGINAL_MS.stem}_source_copy.ms"
    if output_ms is None:
        PIPE_INPUT_MS = WORKDIR / f"{ORIGINAL_MS.stem}_pipeline_input.ms"
    else:
        PIPE_INPUT_MS = Path(output_ms).expanduser()
        if not PIPE_INPUT_MS.is_absolute():
            PIPE_INPUT_MS = WORKDIR / PIPE_INPUT_MS
        PIPE_INPUT_MS.parent.mkdir(parents=True, exist_ok=True)

    pipeline_tasks = load_pipeline_tasks()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def remove_ms(path: Path) -> None:
        for candidate in (path, Path(f"{path}.flagversions")):
            if candidate.exists():
                print(f"[WARN] Removing existing MS artifact: {candidate}")
                shutil.rmtree(candidate)

    def ms_columns(ms_path: Path) -> list[str]:
        tb = table()
        tb.open(str(ms_path))
        try:
            return list(tb.colnames())
        finally:
            tb.close()

    def has_column(ms_path: Path, colname: str) -> bool:
        return colname in ms_columns(ms_path)

    def ms_field_names(ms_path: Path) -> list[str]:
        tb = table()
        tb.open(str(ms_path / "FIELD"))
        try:
            return [str(name) for name in tb.getcol("NAME")]
        finally:
            tb.close()

    def promote_all_states_to_target(ms_path: Path) -> None:
        """Make the working MS look like target data to pipeline selfcal."""
        state_table = ms_path / "STATE"
        if not state_table.exists():
            raise RuntimeError(f"{ms_path} does not have a STATE table")

        tb = table()
        tb.open(str(state_table), nomodify=False)
        try:
            if "OBS_MODE" not in tb.colnames():
                raise RuntimeError(f"{state_table} does not have OBS_MODE")
            old_modes = [str(mode) for mode in tb.getcol("OBS_MODE")]
            new_modes = np.array(["OBSERVE_TARGET#ON_SOURCE"] * len(old_modes))
            tb.putcol("OBS_MODE", new_modes)
        finally:
            tb.close()

        print("[WARN] Rewrote STATE/OBS_MODE in the working pipeline input MS.")
        print(f"[WARN] Old intents: {sorted(set(old_modes))}")
        print("[WARN] New intent : OBSERVE_TARGET#ON_SOURCE")


    # ---------------------------------------------------------------------
    # Prepare a pipeline-friendly input MS
    # ---------------------------------------------------------------------
    if not ORIGINAL_MS.exists():
        raise RuntimeError(f"Original MS not found: {ORIGINAL_MS}")

    print(f"[INFO] Working directory: {WORKDIR}")
    print(f"[INFO] Original MS: {ORIGINAL_MS}")

    if overwrite:
        remove_ms(SOURCE_COPY_MS)
        remove_ms(PIPE_INPUT_MS)
    elif SOURCE_COPY_MS.exists() or PIPE_INPUT_MS.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing selfcal artifacts: {SOURCE_COPY_MS}, {PIPE_INPUT_MS}"
        )

    # 1) Copy original MS locally so we never touch the original
    copy_ms(str(ORIGINAL_MS), str(SOURCE_COPY_MS))

    cols = ms_columns(SOURCE_COPY_MS)
    print(f"[INFO] Columns in source copy: {cols}")

    # Feed the imaging pipeline an MS whose DATA column contains the
    # visibilities we want to image. Some extracted MSs already have only DATA.
    if "CORRECTED_DATA" in cols:
        split_datacolumn = "corrected"
        print("[INFO] Using CORRECTED_DATA as the pipeline input data.")
    elif "DATA" in cols:
        split_datacolumn = "data"
        print(
            "[WARN] CORRECTED_DATA is missing; using DATA as the pipeline input. "
            "This assumes DATA already contains the calibrated/extracted visibilities."
        )
    else:
        raise RuntimeError(
            f"{SOURCE_COPY_MS} has neither CORRECTED_DATA nor DATA; "
            "cannot prepare a pipeline input MS."
        )

    # 2) Split the selected data column into a new MS with DATA
    print(
        f"[INFO] Splitting datacolumn='{split_datacolumn}' into pipeline input: "
        f"{PIPE_INPUT_MS}"
    )
    split(
        vis=str(SOURCE_COPY_MS),
        outputvis=str(PIPE_INPUT_MS),
        datacolumn=split_datacolumn,
    )

    pipe_cols = ms_columns(PIPE_INPUT_MS)
    print(f"[INFO] Columns in pipeline input MS: {pipe_cols}")
    if "DATA" not in pipe_cols:
        raise RuntimeError(f"{PIPE_INPUT_MS} does not have a DATA column after split; aborting.")

    field_names = ms_field_names(PIPE_INPUT_MS)
    if not field_names:
        raise RuntimeError(f"{PIPE_INPUT_MS} has no FIELD table names")
    target_field = FIELD_NAME or field_names[0]
    if target_field not in field_names:
        raise RuntimeError(
            f"Requested FIELD_NAME={target_field!r}, but {PIPE_INPUT_MS} fields are {field_names}"
        )

    print(f"[INFO] Field names in pipeline input MS: {field_names}")
    print(f"[INFO] Pipeline selfcal field: {target_field}")

    if PROMOTE_CALIBRATOR_TO_TARGET:
        promote_all_states_to_target(PIPE_INPUT_MS)

    print("[INFO] Prepared pipeline input MS.")
    print(f"[INFO] Source copy   : {SOURCE_COPY_MS}")
    print(f"[INFO] Pipeline input: {PIPE_INPUT_MS}")


    # ---------------------------------------------------------------------
    # VLA imaging pipeline
    #
    # This follows the NRAO guidance for a manually split calibrated MS:
    # - import with datacolumns={'data': 'regcal_contline_science'}
    # - do NOT run hif_mstransform()
    #
    # Caveat: hif_selfcal is intended for science targets, so this is a
    # calibrator-as-target experiment and may still fail depending on metadata.
    # ---------------------------------------------------------------------
    h_init = pipeline_tasks["h_init"]
    h_save = pipeline_tasks["h_save"]
    hifv_importdata = pipeline_tasks["hifv_importdata"]
    hifv_flagtargetsdata = pipeline_tasks["hifv_flagtargetsdata"]
    hif_checkproductsize = pipeline_tasks["hif_checkproductsize"]
    hif_makeimlist = pipeline_tasks["hif_makeimlist"]
    hif_makeimages = pipeline_tasks["hif_makeimages"]
    hif_selfcal = pipeline_tasks["hif_selfcal"]
    hifv_pbcor = pipeline_tasks["hifv_pbcor"]

    did_selfcal = False
    original_cwd = Path.cwd()
    os.chdir(WORKDIR)

    try:
        context = h_init()
        context.set_state("ProjectSummary", "observatory", "Karl G. Jansky Very Large Array")
        context.set_state("ProjectSummary", "telescope", "EVLA")

        try:
            hifv_importdata(
                vis=[str(PIPE_INPUT_MS)],
                datacolumns={"data": "regcal_contline_science"},
                specline_spws="none",
            )

            # Keep this because it is part of the standard imaging-only recipe.
            # If it complains but continues, that is fine for this experiment.
            hifv_flagtargetsdata()

            # NRAO docs: for a previously split calibrated MS with only DATA,
            # remove/comment out hif_mstransform().
            # hif_mstransform()

            hif_checkproductsize(maximsize=16384)

            # First make the non-selfcal images
            hif_makeimlist(specmode="cont", datatype="regcal", field=target_field)
            hif_makeimages(hm_cyclefactor=3.0)

            # Then try automated selfcal on the calibrator as if it were a target
            try:
                hif_selfcal(
                    field=target_field,
                    apply=True,
                    amplitude_selfcal=False,   # safer first attempt
                )
                did_selfcal = True
                print("[INFO] hif_selfcal completed.")
            except Exception as exc:
                print(f"[WARN] hif_selfcal failed: {exc}")
                print("[WARN] Keeping only the non-selfcal imaging products.")

            # If selfcal worked, make the selfcal images too
            if did_selfcal:
                hif_makeimlist(specmode="cont", datatype="selfcal", field=target_field)
                hif_makeimages(hm_cyclefactor=3.0)

            # PB correction on whatever the latest images are
            hifv_pbcor()

        finally:
            h_save()
    finally:
        os.chdir(original_cwd)

    print("[DONE] Pipeline run finished.")
    print(f"[DONE] Working directory: {WORKDIR}")
    print("[DONE] Check the pipeline-*/html/index.html weblog and casa_commands.log")
    return {
        "original_ms": str(ORIGINAL_MS),
        "workdir": str(WORKDIR),
        "source_copy_ms": str(SOURCE_COPY_MS),
        "selfcal_ms": str(PIPE_INPUT_MS),
        "field": target_field,
        "did_selfcal": did_selfcal,
    }
