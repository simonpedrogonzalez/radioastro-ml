from __future__ import annotations

import csv
import importlib
from pathlib import Path

from scripts import selfcal_compare, vla_pipe
from scripts.sample_groups import UV_LIM


# Run this in the CASA pipeline app, e.g.:
# casa_run2 --pipeline
# CASA <1>: from scripts import selfcal_extracted
# CASA <2>: selfcal_extracted.main()

DEFAULT_EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
SELECTED_FOLDERS: list[str] | None = None
OVERWRITE = True
COMPUTE_IMPROVEMENT_METRICS = True

SELFCAL_DIRNAME = "selfcal"
SELFCAL_MS_SUFFIX = "_selfcal.ms"
SUMMARY_CSV_NAME = "selfcal_summary.csv"

IMPROVEMENT_METRIC_KEYS = [
    "interpretation",
    "Interpretation",
    "selfcal_metrics_csv",
    "comparison_png",
    "original_residual_rms_all_jy_per_beam",
    "selfcal_residual_rms_all_jy_per_beam",
    "residual_rms_all_ratio_selfcal_over_original",
    "original_residual_robust_sigma_jy_per_beam",
    "selfcal_residual_robust_sigma_jy_per_beam",
    "residual_robust_sigma_ratio_selfcal_over_original",
    "original_residual_max_abs_over_robust_sigma",
    "selfcal_residual_max_abs_over_robust_sigma",
    "original_residual_p995_abs_over_sigma",
    "selfcal_residual_p995_abs_over_sigma",
    "residual_p995_abs_over_sigma_ratio_selfcal_over_original",
    "original_residual_p99_abs_over_sigma",
    "selfcal_residual_p99_abs_over_sigma",
    "residual_p99_abs_over_sigma_ratio_selfcal_over_original",
    "original_residual_peak_to_sigma",
    "selfcal_residual_peak_to_sigma",
    "residual_peak_to_sigma_ratio_selfcal_over_original",
    "residual_max_abs_over_robust_sigma_ratio_selfcal_over_original",
    "original_dynamic_range",
    "selfcal_dynamic_range",
    "original_residual_dynamic_range",
    "selfcal_residual_dynamic_range",
    "residual_dynamic_range_ratio_selfcal_over_original",
    "robust_sigma_ratio_selfcal_over_original",
    "rms_offsource_ratio_selfcal_over_original",
    "dynamic_range_ratio_selfcal_over_original",
    "vis_mean_abs_delta_frac",
]


def resolve_extracted_dir(
    selected_folders: list[str] | None,
    default_dir: Path = DEFAULT_EXTRACTED_DIR,
) -> Path:
    if not selected_folders:
        return default_dir

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    if not wanted:
        return default_dir

    missing = [folder for folder in wanted if not (default_dir / folder).exists()]
    if missing:
        print(f"[WARN] requested folders not found under {default_dir}: {missing}")

    return default_dir


def sample_top_for_ms(ms_path: Path) -> Path:
    return ms_path.parent.parent


def selfcal_dir_for_sample(sample_top: Path) -> Path:
    return sample_top / SELFCAL_DIRNAME


def selfcal_ms_for_sample(sample_top: Path) -> Path:
    return selfcal_dir_for_sample(sample_top) / f"{sample_top.name}{SELFCAL_MS_SUFFIX}"


def is_inside_selfcal(path: Path, sample_dir: Path) -> bool:
    try:
        rel_parts = path.relative_to(sample_dir).parts
    except ValueError:
        return False
    return SELFCAL_DIRNAME in rel_parts


def find_extracted_ms_paths(extracted_dir: Path) -> list[Path]:
    hits = []
    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        folder = sample_dir.name
        expected_ms = sample_dir / folder / f"{folder}.ms"
        if expected_ms.exists():
            hits.append(expected_ms)
            continue

        ms_list = sorted(
            ms_path
            for ms_path in sample_dir.rglob("*.ms")
            if not is_inside_selfcal(ms_path, sample_dir)
        )
        if ms_list:
            hits.append(ms_list[0])

    return hits


def filter_ms_paths(ms_paths: list[Path], selected_folders: list[str] | None) -> list[Path]:
    if not selected_folders:
        return ms_paths

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    wanted_set = set(wanted)
    filtered = [ms_path for ms_path in ms_paths if sample_top_for_ms(ms_path).name in wanted_set]

    found_folders = {sample_top_for_ms(ms_path).name for ms_path in filtered}
    missing = [folder for folder in wanted if folder not in found_folders]
    if missing:
        print(f"[WARN] requested folders not found in extracted MS paths: {missing}")

    order = {folder: i for i, folder in enumerate(wanted)}
    filtered.sort(key=lambda ms_path: order.get(sample_top_for_ms(ms_path).name, 10**9))
    return filtered


def selfcal_one_ms(
    ms_path: Path,
    *,
    overwrite: bool = OVERWRITE,
    compute_improvement_metrics: bool = COMPUTE_IMPROVEMENT_METRICS,
    project_rows: list[dict] | None = None,
    calibrator_rows: list[dict] | None = None,
) -> dict:
    sample_top = sample_top_for_ms(ms_path)
    folder = sample_top.name
    workdir = selfcal_dir_for_sample(sample_top)
    output_ms = selfcal_ms_for_sample(sample_top)

    print(f"[SELFCAL] {folder}")
    print(f"[SELFCAL] input : {ms_path}")
    print(f"[SELFCAL] work  : {workdir}")
    print(f"[SELFCAL] output: {output_ms}")

    result = vla_pipe.main(
        original_ms=ms_path,
        workdir=workdir,
        output_ms=output_ms,
        promote_calibrator_to_target=True,
        overwrite=overwrite,
    )

    did_selfcal = bool(result.get("did_selfcal"))
    status = "ok" if did_selfcal else "no_selfcal"
    row = {
        "folder": folder,
        "status": status,
        "did_selfcal": did_selfcal,
        "original_ms": str(ms_path),
        "selfcal_dir": str(workdir),
        "selfcal_ms": str(output_ms),
        "pipeline_field": result.get("field", ""),
        "source_copy_ms": result.get("source_copy_ms", ""),
    }

    if did_selfcal and compute_improvement_metrics:
        try:
            if project_rows is None:
                project_rows = selfcal_compare.read_csv_rows(selfcal_compare.PROJECT_LIST)
            if calibrator_rows is None:
                calibrator_rows = selfcal_compare.read_csv_rows(selfcal_compare.CALIBRATOR_BANDS_CSV)
            metrics = selfcal_compare.process_one(
                ms_path,
                project_rows=project_rows,
                calibrator_rows=calibrator_rows,
            )
            for key in IMPROVEMENT_METRIC_KEYS:
                row[key] = metrics.get(key, "")
        except Exception as exc:
            row["metrics_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] improvement metrics failed for {folder}: {row['metrics_error']}")

    return row


def main(
    extracted_dir: str | Path | None = None,
    *,
    selected_folders: list[str] | None = SELECTED_FOLDERS,
    overwrite: bool = OVERWRITE,
) -> list[dict]:
    importlib.reload(vla_pipe)
    importlib.reload(selfcal_compare)

    extracted_dir = (
        resolve_extracted_dir(selected_folders)
        if extracted_dir is None
        else Path(extracted_dir).expanduser()
    )

    ms_paths = find_extracted_ms_paths(extracted_dir)
    ms_paths = filter_ms_paths(ms_paths, selected_folders)

    print(f"[INFO] using extracted dir: {extracted_dir}")
    print(f"[INFO] found {len(ms_paths)} extracted MS files")
    print(f"[INFO] COMPUTE_IMPROVEMENT_METRICS={COMPUTE_IMPROVEMENT_METRICS}")

    project_rows = None
    calibrator_rows = None
    if COMPUTE_IMPROVEMENT_METRICS:
        project_rows = selfcal_compare.read_csv_rows(selfcal_compare.PROJECT_LIST)
        calibrator_rows = selfcal_compare.read_csv_rows(selfcal_compare.CALIBRATOR_BANDS_CSV)

    rows = []
    for ms_path in ms_paths:
        folder = sample_top_for_ms(ms_path).name
        try:
            rows.append(
                selfcal_one_ms(
                    ms_path,
                    overwrite=overwrite,
                    compute_improvement_metrics=COMPUTE_IMPROVEMENT_METRICS,
                    project_rows=project_rows,
                    calibrator_rows=calibrator_rows,
                )
            )
        except Exception as exc:
            print(f"[ERROR] {folder}: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "folder": folder,
                    "status": "error",
                    "did_selfcal": False,
                    "original_ms": str(ms_path),
                    "selfcal_dir": str(selfcal_dir_for_sample(sample_top_for_ms(ms_path))),
                    "selfcal_ms": str(selfcal_ms_for_sample(sample_top_for_ms(ms_path))),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_csv = extracted_dir / SUMMARY_CSV_NAME
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote summary to {summary_csv}")
    return rows


# if __name__ == "__main__":
#     main()
