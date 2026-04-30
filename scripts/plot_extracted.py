from pathlib import Path
from math import ceil
import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scripts.experiment_outputs import (
    artifact_label_and_suffix,
    find_latest_experiment_artifact,
    setup_experiment_output_layout,
    write_json,
)

from scripts.sample_groups import (
    BAD_ANT,
    BAD_BASELINE,
    BAD_DATA,
    BEAM_SIZE_ISSUE,
    EXTRA_SOURCE,
    GOOD_ONES,
    NEEDS_BIGGER_IMAGE,
    NEEDS_MULTITERM,
    RESOLVED,
    UV_LIM,
)


DEFAULT_EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
EXPERIMENT_NAME = "plot_extracted"
INPUT_REPORT_EXPERIMENT_NAME = "image_extracted"
INPUT_REPORT_JSON: Path | None = None
IMAGE_PREFIX = "clean_corrected"
SELECTED_FOLDERS: list[str] | None = ["0653+370"]
CUSTOM_TITLE: str | None = None
OUTPUT_FIGURE_NAME: str | None = "requires_box_48_after.png"
OUTPUT_MANIFEST_NAME: str | None = None
INCLUDE_SELFCAL = True
SELFCAL_DIRNAME = "selfcal"
SELFCAL_COMPARE_DIRNAME = "compare_original_selfcal"
SELFCAL_METRICS_CSV_NAME = "selfcal_improvement_metrics.csv"
GLOBAL_SELFCAL_SUMMARY_CSV_NAME = "selfcal_compare_uvlim_summary.csv"

PANEL_SPECS = [
    (
        "selfcal",
        "selfcal final",
        [
            "*_selfcal_final.png",
            "*selfcal_final*.png",
        ],
        False,
    ),
    (
        "clean",
        "clean",
        [
            f"{IMAGE_PREFIX}_clean.png",
            "*_corrected_clean.png",
        ],
        True,
    ),
    (
        "dirty",
        "dirty",
        [
            f"{IMAGE_PREFIX}_dirty.png",
            "*_corrected_dirty.png",
        ],
        True,
    ),
    (
        "residual",
        "residual",
        [
            f"{IMAGE_PREFIX}_residual.png",
            "*_corrected_residual.png",
            "*_residual.png",
        ],
        False,
    ),
    (
        "uv",
        "uv coverage",
        [
            f"{IMAGE_PREFIX}_uv.png",
            "*_corrected_uv.png",
            "*_uv.png",
            "*uv_coverage*.png",
        ],
        False,
    ),
    (
        "amp_uvdist",
        "amp vs uv-dist",
        [
            f"{IMAGE_PREFIX}_amp_vs_uvdist.png",
            "*_corrected_amp_vs_uvdist.png",
            "*amp*uvdist*.png",
            "*amp*uv-dist*.png",
            "*amp*uv*.png",
            "*uvdist*.png",
        ],
        False,
    ),
    (
        "amp_uvdist_norm",
        "amp / median(A)",
        [
            f"{IMAGE_PREFIX}_amp_vs_uvdist_norm.png",
            "*_corrected_amp_vs_uvdist_norm.png",
            "*amp*uvdist*norm*.png",
            "*amp*uv-dist*norm*.png",
            "*amp*median*.png",
        ],
        False,
    ),
    (
        "spectrum_by_ant",
        "spectrum",
        [
            f"{IMAGE_PREFIX}_spectrum_by_ant.png",
            "*_corrected_spectrum_by_ant.png",
            "*spectrum*by_ant*.png",
        ],
        False,
    ),
    (
        "spectrum",
        "spectrum avg baseline",
        [
            f"{IMAGE_PREFIX}_spectrum.png",
            "*_corrected_spectrum.png",
            "*spectrum.png",
        ],
        False,
    ),
]


def active_panel_specs() -> list[tuple[str, str, list[str], bool]]:
    if INCLUDE_SELFCAL:
        return PANEL_SPECS
    return [spec for spec in PANEL_SPECS if spec[0] != "selfcal"]


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


PANEL_ARTIFACT_KEYS = {
    "clean": "clean_png",
    "dirty": "dirty_png",
    "residual": "residual_png",
    "uv": "uv_png",
    "amp_uvdist": "amp_vs_uvdist_png",
    "amp_uvdist_norm": "amp_vs_uvdist_norm_png",
    "spectrum_by_ant": "spectrum_by_ant_png",
    "spectrum": "spectrum_png",
}


def load_meta_map(report_json: Path) -> dict:
    with report_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta_map = {}
    for sample in payload.get("targets", []):
        folder = str(sample.get("target_id", "")).strip()
        if not folder:
            continue

        beam = sample.get("beam", {}) or {}
        config = sample.get("configuration", {}) or {}
        uv = sample.get("uv", {}) or {}
        imaging = sample.get("imaging", {}) or {}
        metrics = sample.get("metrics", {}) or {}
        meta_map[folder] = {
            "name": str(sample.get("target_name", folder)).strip() or folder,
            "minutes": sample.get("minutes"),
            "beam_major_arcsec": beam.get("major_arcsec"),
            "beam_minor_arcsec": beam.get("minor_arcsec"),
            "beam_pa_deg": beam.get("pa_deg"),
            "band_used_for_firstpass": str(config.get("band_used_for_firstpass", "")).strip(),
            "gain_array_config": str(config.get("gain_array_config", "")).strip(),
            "clean_mode": str(config.get("clean_mode", "")).strip(),
            "catalog_uvrange": str(uv.get("catalog_uvrange", "")).strip(),
            "catalog_uvmin_kl": uv.get("catalog_uvmin_kl"),
            "catalog_uvmax_kl": uv.get("catalog_uvmax_kl"),
            "applied_uvrange": str(uv.get("applied_uvrange", "")).strip(),
            "uv_fraction_inside_limits": uv.get("fraction_inside_limits"),
            "cell_arcsec": imaging.get("cell_arcsec"),
            "imsize": imaging.get("imsize_pixels"),
            "fov_arcsec": imaging.get("fov_arcsec"),
            "fov_in_beams_minor": imaging.get("fov_in_beams_minor"),
            "pixels_per_beam_minor": imaging.get("pixels_per_beam_minor"),
            "final_clean_mask_mode": str(config.get("final_clean_mask_mode", "")).strip(),
            "final_clean_box_mask_nbeams": config.get("final_clean_box_mask_nbeams"),
            "status_csv": str(sample.get("source_status", "")).strip(),
            "ms": str(sample.get("ms", "")).strip(),
            "image_output_dir": str(sample.get("image_output_dir", "")).strip(),
            "residual_robust_sigma_jy_per_beam": metrics.get("residual_robust_sigma_jy_per_beam"),
            "residual_p99_abs_over_sigma": metrics.get("residual_p99_abs_over_sigma"),
            "residual_p995_abs_over_sigma": metrics.get("residual_p995_abs_over_sigma"),
            "residual_peak_to_sigma": metrics.get("residual_peak_to_sigma"),
            "dynamic_range": metrics.get("dynamic_range"),
            "selfcal_residual_robust_sigma_jy_per_beam": metrics.get("selfcal_residual_robust_sigma_jy_per_beam"),
            "selfcal_residual_p99_abs_over_sigma": metrics.get("selfcal_residual_p99_abs_over_sigma"),
            "selfcal_residual_p995_abs_over_sigma": metrics.get("selfcal_residual_p995_abs_over_sigma"),
            "selfcal_residual_peak_to_sigma": metrics.get("selfcal_residual_peak_to_sigma"),
            "selfcal_dynamic_range": metrics.get("selfcal_dynamic_range"),
            "residual_robust_sigma_ratio_selfcal_over_original": metrics.get("residual_robust_sigma_ratio_selfcal_over_original"),
            "residual_p99_abs_over_sigma_ratio_selfcal_over_original": metrics.get("residual_p99_abs_over_sigma_ratio_selfcal_over_original"),
            "residual_p995_abs_over_sigma_ratio_selfcal_over_original": metrics.get("residual_p995_abs_over_sigma_ratio_selfcal_over_original"),
            "residual_peak_to_sigma_ratio_selfcal_over_original": metrics.get("residual_peak_to_sigma_ratio_selfcal_over_original"),
            "dynamic_range_ratio_selfcal_over_original": metrics.get("dynamic_range_ratio_selfcal_over_original"),
            "artifacts": sample.get("artifacts", {}) or {},
        }

    return meta_map


def find_first_matching(sample_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        candidate = sample_dir / pattern
        if candidate.exists():
            return candidate

        hits = sorted(sample_dir.glob(pattern))
        if hits:
            return hits[0]

    return None


def image_dir_for_sample(sample_dir: Path) -> Path:
    return sample_dir


def find_selfcal_panel(sample_dir: Path, patterns: list[str]) -> Path | None:
    compare_dir = sample_dir / SELFCAL_DIRNAME / SELFCAL_COMPARE_DIRNAME
    expected = compare_dir / f"{sample_dir.name}_selfcal_final.png"
    if expected.exists():
        return expected
    return find_first_matching(compare_dir, patterns)


def load_selfcal_metrics_map(extracted_dir: Path) -> dict:
    if not INCLUDE_SELFCAL:
        return {}
    metrics_map: dict[str, dict] = {}

    global_summary = extracted_dir / GLOBAL_SELFCAL_SUMMARY_CSV_NAME
    if global_summary.exists():
        try:
            df = pd.read_csv(global_summary)
            for _, row in df.iterrows():
                folder = str(row.get("folder", "")).strip()
                if folder:
                    metrics_map[folder] = dict(row)
        except Exception as exc:
            print(f"[WARN] could not read selfcal summary {global_summary}: {exc}")

    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        metrics_csv = sample_dir / SELFCAL_DIRNAME / SELFCAL_METRICS_CSV_NAME
        if not metrics_csv.exists():
            continue
        try:
            df = pd.read_csv(metrics_csv)
            if not df.empty:
                metrics_map[sample_dir.name] = dict(df.iloc[0])
        except Exception as exc:
            print(f"[WARN] could not read selfcal metrics {metrics_csv}: {exc}")

    return metrics_map


def find_valid_samples(extracted_dir: Path, meta_map: dict, selfcal_metrics_map: dict):
    samples = []

    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        folder = sample_dir.name
        image_dir = image_dir_for_sample(sample_dir)
        meta = meta_map.get(folder, {})
        artifact_map = meta.get("artifacts", {}) if isinstance(meta, dict) else {}

        panel_paths: dict[str, Path] = {}
        missing_required = False
        for key, _, patterns, required in active_panel_specs():
            match = None
            if key == "selfcal":
                match = find_selfcal_panel(sample_dir, patterns)
            else:
                artifact_key = PANEL_ARTIFACT_KEYS.get(key)
                artifact_path = None if artifact_key is None else artifact_map.get(artifact_key)
                if artifact_path:
                    candidate = Path(artifact_path)
                    if candidate.exists():
                        match = candidate
                if match is None:
                    match = find_first_matching(image_dir, patterns)
            if match is not None:
                panel_paths[key] = match
            elif required:
                missing_required = True
                break

        if missing_required:
            continue

        samples.append(
            {
                "folder": folder,
                "name": meta.get("name", folder),
                "minutes": meta.get("minutes"),
                "beam_major_arcsec": meta.get("beam_major_arcsec"),
                "beam_minor_arcsec": meta.get("beam_minor_arcsec"),
                "beam_pa_deg": meta.get("beam_pa_deg"),
                "band_used_for_firstpass": meta.get("band_used_for_firstpass", ""),
                "gain_array_config": meta.get("gain_array_config", ""),
                "clean_mode": meta.get("clean_mode", ""),
                "catalog_uvrange": meta.get("catalog_uvrange", ""),
                "catalog_uvmin_kl": meta.get("catalog_uvmin_kl"),
                "catalog_uvmax_kl": meta.get("catalog_uvmax_kl"),
                "applied_uvrange": meta.get("applied_uvrange", ""),
                "uv_fraction_inside_limits": meta.get("uv_fraction_inside_limits"),
                "cell_arcsec": meta.get("cell_arcsec"),
                "imsize": meta.get("imsize"),
                "fov_arcsec": meta.get("fov_arcsec"),
                "fov_in_beams_minor": meta.get("fov_in_beams_minor"),
                "pixels_per_beam_minor": meta.get("pixels_per_beam_minor"),
                "status_csv": meta.get("status_csv", ""),
                "ms": meta.get("ms", ""),
                "image_dir": str(image_dir),
                "residual_robust_sigma_jy_per_beam": meta.get("residual_robust_sigma_jy_per_beam"),
                "residual_p99_abs_over_sigma": meta.get("residual_p99_abs_over_sigma"),
                "residual_p995_abs_over_sigma": meta.get("residual_p995_abs_over_sigma"),
                "residual_peak_to_sigma": meta.get("residual_peak_to_sigma"),
                "dynamic_range": meta.get("dynamic_range"),
                "selfcal_residual_robust_sigma_jy_per_beam": meta.get("selfcal_residual_robust_sigma_jy_per_beam"),
                "selfcal_residual_p99_abs_over_sigma": meta.get("selfcal_residual_p99_abs_over_sigma"),
                "selfcal_residual_p995_abs_over_sigma": meta.get("selfcal_residual_p995_abs_over_sigma"),
                "selfcal_residual_peak_to_sigma": meta.get("selfcal_residual_peak_to_sigma"),
                "selfcal_dynamic_range": meta.get("selfcal_dynamic_range"),
                "residual_robust_sigma_ratio_selfcal_over_original": meta.get("residual_robust_sigma_ratio_selfcal_over_original"),
                "residual_p99_abs_over_sigma_ratio_selfcal_over_original": meta.get("residual_p99_abs_over_sigma_ratio_selfcal_over_original"),
                "residual_p995_abs_over_sigma_ratio_selfcal_over_original": meta.get("residual_p995_abs_over_sigma_ratio_selfcal_over_original"),
                "residual_peak_to_sigma_ratio_selfcal_over_original": meta.get("residual_peak_to_sigma_ratio_selfcal_over_original"),
                "dynamic_range_ratio_selfcal_over_original": meta.get("dynamic_range_ratio_selfcal_over_original"),
                "selfcal_metrics": selfcal_metrics_map.get(folder, {}),
                "panels": panel_paths,
            }
        )

    return samples


def filter_samples(samples: list[dict], selected_folders: list[str] | None) -> list[dict]:
    if not selected_folders:
        return samples

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    wanted_set = set(wanted)
    filtered = [sample for sample in samples if sample["folder"] in wanted_set]

    missing = [folder for folder in wanted if folder not in {sample["folder"] for sample in filtered}]
    if missing:
        print(f"[WARN] requested folders not found in valid samples: {missing}")

    order = {folder: i for i, folder in enumerate(wanted)}
    filtered.sort(key=lambda sample: order.get(sample["folder"], 10**9))
    return filtered


def fmt_float(x, fmt: str, fallback: str = "?") -> str:
    try:
        x = float(x)
        if pd.notna(x):
            return format(x, fmt)
    except Exception:
        pass
    return fallback


def parse_metric(x) -> float:
    try:
        x = float(x)
    except Exception:
        return float("nan")
    return x if pd.notna(x) else float("nan")


def format_residual_metric(x, fallback: str = "?") -> str:
    value = parse_metric(x)
    if not pd.notna(value):
        return fallback
    if abs(value) >= 1e-4:
        return f"{value:.4f}"
    return f"{value:.6f}"


def first_metric(values) -> float:
    for value in values:
        parsed = parse_metric(value)
        if pd.notna(parsed):
            return parsed
    return float("nan")


def interpret_selfcal_metrics(metrics: dict) -> str:
    robust_ratio = parse_metric(metrics.get("residual_robust_sigma_ratio_selfcal_over_original"))
    peak_ratio = first_metric(
        [
            metrics.get("residual_peak_to_sigma_ratio_selfcal_over_original"),
            metrics.get("residual_max_abs_over_robust_sigma_ratio_selfcal_over_original"),
        ]
    )
    dr_ratio = first_metric(
        [
            metrics.get("dynamic_range_ratio_selfcal_over_original"),
            metrics.get("residual_dynamic_range_ratio_selfcal_over_original"),
        ]
    )

    if (
        pd.notna(robust_ratio)
        and pd.notna(peak_ratio)
        and pd.notna(dr_ratio)
        and robust_ratio < 1.0
        and peak_ratio < 1.0
        and dr_ratio > 1.0
    ):
        return "improved"
    if (
        (pd.notna(robust_ratio) and robust_ratio > 1.0)
        or (pd.notna(peak_ratio) and peak_ratio > 1.0)
        or (pd.notna(dr_ratio) and dr_ratio < 1.0)
    ):
        return "worse"
    return "mixed / small change"


def residual_p995_abs_over_sigma_sort_key(sample: dict) -> tuple[float, str]:
    value = parse_metric(sample.get("residual_p995_abs_over_sigma"))
    if pd.isna(value):
        metrics = sample.get("selfcal_metrics") or {}
        value = first_metric(
            [
                metrics.get("original_residual_p995_abs_over_sigma"),
            ]
        )
    if pd.isna(value):
        value = float("inf")
    return value, str(sample.get("folder", ""))


def sort_samples_by_residual_p995_abs_over_sigma(samples: list[dict]) -> list[dict]:
    return sorted(samples, key=residual_p995_abs_over_sigma_sort_key)


def make_title(i:int, sample: dict) -> str:
    name = sample["name"]

    beam_maj = fmt_float(sample["beam_major_arcsec"], ".2f")
    beam_min = fmt_float(sample["beam_minor_arcsec"], ".2f")
    beam_pa = fmt_float(sample["beam_pa_deg"], ".1f")
    band = str(sample.get("band_used_for_firstpass", "")).strip() or "?"
    config = str(sample.get("gain_array_config", "")).strip() or "?"
    clean_mode = str(sample.get("clean_mode", "")).strip() or "standard"
    applied_uvrange = str(sample.get("applied_uvrange", "")).strip() or "all"
    catalog_uvrange = str(sample.get("catalog_uvrange", "")).strip() or "none"
    uvmin = fmt_float(sample.get("catalog_uvmin_kl"), ".1f", fallback="-")
    uvmax = fmt_float(sample.get("catalog_uvmax_kl"), ".1f", fallback="-")
    uv_inside = fmt_float(
        None if pd.isna(sample.get("uv_fraction_inside_limits")) else 100.0 * float(sample.get("uv_fraction_inside_limits")),
        ".1f",
        fallback="?",
    )
    cell = fmt_float(sample["cell_arcsec"], ".3f")
    ppb = fmt_float(sample["pixels_per_beam_minor"], ".1f")
    fov_arcsec = fmt_float(sample["fov_arcsec"], ".1f")
    fov_beams = fmt_float(sample["fov_in_beams_minor"], ".1f")
    minutes = fmt_float(sample["minutes"], ".1f", fallback="?")
    mask_mode = str(sample.get("final_clean_mask_mode", "")).strip() or "none"
    mask_nbeams = fmt_float(sample.get("final_clean_box_mask_nbeams"), ".1f", fallback="?")
    mask_label = f"box {mask_nbeams} beams" if mask_mode == "beam_box" else "none"

    return (
        f"# {i}\n"
        f"{name}\n"
        f"source=original\n"
        f"beam={beam_maj}\"x{beam_min}\"  pa={beam_pa}°\n"
        f"band={band}  config={config}\n"
        f"clean={clean_mode}\n"
        f"uv={applied_uvrange}  catalog={catalog_uvrange}\n"
        f"in={uv_inside}%  [{uvmin},{uvmax}] kl\n"
        f"cell={cell}\"/pix  ppb={ppb}\n"
        f"mask={mask_label}\n"
        f"FoV={fov_arcsec}\"  ({fov_beams} beams)\n"
        f"time={minutes} min"
    )


def make_selfcal_title(sample: dict) -> str:
    metrics = sample.get("selfcal_metrics") or {}
    if not metrics:
        return "selfcal"

    robust = fmt_float(metrics.get("residual_robust_sigma_ratio_selfcal_over_original"), ".3g")
    peak = fmt_float(
        first_metric(
            [
                metrics.get("residual_peak_to_sigma_ratio_selfcal_over_original"),
                metrics.get("residual_max_abs_over_robust_sigma_ratio_selfcal_over_original"),
            ]
        ),
        ".3g",
    )
    dr = fmt_float(
        first_metric(
            [
                metrics.get("dynamic_range_ratio_selfcal_over_original"),
                metrics.get("residual_dynamic_range_ratio_selfcal_over_original"),
            ]
        ),
        ".3g",
    )
    vis_delta = fmt_float(metrics.get("vis_mean_abs_delta_frac"), ".3g")
    interpretation = str(metrics.get("interpretation", "")).strip()
    if not interpretation or interpretation.lower() == "nan":
        interpretation = str(metrics.get("Interpretation", "")).strip()
    if not interpretation or interpretation.lower() == "nan":
        interpretation = interpret_selfcal_metrics(metrics)

    return (
        "selfcal final\n"
        f"ratios: sigma={robust}  peak/sigma={peak}\n"
        f"DR={dr}\n"
        f"vis={vis_delta}  {interpretation}"
    )


def make_residual_title(sample: dict) -> str:
    p995 = format_residual_metric(sample.get("residual_p995_abs_over_sigma"))
    p99 = format_residual_metric(sample.get("residual_p99_abs_over_sigma"))
    dr = format_residual_metric(sample.get("dynamic_range"))
    sigma = format_residual_metric(sample.get("residual_robust_sigma_jy_per_beam"))
    peak = format_residual_metric(sample.get("residual_peak_to_sigma"))
    return (
        "residual\n"
        f"sigma=1.4826*MAD(residual)={sigma} Jy/bm\n"
        f"max=max(|residual|)/sigma={peak}\n"
        f"p99=P99(|residual|)/sigma={p99}\n"
        f"p995=P99.5(|residual|)/sigma={p995}\n"
        f"DR=max(|clean|)/sigma={dr}"
    )


def make_contact_sheet(
    samples,
    out_path: Path,
    samples_per_row: int = 3,
    figure_title: str | None = None,
):
    if not samples:
        print("[INFO] No valid samples found.")
        return

    active_panels = [
        (key, label)
        for key, label, _, _ in active_panel_specs()
    ]

    n = len(samples)
    nrows = ceil(n / samples_per_row)
    panels_per_sample = len(active_panels)
    ncols = samples_per_row * panels_per_sample

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 5.2 * nrows),
        squeeze=False,
    )

    for axrow in axes:
        for ax in axrow:
            ax.axis("off")

    for i, sample in enumerate(samples):
        row = i // samples_per_row
        slot = i % samples_per_row
        base_col = slot * panels_per_sample

        for panel_offset, (panel_key, panel_label) in enumerate(active_panels):
            ax = axes[row, base_col + panel_offset]
            panel_path = sample["panels"].get(panel_key)

            if panel_path is not None:
                img = mpimg.imread(panel_path)
                ax.imshow(img, aspect="auto")
            elif panel_key == "selfcal":
                pass
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{panel_label}\nnot found",
                    ha="center",
                    va="center",
                    fontsize=11,
                )

            ax.axis("off")
            if panel_key == "selfcal":
                if panel_path is not None or sample.get("selfcal_metrics"):
                    ax.set_title(make_selfcal_title(sample), fontsize=9)
            elif panel_key == "clean":
                ax.set_title(make_title(i, sample), fontsize=9)
            elif panel_key == "residual":
                ax.set_title(make_residual_title(sample), fontsize=9)
            else:
                ax.set_title(panel_label, fontsize=10)

    fig.suptitle(
        figure_title or f"Extracted samples overview ({len(samples)} samples)",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved figure to: {out_path}")


def build_plot_report(
    samples: list[dict],
    *,
    extracted_dir: Path,
    output_layout,
    source_report_json: Path,
    output_figure: Path,
) -> dict:
    return {
        "experiment_name": output_layout.experiment_name,
        "timestamp": output_layout.timestamp,
        "paths": {
            "extracted_dir": str(extracted_dir),
            "source_report_json": str(source_report_json),
            "output_dir": str(output_layout.root_dir),
            "report_json": str(output_layout.artifact_path("report", ".json")),
            "figure_png": str(output_figure),
        },
        "configuration": {
            "selected_folders": SELECTED_FOLDERS,
            "custom_title": CUSTOM_TITLE,
            "panel_specs": [
                {"key": key, "label": label, "required": required}
                for key, label, _, required in active_panel_specs()
            ],
        },
        "summary": {
            "sample_count": len(samples),
        },
        "samples": [
            {
                "target_id": sample["folder"],
                "target_name": sample.get("name", sample["folder"]),
                "titles": (
                    {
                        "clean": make_title(index, sample),
                        "selfcal": make_selfcal_title(sample),
                        "residual": make_residual_title(sample),
                    }
                    if INCLUDE_SELFCAL
                    else {
                        "clean": make_title(index, sample),
                        "residual": make_residual_title(sample),
                    }
                ),
                "panels": {key: str(path) for key, path in sample.get("panels", {}).items()},
                "ms": sample.get("ms", ""),
                "image_dir": sample.get("image_dir", ""),
                "minutes": sample.get("minutes"),
                "beam_major_arcsec": sample.get("beam_major_arcsec"),
                "beam_minor_arcsec": sample.get("beam_minor_arcsec"),
                "beam_pa_deg": sample.get("beam_pa_deg"),
                "cell_arcsec": sample.get("cell_arcsec"),
                "fov_in_beams_minor": sample.get("fov_in_beams_minor"),
                "metrics": {
                    "residual_robust_sigma_jy_per_beam": sample.get("residual_robust_sigma_jy_per_beam"),
                    "residual_p99_abs_over_sigma": sample.get("residual_p99_abs_over_sigma"),
                    "residual_p995_abs_over_sigma": sample.get("residual_p995_abs_over_sigma"),
                    "residual_peak_to_sigma": sample.get("residual_peak_to_sigma"),
                    "dynamic_range": sample.get("dynamic_range"),
                },
                "selfcal_metrics": sample.get("selfcal_metrics", {}) if INCLUDE_SELFCAL else {},
            }
            for index, sample in enumerate(samples)
        ],
    }


def main():
    extracted_dir = resolve_extracted_dir(SELECTED_FOLDERS)
    output_layout = setup_experiment_output_layout(extracted_dir, EXPERIMENT_NAME)
    source_report_json = (
        Path(INPUT_REPORT_JSON).expanduser()
        if INPUT_REPORT_JSON is not None
        else find_latest_experiment_artifact(
            extracted_dir,
            INPUT_REPORT_EXPERIMENT_NAME,
            label="report",
            suffix=".json",
        )
    )
    figure_label, figure_suffix = artifact_label_and_suffix(
        OUTPUT_FIGURE_NAME,
        default_label="contact_sheet",
        default_suffix=".png",
    )
    report_label, _ = artifact_label_and_suffix(
        OUTPUT_MANIFEST_NAME,
        default_label="report",
        default_suffix=".json",
    )
    out_fig = output_layout.artifact_path(figure_label, figure_suffix)
    report_json = output_layout.artifact_path(report_label, ".json")

    print(f"[INFO] using extracted dir: {extracted_dir}")
    print(f"[INFO] experiment output dir: {output_layout.root_dir}")
    print(f"[INFO] source report: {source_report_json}")
    print(f"[INFO] output figure: {out_fig}")
    meta_map = load_meta_map(source_report_json)
    selfcal_metrics_map = load_selfcal_metrics_map(extracted_dir)
    samples = find_valid_samples(extracted_dir, meta_map, selfcal_metrics_map)
    samples = filter_samples(samples, SELECTED_FOLDERS)
    samples = sort_samples_by_residual_p995_abs_over_sigma(samples)
    print(f"[INFO] valid samples found: {len(samples)}")
    figure_title = CUSTOM_TITLE
    make_contact_sheet(samples, out_fig, samples_per_row=1, figure_title=figure_title)
    report = build_plot_report(
        samples,
        extracted_dir=extracted_dir,
        output_layout=output_layout,
        source_report_json=source_report_json,
        output_figure=out_fig,
    )
    write_json(report_json, report)
    print(f"[OK] wrote plot report to: {report_json}")
