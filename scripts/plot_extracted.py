from pathlib import Path
from math import ceil

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    GOOD_ONES_2,
    UV_LIM,
    NEED_SELFCAL_2
)


DEFAULT_EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
IMAGE_PREFIX = "clean_corrected"
SELECTED_FOLDERS: list[str] | None = ["0653+370"]
CUSTOM_TITLE: str | None = None
OUTPUT_FIGURE_NAME: str | None = "requires_box_48_after.png"
OUTPUT_MANIFEST_NAME: str | None = None
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


def load_meta_map(summary_csv: Path) -> dict:
    df = pd.read_csv(summary_csv)

    meta_map = {}
    for _, row in df.iterrows():
        folder = str(row.get("folder", "")).strip()
        if not folder:
            continue

        meta_map[folder] = {
            "name": str(row.get("name", folder)).strip() or folder,
            "minutes": row.get("minutes"),
            "beam_major_arcsec": row.get("beam_major_arcsec"),
            "beam_minor_arcsec": row.get("beam_minor_arcsec"),
            "beam_pa_deg": row.get("beam_pa_deg"),
            "band_used_for_firstpass": str(row.get("band_used_for_firstpass", "")).strip(),
            "gain_array_config": str(row.get("gain_array_config", "")).strip(),
            "clean_mode": str(row.get("clean_mode", "")).strip(),
            "catalog_uvrange": str(row.get("catalog_uvrange", "")).strip(),
            "catalog_uvmin_kl": row.get("catalog_uvmin_kl"),
            "catalog_uvmax_kl": row.get("catalog_uvmax_kl"),
            "applied_uvrange": str(row.get("applied_uvrange", "")).strip(),
            "uv_fraction_inside_limits": row.get("uv_fraction_inside_limits"),
            "cell_arcsec": row.get("cell_arcsec"),
            "imsize": row.get("imsize"),
            "fov_arcsec": row.get("fov_arcsec"),
            "fov_in_beams_minor": row.get("fov_in_beams_minor"),
            "pixels_per_beam_minor": row.get("pixels_per_beam_minor"),
            "final_clean_mask_mode": str(row.get("final_clean_mask_mode", "")).strip(),
            "final_clean_box_mask_nbeams": row.get("final_clean_box_mask_nbeams"),
            "status_csv": str(row.get("status_csv", "")).strip(),
            "ms": str(row.get("ms", "")).strip(),
            "image_output_dir": str(row.get("image_output_dir", "")).strip(),
            "residual_robust_sigma_jy_per_beam": row.get("residual_robust_sigma_jy_per_beam"),
            "residual_p99_abs_over_sigma": row.get("residual_p99_abs_over_sigma"),
            "residual_p995_abs_over_sigma": row.get("residual_p995_abs_over_sigma"),
            "residual_peak_to_sigma": row.get("residual_peak_to_sigma"),
            "dynamic_range": row.get("dynamic_range"),
            "selfcal_residual_robust_sigma_jy_per_beam": row.get("selfcal_residual_robust_sigma_jy_per_beam"),
            "selfcal_residual_p99_abs_over_sigma": row.get("selfcal_residual_p99_abs_over_sigma"),
            "selfcal_residual_p995_abs_over_sigma": row.get("selfcal_residual_p995_abs_over_sigma"),
            "selfcal_residual_peak_to_sigma": row.get("selfcal_residual_peak_to_sigma"),
            "selfcal_dynamic_range": row.get("selfcal_dynamic_range"),
            "residual_robust_sigma_ratio_selfcal_over_original": row.get("residual_robust_sigma_ratio_selfcal_over_original"),
            "residual_p99_abs_over_sigma_ratio_selfcal_over_original": row.get("residual_p99_abs_over_sigma_ratio_selfcal_over_original"),
            "residual_p995_abs_over_sigma_ratio_selfcal_over_original": row.get("residual_p995_abs_over_sigma_ratio_selfcal_over_original"),
            "residual_peak_to_sigma_ratio_selfcal_over_original": row.get("residual_peak_to_sigma_ratio_selfcal_over_original"),
            "dynamic_range_ratio_selfcal_over_original": row.get("dynamic_range_ratio_selfcal_over_original"),
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

        panel_paths: dict[str, Path] = {}
        missing_required = False
        for key, _, patterns, required in PANEL_SPECS:
            match = (
                find_selfcal_panel(sample_dir, patterns)
                if key == "selfcal"
                else find_first_matching(image_dir, patterns)
            )
            if match is not None:
                panel_paths[key] = match
            elif required:
                missing_required = True
                break

        if missing_required:
            continue

        meta = meta_map.get(folder, {})
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
        for key, label, _, _ in PANEL_SPECS
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


def write_manifest(samples: list[dict], out_path: Path) -> None:
    rows = []
    for sample in samples:
        base = {
            "folder": sample["folder"],
            "source": "original",
            "ms": sample.get("ms", ""),
            "image_dir": sample.get("image_dir", ""),
        }
        for panel_key, panel_path in sample["panels"].items():
            row = dict(base)
            row["panel"] = panel_key
            row["png"] = str(panel_path)
            rows.append(row)

    if not rows:
        return

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] wrote plot manifest to: {out_path}")


def main():
    extracted_dir = resolve_extracted_dir(SELECTED_FOLDERS)
    summary_name = "beam_imaging_summary.csv"
    figure_name = OUTPUT_FIGURE_NAME or "uv_lim.png"
    manifest_name = OUTPUT_MANIFEST_NAME or "plot_manifest.csv"
    summary_csv = extracted_dir / summary_name
    out_fig = extracted_dir / figure_name
    manifest_csv = extracted_dir / manifest_name

    print(f"[INFO] using extracted dir: {extracted_dir}")
    print(f"[INFO] summary: {summary_csv}")
    print(f"[INFO] output figure: {out_fig}")
    meta_map = load_meta_map(summary_csv)
    selfcal_metrics_map = load_selfcal_metrics_map(extracted_dir)
    samples = find_valid_samples(extracted_dir, meta_map, selfcal_metrics_map)
    samples = filter_samples(samples, SELECTED_FOLDERS)
    samples = sort_samples_by_residual_p995_abs_over_sigma(samples)
    print(f"[INFO] valid samples found: {len(samples)}")
    figure_title = CUSTOM_TITLE
    write_manifest(samples, manifest_csv)
    make_contact_sheet(samples, out_fig, samples_per_row=1, figure_title=figure_title)
