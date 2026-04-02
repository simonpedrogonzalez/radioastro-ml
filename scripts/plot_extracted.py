from pathlib import Path
from math import ceil

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
SUMMARY_CSV = EXTRACTED_DIR / "beam_imaging_summary.csv"
OUT_FIG = EXTRACTED_DIR / "all_samples_contact_sheet.png"


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
            "catalog_uvrange": str(row.get("catalog_uvrange", "")).strip(),
            "catalog_uvmin_kl": row.get("catalog_uvmin_kl"),
            "catalog_uvmax_kl": row.get("catalog_uvmax_kl"),
            "uv_fraction_inside_limits": row.get("uv_fraction_inside_limits"),
            "cell_arcsec": row.get("cell_arcsec"),
            "imsize": row.get("imsize"),
            "fov_arcsec": row.get("fov_arcsec"),
            "fov_in_beams_minor": row.get("fov_in_beams_minor"),
            "pixels_per_beam_minor": row.get("pixels_per_beam_minor"),
            "status_csv": str(row.get("status_csv", "")).strip(),
        }

    return meta_map


def find_valid_samples(extracted_dir: Path, meta_map: dict):
    samples = []

    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        folder = sample_dir.name

        clean_png = sample_dir / "clean_corrected_clean.png"
        dirty_png = sample_dir / "clean_corrected_dirty.png"

        if not clean_png.exists():
            hits = sorted(sample_dir.glob("*_corrected_clean.png"))
            if hits:
                clean_png = hits[0]

        if not dirty_png.exists():
            hits = sorted(sample_dir.glob("*_corrected_dirty.png"))
            if hits:
                dirty_png = hits[0]

        if not (clean_png.exists() and dirty_png.exists()):
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
                "clean": clean_png,
                "dirty": dirty_png,
            }
        )

    return samples


def fmt_float(x, fmt: str, fallback: str = "?") -> str:
    try:
        x = float(x)
        if pd.notna(x):
            return format(x, fmt)
    except Exception:
        pass
    return fallback


def make_title(i:int, sample: dict) -> str:
    name = sample["name"]

    beam_maj = fmt_float(sample["beam_major_arcsec"], ".2f")
    beam_min = fmt_float(sample["beam_minor_arcsec"], ".2f")
    beam_pa = fmt_float(sample["beam_pa_deg"], ".1f")
    band = str(sample.get("band_used_for_firstpass", "")).strip() or "?"
    config = str(sample.get("gain_array_config", "")).strip() or "?"
    uvrange = str(sample.get("catalog_uvrange", "")).strip() or "all"
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

    return (
        f"# {i}\n"
        f"{name}\n"
        f"beam={beam_maj}\"x{beam_min}\"  pa={beam_pa}°\n"
        f"band={band}  config={config}\n"
        f"uv={uvrange}  in={uv_inside}%  [{uvmin},{uvmax}] kl\n"
        f"cell={cell}\"/pix  ppb={ppb}\n"
        f"FoV={fov_arcsec}\"  ({fov_beams} beams)\n"
        f"time={minutes} min"
    )


def make_contact_sheet(samples, out_path: Path, samples_per_row: int = 3):
    if not samples:
        print("[INFO] No valid samples found.")
        return

    n = len(samples)
    nrows = ceil(n / samples_per_row)
    ncols = samples_per_row * 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 5.2 * nrows),
        squeeze=False,
    )

    for axrow in axes:
        for ax in axrow:
            ax.axis("off")

    for i, sample in enumerate(samples):
        row = i // samples_per_row
        slot = i % samples_per_row
        ccol = slot * 2
        dcol = ccol + 1

        ax_clean = axes[row, ccol]
        ax_dirty = axes[row, dcol]

        img_clean = mpimg.imread(sample["clean"])
        img_dirty = mpimg.imread(sample["dirty"])

        ax_clean.imshow(img_clean)
        ax_dirty.imshow(img_dirty)

        ax_clean.axis("off")
        ax_dirty.axis("off")

        ax_clean.set_title(make_title(i, sample), fontsize=9)
        ax_dirty.set_title("dirty", fontsize=10)

    fig.suptitle(
        f"Extracted samples overview ({len(samples)} samples)",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved figure to: {out_path}")


def main():
    meta_map = load_meta_map(SUMMARY_CSV)
    samples = find_valid_samples(EXTRACTED_DIR, meta_map)
    print(f"[INFO] valid samples found: {len(samples)}")
    make_contact_sheet(samples, OUT_FIG, samples_per_row=1)
