from pathlib import Path
from math import ceil

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PROJECT_LIST = Path("/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv")
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
OUT_FIG = EXTRACTED_DIR / "all_samples_contact_sheet.png"


def load_minutes_map(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    # keep only rows that have a folder name
    df = df[df["folder"].notna()].copy()

    minutes_map = {}
    for _, row in df.iterrows():
        folder = str(row["folder"]).strip()
        if not folder:
            continue

        minutes = row.get("extracted_gain_onsource_min", None)
        name = str(row.get("name", folder)).strip() or folder
        status = str(row.get("status", "")).strip().lower()

        minutes_map[folder] = {
            "minutes": minutes,
            "name": name,
            "status": status,
        }

    return minutes_map


def find_valid_samples(extracted_dir: Path, minutes_map: dict):
    """
    Return a list of dicts:
      {
        "folder": folder_name,
        "name": display_name,
        "minutes": float or nan,
        "clean": Path(..._corrected_clean.png),
        "dirty": Path(..._corrected_dirty.png),
      }
    """
    samples = []

    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        folder = sample_dir.name

        # expected outputs from your pipeline
        clean_png = sample_dir / "clean_corrected_clean.png"
        dirty_png = sample_dir / "clean_corrected_dirty.png"

        # fallback in case exact names differ
        if not clean_png.exists():
            hits = sorted(sample_dir.glob("*_corrected_clean.png"))
            if hits:
                clean_png = hits[0]

        if not dirty_png.exists():
            hits = sorted(sample_dir.glob("*_corrected_dirty.png"))
            if hits:
                dirty_png = hits[0]

        # "correctly extracted thing" = has both images
        if not (clean_png.exists() and dirty_png.exists()):
            continue

        meta = minutes_map.get(folder, {})
        samples.append(
            {
                "folder": folder,
                "name": meta.get("name", folder),
                "minutes": meta.get("minutes", float("nan")),
                "status": meta.get("status", ""),
                "clean": clean_png,
                "dirty": dirty_png,
            }
        )

    return samples


def fmt_minutes(x) -> str:
    try:
        x = float(x)
        if pd.notna(x):
            return f"{x:.1f} min"
    except Exception:
        pass
    return "min ?"


def make_contact_sheet(samples, out_path: Path, samples_per_row: int = 3):
    """
    Layout:
      6 columns total = (clean, dirty) x 3 samples per row
      each sample occupies 2 columns
    """
    if not samples:
        print("[INFO] No valid samples found.")
        return

    n = len(samples)
    nrows = ceil(n / samples_per_row)
    ncols = samples_per_row * 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
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

        title = f'{sample["name"]}\n{fmt_minutes(sample["minutes"])}'
        ax_clean.set_title(title, fontsize=10)
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
    minutes_map = load_minutes_map(PROJECT_LIST)
    samples = find_valid_samples(EXTRACTED_DIR, minutes_map)
    print(f"[INFO] valid samples found: {len(samples)}")
    make_contact_sheet(samples, OUT_FIG, samples_per_row=3)
