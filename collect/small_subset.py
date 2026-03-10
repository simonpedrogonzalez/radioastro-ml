from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord


# =============================================================================
# Config
# =============================================================================

IN_CSV = "vla_calibrators_one_hit.csv"

OUT_ROOT = Path("small_subset")
ALL_PLOTS_DIR = OUT_ROOT / "all_plots"
SMALL_PLOTS_DIR = OUT_ROOT / "small_plots"
SMALL_CSV = OUT_ROOT / "small_selection.csv"

SIZE_CUT_GB = 15.0  # selection threshold
sns.set_theme(style="whitegrid")


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs():
    ALL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SMALL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def load_df(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # numeric
    df["size_gb"] = pd.to_numeric(df.get("size_gb"), errors="coerce")
    df["gain_onsource_min"] = pd.to_numeric(df.get("gain_onsource_min"), errors="coerce")

    # date
    if "date_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)

    # coords
    try:
        sc = SkyCoord(
            df["ra"].astype(str).values,
            df["dec"].astype(str).values,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        df["ra_deg"] = sc.ra.deg
        df["dec_deg"] = sc.dec.deg
        df["ra_rad"] = sc.ra.wrap_at(180 * u.deg).radian  # for mollweide
        df["dec_rad"] = sc.dec.radian
    except Exception:
        df["ra_deg"] = np.nan
        df["dec_deg"] = np.nan
        df["ra_rad"] = np.nan
        df["dec_rad"] = np.nan

    return df


# =============================================================================
# Plots
# =============================================================================

def plot_file_size_hist(df: pd.DataFrame, outdir: Path, *, tag: str):
    plt.figure()
    sns.histplot(df["size_gb"].dropna(), bins=30)
    plt.xlabel("File size (GB)")
    plt.ylabel("Count")
    plt.title(f"File size distribution ({tag})")
    savefig(outdir / f"file_size_hist_{tag}.png")


def plot_file_size_hist_restricted(df: pd.DataFrame, outdir: Path, *, tag: str, xmax: float):
    d = df[df["size_gb"].notna() & (df["size_gb"] <= xmax)]
    plt.figure()
    sns.histplot(d["size_gb"], bins=30)
    plt.xlabel(f"File size (GB) (<= {xmax})")
    plt.ylabel("Count")
    plt.title(f"File size distribution (<= {xmax} GB) ({tag})")
    savefig(outdir / f"file_size_hist_le_{int(xmax)}gb_{tag}.png")


def plot_onsource_hist(df: pd.DataFrame, outdir: Path, *, tag: str):
    plt.figure()
    sns.histplot(df["gain_onsource_min"].dropna(), bins=30)
    plt.xlabel("Gain calibrator on-source time (min)")
    plt.ylabel("Count")
    plt.title(f"Gain calibrator on-source time ({tag})")
    savefig(outdir / f"onsource_hist_{tag}.png")


def plot_size_vs_onsource(df: pd.DataFrame, outdir: Path, *, tag: str):
    d = df[df["size_gb"].notna() & df["gain_onsource_min"].notna()]
    plt.figure()
    sns.scatterplot(data=d, x="gain_onsource_min", y="size_gb", s=30)
    sns.regplot(data=d, x="gain_onsource_min", y="size_gb", scatter=False)
    plt.xlabel("Gain calibrator on-source time (min)")
    plt.ylabel("File size (GB)")
    plt.title(f"File size vs on-source time ({tag})")
    savefig(outdir / f"scatter_size_vs_onsource_{tag}.png")


def plot_ra_dec_mollweide(df: pd.DataFrame, outdir: Path, *, tag: str):
    """
    Global sky map in a spherical-like projection (Mollweide).
    Good for seeing concentration patterns across the full sky.
    """
    d = df[df["ra_rad"].notna() & df["dec_rad"].notna()].copy()
    if d.empty:
        return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide")

    ax.scatter(d["ra_rad"].values, d["dec_rad"].values, s=12, alpha=0.8)

    ax.grid(True)
    ax.set_title(f"Sky positions (Mollweide, ICRS) ({tag})")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    # nicer RA tick labels (in hours)
    # Mollweide x ticks are in radians, centered at 0. Use custom hour labels.
    xticks = np.radians(np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]))
    ax.set_xticks(xticks)
    ax.set_xticklabels(["10h", "8h", "6h", "4h", "2h", "0h", "22h", "20h", "18h", "16h", "14h"])

    plt.tight_layout()
    plt.savefig(outdir / f"sky_mollweide_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_date_hist_yearly(df: pd.DataFrame, outdir: Path, *, tag: str):
    """
    Date histogram with big bins: 1 bin per year.
    """
    d = df[df["date_utc"].notna()].copy()
    if d.empty:
        return

    years = d["date_utc"].dt.year

    # build integer year bins [min_year..max_year]
    y0 = int(years.min())
    y1 = int(years.max())
    bins = np.arange(y0, y1 + 2)  # +2 so last year included

    plt.figure(figsize=(10, 4))
    sns.histplot(years, bins=bins)
    plt.xlabel("Year (UTC)")
    plt.ylabel("Count")
    plt.title(f"Observation year histogram ({tag})")
    plt.xticks(np.arange(y0, y1 + 1), rotation=0)
    savefig(outdir / f"date_hist_yearly_{tag}.png")


def plot_category_hist(df: pd.DataFrame, outdir: Path, col: str, *, tag: str):
    if col not in df.columns:
        return
    d = df[df[col].notna()].copy()
    if d.empty:
        return

    plt.figure(figsize=(8, 4))
    order = d[col].value_counts().index
    sns.countplot(data=d, x=col, order=order)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"{col} histogram ({tag})")
    plt.xticks(rotation=30, ha="right")
    savefig(outdir / f"hist_{col}_{tag}.png")


def plot_all_hists_grid(df: pd.DataFrame, outdir: Path, *, tag: str):
    """
    One figure with many lil plots:
      - size_gb
      - gain_onsource_min
      - yearly date hist
      - gain_array_config
      - band_code
      - band_guess
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    sns.histplot(df["size_gb"].dropna(), bins=25, ax=axes[0])
    axes[0].set_title("size_gb")
    axes[0].set_xlabel("GB")

    sns.histplot(df["gain_onsource_min"].dropna(), bins=25, ax=axes[1])
    axes[1].set_title("gain_onsource_min")
    axes[1].set_xlabel("min")

    ddate = df[df["date_utc"].notna()].copy()
    if not ddate.empty:
        years = ddate["date_utc"].dt.year
        y0 = int(years.min())
        y1 = int(years.max())
        bins = np.arange(y0, y1 + 2)
        sns.histplot(years, bins=bins, ax=axes[2])
        axes[2].set_xticks(np.arange(y0, y1 + 1))
    axes[2].set_title("date_utc (year)")
    axes[2].set_xlabel("year")

    if "gain_array_config" in df.columns:
        order = df["gain_array_config"].dropna().value_counts().index
        sns.countplot(data=df, x="gain_array_config", order=order, ax=axes[3])
        axes[3].tick_params(axis="x", rotation=30)
    axes[3].set_title("gain_array_config")

    if "band_code" in df.columns:
        order = df["band_code"].dropna().value_counts().index
        sns.countplot(data=df, x="band_code", order=order, ax=axes[4])
        axes[4].tick_params(axis="x", rotation=30)
    axes[4].set_title("band_code")

    if "band_guess" in df.columns:
        order = df["band_guess"].dropna().value_counts().index
        sns.countplot(data=df, x="band_guess", order=order, ax=axes[5])
        axes[5].tick_params(axis="x", rotation=30)
        axes[5].set_title("band_guess")
    else:
        axes[5].axis("off")

    fig.suptitle(f"Summary histograms ({tag})", y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / f"hists_grid_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(df: pd.DataFrame, outdir: Path, *, tag: str):
    plot_file_size_hist(df, outdir, tag=tag)
    plot_file_size_hist_restricted(df, outdir, tag=tag, xmax=SIZE_CUT_GB)
    plot_onsource_hist(df, outdir, tag=tag)
    plot_size_vs_onsource(df, outdir, tag=tag)
    plot_ra_dec_mollweide(df, outdir, tag=tag)
    plot_date_hist_yearly(df, outdir, tag=tag)
    plot_category_hist(df, outdir, "gain_array_config", tag=tag)
    plot_category_hist(df, outdir, "band_code", tag=tag)
    plot_all_hists_grid(df, outdir, tag=tag)


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dirs()

    df = load_df(IN_CSV)

    # 1) All plots
    make_all_plots(df, ALL_PLOTS_DIR, tag="all")

    # 2) Selection: all < 15 GB, sorted by on-source desc
    df_small = df[df["size_gb"].notna() & (df["size_gb"] < SIZE_CUT_GB)].copy()
    df_small = df_small.sort_values("gain_onsource_min", ascending=False)

    df_small.to_csv(SMALL_CSV, index=False)

    # 3) Plots for the small subset
    make_all_plots(df_small, SMALL_PLOTS_DIR, tag=f"lt_{int(SIZE_CUT_GB)}gb_sorted_by_onsource")

    print(f"[OK] Wrote: {SMALL_CSV}")
    print(f"[OK] Plots: {ALL_PLOTS_DIR} and {SMALL_PLOTS_DIR}")
    print(f"[OK] Small selection rows: {len(df_small)}")


if __name__ == "__main__":
    main()