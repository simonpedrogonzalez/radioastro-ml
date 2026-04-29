from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OLD_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_bands.csv")
NEW_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_bands_v2.csv")
OUTDIR = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_v2_review")

KEY_COLS = ["name", "frame", "ra", "dec", "band_idx"]
UV_COLS = ["uvmin_kl", "uvmax_kl"]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(KEY_COLS).reset_index(drop=True)


def _norm_value(x):
    if pd.isna(x):
        return None
    return x


def compare_non_uv_columns(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    merged = old_df.merge(new_df, on=KEY_COLS, how="outer", suffixes=("_old", "_new"), indicator=True)
    diffs = []

    base_cols = [c for c in old_df.columns if c not in KEY_COLS + UV_COLS]
    for _, row in merged.iterrows():
        if row["_merge"] != "both":
            diffs.append(
                {
                    **{k: row.get(k) for k in KEY_COLS},
                    "column": "__row_presence__",
                    "old": row["_merge"],
                    "new": row["_merge"],
                }
            )
            continue

        for col in base_cols:
            old_val = _norm_value(row.get(f"{col}_old"))
            new_val = _norm_value(row.get(f"{col}_new"))
            if old_val != new_val:
                diffs.append(
                    {
                        **{k: row.get(k) for k in KEY_COLS},
                        "column": col,
                        "old": old_val,
                        "new": new_val,
                    }
                )

    return pd.DataFrame(diffs)


def build_uv_review(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    merged = old_df.merge(new_df, on=KEY_COLS, how="outer", suffixes=("_old", "_new"), indicator=True)
    mask = (
        merged["uvmin_kl_old"].notna()
        | merged["uvmax_kl_old"].notna()
        | merged["uvmin_kl_new"].notna()
        | merged["uvmax_kl_new"].notna()
    )
    uv_df = merged.loc[mask].copy()
    keep_cols = (
        KEY_COLS
        + [
            "wavelength_old", "receiver_old", "cfg_A_old", "cfg_B_old", "cfg_C_old", "cfg_D_old",
            "uvmin_kl_old", "uvmax_kl_old",
            "wavelength_new", "receiver_new", "cfg_A_new", "cfg_B_new", "cfg_C_new", "cfg_D_new",
            "uvmin_kl_new", "uvmax_kl_new",
        ]
    )
    uv_df = uv_df[keep_cols].sort_values(KEY_COLS).reset_index(drop=True)
    return uv_df


def config_string(row: pd.Series, suffix: str) -> str:
    vals = [row.get(f"cfg_{cfg}_{suffix}") for cfg in ["A", "B", "C", "D"]]
    vals = ["-" if pd.isna(v) else str(v) for v in vals]
    return f"A={vals[0]} B={vals[1]} C={vals[2]} D={vals[3]}"


def make_review_pages(uv_df: pd.DataFrame, outdir: Path, rows_per_page: int = 28) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    if uv_df.empty:
        return []

    paths = []
    n_pages = ceil(len(uv_df) / rows_per_page)
    for page in range(n_pages):
        start = page * rows_per_page
        end = min(len(uv_df), start + rows_per_page)
        chunk = uv_df.iloc[start:end]

        fig_h = max(8.0, 0.42 * len(chunk) + 1.6)
        fig, ax = plt.subplots(figsize=(18, fig_h))
        ax.axis("off")

        y = 0.98
        ax.text(
            0.01,
            y,
            f"V1 vs V2 uv-limit review (page {page + 1}/{n_pages})",
            fontsize=14,
            fontweight="bold",
            va="top",
            family="monospace",
        )
        y -= 0.05

        for _, row in chunk.iterrows():
            old_band = row.get("wavelength_old")
            new_band = row.get("wavelength_new")
            receiver = row.get("receiver_new") if pd.notna(row.get("receiver_new")) else row.get("receiver_old")
            header = (
                f"{row['name']} | band_idx={int(row['band_idx'])} | "
                f"band={new_band if pd.notna(new_band) else old_band} | receiver={receiver}"
            )
            old_line = (
                f"old  {config_string(row, 'old')} | "
                f"uvmin={row.get('uvmin_kl_old')} uvmax={row.get('uvmax_kl_old')}"
            )
            new_line = (
                f"new  {config_string(row, 'new')} | "
                f"uvmin={row.get('uvmin_kl_new')} uvmax={row.get('uvmax_kl_new')}"
            )
            ax.text(0.01, y, header, fontsize=9.8, va="top", family="monospace")
            y -= 0.028
            ax.text(0.03, y, old_line, fontsize=9.3, va="top", family="monospace", color="dimgray")
            y -= 0.024
            ax.text(0.03, y, new_line, fontsize=9.3, va="top", family="monospace", color="darkgreen")
            y -= 0.03

        out_path = outdir / f"uv_review_page_{page + 1:03d}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    old_df = load_csv(OLD_CSV)
    new_df = load_csv(NEW_CSV)

    non_uv_diff = compare_non_uv_columns(old_df, new_df)
    uv_review = build_uv_review(old_df, new_df)

    non_uv_diff_path = OUTDIR / "non_uv_differences.csv"
    uv_review_path = OUTDIR / "uv_review.csv"
    non_uv_diff.to_csv(non_uv_diff_path, index=False)
    uv_review.to_csv(uv_review_path, index=False)

    page_paths = make_review_pages(uv_review, OUTDIR / "pages")

    print(f"[OK] wrote {non_uv_diff_path}")
    print(f"[OK] wrote {uv_review_path}")
    if non_uv_diff.empty:
        print("[OK] old/new CSVs match on all non-uv columns")
    else:
        print(f"[WARN] found {len(non_uv_diff)} non-uv differences")
    print(f"[OK] wrote {len(page_paths)} review page(s)")


if __name__ == "__main__":
    main()
