from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def build_wget_command(url: str, folder: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    return f"wget -c -P {folder} '{url}'"


def add_processing_cols(input_csv: str | Path, output_csv: str | Path | None = None):
    input_csv = Path(input_csv)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_with_processing.csv")
    else:
        output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    # --- new columns ---
    df["folder"] = ""

    df["wget_command"] = ""

    df["status"] = ""   # to be filled later during processing
    df["new_size"] = None   # placeholder for post-download size

    df.to_csv(output_csv, index=False)

    print(f"[OK] Wrote: {output_csv}")


if __name__ == "__main__":

    input_csv = "small_subset/small_selection.csv"
    output_csv = "small_subset/small_selection.csv"

    add_processing_cols(input_csv, output_csv)