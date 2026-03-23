from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from casatasks import tclean, flagdata, imhead
from scripts.img_utils import casa_image_to_png


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_LIST = Path("/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv")
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")

# -------------------------------------------------------------------
# Imaging policy
# -------------------------------------------------------------------
MODIFY_MS_IN_PLACE = True

# first-pass dirty image only to estimate beam
FIRSTPASS_CELL_ARCSEC = 0.5
FIRSTPASS_IMSIZE = 256

# beam-normalized final image policy
PIXELS_PER_BEAM = 4.0
FOV_IN_BEAMS = 64.0

# optional guardrails
MIN_IMSIZE = 128
MAX_IMSIZE = 1024
MIN_CELL_ARCSEC = 0.02
MAX_CELL_ARCSEC = 50.0

# clean config
TCLEAN_BASE = dict(
    specmode="mfs",
    weighting="briggs",
    robust=0.5,
    stokes="I",
    deconvolver="hogbom",
    gridder="standard",
    interactive=False,
)

FIRSTPASS_NITER = 0
FINAL_CLEAN_NITER = 100


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_projects(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


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

        ms_list = sorted(sample_dir.rglob("*.ms"))
        if ms_list:
            hits.append(ms_list[0])

    return hits


def row_for_folder(df: pd.DataFrame, folder_name: str) -> Optional[pd.Series]:
    if "folder" not in df.columns:
        return None
    m = df["folder"].astype("string").fillna("").str.strip() == folder_name
    if not m.any():
        return None
    return df.loc[m].iloc[0]


def remove_casa_products(imagename: str) -> None:
    stem = Path(imagename)
    parent = stem.parent
    prefix = stem.name

    if not parent.exists():
        return

    for p in parent.glob(prefix + ".*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except OSError:
                pass


def ensure_imaging_ms(ms_path: Path, modify_in_place: bool = True) -> Path:
    if modify_in_place:
        return ms_path

    out_ms = ms_path.with_name(ms_path.stem + "_imgprep.ms")
    if out_ms.exists():
        shutil.rmtree(out_ms)
    shutil.copytree(ms_path, out_ms)
    return out_ms


def flag_zero_visibilities(ms_path: Path) -> None:
    print(f"[FLAG] flagging exact zeros in {ms_path}")
    flagdata(
        vis=str(ms_path),
        mode="clip",
        clipzeros=True,
        flagbackup=False,
    )


def run_tclean(
    ms_path: Path,
    imagename: str,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
) -> None:
    remove_casa_products(imagename)

    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=imagename,
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
    )

    print(
        f"[TCLEAN] vis={ms_path.name} "
        f"imagename={imagename} "
        f"cell={cell_arcsec:.6f}arcsec "
        f"imsize={imsize} "
        f"niter={niter}"
    )
    tclean(**cfg)


def _beam_value_to_arcsec(x) -> float:
    if isinstance(x, dict):
        value = float(x["value"])
        unit = str(x.get("unit", "")).strip().lower()

        if unit in ("arcsec", "arcseconds", "asec"):
            return value
        if unit in ("arcmin", "arcminutes"):
            return value * 60.0
        if unit in ("deg", "degree", "degrees"):
            return value * 3600.0

        raise ValueError(f"Unsupported beam unit: {unit!r}")

    return float(x)


def read_restoring_beam_arcsec(image_path: str | Path) -> Tuple[float, float, float]:
    info = imhead(imagename=str(image_path), mode="summary")
    if not isinstance(info, dict):
        raise RuntimeError(f"imhead summary failed for {image_path}")

    rb = info.get("restoringbeam")
    if rb is None:
        raise RuntimeError(f"No restoringbeam found in {image_path}")

    bmaj = _beam_value_to_arcsec(rb["major"])
    bmin = _beam_value_to_arcsec(rb["minor"])

    pa = rb.get("positionangle", {})
    if isinstance(pa, dict):
        pa_deg = float(pa.get("value", np.nan))
        pa_unit = str(pa.get("unit", "deg")).strip().lower()
        if pa_unit.startswith("rad"):
            pa_deg = np.rad2deg(pa_deg)
    else:
        pa_deg = float(pa)

    return bmaj, bmin, pa_deg


def choose_beam_based_cell_arcsec(
    bmaj_arcsec: float,
    bmin_arcsec: float,
    pixels_per_beam: float = PIXELS_PER_BEAM,
) -> float:
    beam_ref = min(bmaj_arcsec, bmin_arcsec)
    cell = beam_ref / pixels_per_beam
    cell = max(MIN_CELL_ARCSEC, min(MAX_CELL_ARCSEC, cell))
    return cell


def choose_imsize_for_beam_normalized_fov(
    bmin_arcsec: float,
    cell_arcsec: float,
    fov_in_beams: float = FOV_IN_BEAMS,
) -> tuple[int, float]:
    target_fov_arcsec = fov_in_beams * bmin_arcsec
    imsize = int(np.ceil(target_fov_arcsec / cell_arcsec))

    if imsize % 2 == 1:
        imsize += 1

    imsize = max(MIN_IMSIZE, min(MAX_IMSIZE, imsize))
    actual_fov_arcsec = imsize * cell_arcsec
    return imsize, actual_fov_arcsec


def image_to_png_if_exists(casa_image: Path, png_path: Path) -> None:
    if casa_image.exists():
        casa_image_to_png(str(casa_image), str(png_path))


def process_one_sample(ms_path: Path) -> dict:
    sample_leaf = ms_path.parent
    sample_top = sample_leaf.parent
    folder_name = sample_top.name

    ms_for_imaging = ensure_imaging_ms(ms_path, modify_in_place=MODIFY_MS_IN_PLACE)

    # 1) flag zeros
    flag_zero_visibilities(ms_for_imaging)

    # 2) first-pass dirty image to estimate beam
    firstpass_base = sample_top / "beam_firstpass_dirty"
    run_tclean(
        ms_for_imaging,
        str(firstpass_base),
        cell_arcsec=FIRSTPASS_CELL_ARCSEC,
        imsize=FIRSTPASS_IMSIZE,
        niter=FIRSTPASS_NITER,
    )

    firstpass_image = firstpass_base.with_suffix(".image")
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)

    # 3) beam-based cell, beam-normalized FoV
    final_cell = choose_beam_based_cell_arcsec(
        bmaj, bmin, pixels_per_beam=PIXELS_PER_BEAM
    )
    final_imsize, final_fov_arcsec = choose_imsize_for_beam_normalized_fov(
        bmin, final_cell, fov_in_beams=FOV_IN_BEAMS
    )

    # 4) final dirty
    dirty_base = sample_top / "clean_corrected_dirty"
    run_tclean(
        ms_for_imaging,
        str(dirty_base),
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=0,
    )

    # 5) final clean
    clean_base = sample_top / "clean_corrected_clean"
    run_tclean(
        ms_for_imaging,
        str(clean_base),
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=FINAL_CLEAN_NITER,
    )

    # 6) export pngs
    image_to_png_if_exists(dirty_base.with_suffix(".image"), sample_top / "clean_corrected_dirty.png")
    image_to_png_if_exists(clean_base.with_suffix(".image"), sample_top / "clean_corrected_clean.png")

    result = {
        "folder": folder_name,
        "ms": str(ms_path),
        "beam_major_arcsec": bmaj,
        "beam_minor_arcsec": bmin,
        "beam_pa_deg": bpa,
        "cell_arcsec": final_cell,
        "imsize": final_imsize,
        "fov_arcsec": final_fov_arcsec,
        "fov_in_beams_minor": final_fov_arcsec / bmin if bmin > 0 else np.nan,
        "pixels_per_beam_minor": bmin / final_cell if final_cell > 0 else np.nan,
    }

    print(
        f"[DONE] {folder_name} | "
        f"beam=({bmaj:.3f}\", {bmin:.3f}\", {bpa:.1f} deg) | "
        f"cell={final_cell:.4f}\" | imsize={final_imsize} | "
        f"FoV={final_fov_arcsec:.2f}\" | "
        f"FoV/beams={final_fov_arcsec / bmin:.2f}"
    )

    return result


def main():
    df = load_projects(PROJECT_LIST)
    ms_paths = find_extracted_ms_paths(EXTRACTED_DIR)

    print(f"[INFO] found {len(ms_paths)} extracted MS files")

    rows = []
    for ms_path in ms_paths:
        try:
            out = process_one_sample(ms_path)

            folder = ms_path.parent.parent.name
            row = row_for_folder(df, folder)
            if row is not None:
                out["name"] = str(row.get("name", folder))
                out["minutes"] = row.get("extracted_gain_onsource_min", np.nan)
                out["status_csv"] = str(row.get("status", ""))
            else:
                out["name"] = folder
                out["minutes"] = np.nan
                out["status_csv"] = ""

            rows.append(out)

        except Exception as e:
            print(f"[ERROR] {ms_path}: {type(e).__name__}: {e}")
            rows.append(
                {
                    "folder": ms_path.parent.parent.name,
                    "ms": str(ms_path),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    summary_csv = EXTRACTED_DIR / "beam_imaging_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[OK] wrote summary to {summary_csv}")


# if __name__ == "__main__":
#     main()