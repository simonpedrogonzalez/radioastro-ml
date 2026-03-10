# corrupt_done_projects.py
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from casatasks import gaincal, applycal, rmtables
from casatools import table

from scripts.io_utils import copy_ms, col_diff
from scripts.corruption import AntennaGainCorruption
from scripts.corrfn import fBM
from scripts.timegrid import TimeGrid
from scripts.corrtab_utils import GTabQuery, GCOLS, get_unflagged_antennas
from scripts.img_utils import make_clean, make_frac_residuals, fracres_before_after_png


PROJECT_LIST = "/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv"
OUT_DIR = Path("corruption_runs")
REFANT = "ea21"

TCLEAN_KW = dict(
    specmode="mfs",
    imsize=512,
    cell="0.5arcsec",
    weighting="briggs",
    robust=0.5,
    stokes="I",
)


def load_projects(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def put_corrected_into_data(ms_path: str | Path) -> None:
    tb = table()
    tb.open(str(ms_path), nomodify=False)
    tb.putcol("DATA", tb.getcol("CORRECTED_DATA"))
    tb.close()


def process_row(row: pd.Series) -> None:
    name = str(row["name"]).strip()
    folder = str(row["folder"]).strip()
    ms_src = Path(str(row["extracted_ms"]).strip())
    field = str(row["gain_calibrator_name"]).strip()
    spw = str(row.get("spw_selected", "0")).strip() or "0"

    if not ms_src.exists():
        print(f"[SKIP] {name}: extracted_ms not found -> {ms_src}")
        return

    work = OUT_DIR / folder
    imgdir = work / "images"
    work.mkdir(parents=True, exist_ok=True)
    imgdir.mkdir(parents=True, exist_ok=True)

    ms_in = work / f"{folder}.recal.ms"
    ms_out = work / f"{folder}.corrupt.ms"
    ms_rec = work / f"{folder}.recover.ms"

    gtab_base = str(work / f"{folder}.initial-recalibration.G")
    gtab_inj = str(work / f"{folder}.Gcorr")
    gtab_sol = str(work / f"{folder}.recover.G")

    print(f"\n[PROCESS] {name}")

    # 1) copy extracted ms -> recal copy
    if ms_in.exists():
        rmtables(str(ms_in))
        shutil.rmtree(ms_in, ignore_errors=True)
    copy_ms(str(ms_src), str(ms_in))

    # 2) recalibrate on copy
    rmtables(gtab_base)
    gaincal(
        vis=str(ms_in),
        caltable=gtab_base,
        field=field,
        spw=spw,
        refant=REFANT,
        refantmode="strict",
        gaintype="G",
        calmode="p",
        solint="int",
        combine="scan",
        minsnr=10,
        solnorm=False,
        smodel=[1, 0, 0, 0],
    )
    applycal(
        vis=str(ms_in),
        field=field,
        spw=spw,
        gaintable=[gtab_base],
        interp=["linear"],
        calwt=False,
    )
    put_corrected_into_data(ms_in)

    # 3) copy recal ms -> corrupt ms
    if ms_out.exists():
        rmtables(str(ms_out))
        shutil.rmtree(ms_out, ignore_errors=True)
    copy_ms(str(ms_in), str(ms_out))

    ants, _ = get_unflagged_antennas(str(ms_in))
    ant_ids = list(ants.values())[:2]

    rmtables(gtab_inj)
    AntennaGainCorruption(
        timegrid=TimeGrid(solint="10m", interp="linear"),
        amp_fn=None,
        query=GTabQuery().where_in(GCOLS.ANTENNA1, ant_ids).group_by([GCOLS.ANTENNA1]),
        phase_fn=fBM(max_amp=0.15 * np.pi, H=0.05),
    ).build_corrtable(str(ms_out), gtab_inj, seed=0).apply_corrtable(str(ms_out), gtab_inj)

    print("[DIFF] in vs corrupt")
    col_diff(str(ms_in), str(ms_out))

    # 4) solve gains back
    rmtables(gtab_sol)
    gaincal(
        vis=str(ms_out),
        caltable=gtab_sol,
        field=field,
        spw=spw,
        refant=REFANT,
        refantmode="strict",
        gaintype="G",
        calmode="p",
        solint="int",
        combine="scan",
        minsnr=10,
        solnorm=False,
        smodel=[1, 0, 0, 0],
        interp=["linear"],
    )

    # 5) recovered ms
    if ms_rec.exists():
        rmtables(str(ms_rec))
        shutil.rmtree(ms_rec, ignore_errors=True)
    copy_ms(str(ms_out), str(ms_rec))

    applycal(
        vis=str(ms_rec),
        field=field,
        spw=spw,
        gaintable=[gtab_sol],
        interp=["linear"],
        calwt=False,
    )
    put_corrected_into_data(ms_rec)

    print("[DIFF] corrupt vs recovered")
    col_diff(str(ms_out), str(ms_rec))
    print("[DIFF] recalibrated vs recovered")
    col_diff(str(ms_in), str(ms_rec))

    # 6) fracres plots
    img_base = str(work / "img_base_clean")
    img_corr = str(work / "img_corrupted_clean")
    img_rec = str(work / "img_recovered_clean")

    make_clean(str(ms_in), img_base, TCLEAN_KW)
    make_clean(str(ms_out), img_corr, TCLEAN_KW)
    make_clean(str(ms_rec), img_rec, TCLEAN_KW)

    frac_base = str(work / "img_base_fracres")
    frac_corr = str(work / "img_corrupted_fracres")
    frac_rec = str(work / "img_recovered_fracres")

    make_frac_residuals(
        residual_im=img_base,
        reference_im=img_base,
        out_im=frac_base,
    )
    make_frac_residuals(
        residual_im=img_corr,
        reference_im=img_base,
        out_im=frac_corr,
    )
    make_frac_residuals(
        residual_im=img_rec,
        reference_im=img_base,
        out_im=frac_rec,
    )

    fracres_before_after_png(
        before_im=f"{frac_base}.image",
        after_im=f"{frac_corr}.image",
        out_png=str(imgdir / "fracres_base_corrupted.png"),
        crop_half=64,
    )
    fracres_before_after_png(
        before_im=f"{frac_base}.image",
        after_im=f"{frac_rec}.image",
        out_png=str(imgdir / "fracres_base_recovered.png"),
        crop_half=64,
    )
    fracres_before_after_png(
        before_im=f"{frac_corr}.image",
        after_im=f"{frac_rec}.image",
        out_png=str(imgdir / "fracres_corrupted_recovered.png"),
        crop_half=64,
    )

    print("[DONE]")
    print(f"  corrupted MS : {ms_out}")
    print(f"  recovered MS : {ms_rec}")
    print(f"  images dir   : {imgdir}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_projects(PROJECT_LIST)

    done = df[df["status"].astype(str).str.strip().str.lower() == "done"].copy()
    print(f"[INFO] done rows = {len(done)}")

    for _, row in done.iterrows():
        try:
            process_row(row)
        except Exception as e:
            print(f"[ERROR] {row.get('name', '<unknown>')}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()