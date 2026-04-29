from __future__ import annotations

import shutil
from math import ceil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casatasks import (
    applycal,
    clearcal,
    flagdata,
    gaincal,
    imhead,
    imstat,
    tclean,
)
from scripts.img_utils import casa_image_to_png


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")

FIRSTPASS_CELL_ARCSEC = 0.5
FIRSTPASS_IMSIZE = 256

PIXELS_PER_BEAM = 4.0
FOV_IN_BEAMS = 64.0

MIN_CELL_ARCSEC = 0.02
MAX_CELL_ARCSEC = 20.0
MIN_IMSIZE = 128
MAX_IMSIZE = 1024

OUTDIR_NAME = "selfcal_test"
PLOT_COLS = 2

TCLEAN_BASE = dict(
    specmode="mfs",
    weighting="briggs",
    robust=0.5,
    stokes="I",
    deconvolver="hogbom",
    gridder="standard",
    interactive=False,
)

INITIAL_CLEAN_NITER = 1000
POSTSELFCAL_CLEAN_NITER = 1000
PHASE_MINSNR = 3.0
PHASE_GAINTYPE = "G"
PHASE_CALMODE = "p"

USE_UVMODEL_DATA_COLUMN = "corrected"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def find_sample_ms(folder_name: str) -> tuple[Path, Path]:
    sample_top = EXTRACTED_DIR / folder_name
    if not sample_top.exists():
        raise FileNotFoundError(f"Folder not found: {sample_top}")

    expected_ms = sample_top / folder_name / f"{folder_name}.ms"
    if expected_ms.exists():
        print(f"[INFO] using expected MS: {expected_ms}")
        return sample_top, expected_ms

    ms_hits = sorted(sample_top.rglob("*.ms"))
    if not ms_hits:
        raise FileNotFoundError(f"No .ms found under: {sample_top}")

    print("[WARN] expected MS path not found, falling back to first .ms found")
    for p in ms_hits:
        print(f"    candidate: {p}")
    print(f"[INFO] using fallback MS: {ms_hits[0]}")
    return sample_top, ms_hits[0]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def remove_casa_products(imagename: str) -> None:
    stem = Path(imagename)
    parent = stem.parent
    prefix = stem.name
    if not parent.exists():
        return
    for p in parent.glob(prefix + ".*"):
        remove_path(p)


def copy_ms(ms_path: Path, out_ms: Path) -> Path:
    if out_ms.exists():
        shutil.rmtree(out_ms)
    print(f"[COPY] {ms_path} -> {out_ms}")
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


def read_restoring_beam_arcsec(image_path: str | Path) -> tuple[float, float, float]:
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
    return imsize, imsize * cell_arcsec


def choose_multiscale_scales(cell_arcsec: float, bmaj_arcsec: float, bmin_arcsec: float) -> list[int]:
    beam_pix = max(1, int(round(min(bmaj_arcsec, bmin_arcsec) / cell_arcsec)))
    s1 = max(1, beam_pix)
    s2 = max(s1 + 1, 3 * beam_pix)
    return [0, s1, s2]


def export_png(casa_image: Path, out_png: Path) -> None:
    if not casa_image.exists():
        raise FileNotFoundError(f"CASA image missing: {casa_image}")
    if out_png.exists():
        out_png.unlink()
    casa_image_to_png(str(casa_image), str(out_png), symmetric=False, cmap="inferno")


def image_stats(image_path: Path) -> dict:
    s = imstat(imagename=str(image_path))
    peak = float(np.nanmax(s["max"])) if "max" in s else np.nan
    rms = float(np.nanmax(s["rms"])) if "rms" in s else np.nan
    dyn = peak / rms if np.isfinite(peak) and np.isfinite(rms) and rms > 0 else np.nan
    return {
        "peak_jy_per_beam": peak,
        "rms_jy_per_beam": rms,
        "dynrange": dyn,
    }


def run_tclean(
    ms_path: Path,
    imagename: str,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    datacolumn: str = "corrected",
    weighting: str = "briggs",
    robust: float = 0.5,
    deconvolver: str = "hogbom",
    scales: list[int] | None = None,
    usemask: str = "auto-multithresh",
    mask: str = "",
) -> None:
    remove_casa_products(imagename)

    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=imagename,
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        datacolumn=datacolumn,
        weighting=weighting,
        deconvolver=deconvolver,
    )

    if weighting == "briggs":
        cfg["robust"] = robust

    if deconvolver == "multiscale":
        cfg["scales"] = [0, 5, 15] if scales is None else scales

    if usemask:
        cfg["usemask"] = usemask
    if mask:
        cfg["mask"] = mask

    print(
        f"[TCLEAN] {Path(imagename).name} | cell={cell_arcsec:.4f}\" | imsize={imsize} | "
        f"niter={niter} | datacolumn={datacolumn} | weighting={weighting} | deconvolver={deconvolver}"
    )
    tclean(**cfg)


def estimate_grid(ms_path: Path, outdir: Path) -> tuple[float, int, float, float, float]:
    firstpass_base = outdir / "beam_firstpass_dirty"
    run_tclean(
        ms_path,
        str(firstpass_base),
        cell_arcsec=FIRSTPASS_CELL_ARCSEC,
        imsize=FIRSTPASS_IMSIZE,
        niter=0,
        datacolumn="corrected",
        usemask="",
    )

    firstpass_image = firstpass_base.with_suffix(".image")
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)

    cell = choose_beam_based_cell_arcsec(bmaj, bmin)
    imsize, fov_arcsec = choose_imsize_for_beam_normalized_fov(bmin, cell)

    print(
        f"[GRID] beam=({bmaj:.3f}\", {bmin:.3f}\", pa={bpa:.1f} deg) | "
        f"cell={cell:.4f}\" | imsize={imsize} | FoV={fov_arcsec:.2f}\""
    )
    return cell, imsize, bmaj, bmin, bpa


def central_mask_string(imsize: int, radius_frac: float = 0.12) -> str:
    cx = imsize // 2
    cy = imsize // 2
    r = max(6, int(round(imsize * radius_frac)))
    return f"circle[[{cx}pix,{cy}pix],{r}pix]"


def build_model_image(
    ms_path: Path,
    outdir: Path,
    key: str,
    cell: float,
    imsize: int,
    *,
    weighting: str = "briggs",
    robust: float = 0.5,
    deconvolver: str = "hogbom",
    scales: list[int] | None = None,
    usemask: str = "auto-multithresh",
    mask: str = "",
    niter: int = INITIAL_CLEAN_NITER,
) -> Path:
    base = outdir / f"{key}_model"
    run_tclean(
        ms_path,
        str(base),
        cell_arcsec=cell,
        imsize=imsize,
        niter=niter,
        datacolumn=USE_UVMODEL_DATA_COLUMN,
        weighting=weighting,
        robust=robust,
        deconvolver=deconvolver,
        scales=scales,
        usemask=usemask,
        mask=mask,
    )
    return base


def run_phase_selfcal(ms_path: Path, caltable: Path, *, solint: str) -> None:
    remove_path(caltable)

    gaincal(
        vis=str(ms_path),
        caltable=str(caltable),
        gaintype=PHASE_GAINTYPE,
        calmode=PHASE_CALMODE,
        solint=solint,
        minsnr=PHASE_MINSNR,
        refant="",
    )

    applycal(
        vis=str(ms_path),
        gaintable=[str(caltable)],
        interp=["linear"],
        calwt=False,
        applymode="calonly",
    )


def image_dirty_and_clean(
    ms_path: Path,
    outdir: Path,
    key: str,
    title: str,
    cell: float,
    imsize: int,
) -> dict:
    dirty_base = outdir / f"{key}_dirty"
    clean_base = outdir / f"{key}_clean"

    run_tclean(
        ms_path, str(dirty_base),
        cell_arcsec=cell, imsize=imsize, niter=0,
        datacolumn="corrected", usemask=""
    )
    run_tclean(
        ms_path, str(clean_base),
        cell_arcsec=cell, imsize=imsize, niter=POSTSELFCAL_CLEAN_NITER,
        datacolumn="corrected", usemask="auto-multithresh"
    )

    dirty_png = outdir / f"{key}_dirty.png"
    clean_png = outdir / f"{key}_clean.png"
    export_png(dirty_base.with_suffix(".image"), dirty_png)
    export_png(clean_base.with_suffix(".image"), clean_png)

    stats = image_stats(clean_base.with_suffix(".image"))

    return {
        "key": key,
        "title": title,
        "dirty_png": dirty_png,
        "clean_png": clean_png,
        "dirty_image": dirty_base.with_suffix(".image"),
        "clean_image": clean_base.with_suffix(".image"),
        **stats,
    }


def run_variant(
    src_ms: Path,
    outdir: Path,
    folder_name: str,
    key: str,
    title: str,
    cell: float,
    imsize: int,
    *,
    selfcal: bool,
    solint: str = "inf",
    model_weighting: str = "briggs",
    model_robust: float = 0.5,
    model_deconvolver: str = "hogbom",
    model_scales: list[int] | None = None,
    model_usemask: str = "auto-multithresh",
    model_mask: str = "",
) -> dict:
    work_ms = outdir / f"{folder_name}_{key}.ms"
    ms = copy_ms(src_ms, work_ms)

    clearcal(vis=str(ms))
    flag_zero_visibilities(ms)

    if selfcal:
        build_model_image(
            ms, outdir, key, cell, imsize,
            weighting=model_weighting,
            robust=model_robust,
            deconvolver=model_deconvolver,
            scales=model_scales,
            usemask=model_usemask,
            mask=model_mask,
            niter=INITIAL_CLEAN_NITER,
        )
        caltable = outdir / f"{key}.g"
        run_phase_selfcal(ms, caltable, solint=solint)

    result = image_dirty_and_clean(ms, outdir, key, title, cell, imsize)
    result["selfcal"] = selfcal
    result["solint"] = solint if selfcal else ""
    result["model_weighting"] = model_weighting
    result["model_deconvolver"] = model_deconvolver
    result["model_usemask"] = model_usemask
    result["model_mask"] = model_mask
    return result


def make_summary_plot(
    *,
    folder_name: str,
    out_png: Path,
    bmaj: float,
    bmin: float,
    bpa: float,
    cell: float,
    imsize: int,
    results: list[dict],
) -> None:
    panels = []
    for r in results:
        subtitle = (
            f'peak={r["peak_jy_per_beam"]:.3e} Jy/bm\n'
            f'rms={r["rms_jy_per_beam"]:.3e} Jy/bm  DR={r["dynrange"]:.1f}'
            if np.isfinite(r["peak_jy_per_beam"]) and np.isfinite(r["rms_jy_per_beam"])
            else "stats unavailable"
        )
        panels.append((f'{r["title"]} dirty', r["dirty_png"], ""))
        panels.append((f'{r["title"]} clean', r["clean_png"], subtitle))

    n = len(panels)
    ncols = PLOT_COLS
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.1 * ncols, 5.6 * nrows),
        squeeze=False,
    )

    for axrow in axes:
        for ax in axrow:
            ax.axis("off")

    for i, (title, png_path, extra) in enumerate(panels):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        img = mpimg.imread(png_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title + (f"\n{extra}" if extra else ""), fontsize=11)

    fig.suptitle(
        (
            f"{folder_name} selfcal experiments\n"
            f"beam=({bmaj:.3f}\", {bmin:.3f}\", pa={bpa:.1f}°) | "
            f"cell={cell:.4f}\" | imsize={imsize}"
        ),
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote comparison PNG: {out_png}")


def run_one(folder_name: str) -> tuple[Path, Path]:
    sample_top, src_ms = find_sample_ms(folder_name)
    outdir = sample_top / OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    # one copy just to estimate beam/grid consistently
    probe_ms = outdir / f"{folder_name}_probe.ms"
    probe = copy_ms(src_ms, probe_ms)
    clearcal(vis=str(probe))
    flag_zero_visibilities(probe)
    cell, imsize, bmaj, bmin, bpa = estimate_grid(probe, outdir)

    scales = choose_multiscale_scales(cell, bmaj, bmin)
    mask_str = central_mask_string(imsize)

    results = []

    results.append(
        run_variant(
            src_ms, outdir, folder_name,
            key="baseline",
            title="baseline",
            cell=cell, imsize=imsize,
            selfcal=False,
        )
    )

    results.append(
        run_variant(
            src_ms, outdir, folder_name,
            key="selfcal_phase_inf",
            title="selfcal phase inf",
            cell=cell, imsize=imsize,
            selfcal=True,
            solint="inf",
        )
    )

    results.append(
        run_variant(
            src_ms, outdir, folder_name,
            key="selfcal_phase_60s",
            title="selfcal phase 60s",
            cell=cell, imsize=imsize,
            selfcal=True,
            solint="60s",
        )
    )

    results.append(
        run_variant(
            src_ms, outdir, folder_name,
            key="selfcal_natural_model",
            title="selfcal natural model",
            cell=cell, imsize=imsize,
            selfcal=True,
            solint="inf",
            model_weighting="natural",
        )
    )

    results.append(
        run_variant(
            src_ms, outdir, folder_name,
            key="selfcal_multiscale_model",
            title="selfcal multiscale model",
            cell=cell, imsize=imsize,
            selfcal=True,
            solint="inf",
            model_deconvolver="multiscale",
            model_scales=scales,
        )
    )

    results.append(
        run_variant(
            src_ms, outdir, folder_name,
            key="selfcal_masked_model",
            title="selfcal masked model",
            cell=cell, imsize=imsize,
            selfcal=True,
            solint="inf",
            model_usemask="user",
            model_mask=mask_str,
        )
    )

    out_png = outdir / f"{folder_name}_selfcal_experiments.png"
    make_summary_plot(
        folder_name=folder_name,
        out_png=out_png,
        bmaj=bmaj,
        bmin=bmin,
        bpa=bpa,
        cell=cell,
        imsize=imsize,
        results=results,
    )

    summary_csv = outdir / f"{folder_name}_selfcal_experiments.csv"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    print(f"[OK] wrote CSV: {summary_csv}")

    return out_png, summary_csv


def main(folder_name: str) -> None:
    out_png, summary_csv = run_one(folder_name)
    print(f"[DONE] comparison saved to: {out_png}")
    print(f"[DONE] stats saved to: {summary_csv}")


# if __name__ == "__main__":
#     main("0739+016")