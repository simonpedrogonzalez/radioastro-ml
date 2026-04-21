from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

from casatasks import imhead, imstat, rmtables, tclean

from scripts.img_utils import casa_image_to_png


DEFAULT_VIS = Path("/Users/u1528314/repos/radioastro-ml/runs/vla_pipe_test/0205+322_pipeline_input.ms")
DEFAULT_OUTDIR = Path("/Users/u1528314/repos/radioastro-ml/runs/vla_pipe_test")
DEFAULT_IMAGENAME = "reproduce_before_data_clean"
TCLEAN_IMSIZE = 256
TCLEAN_CELL_ARCSEC = 1.854640


def remove_products(imagename: Path) -> None:
    for suffix in (
        ".image",
        ".model",
        ".psf",
        ".residual",
        ".sumwt",
        ".pb",
        ".mask",
        ".image.tt0",
        ".model.tt0",
        ".psf.tt0",
        ".residual.tt0",
        ".sumwt.tt0",
        ".pb.tt0",
        ".mask",
    ):
        rmtables(str(imagename) + suffix)


def print_stats(image_path: Path) -> dict:
    stats = imstat(imagename=str(image_path))
    peak = float(max(stats["max"]))
    rms = float(max(stats["rms"]))
    dynrange = peak / rms if rms > 0 else float("nan")
    print(f"[STATS] {image_path}")
    print(f"[STATS] peak Jy/beam: {peak}")
    print(f"[STATS] rms  Jy/beam: {rms}")
    print(f"[STATS] dynrange    : {dynrange}")
    return {
        "peak_jy_per_beam": peak,
        "rms_jy_per_beam": rms,
        "dynrange": dynrange,
    }


def beam_value_to_arcsec(value) -> float:
    if isinstance(value, dict):
        number = float(value["value"])
        unit = str(value.get("unit", "")).strip().lower()
        if unit in {"arcsec", "arcseconds", "asec"}:
            return number
        if unit in {"arcmin", "arcminutes"}:
            return number * 60.0
        if unit in {"deg", "degree", "degrees"}:
            return number * 3600.0
        raise ValueError(f"Unsupported beam unit: {unit!r}")
    return float(value)


def read_restoring_beam_arcsec(image_path: Path) -> tuple[float, float, float]:
    info = imhead(imagename=str(image_path), mode="summary")
    beam = info.get("restoringbeam")
    if beam is None:
        raise RuntimeError(f"No restoring beam in {image_path}")

    bmaj = beam_value_to_arcsec(beam["major"])
    bmin = beam_value_to_arcsec(beam["minor"])
    pa = beam.get("positionangle", {})
    if isinstance(pa, dict):
        pa_deg = float(pa.get("value", 0.0))
        if str(pa.get("unit", "deg")).strip().lower().startswith("rad"):
            pa_deg = math.degrees(pa_deg)
    else:
        pa_deg = float(pa)
    return bmaj, bmin, pa_deg


def main(
    vis: str | Path = DEFAULT_VIS,
    *,
    outdir: str | Path = DEFAULT_OUTDIR,
    imagename: str = DEFAULT_IMAGENAME,
) -> dict:
    vis = Path(vis).expanduser()
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    image_base = outdir / imagename
    image_path = image_base.with_suffix(".image")
    png_path = outdir / f"{imagename}.png"

    print("[INFO] Reproducing uvlim_recal before_data_clean.image")
    print(f"[INFO] vis       : {vis}")
    print(f"[INFO] imagename : {image_base}")
    print("[INFO] datacolumn: data")
    print("[INFO] uvrange   : all")
    print(f"[INFO] imsize    : {TCLEAN_IMSIZE}")
    print(f"[INFO] cell      : {TCLEAN_CELL_ARCSEC:.6f}arcsec")
    print("[INFO] niter     : 100")

    remove_products(image_base)
    if png_path.exists():
        png_path.unlink()

    tclean(
        vis=str(vis),
        imagename=str(image_base),
        datacolumn="data",
        imsize=TCLEAN_IMSIZE,
        cell=f"{TCLEAN_CELL_ARCSEC:.6f}arcsec",
        niter=100,
        uvrange="",
        specmode="mfs",
        weighting="briggs",
        robust=0.5,
        stokes="I",
        deconvolver="hogbom",
        gridder="standard",
        interactive=False,
    )

    bmaj, bmin, bpa = read_restoring_beam_arcsec(image_path)
    fov_arcsec = TCLEAN_IMSIZE * TCLEAN_CELL_ARCSEC
    plot_title = (
        "reproduction | DATA | "
        f"cell={TCLEAN_CELL_ARCSEC:.6f}arcsec | imsize={TCLEAN_IMSIZE} | "
        f"beam={bmaj:.2f}\"x{bmin:.2f}\" pa={bpa:.1f}deg | "
        f"FoV={fov_arcsec:.1f}\""
    )

    casa_image_to_png(
        str(image_path),
        str(png_path),
        title=plot_title,
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )

    stats = print_stats(image_path)
    print(f"[DONE] image: {image_path}")
    print(f"[DONE] png  : {png_path}")
    return {
        "vis": str(vis),
        "image": str(image_path),
        "png": str(png_path),
        "beam_major_arcsec": bmaj,
        "beam_minor_arcsec": bmin,
        "beam_pa_deg": bpa,
        "fov_arcsec": fov_arcsec,
        **stats,
    }


# if __name__ == "__main__":
#     main()
