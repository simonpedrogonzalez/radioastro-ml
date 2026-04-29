from __future__ import annotations

import shutil
from math import ceil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from casatasks import flagdata, imhead, tclean

from scripts.img_utils import casa_image_to_png


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")

MODIFY_MS_IN_PLACE = True

# first pass only to estimate beam
FIRSTPASS_CELL_ARCSEC = 0.5
FIRSTPASS_IMSIZE = 256

# final imaging policy
FINAL_IMSIZE = 256
PIXELS_PER_BEAM = 4.0

MIN_CELL_ARCSEC = 0.02
MAX_CELL_ARCSEC = 10.0

OUTDIR_NAME = "imaging_variants"

# plotting
PLOT_COLS = 3

# base tclean config
TCLEAN_BASE = dict(
    specmode="mfs",
    stokes="I",
    gridder="standard",
    interactive=False,
)

# variants to try
VARIANTS = [
    dict(
        key="dirty",
        title="dirty",
        niter=0,
        weighting="briggs",
        robust=0.5,
        deconvolver="hogbom",
    ),
    dict(
        key="standard",
        title="standard",
        niter=100,
        weighting="briggs",
        robust=0.5,
        deconvolver="hogbom",
    ),
    dict(
        key="more_iterations",
        title="more iterations",
        niter=1000,
        weighting="briggs",
        robust=0.5,
        deconvolver="hogbom",
    ),
    dict(
        key="clark",
        title="clark",
        niter=1000,
        weighting="briggs",
        robust=0.5,
        deconvolver="clark",
    ),
    dict(
        key="natural_weighting",
        title="natural weighting",
        niter=1000,
        weighting="natural",
        deconvolver="hogbom",
    ),
    dict(
        key="uniform_weighting",
        title="uniform weighting",
        niter=1000,
        weighting="uniform",
        deconvolver="hogbom",
    ),
    dict(
        key="multiscale",
        title="multiscale",
        niter=1000,
        weighting="briggs",
        robust=0.5,
        deconvolver="multiscale",
    ),
    dict(
        key="multiscale_natural",
        title="multiscale + natural",
        niter=1000,
        weighting="natural",
        deconvolver="multiscale",
    ),
]


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


def run_tclean(
    ms_path: Path,
    imagename: str,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    weighting: str,
    deconvolver: str,
    robust: float | None = None,
    scales: list[int] | None = None,
) -> None:
    remove_casa_products(imagename)

    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=imagename,
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        weighting=weighting,
        deconvolver=deconvolver,
    )

    if weighting == "briggs":
        cfg["robust"] = 0.5 if robust is None else robust

    if deconvolver == "multiscale":
        cfg["scales"] = scales if scales is not None else [0, 5, 15]

    print(
        f"[TCLEAN] {Path(imagename).name} | "
        f"cell={cell_arcsec:.4f}\" | imsize={imsize} | "
        f"niter={niter} | weighting={weighting} | deconvolver={deconvolver}"
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


def choose_multiscale_scales(cell_arcsec: float, bmaj_arcsec: float, bmin_arcsec: float) -> list[int]:
    beam_pix = max(1, int(round(min(bmaj_arcsec, bmin_arcsec) / cell_arcsec)))
    s1 = max(1, beam_pix)
    s2 = max(s1 + 1, 3 * beam_pix)
    return [0, s1, s2]


def estimate_beam_and_cell(ms_path: Path, outdir: Path) -> tuple[float, float, float, float]:
    firstpass_base = outdir / "beam_firstpass_dirty"

    run_tclean(
        ms_path,
        str(firstpass_base),
        cell_arcsec=FIRSTPASS_CELL_ARCSEC,
        imsize=FIRSTPASS_IMSIZE,
        niter=0,
        weighting="briggs",
        robust=0.5,
        deconvolver="hogbom",
    )

    firstpass_image = firstpass_base.with_suffix(".image")
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)
    cell = choose_beam_based_cell_arcsec(bmaj, bmin)
    return bmaj, bmin, bpa, cell


def export_variant_png(image_path: Path, png_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"CASA image not found: {image_path}")

    if png_path.exists():
        png_path.unlink()

    casa_image_to_png(str(image_path), str(png_path))


def run_all_variants(folder_name: str) -> Path:
    sample_top, ms_path = find_sample_ms(folder_name)
    outdir = sample_top / OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    ms_for_imaging = ensure_imaging_ms(ms_path, modify_in_place=MODIFY_MS_IN_PLACE)

    flag_zero_visibilities(ms_for_imaging)

    bmaj, bmin, bpa, cell = estimate_beam_and_cell(ms_for_imaging, outdir)
    scales = choose_multiscale_scales(cell, bmaj, bmin)

    print(
        f"[INFO] beam major={bmaj:.3f}\" minor={bmin:.3f}\" pa={bpa:.1f} deg | "
        f"cell={cell:.4f}\" | multiscale scales={scales}"
    )

    results = []

    for variant in VARIANTS:
        base = outdir / variant["key"]

        run_tclean(
            ms_for_imaging,
            str(base),
            cell_arcsec=cell,
            imsize=FINAL_IMSIZE,
            niter=variant["niter"],
            weighting=variant["weighting"],
            deconvolver=variant["deconvolver"],
            robust=variant.get("robust"),
            scales=scales if variant["deconvolver"] == "multiscale" else None,
        )

        image_path = base.with_suffix(".image")
        png_path = outdir / f"{variant['key']}.png"
        export_variant_png(image_path, png_path)

        item = {
            "title": variant["title"],
            "png_path": png_path,
        }
        results.append(item)

    out_png = outdir / f"{folder_name}_imaging_comparison.png"
    make_summary_plot(
        folder_name=folder_name,
        out_png=out_png,
        bmaj=bmaj,
        bmin=bmin,
        bpa=bpa,
        cell=cell,
        scales=scales,
        results=results,
    )
    return out_png


def make_summary_plot(
    *,
    folder_name: str,
    out_png: Path,
    bmaj: float,
    bmin: float,
    bpa: float,
    cell: float,
    scales: list[int],
    results: list[dict],
) -> None:
    if not results:
        raise RuntimeError("No results to plot")

    n = len(results)
    ncols = PLOT_COLS
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 5.2 * nrows),
        squeeze=False,
    )

    for axrow in axes:
        for ax in axrow:
            ax.axis("off")

    for i, result in enumerate(results):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        img = mpimg.imread(result["png_path"])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(result["title"], fontsize=11)

    fig.suptitle(
        (
            f"{folder_name} imaging comparison\n"
            f"beam=({bmaj:.3f}\", {bmin:.3f}\", pa={bpa:.1f}°) | "
            f"cell={cell:.4f}\" | imsize={FINAL_IMSIZE} | "
            f"multiscale scales={scales}"
        ),
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote comparison PNG: {out_png}")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
def main(folder_name: str) -> None:
    out_png = run_all_variants(folder_name)
    print(f"[DONE] comparison figure saved to: {out_png}")

