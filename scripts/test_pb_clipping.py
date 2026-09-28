#!/usr/bin/env python3
"""Image one square and one circular MS with/without CASA's PB image mask.

Run from the repository root with CASA:
    casa --nogui --nologger --no-auto-update -c scripts/test_pb_clipping.py --output-dir collect/experiments/pb_clipping_test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.imaging import BeamRegion, DefaultImagingConfig, image_ms
from scripts.imaging.plot_utils import casa_image_to_png, shared_fits_display_limits
from scripts.reporting import QuartoReporter


THERMAL = ROOT / "collect/experiments/extracted_thermal_simulation_comparison_20260909T005739"
SOURCES = ("0005+383", "0029+349")
PRODUCTS = ("dirty", "clean", "residual", "psf")
CASES = (("default", None), ("no_pb_mask", -0.1))


def _paths(run):
    return {name: run / f"{name}.fits.gz" for name in PRODUCTS}


def _native_paths(run):
    return {"dirty": run / "dirty.image.tt0", "clean": run / "clean.image.tt0",
            "residual": run / "clean.residual.tt0", "psf": run / "clean.psf.tt0"}


def _support(native, exported):
    from casatools import image

    ia = image()
    ia.open(str(native))
    try:
        values = np.squeeze(ia.getchunk())
        valid = np.squeeze(ia.getchunk(getmask=True))
    finally:
        ia.close()
    with fits.open(exported) as hdul:
        saved = np.squeeze(hdul[0].data)
    return f"{np.count_nonzero(valid)}/{valid.size} valid; {np.count_nonzero(saved == 0)} zeros"


def _plot_psf(path, destination, limits, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with fits.open(path) as hdul:
        values = np.squeeze(hdul[0].data)
        wcs = WCS(hdul[0].header).celestial
    figure = plt.figure()
    axis = figure.add_subplot(projection=wcs)
    artist = axis.imshow(values, origin="lower", cmap="inferno", vmin=-limits, vmax=limits)
    axis.set(xlabel="RA", ylabel="Dec", title=title)
    figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04).set_label("Relative response")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _report(output):
    lines = ["---", 'title: "PB clipping: square and circular MS comparison"',
             "format:", "  html:", "    page-layout: full", "    embed-resources: true", "---", "",
             "<style>",
             ".comparison {width:100%;table-layout:fixed;border-collapse:collapse}",
             ".comparison th:first-child,.comparison td:first-child {width:110px}",
             ".comparison td {vertical-align:top}",
             ".comparison img {display:block;width:100%;height:auto}",
             "</style>", ""]
    for source in SOURCES:
        runs = {case: output / source / case for case, _ in CASES}
        paths = {case: _paths(run) for case, run in runs.items()}
        scales = {name: shared_fits_display_limits([paths[case][name] for case in runs])
                  for name in PRODUCTS if name != "psf"}
        psf_values = []
        for case in runs:
            with fits.open(paths[case]["psf"]) as hdul:
                psf_values.append(np.squeeze(hdul[0].data).ravel())
        psf_limit = float(np.percentile(np.abs(np.concatenate(psf_values)), 99.5))
        if psf_limit <= 0:
            raise ValueError(f"Invalid PSF display limit for {source}")
        lines += [f"## {source}", "", '<table class="comparison">',
                  "<thead><tr><th>Setting</th>" + "".join(f"<th>{name.title()}</th>" for name in PRODUCTS) + "</tr></thead>",
                  "<tbody>"]
        for case, _ in CASES:
            label = "Current" if case == "default" else "pblimit=-0.1"
            lines.append(f"<tr><th>{label}</th>")
            for name in PRODUCTS:
                png = output / "plots" / f"{source}_{case}_{name}.png"
                if name == "psf":
                    _plot_psf(paths[case][name], png, psf_limit, name.title())
                else:
                    casa_image_to_png(paths[case][name], png, title=name.title(),
                                      draw_beam=True, display_limits_mjy_per_beam=scales[name])
                lines.append(f'<td><img src="plots/{png.name}" alt="{source} {case} {name}"></td>')
            lines.append("</tr>")
        lines += ["</tbody></table>", "", "| Run | Dirty | CLEAN | Residual | PSF |",
                  "| --- | --- | --- | --- | --- |"]
        for case, _ in CASES:
            native = _native_paths(runs[case])
            counts = [_support(native[name], paths[case][name]) for name in PRODUCTS]
            lines.append(f"| {case} | " + " | ".join(counts) + " |")
        lines.append("")
        if source == "0005+383":
            lines += ["### FITS pixel differences (pblimit=-0.1 minus current)", "",
                      "| Product | Different pixels | Maximum absolute difference |",
                      "| --- | ---: | ---: |"]
            for name in PRODUCTS:
                current = np.squeeze(fits.getdata(paths["default"][name]))
                changed = np.squeeze(fits.getdata(paths["no_pb_mask"][name]))
                if current.shape != changed.shape:
                    raise ValueError(f"Mismatched {name} FITS shapes for {source}")
                difference = changed - current
                lines.append(f"| {name.title()} | {np.count_nonzero(changed != current)} | "
                             f"{np.max(np.abs(difference)):.6g} |")
            lines.append("")
    qmd = output / "report.qmd"
    qmd.write_text("\n".join(lines), encoding="utf-8")
    QuartoReporter(qmd).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError("Quarto did not render report.html")
    return qmd.with_suffix(".html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true", help="Rebuild the report from existing imaging runs")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.report_only:
        if not output.is_dir():
            raise FileNotFoundError(output)
        print(f"Report: {_report(output)}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty comparison directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir()
    for source in SOURCES:
        thermal = THERMAL / source / "simulation"
        ms = thermal / f"{source}_matched_snr.ms"
        original = json.loads((thermal / "default_imaging/qa.json").read_text())
        region = BeamRegion(**original["metrics"]["region"])
        imsize = original["effective_imaging_parameters"]["imsize"]
        for case, pblimit in CASES:
            destination = output / source / case
            print(f"Imaging {source} / {case}: {destination}", flush=True)
            image_ms(ms, DefaultImagingConfig, destination, imsize=imsize,
                     metric_region=region, keep_intermediate_products=True,
                     fits_invalid_policy="fill", fits_fill_value=0.0, pblimit=pblimit)
    print(f"Report: {_report(output)}", flush=True)


if __name__ == "__main__":
    main()
