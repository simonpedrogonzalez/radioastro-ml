from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from casatasks import imhead


DEFAULT_REGULAR_IMAGE = Path(
    "/Users/u1528314/repos/radioastro-ml/collect/extracted/0205+322/clean_corrected_clean.image"
)
DEFAULT_REPRODUCTION_IMAGE = Path(
    "/Users/u1528314/repos/radioastro-ml/runs/vla_pipe_test/reproduce_before_data_clean.image"
)
DEFAULT_SUMMARY_CSV = Path(
    "/Users/u1528314/repos/radioastro-ml/collect/extracted/beam_imaging_summary.csv"
)


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
        if unit in {"rad", "radian", "radians"}:
            return float(np.rad2deg(number) * 3600.0)
        raise ValueError(f"Unsupported angular unit: {unit!r}")
    return float(value)


def angle_increment_to_arcsec(value: float, unit: str) -> float:
    unit = str(unit).strip().lower()
    value = abs(float(value))
    if unit in {"rad", "radian", "radians"}:
        return float(np.rad2deg(value) * 3600.0)
    if unit in {"deg", "degree", "degrees"}:
        return value * 3600.0
    if unit in {"arcmin", "arcminutes"}:
        return value * 60.0
    if unit in {"arcsec", "arcseconds", "asec"}:
        return value
    raise ValueError(f"Unsupported increment unit: {unit!r}")


def image_geometry(image_path: Path) -> dict:
    info = imhead(imagename=str(image_path), mode="summary")
    if not isinstance(info, dict):
        raise RuntimeError(f"imhead failed for {image_path}")

    shape = list(info["shape"])
    incr = list(info["incr"])
    units = list(info.get("axisunits", []))
    if len(shape) < 2 or len(incr) < 2 or len(units) < 2:
        raise RuntimeError(f"Image summary missing spatial axes for {image_path}")

    cell_ra_arcsec = angle_increment_to_arcsec(incr[0], units[0])
    cell_dec_arcsec = angle_increment_to_arcsec(incr[1], units[1])
    cell_arcsec = 0.5 * (cell_ra_arcsec + cell_dec_arcsec)
    imsize_x = int(shape[0])
    imsize_y = int(shape[1])

    beam = info.get("restoringbeam")
    if beam is None:
        raise RuntimeError(f"No restoring beam in {image_path}")
    beam_major_arcsec = beam_value_to_arcsec(beam["major"])
    beam_minor_arcsec = beam_value_to_arcsec(beam["minor"])
    pa = beam.get("positionangle", {})
    if isinstance(pa, dict):
        beam_pa_deg = float(pa.get("value", np.nan))
        if str(pa.get("unit", "deg")).strip().lower().startswith("rad"):
            beam_pa_deg = float(np.rad2deg(beam_pa_deg))
    else:
        beam_pa_deg = float(pa)

    return {
        "image": str(image_path),
        "imsize_x": imsize_x,
        "imsize_y": imsize_y,
        "cell_ra_arcsec": cell_ra_arcsec,
        "cell_dec_arcsec": cell_dec_arcsec,
        "cell_arcsec": cell_arcsec,
        "fov_x_arcsec": imsize_x * cell_ra_arcsec,
        "fov_y_arcsec": imsize_y * cell_dec_arcsec,
        "beam_major_arcsec": beam_major_arcsec,
        "beam_minor_arcsec": beam_minor_arcsec,
        "beam_pa_deg": beam_pa_deg,
        "pixels_per_beam_minor": beam_minor_arcsec / cell_arcsec if cell_arcsec > 0 else np.nan,
    }


def diff_row(regular: dict, reproduction: dict) -> dict:
    row = {}
    keys = [
        "imsize_x",
        "imsize_y",
        "cell_arcsec",
        "fov_x_arcsec",
        "fov_y_arcsec",
        "beam_major_arcsec",
        "beam_minor_arcsec",
        "beam_pa_deg",
        "pixels_per_beam_minor",
    ]
    for key in keys:
        reg = float(regular[key])
        rep = float(reproduction[key])
        row[f"{key}_regular"] = reg
        row[f"{key}_reproduction"] = rep
        row[f"{key}_diff_regular_minus_reproduction"] = reg - rep
        row[f"{key}_ratio_regular_over_reproduction"] = reg / rep if rep != 0 else np.nan
    return row


def print_report(regular: dict, reproduction: dict, diff: dict) -> None:
    print("\n[GEOMETRY] regular image")
    print(f"  image: {regular['image']}")
    print(f"  imsize: {regular['imsize_x']} x {regular['imsize_y']}")
    print(f"  cell: {regular['cell_arcsec']:.6f} arcsec")
    print(f"  FoV: {regular['fov_x_arcsec']:.3f}\" x {regular['fov_y_arcsec']:.3f}\"")
    print(
        f"  beam: {regular['beam_major_arcsec']:.3f}\" x "
        f"{regular['beam_minor_arcsec']:.3f}\"  pa={regular['beam_pa_deg']:.2f} deg"
    )

    print("\n[GEOMETRY] reproduction image")
    print(f"  image: {reproduction['image']}")
    print(f"  imsize: {reproduction['imsize_x']} x {reproduction['imsize_y']}")
    print(f"  cell: {reproduction['cell_arcsec']:.6f} arcsec")
    print(f"  FoV: {reproduction['fov_x_arcsec']:.3f}\" x {reproduction['fov_y_arcsec']:.3f}\"")
    print(
        f"  beam: {reproduction['beam_major_arcsec']:.3f}\" x "
        f"{reproduction['beam_minor_arcsec']:.3f}\"  pa={reproduction['beam_pa_deg']:.2f} deg"
    )

    print("\n[DIFF] regular - reproduction")
    for key in [
        "imsize_x",
        "imsize_y",
        "cell_arcsec",
        "fov_x_arcsec",
        "fov_y_arcsec",
        "beam_major_arcsec",
        "beam_minor_arcsec",
        "beam_pa_deg",
        "pixels_per_beam_minor",
    ]:
        print(
            f"  {key}: "
            f"diff={diff[f'{key}_diff_regular_minus_reproduction']:.6g}, "
            f"ratio={diff[f'{key}_ratio_regular_over_reproduction']:.6g}"
        )


def main(
    regular_image: str | Path = DEFAULT_REGULAR_IMAGE,
    reproduction_image: str | Path = DEFAULT_REPRODUCTION_IMAGE,
    *,
    out_csv: str | Path | None = None,
) -> dict:
    regular_image = Path(regular_image).expanduser()
    reproduction_image = Path(reproduction_image).expanduser()

    if not regular_image.exists():
        raise FileNotFoundError(f"regular image not found: {regular_image}")
    if not reproduction_image.exists():
        raise FileNotFoundError(f"reproduction image not found: {reproduction_image}")

    regular = image_geometry(regular_image)
    reproduction = image_geometry(reproduction_image)
    diff = diff_row(regular, reproduction)
    print_report(regular, reproduction, diff)

    if out_csv is None:
        out_csv = regular_image.parent / "0205+322_regular_vs_reproduction_geometry.csv"
    out_csv = Path(out_csv).expanduser()
    pd.DataFrame([{**diff, "regular_image": str(regular_image), "reproduction_image": str(reproduction_image)}]).to_csv(
        out_csv,
        index=False,
    )
    print(f"\n[OK] wrote geometry comparison CSV: {out_csv}")
    return {
        "regular": regular,
        "reproduction": reproduction,
        "diff": diff,
        "csv": str(out_csv),
    }


# if __name__ == "__main__":
#     main()
