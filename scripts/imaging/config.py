"""User configuration and staged resolution for direct imaging."""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from scripts.vla_config import estimate_synthesized_beam_arcsec

from .metadata import (
    DEFAULT_CALIBRATOR_META_CSV,
    DEFAULT_EXTRACTED_MS_ROOT,
    repository_path,
    resolve_csv_meta,
    resolve_ms_band,
    resolve_path,
    validate_data_column,
)
from .models import (
    Beam,
    CSVMeta,
    DataColumnValidation,
    ImageGrid,
    MSBandMeta,
    ResolvedCleanControls,
    ResolvedGrid,
    ResolvedImagingConfig,
    ResolvedMS,
)


DEFAULT_IMSIZE = (256, 256)
DEFAULT_MASK_NBEAMS = 6.0


def normalize_imsize(imsize: int | Sequence[int]) -> Tuple[int, int]:
    """Return a positive two-axis CASA image size from a scalar or pair."""
    if isinstance(imsize, bool):
        raise TypeError("imsize must be an integer or a two-integer sequence")
    try:
        scalar = operator.index(imsize)
    except TypeError:
        if isinstance(imsize, (str, bytes)):
            raise TypeError("imsize must be an integer or a two-integer sequence")
        try:
            values = tuple(imsize)
        except TypeError as exc:
            raise TypeError("imsize must be an integer or a two-integer sequence") from exc
        if len(values) != 2:
            raise ValueError("imsize sequence must contain exactly two values")
        normalized = []
        for value in values:
            if isinstance(value, bool):
                raise TypeError("imsize values must be integers")
            try:
                normalized.append(operator.index(value))
            except TypeError as exc:
                raise TypeError("imsize values must be integers") from exc
        result = (normalized[0], normalized[1])
    else:
        result = (scalar, scalar)
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("imsize values must be positive")
    return result


def central_circle_mask(
    imsize: int | Sequence[int],
    beam_major_arcsec: float,
    diameter_beams: float,
) -> str:
    """Return a centered CASA circle with a beam-scaled diameter."""
    nx, ny = normalize_imsize(imsize)
    radius_arcsec = 0.5 * float(diameter_beams) * float(beam_major_arcsec)
    if radius_arcsec <= 0:
        raise ValueError("beam size and mask diameter must be positive")
    return f"circle[[{nx // 2}pix,{ny // 2}pix],{radius_arcsec:.12g}arcsec]"


def _image_product(base: Path, suffix: str, deconvolver: str) -> Path:
    tail = f".{suffix}.tt0" if deconvolver == "mtmfs" and suffix != "mask" else f".{suffix}"
    return Path(f"{base}{tail}")


def _assert_prefix_available(base: Path) -> None:
    existing = sorted(base.parent.glob(base.name + ".*"))
    if existing:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing CASA products for {base}:\n{rendered}"
        )


def _angle_arcsec(value: object) -> float:
    if not isinstance(value, dict):
        return float(value)
    number = float(value["value"])
    unit = str(value.get("unit", "arcsec")).strip().lower()
    if unit.startswith("rad"):
        return math.degrees(number) * 3600.0
    if unit.startswith("deg"):
        return number * 3600.0
    if unit.startswith("arcmin"):
        return number * 60.0
    if unit.startswith("arcsec") or unit == "asec":
        return number
    raise ValueError(f"Unsupported angular unit {unit!r}")


def read_beam(image_path: Path, *, imhead_task=None) -> Beam:
    if imhead_task is None:
        try:
            from casatasks import imhead as imhead_task
        except ImportError as exc:
            raise RuntimeError("CASA casatasks is required to read image beams") from exc
    info = imhead_task(imagename=str(image_path), mode="summary")
    if not isinstance(info, dict) or not isinstance(info.get("restoringbeam"), dict):
        raise RuntimeError(f"No restoring beam found in CASA image {image_path}")
    restoring = info["restoringbeam"]
    major = _angle_arcsec(restoring["major"])
    minor = _angle_arcsec(restoring["minor"])
    pa_value = restoring.get("positionangle", 0.0)
    if isinstance(pa_value, dict):
        pa = float(pa_value.get("value", 0.0))
        if str(pa_value.get("unit", "deg")).lower().startswith("rad"):
            pa = math.degrees(pa)
    else:
        pa = float(pa_value)
    if not all(math.isfinite(value) for value in (major, minor, pa)) or major <= 0 or minor <= 0:
        raise RuntimeError(f"Invalid restoring beam in {image_path}: {restoring!r}")
    return Beam(major, minor, pa)


@dataclass(frozen=True)
class _ResolutionContext:
    csv_meta: Optional[CSVMeta]
    ms_band_meta: MSBandMeta
    data_validation: DataColumnValidation
    array_configuration: Optional[str]
    tclean_parameters: Dict[str, Any]


@dataclass(frozen=True)
class GridConfig:
    pixels_per_beam: float = 4.0
    field_of_view_in_beams: float = 64.0
    min_imsize: int = 128
    max_imsize: int = 1024
    min_cell_arcsec: float = 0.02
    max_cell_arcsec: float = 50.0

    def __post_init__(self) -> None:
        if self.pixels_per_beam <= 0 or self.field_of_view_in_beams <= 0:
            raise ValueError("pixels_per_beam and field_of_view_in_beams must be positive")
        if self.min_imsize <= 0 or self.max_imsize < self.min_imsize:
            raise ValueError("invalid min_imsize/max_imsize bounds")
        if self.min_cell_arcsec <= 0 or self.max_cell_arcsec < self.min_cell_arcsec:
            raise ValueError("invalid min_cell_arcsec/max_cell_arcsec bounds")

    def _grid_for_beam(self, beam_arcsec: float) -> ImageGrid:
        cell = min(self.max_cell_arcsec, max(self.min_cell_arcsec, beam_arcsec / self.pixels_per_beam))
        size = int(math.ceil(self.field_of_view_in_beams * beam_arcsec / cell))
        if size % 2:
            size += 1
        size = min(self.max_imsize, max(self.min_imsize, size))
        if size % 2:
            size = size - 1 if size == self.max_imsize else size + 1
        fov = size * cell
        return ImageGrid((size, size), (cell, cell), (fov, fov))

    def resolve(
        self,
        ms: ResolvedMS,
        *,
        workspace: Path,
        imsize: int | Sequence[int] = DEFAULT_IMSIZE,
        _context: Optional[_ResolutionContext] = None,
        tclean_task=None,
        imhead_task=None,
    ) -> ResolvedGrid:
        """Run a safe first pass, measure its beam, and derive the final grid."""
        workspace = Path(workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if _context is None:
            csv_meta, preferred_spw = resolve_csv_meta(ms.visibility_id, DEFAULT_CALIBRATOR_META_CSV)
            ms_band = resolve_ms_band(ms.path, csv_meta, preferred_spw=preferred_spw)
            validation = validate_data_column(ms.path, "corrected")
            _context = _ResolutionContext(
                csv_meta,
                ms_band,
                validation,
                None if csv_meta is None else csv_meta.array_configuration,
                {
                    "datacolumn": "corrected",
                    "specmode": "mfs",
                    "stokes": "I",
                    "gridder": "standard",
                    "deconvolver": "mtmfs",
                    "nterms": 2,
                    "weighting": "briggs",
                    "robust": 0.5,
                    "interactive": False,
                    "parallel": False,
                },
            )

        estimate = estimate_synthesized_beam_arcsec(
            _context.array_configuration,
            _context.ms_band_meta.representative_frequency_ghz,
        )
        provisional = self._grid_for_beam(estimate) if estimate and estimate > 0 else ImageGrid(
            (256, 256), (0.5, 0.5), (128.0, 128.0)
        )
        base = workspace / "firstpass"
        _assert_prefix_available(base)
        if tclean_task is None:
            try:
                from casatasks import tclean as tclean_task
            except ImportError as exc:
                raise RuntimeError("CASA casatasks is required for grid resolution") from exc
        parameters = dict(_context.tclean_parameters)
        parameters.update(
            vis=str(ms.path),
            imagename=str(base),
            imsize=list(provisional.imsize),
            cell=[f"{provisional.cell_arcsec[0]:.12g}arcsec"],
            niter=0,
        )
        tclean_task(**parameters)
        image = _image_product(base, "image", str(parameters["deconvolver"]))
        if not image.exists():
            raise RuntimeError(f"First-pass tclean did not create expected image {image}")
        beam = read_beam(image, imhead_task=imhead_task)
        derived = self._grid_for_beam(min(beam.major_arcsec, beam.minor_arcsec))
        requested_imsize = normalize_imsize(imsize)
        cell = derived.cell_arcsec
        grid = ImageGrid(
            requested_imsize,
            cell,
            (requested_imsize[0] * cell[0], requested_imsize[1] * cell[1]),
        )
        return ResolvedGrid(beam, grid, base)


@dataclass(frozen=True)
class CleanIterationsConfig:
    niter: int = 1_000_000
    threshold: Optional[str] = None
    dirty_peak_fraction: Optional[float] = None
    nsigma: Optional[float] = None
    cycleniter: Optional[int] = None

    def __post_init__(self) -> None:
        if self.niter < 0:
            raise ValueError("niter cannot be negative")
        if self.threshold is not None and self.dirty_peak_fraction is not None:
            raise ValueError("threshold and dirty_peak_fraction are mutually exclusive")
        if self.dirty_peak_fraction is not None and self.dirty_peak_fraction < 0:
            raise ValueError("dirty_peak_fraction cannot be negative")
        if self.nsigma is not None and self.nsigma < 0:
            raise ValueError("nsigma cannot be negative")
        if self.cycleniter is not None and self.cycleniter < 0:
            raise ValueError("cycleniter cannot be negative")
        if self.threshold is not None:
            match = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)", self.threshold)
            if match is None:
                raise ValueError(f"threshold is not a CASA numeric quantity: {self.threshold!r}")
            if float(match.group(1)) < 0:
                raise ValueError("threshold cannot be negative")
        if self.niter > 0 and self.threshold is None and self.dirty_peak_fraction is None and self.nsigma is None:
            raise ValueError("a positive niter requires threshold, dirty_peak_fraction, or nsigma")

    def resolve(self, dirty_residual: Path, *, imstat_task=None) -> ResolvedCleanControls:
        threshold = self.threshold
        if self.dirty_peak_fraction is not None:
            if imstat_task is None:
                try:
                    from casatasks import imstat as imstat_task
                except ImportError as exc:
                    raise RuntimeError("CASA casatasks is required to measure the dirty peak") from exc
            stats = imstat_task(imagename=str(dirty_residual))
            try:
                low = _first_number(stats["min"])
                high = _first_number(stats["max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Could not read dirty residual extrema from {dirty_residual}") from exc
            peak = max(abs(low), abs(high))
            if not math.isfinite(peak) or peak <= 0:
                raise RuntimeError(f"Invalid dirty residual peak {peak!r} in {dirty_residual}")
            threshold = f"{peak * self.dirty_peak_fraction:.12g}Jy"
        return ResolvedCleanControls(self.niter, threshold, self.nsigma, self.cycleniter)


def _first_number(value: object) -> float:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty numeric value")
        value = value[0]
    return float(value)


@dataclass(frozen=True)
class ImagingConfig:
    datacolumn: Literal["corrected", "data"] = "corrected"
    specmode: str = "mfs"
    stokes: str = "I"
    gridder: str = "standard"
    deconvolver: str = "mtmfs"
    nterms: int = 2
    weighting: str = "briggs"
    robust: float = 0.5
    gain: float = 0.1
    mask_nbeams: Optional[float] = DEFAULT_MASK_NBEAMS
    grid: GridConfig = GridConfig()
    clean: CleanIterationsConfig = CleanIterationsConfig(nsigma=3.0, cycleniter=100)
    calibrator_meta_csv: Path = DEFAULT_CALIBRATOR_META_CSV
    extracted_ms_root: Path = DEFAULT_EXTRACTED_MS_ROOT
    uvrange: str = ""
    savemodel: str = "none"
    interactive: bool = False
    parallel: bool = False

    def __post_init__(self) -> None:
        if self.datacolumn not in ("corrected", "data"):
            raise ValueError("datacolumn must be exactly 'corrected' or 'data'")
        if self.nterms <= 0:
            raise ValueError("nterms must be positive")
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.mask_nbeams is not None and self.mask_nbeams <= 0:
            raise ValueError("mask_nbeams must be positive or None")

    def _base_parameters(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "datacolumn": self.datacolumn,
            "specmode": self.specmode,
            "stokes": self.stokes,
            "gridder": self.gridder,
            "deconvolver": self.deconvolver,
            "weighting": self.weighting,
            "robust": self.robust,
            "gain": self.gain,
            "uvrange": self.uvrange,
            "savemodel": self.savemodel,
            "interactive": self.interactive,
            "parallel": self.parallel,
        }
        if self.deconvolver == "mtmfs":
            result["nterms"] = self.nterms
        return result

    def resolve(
        self,
        ms: str | Path,
        workspace: str | Path,
        *,
        imsize: int | Sequence[int] = DEFAULT_IMSIZE,
    ) -> ResolvedImagingConfig:
        resolved_ms = resolve_path(ms, repository_path(self.extracted_ms_root))
        csv_meta, preferred_spw = resolve_csv_meta(
            resolved_ms.visibility_id, repository_path(self.calibrator_meta_csv)
        )
        band_meta = resolve_ms_band(resolved_ms.path, csv_meta, preferred_spw=preferred_spw)
        validation = validate_data_column(resolved_ms.path, self.datacolumn)
        warnings = []
        if csv_meta is None:
            warnings.append(
                "No calibrator metadata row was found; MS-derived band metadata and fallback "
                "first-pass sizing were used."
            )
        elif band_meta.catalog_band_matches is False:
            warnings.append(
                f"MS-derived band {band_meta.selected_band} does not match catalog bands "
                f"{', '.join(csv_meta.catalog_band_codes)}."
            )
        base = self._base_parameters()
        context = _ResolutionContext(
            csv_meta,
            band_meta,
            validation,
            None if csv_meta is None else csv_meta.array_configuration,
            base,
        )
        grid = self.grid.resolve(
            resolved_ms,
            workspace=Path(workspace),
            imsize=imsize,
            _context=context,
        )
        effective = dict(base)
        effective.update(
            imsize=list(grid.image_grid.imsize),
            cell=[f"{grid.image_grid.cell_arcsec[0]:.12g}arcsec"],
            mask_nbeams=self.mask_nbeams,
            requested_clean={
                "niter": self.clean.niter,
                "threshold": self.clean.threshold,
                "dirty_peak_fraction": self.clean.dirty_peak_fraction,
                "nsigma": self.clean.nsigma,
                "cycleniter": self.clean.cycleniter,
            },
            calibrator_meta_csv=str(repository_path(self.calibrator_meta_csv)),
            extracted_ms_root=str(repository_path(self.extracted_ms_root)),
        )
        return ResolvedImagingConfig(
            resolved_ms.path,
            resolved_ms.visibility_id,
            csv_meta,
            band_meta,
            self.datacolumn,
            validation,
            grid,
            None,
            effective,
            tuple(warnings),
        )


DefaultImagingConfig = ImagingConfig(
    datacolumn="data",
    specmode="mfs",
    stokes="I",
    gridder="standard",
    deconvolver="mtmfs",
    nterms=1,
    weighting="briggs",
    robust=0.5,
    gain=0.1,
    mask_nbeams=DEFAULT_MASK_NBEAMS,
    grid=GridConfig(
        pixels_per_beam=4.0,
        field_of_view_in_beams=64.0,
        min_imsize=128,
        max_imsize=1024,
        min_cell_arcsec=0.02,
        max_cell_arcsec=50.0,
    ),
    clean=CleanIterationsConfig( # Kinda deep cleaning
        niter=1_000_000,
        nsigma=3.0,
        cycleniter=None,
        dirty_peak_fraction=1e-7,
    ),
    calibrator_meta_csv=DEFAULT_CALIBRATOR_META_CSV,
    extracted_ms_root=DEFAULT_EXTRACTED_MS_ROOT,
)


__all__ = [
    "CleanIterationsConfig",
    "DEFAULT_IMSIZE",
    "DefaultImagingConfig",
    "GridConfig",
    "ImagingConfig",
    "normalize_imsize",
]
