"""Immutable public data models for the imaging package."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass(frozen=True)
class ResolvedMS:
    path: Path
    visibility_id: Optional[str]


@dataclass(frozen=True)
class CSVMeta:
    visibility_id: str
    array_configuration: Optional[str]
    catalog_band_codes: Tuple[str, ...]
    catalog_frequency_ghz: Optional[float]


@dataclass(frozen=True)
class MSBandMeta:
    selected_band: str
    representative_frequency_ghz: float
    frequency_reference_spw: int
    catalog_band_matches: Optional[bool]


@dataclass(frozen=True)
class Beam:
    major_arcsec: float
    minor_arcsec: float
    position_angle_deg: float


@dataclass(frozen=True)
class ImageGrid:
    imsize: Tuple[int, int]
    cell_arcsec: Tuple[float, float]
    field_of_view_arcsec: Tuple[float, float]


@dataclass(frozen=True)
class DataColumnValidation:
    requested_name: str
    table_column: str
    available_columns: Tuple[str, ...]
    rows_examined: int
    unflagged_samples: int
    finite_unflagged_samples: int
    nonzero_finite_unflagged_samples: int


@dataclass(frozen=True)
class ResolvedGrid:
    beam: Beam
    image_grid: ImageGrid
    first_pass_imagename: Path


@dataclass(frozen=True)
class ResolvedCleanControls:
    niter: int
    threshold: Optional[str]
    nsigma: Optional[float]
    cycleniter: Optional[int]


@dataclass(frozen=True)
class ResolvedImagingConfig:
    ms_path: Path
    visibility_id: Optional[str]
    csv_meta: Optional[CSVMeta]
    ms_band_meta: MSBandMeta
    datacolumn: str
    data_column_validation: DataColumnValidation
    grid: ResolvedGrid
    clean: Optional[ResolvedCleanControls]
    effective_imaging_parameters: Dict[str, Any]
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TcleanRunSummary:
    source: Literal["return_record", "pipeline_context", "pipeline_log"]
    iterdone: Optional[int]
    nmajordone: Optional[int]
    stopcode: Optional[int]
    stop_description: Optional[str]
    major_cycle_iteration_counts: Tuple[int, ...]
    final_peak_residual_jy_per_beam: Optional[float]
    final_model_flux_jy: Optional[float]
    final_cycle_threshold_jy_per_beam: Optional[float]
    minor_cycle_history: Optional[Dict[str, Any]]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class BeamRegion:
    """A circular radial selection measured in synthesized-beam major FWHM."""

    min_radius_beams: Optional[float] = None
    max_radius_beams: Optional[float] = None

    def __post_init__(self) -> None:
        for name, value in (
            ("min_radius_beams", self.min_radius_beams),
            ("max_radius_beams", self.max_radius_beams),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative, or None")
        if (
            self.min_radius_beams is not None
            and self.max_radius_beams is not None
            and self.max_radius_beams <= self.min_radius_beams
        ):
            raise ValueError("max_radius_beams must be greater than min_radius_beams")


@dataclass(frozen=True)
class MetricDefinition:
    """Stable human- and report-facing definition of one image metric."""

    key: str
    name: str
    formula: str
    latex: str
    description: str
    unit: str

    def __repr__(self) -> str:
        return f"{self.name}={self.formula}"

    def to_report_dict(self) -> Dict[str, str]:
        return {
            "key": self.key,
            "name": self.name,
            "formula": self.formula,
            "latex": self.latex,
            "description": self.description,
            "unit": self.unit,
        }

    def format_value(self, value: str) -> str:
        unit = "" if self.unit == "dimensionless" else f" {self.unit}"
        return f"{self!r}={value}{unit}"


@dataclass(frozen=True)
class RegionMetrics:
    n_pixels: int
    area_synthesized_beams: float
    rms_jy_per_beam: float
    scaled_mad_jy_per_beam: float
    residual_abs_peak_jy_per_beam: float
    residual_min_jy_per_beam: float
    residual_max_jy_per_beam: float
    peak_over_scaled_mad: float
    p99_over_scaled_mad: float
    p99_5_over_scaled_mad: float
    rms_over_scaled_mad: float


@dataclass(frozen=True)
class ImageMetrics:
    region: BeamRegion
    clean_peak_jy_per_beam: float
    residual: RegionMetrics
    dynamic_range_rms: float
    dynamic_range_scaled_mad: float


@dataclass(frozen=True)
class PipelineBackground:
    """Specialized VLA Pipeline background statistic and its provenance."""

    rms_jy_per_beam: float
    region: str
    algorithm: str


@dataclass(frozen=True)
class QAReport:
    schema_version: int
    engine: Literal["direct", "vla_pipeline"]
    input_value: str
    ms_path: Path
    visibility_id: Optional[str]
    resolved_config: Optional[ResolvedImagingConfig]
    effective_imaging_parameters: Dict[str, Any]
    products: Dict[str, Optional[Path]]
    metrics: ImageMetrics
    metric_units: Dict[str, Any]
    metric_validity: Dict[str, Any]
    tclean_summary: Optional[TcleanRunSummary]
    pipeline_background: Optional[PipelineBackground]
    warnings: Tuple[str, ...] = ()
    plot_recipes: Dict[str, Any] = field(default_factory=dict)
    temporary_products: Dict[str, Optional[Path]] = field(default_factory=dict)


@dataclass(frozen=True)
class ImagingResult:
    engine: Literal["direct", "vla_pipeline"]
    ms_path: Path
    visibility_id: Optional[str]
    output_dir: Path
    resolved_config: Optional[ResolvedImagingConfig]
    effective_imaging_parameters: Dict[str, Any]
    dirty_image: Optional[Path]
    clean_image: Optional[Path]
    residual_image: Optional[Path]
    dirty_fits: Path
    clean_fits: Path
    residual_fits: Path
    psf_fits: Path
    model_image: Optional[Path]
    mask_image: Optional[Path]
    psf_image: Optional[Path]
    dirty_png: Optional[Path]
    clean_png: Optional[Path]
    residual_png: Optional[Path]
    qa_text: Path
    qa_json: Path
    tclean_summary: Optional[TcleanRunSummary]
    qa: QAReport
    pipeline_background: Optional[PipelineBackground] = None
