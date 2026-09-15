"""Detectability inputs, results, and bounded-memory MS power measurement."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


_PARALLEL_HAND_CORRELATIONS = frozenset((5, 8, 9, 12))
_POWER_ESTIMATOR = "noise_debiased_data_power"
_DEFAULT_CHUNK_ROWS = 4096
_WEIGHT_RELATIVE_TOLERANCE = 1e-6


def _finite_positive(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return result


@dataclass(frozen=True)
class ConstantGainParameters:
    """Inputs for a one-antenna constant corruption at a target detectability."""

    ms: Path
    antenna_id: int
    corruption_type: Literal["amp", "phase"]
    target_rho_corr: float
    thermal_noise_jy: float
    sign: Literal[-1, 1] = 1

    def __post_init__(self) -> None:
        ms = Path(self.ms).expanduser().resolve()
        if not ms.is_dir() or ms.suffix.lower() != ".ms":
            raise ValueError(f"ms must be an existing .ms directory, got {ms}")
        if isinstance(self.antenna_id, bool):
            raise ValueError(
                f"antenna_id must be a nonnegative integer, got {self.antenna_id!r}"
            )
        try:
            antenna_id = operator.index(self.antenna_id)
        except TypeError as exc:
            raise ValueError(
                f"antenna_id must be a nonnegative integer, got {self.antenna_id!r}"
            ) from exc
        if antenna_id < 0:
            raise ValueError(
                f"antenna_id must be a nonnegative integer, got {self.antenna_id!r}"
            )
        if self.corruption_type not in ("amp", "phase"):
            raise ValueError(
                "corruption_type must be exactly 'amp' or 'phase', "
                f"got {self.corruption_type!r}"
            )
        if isinstance(self.sign, bool):
            raise ValueError(f"sign must be exactly -1 or 1, got {self.sign!r}")
        try:
            sign = operator.index(self.sign)
        except TypeError as exc:
            raise ValueError(f"sign must be exactly -1 or 1, got {self.sign!r}") from exc
        if sign not in (-1, 1):
            raise ValueError(f"sign must be exactly -1 or 1, got {self.sign!r}")
        object.__setattr__(self, "ms", ms)
        object.__setattr__(self, "antenna_id", int(antenna_id))
        object.__setattr__(self, "sign", int(sign))
        object.__setattr__(
            self,
            "target_rho_corr",
            _finite_positive(self.target_rho_corr, name="target_rho_corr"),
        )
        object.__setattr__(
            self,
            "thermal_noise_jy",
            _finite_positive(self.thermal_noise_jy, name="thermal_noise_jy"),
        )

    @classmethod
    def from_simulation_report(
        cls,
        simulation_report,
        *,
        antenna_id,
        corruption_type,
        target_rho_corr,
        sign=1,
    ) -> "ConstantGainParameters":
        """Build parameters from an existing simulation JSON report or mapping."""
        report_path: Path | None = None
        if isinstance(simulation_report, Mapping):
            payload = dict(simulation_report)
        else:
            from scripts.simulation.reporting import load_simulation_report

            report_path = Path(simulation_report).expanduser().resolve()
            payload = load_simulation_report(report_path)

        output_ms = payload.get("output_ms")
        noise = payload.get("noise")
        if not isinstance(output_ms, (str, Path)) or not str(output_ms).strip():
            raise ValueError("simulation report is missing output_ms")
        if not isinstance(noise, Mapping) or "simplenoise_jy" not in noise:
            raise ValueError("simulation report is missing noise.simplenoise_jy")

        ms = Path(output_ms).expanduser()
        if not ms.is_absolute() and report_path is not None:
            ms = report_path.parent / ms
        return cls(
            ms=ms,
            antenna_id=antenna_id,
            corruption_type=corruption_type,
            target_rho_corr=target_rho_corr,
            thermal_noise_jy=noise["simplenoise_jy"],
            sign=sign,
        )


@dataclass(frozen=True)
class DetectabilityMetricDefinition:
    """Stable formula and explanation used by every corruption report."""

    key: str
    name: str
    formula: str
    latex: str
    description: str

    def __repr__(self) -> str:
        return f"{self.name}={self.formula}"

    def to_report_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "name": self.name,
            "formula": self.formula,
            "latex": self.latex,
            "description": self.description,
        }


DETECTABILITY_METRIC_DEFINITIONS: tuple[DetectabilityMetricDefinition, ...] = (
    DetectabilityMetricDefinition(
        key="gain_error_magnitude",
        name="epsilon_g",
        formula="|g-1|",
        latex=r"\epsilon_g=|g-1|",
        description="Physical complex-gain displacement on affected baselines.",
    ),
    DetectabilityMetricDefinition(
        key="epsilon_vis",
        name="epsilon_vis",
        formula="epsilon_g*sqrt(P_A/P_all)",
        latex=r"\epsilon_{\mathrm{vis}}=\epsilon_g\sqrt{P_A/P_{\mathrm{all}}}",
        description="Fractional signal perturbation across the full visibility dataset.",
    ),
    DetectabilityMetricDefinition(
        key="rho_corr",
        name="rho_corr",
        formula="epsilon_g*sqrt(P_A,w)",
        latex=r"\rho_{\mathrm{corr}}=\epsilon_g\sqrt{P_{A,w}}",
        description=(
            "Optimal aggregate visibility-space S/N of the injected signal "
            "perturbation; it is not an image-domain artifact S/N."
        ),
    ),
)


def detectability_metric_definitions() -> tuple[dict[str, str], ...]:
    """Return the canonical detectability definitions for external reports."""
    return tuple(
        definition.to_report_dict()
        for definition in DETECTABILITY_METRIC_DEFINITIONS
    )


@dataclass(frozen=True)
class DetectabilityMetrics:
    """Reported derivation for a detectability-controlled constant gain."""

    corruption_type: Literal["amp", "phase"]
    antenna_id: int
    sign: Literal[-1, 1]
    gain_error_magnitude: float
    amplitude_gain: float | None
    phase_offset_rad: float | None
    phase_offset_deg: float | None
    epsilon_vis: float
    rho_corr: float
    target_rho_corr: float
    unit_gain_detectability: float
    thermal_noise_jy: float
    valid_sample_count: int
    affected_sample_count: int
    total_signal_power: float
    affected_signal_power: float
    weighted_affected_signal_power: float
    power_estimator: str = _POWER_ESTIMATOR

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "constant_one_antenna_detectability",
            "power_estimator": self.power_estimator,
            "corruption_type": self.corruption_type,
            "antenna_id": self.antenna_id,
            "sign": self.sign,
            "thermal_noise_jy": self.thermal_noise_jy,
            "valid_sample_count": self.valid_sample_count,
            "affected_sample_count": self.affected_sample_count,
            "total_signal_power": self.total_signal_power,
            "affected_signal_power": self.affected_signal_power,
            "weighted_affected_signal_power": self.weighted_affected_signal_power,
            "unit_gain_detectability": self.unit_gain_detectability,
            "gain_error_magnitude": self.gain_error_magnitude,
            "amplitude_gain": self.amplitude_gain,
            "phase_offset_rad": self.phase_offset_rad,
            "phase_offset_deg": self.phase_offset_deg,
            "epsilon_vis": self.epsilon_vis,
            "rho_corr": self.rho_corr,
            "target_rho_corr": self.target_rho_corr,
            "metric_definitions": list(detectability_metric_definitions()),
        }

    def to_report_text(self) -> str:
        def number(value: float) -> str:
            return format(float(value), ".17g")

        physical_value = (
            f"amplitude gain = {number(self.amplitude_gain)}"
            if self.amplitude_gain is not None
            else (
                f"phase offset = {number(self.phase_offset_rad)} rad "
                f"({number(self.phase_offset_deg)} deg)"
            )
        )
        definitions = {
            definition.key: definition
            for definition in DETECTABILITY_METRIC_DEFINITIONS
        }
        return "\n".join(
            [
                "Detectability metrics",
                "-" * 80,
                "Power estimator: noise-debiased DATA over the full unflagged MS",
                (
                    f"Thermal sigma: {number(self.thermal_noise_jy)} Jy per "
                    "real/imaginary component"
                ),
                (
                    f"Valid samples: {self.valid_sample_count}; affected samples: "
                    f"{self.affected_sample_count}"
                ),
                f"Constant corruption: {physical_value}; sign = {self.sign:+d}",
                "",
                (
                    f"A_k=sqrt(P_A,w)=sqrt("
                    f"{number(self.weighted_affected_signal_power)})="
                    f"{number(self.unit_gain_detectability)}"
                ),
                "",
                (
                    f"{definitions['gain_error_magnitude']!r}="
                    f"rho_target/A_k={number(self.target_rho_corr)}/"
                    f"{number(self.unit_gain_detectability)}="
                    f"{number(self.gain_error_magnitude)}"
                ),
                f"  {definitions['gain_error_magnitude'].description}",
                "",
                (
                    f"{definitions['epsilon_vis']!r}="
                    f"{number(self.gain_error_magnitude)}*"
                    f"sqrt({number(self.affected_signal_power)} / "
                    f"{number(self.total_signal_power)})="
                    f"{number(self.epsilon_vis)}"
                ),
                f"  {definitions['epsilon_vis'].description}",
                "",
                (
                    f"{definitions['rho_corr']!r}="
                    f"{number(self.gain_error_magnitude)}*"
                    f"sqrt({number(self.weighted_affected_signal_power)})="
                    f"{number(self.rho_corr)} "
                    f"(requested {number(self.target_rho_corr)})"
                ),
                f"  {definitions['rho_corr'].description}",
                "",
                (
                    "Interpretation limitation: rho_corr controls the estimated deterministic "
                    "sky-signal perturbation; it does not guarantee equal image-domain or "
                    "classification difficulty. Amplitude corruption also rescales affected "
                    "noise, while phase corruption rotates circular noise."
                ),
            ]
        )

    def __repr__(self) -> str:
        return (
            "DetectabilityMetrics("
            f"corruption_type={self.corruption_type!r}, antenna_id={self.antenna_id}, "
            f"gain_error_magnitude={self.gain_error_magnitude!r}, "
            f"epsilon_vis={self.epsilon_vis!r}, rho_corr={self.rho_corr!r})"
        )

    def __str__(self) -> str:
        return self.to_report_text()


@dataclass(frozen=True)
class _PowerSummary:
    valid_sample_count: int
    affected_sample_count: int
    total_signal_power: float
    affected_signal_power: float
    weighted_affected_signal_power: float


def _new_table():
    try:
        from casatools import table
    except ImportError as exc:
        raise RuntimeError(
            "CASA casatools is required to calculate corruption detectability"
        ) from exc
    return table()


def _read_correlation_indices(ms: Path) -> dict[int, tuple[int, ...]]:
    tb = _new_table()
    tb.open(str(ms / "DATA_DESCRIPTION"))
    try:
        columns = set(tb.colnames())
        if "POLARIZATION_ID" not in columns:
            raise ValueError("MS DATA_DESCRIPTION table is missing POLARIZATION_ID")
        polarization_ids = np.asarray(tb.getcol("POLARIZATION_ID"), dtype=int).reshape(-1)
    finally:
        tb.close()

    tb = _new_table()
    tb.open(str(ms / "POLARIZATION"))
    try:
        if "CORR_TYPE" not in set(tb.colnames()):
            raise ValueError("MS POLARIZATION table is missing CORR_TYPE")
        correlations_by_polarization = {
            row: np.asarray(tb.getcell("CORR_TYPE", row), dtype=int).reshape(-1)
            for row in range(int(tb.nrows()))
        }
    finally:
        tb.close()

    result: dict[int, tuple[int, ...]] = {}
    for data_description_id, polarization_id in enumerate(polarization_ids):
        correlations = correlations_by_polarization.get(int(polarization_id))
        if correlations is None:
            raise ValueError(
                f"DATA_DESCRIPTION row {data_description_id} references missing "
                f"POLARIZATION_ID {int(polarization_id)}"
            )
        result[data_description_id] = tuple(
            int(index)
            for index, code in enumerate(correlations)
            if int(code) in _PARALLEL_HAND_CORRELATIONS
        )
    if not result or not any(result.values()):
        raise ValueError("MS has no RR, LL, XX, or YY correlations")
    return result


def _broadcast_weights(weight: np.ndarray, data_shape: tuple[int, ...]) -> np.ndarray:
    if weight.ndim != 2 or len(data_shape) != 3:
        raise ValueError(
            f"WEIGHT must have shape (correlation, row), got {weight.shape}"
        )
    expected = (data_shape[0], data_shape[2])
    if weight.shape != expected:
        raise ValueError(f"WEIGHT must have shape {expected}, got {weight.shape}")
    return np.broadcast_to(weight[:, None, :], data_shape)


def _chunk_weights(
    tb,
    columns: set[str],
    data_shape: tuple[int, ...],
    start: int,
    count: int,
) -> np.ndarray:
    spectrum: np.ndarray | None = None
    if "WEIGHT_SPECTRUM" in columns:
        try:
            candidate = np.asarray(
                tb.getcol("WEIGHT_SPECTRUM", startrow=start, nrow=count),
                dtype=np.float64,
            )
            if candidate.shape == data_shape:
                spectrum = candidate
        except Exception:
            spectrum = None

    if spectrum is not None and np.all(np.isfinite(spectrum) & (spectrum > 0.0)):
        return spectrum

    base: np.ndarray | None = None
    if "WEIGHT" in columns:
        try:
            base = _broadcast_weights(
                np.asarray(
                    tb.getcol("WEIGHT", startrow=start, nrow=count),
                    dtype=np.float64,
                ),
                data_shape,
            )
        except Exception:
            base = None

    if spectrum is None and base is None:
        raise ValueError("MS has neither usable WEIGHT_SPECTRUM nor WEIGHT")
    if spectrum is None:
        assert base is not None
        return base
    if base is None:
        return spectrum
    return np.where(np.isfinite(spectrum) & (spectrum > 0.0), spectrum, base)


def _measure_visibility_powers(
    parameters: ConstantGainParameters,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
) -> _PowerSummary:
    """Scan DATA in bounded chunks and return noise-debiased power aggregates."""
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive integer, got {chunk_rows!r}")

    correlation_indices = _read_correlation_indices(parameters.ms)

    tb = _new_table()
    tb.open(str(parameters.ms / "ANTENNA"))
    try:
        antenna_count = int(tb.nrows())
    finally:
        tb.close()
    if parameters.antenna_id >= antenna_count:
        raise ValueError(
            f"antenna_id {parameters.antenna_id} does not exist; MS has "
            f"{antenna_count} antennas"
        )

    valid_count = 0
    affected_count = 0
    total_observed_power = 0.0
    affected_observed_power = 0.0
    affected_weighted_observed_power = 0.0
    affected_weight_sum = 0.0
    expected_weight = 1.0 / parameters.thermal_noise_jy**2

    tb = _new_table()
    tb.open(str(parameters.ms))
    try:
        columns = set(tb.colnames())
        required = {"DATA", "FLAG", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"MS is missing required columns: {', '.join(missing)}")
        if "WEIGHT" not in columns and "WEIGHT_SPECTRUM" not in columns:
            raise ValueError("MS is missing both WEIGHT_SPECTRUM and WEIGHT")

        total_rows = int(tb.nrows())
        for start in range(0, total_rows, chunk_rows):
            count = min(chunk_rows, total_rows - start)
            data = np.asarray(
                tb.getcol("DATA", startrow=start, nrow=count), dtype=np.complex128
            )
            flags = np.asarray(
                tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool
            )
            if data.ndim != 3 or flags.shape != data.shape:
                raise ValueError(
                    f"DATA and FLAG must share shape (correlation, channel, row), got "
                    f"{data.shape} and {flags.shape}"
                )
            antenna1 = np.asarray(
                tb.getcol("ANTENNA1", startrow=start, nrow=count), dtype=int
            ).reshape(-1)
            antenna2 = np.asarray(
                tb.getcol("ANTENNA2", startrow=start, nrow=count), dtype=int
            ).reshape(-1)
            data_description_ids = np.asarray(
                tb.getcol("DATA_DESC_ID", startrow=start, nrow=count), dtype=int
            ).reshape(-1)
            if any(
                array.size != count
                for array in (antenna1, antenna2, data_description_ids)
            ):
                raise ValueError("MS row metadata shapes do not match DATA row count")
            flag_row = (
                np.asarray(
                    tb.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
                ).reshape(-1)
                if "FLAG_ROW" in columns
                else np.zeros(count, dtype=bool)
            )
            if flag_row.size != count:
                raise ValueError("FLAG_ROW shape does not match DATA row count")

            weights = _chunk_weights(tb, columns, data.shape, start, count)
            correlation_mask = np.zeros((data.shape[0], count), dtype=bool)
            for data_description_id in np.unique(data_description_ids):
                indices = correlation_indices.get(int(data_description_id))
                if indices is None:
                    raise ValueError(
                        f"MS row references missing DATA_DESCRIPTION_ID "
                        f"{int(data_description_id)}"
                    )
                if any(index >= data.shape[0] for index in indices):
                    raise ValueError(
                        f"Correlation mapping for DATA_DESCRIPTION_ID "
                        f"{int(data_description_id)} does not match DATA shape {data.shape}"
                    )
                if indices:
                    correlation_mask[
                        np.ix_(indices, data_description_ids == data_description_id)
                    ] = True

            row_mask = (~flag_row) & (antenna1 != antenna2)
            finite_data = np.isfinite(data.real) & np.isfinite(data.imag)
            finite_positive_weight = np.isfinite(weights) & (weights > 0.0)
            valid = (
                row_mask[None, None, :]
                & correlation_mask[:, None, :]
                & ~flags
                & finite_data
                & finite_positive_weight
            )
            if not np.any(valid):
                continue
            selected_weights = weights[valid]
            if not np.allclose(
                selected_weights,
                expected_weight,
                rtol=_WEIGHT_RELATIVE_TOLERANCE,
                atol=0.0,
            ):
                maximum_relative_error = float(
                    np.max(np.abs(selected_weights - expected_weight) / expected_weight)
                )
                raise ValueError(
                    "MS weights are inconsistent with 1 / thermal_noise_jy**2: "
                    f"expected {expected_weight!r}, maximum relative error "
                    f"{maximum_relative_error!r}"
                )

            power = data.real * data.real + data.imag * data.imag
            valid_count += int(np.count_nonzero(valid))
            total_observed_power += float(np.sum(power[valid], dtype=np.float64))

            affected_rows = (antenna1 == parameters.antenna_id) | (
                antenna2 == parameters.antenna_id
            )
            affected = valid & affected_rows[None, None, :]
            if np.any(affected):
                affected_count += int(np.count_nonzero(affected))
                affected_observed_power += float(
                    np.sum(power[affected], dtype=np.float64)
                )
                affected_weighted_observed_power += float(
                    np.sum(weights[affected] * power[affected], dtype=np.float64)
                )
                affected_weight_sum += float(
                    np.sum(weights[affected], dtype=np.float64)
                )
    finally:
        tb.close()

    if valid_count == 0:
        raise ValueError("No valid full-MS parallel-hand visibility samples remain")
    if affected_count == 0:
        raise ValueError(
            f"No valid visibility samples involve antenna_id {parameters.antenna_id}"
        )

    noise_power = 2.0 * parameters.thermal_noise_jy**2
    total_signal_power = total_observed_power - noise_power * valid_count
    affected_signal_power = affected_observed_power - noise_power * affected_count
    weighted_affected_signal_power = (
        affected_weighted_observed_power - noise_power * affected_weight_sum
    )
    aggregates = {
        "total_signal_power": total_signal_power,
        "affected_signal_power": affected_signal_power,
        "weighted_affected_signal_power": weighted_affected_signal_power,
    }
    for name, value in aggregates.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and positive after aggregate thermal-noise "
                f"debiasing, got {value!r}"
            )
    return _PowerSummary(
        valid_sample_count=valid_count,
        affected_sample_count=affected_count,
        total_signal_power=float(total_signal_power),
        affected_signal_power=float(affected_signal_power),
        weighted_affected_signal_power=float(weighted_affected_signal_power),
    )


__all__ = [
    "ConstantGainParameters",
    "DETECTABILITY_METRIC_DEFINITIONS",
    "DetectabilityMetricDefinition",
    "DetectabilityMetrics",
    "detectability_metric_definitions",
]
