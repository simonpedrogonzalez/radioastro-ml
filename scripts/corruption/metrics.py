"""Exact visibility-domain corruption metrics and constant-gain solving."""

from __future__ import annotations

import math
import operator
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


_PARALLEL_HAND_CORRELATIONS = frozenset((5, 8, 9, 12))
_DEFAULT_CHUNK_ROWS = 4096


def _finite_positive(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return result


def _existing_ms(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or path.suffix.lower() != ".ms":
        raise ValueError(f"{name} must be an existing .ms directory, got {path}")
    return path


def _antenna_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"antenna_id must be a nonnegative integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"antenna_id must be a nonnegative integer, got {value!r}"
        ) from exc
    if result < 0:
        raise ValueError(f"antenna_id must be a nonnegative integer, got {value!r}")
    return int(result)


@dataclass(frozen=True)
class ConstantGainSpec:
    """Inputs for a one-antenna constant gain at ``SNR_corr_target``."""

    V_ms: Path
    antenna_id: int
    corruption_type: Literal["amp", "phase"]
    SNR_corr_target: float
    sigma: float
    seed: int
    sign: Literal[-1, 1] = field(init=False)

    def __post_init__(self) -> None:
        if self.corruption_type not in ("amp", "phase"):
            raise ValueError(
                "corruption_type must be exactly 'amp' or 'phase', "
                f"got {self.corruption_type!r}"
            )
        if isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        try:
            seed = operator.index(self.seed)
        except TypeError as exc:
            raise ValueError(f"seed must be an integer, got {self.seed!r}") from exc
        object.__setattr__(self, "V_ms", _existing_ms(self.V_ms, name="V_ms"))
        object.__setattr__(self, "antenna_id", _antenna_id(self.antenna_id))
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "sign", random.Random(seed).choice((-1, 1)))
        object.__setattr__(
            self,
            "SNR_corr_target",
            _finite_positive(self.SNR_corr_target, name="SNR_corr_target"),
        )
        object.__setattr__(self, "sigma", _finite_positive(self.sigma, name="sigma"))


@dataclass(frozen=True)
class CorruptionMetricDefinition:
    key: str
    formula: str
    latex: str
    description: str
    unit: str = "dimensionless"

    def __repr__(self) -> str:
        return f"{self.key}={self.formula}"

    def to_report_dict(self) -> dict[str, str]:
        return asdict(self)


CORRUPTION_METRIC_DEFINITIONS: tuple[CorruptionMetricDefinition, ...] = (
    CorruptionMetricDefinition(
        key="eps_g",
        formula="|g-1|",
        latex=r"\epsilon_g=|g-1|",
        description="Magnitude of the constant antenna gain displacement.",
    ),
    CorruptionMetricDefinition(
        key="eps_vis",
        formula="||Delta_V||_2/||V||_2",
        latex=(
            r"\epsilon_{\mathrm{vis}}="
            r"\frac{\lVert\Delta\mathbf V\rVert_2}{\lVert\mathbf V\rVert_2}"
        ),
        description="Visibility corruption relative to the noiseless sky signal.",
    ),
    CorruptionMetricDefinition(
        key="SNR_corr",
        formula="||Delta_V/sigma||_2",
        latex=(
            r"\mathrm{SNR}_{\mathrm{corr}}="
            r"\left\lVert\frac{\Delta\mathbf V}{\boldsymbol\sigma}\right\rVert_2"
        ),
        description="Noise-weighted L2 magnitude of the visibility corruption.",
    ),
)


def corruption_metric_definitions() -> tuple[dict[str, str], ...]:
    return tuple(item.to_report_dict() for item in CORRUPTION_METRIC_DEFINITIONS)


@dataclass(frozen=True)
class ConstantGainNorms:
    valid_sample_count: int
    affected_sample_count: int
    V_L2: float
    V_Ak_L2: float
    V_Ak_over_sigma_L2: float


@dataclass(frozen=True)
class ConstantGainSolution:
    corruption_type: Literal["amp", "phase"]
    antenna_id: int
    sign: Literal[-1, 1]
    sigma: float
    SNR_corr_target: float
    SNR_corr_expected: float
    eps_g: float
    eps_vis_expected: float
    g_amp: float | None
    phi_rad: float | None
    phi_deg: float | None
    norms: ConstantGainNorms

    @property
    def constant_value(self) -> float:
        value = self.g_amp if self.corruption_type == "amp" else self.phi_rad
        assert value is not None
        return value

    def to_report_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "type": "constant_one_antenna_SNR_corr_solution",
            "visibility_source": "noiseless_predicted_DATA",
            "metric_definitions": list(corruption_metric_definitions()),
        }


@dataclass(frozen=True)
class CorruptionMetrics:
    valid_sample_count: int
    V_L2: float
    Delta_V_L2: float
    Delta_V_over_sigma_L2: float
    eps_vis: float
    SNR_corr: float

    def to_report_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "visibility_source": "noiseless_predicted_DATA",
            "delta_definition": "Delta_V=V_corr-V",
            "metric_definitions": list(corruption_metric_definitions()),
        }


def _new_table():
    try:
        from casatools import table
    except ImportError as exc:
        raise RuntimeError(
            "CASA casatools is required to calculate corruption metrics"
        ) from exc
    return table()


def _read_correlation_indices(ms: Path) -> dict[int, tuple[int, ...]]:
    tb = _new_table()
    tb.open(str(ms / "DATA_DESCRIPTION"))
    try:
        polarization_ids = np.asarray(tb.getcol("POLARIZATION_ID"), dtype=int).reshape(-1)
    finally:
        tb.close()

    tb = _new_table()
    tb.open(str(ms / "POLARIZATION"))
    try:
        correlations = {
            row: np.asarray(tb.getcell("CORR_TYPE", row), dtype=int).reshape(-1)
            for row in range(int(tb.nrows()))
        }
    finally:
        tb.close()

    result: dict[int, tuple[int, ...]] = {}
    for data_description_id, polarization_id in enumerate(polarization_ids):
        if int(polarization_id) not in correlations:
            raise ValueError(
                f"DATA_DESCRIPTION row {data_description_id} references missing "
                f"POLARIZATION_ID {int(polarization_id)}"
            )
        result[data_description_id] = tuple(
            int(index)
            for index, code in enumerate(correlations[int(polarization_id)])
            if int(code) in _PARALLEL_HAND_CORRELATIONS
        )
    if not result or not any(result.values()):
        raise ValueError("MS has no RR, LL, XX, or YY correlations")
    return result


def _correlation_mask(
    data_shape: tuple[int, ...],
    data_description_ids: np.ndarray,
    correlation_indices: dict[int, tuple[int, ...]],
) -> np.ndarray:
    count = data_shape[2]
    mask = np.zeros((data_shape[0], count), dtype=bool)
    for data_description_id in np.unique(data_description_ids):
        indices = correlation_indices.get(int(data_description_id))
        if indices is None:
            raise ValueError(
                f"MS row references missing DATA_DESCRIPTION_ID "
                f"{int(data_description_id)}"
            )
        if any(index >= data_shape[0] for index in indices):
            raise ValueError(
                f"Correlation mapping for DATA_DESCRIPTION_ID "
                f"{int(data_description_id)} does not match DATA shape {data_shape}"
            )
        if indices:
            mask[np.ix_(indices, data_description_ids == data_description_id)] = True
    return mask


def _chunk_metadata(tb, columns: set[str], start: int, count: int):
    antenna1 = np.asarray(
        tb.getcol("ANTENNA1", startrow=start, nrow=count), dtype=int
    ).reshape(-1)
    antenna2 = np.asarray(
        tb.getcol("ANTENNA2", startrow=start, nrow=count), dtype=int
    ).reshape(-1)
    data_description_ids = np.asarray(
        tb.getcol("DATA_DESC_ID", startrow=start, nrow=count), dtype=int
    ).reshape(-1)
    flag_row = (
        np.asarray(
            tb.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
        ).reshape(-1)
        if "FLAG_ROW" in columns
        else np.zeros(count, dtype=bool)
    )
    if any(
        array.size != count
        for array in (antenna1, antenna2, data_description_ids, flag_row)
    ):
        raise ValueError("MS row metadata shapes do not match DATA row count")
    return antenna1, antenna2, data_description_ids, flag_row


def _valid_mask(
    data: np.ndarray,
    flags: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flag_row: np.ndarray,
    data_description_ids: np.ndarray,
    correlation_indices: dict[int, tuple[int, ...]],
) -> np.ndarray:
    if data.ndim != 3 or flags.shape != data.shape:
        raise ValueError(
            "DATA and FLAG must share shape (correlation, channel, row), got "
            f"{data.shape} and {flags.shape}"
        )
    correlations = _correlation_mask(
        data.shape, data_description_ids, correlation_indices
    )
    return (
        ((~flag_row) & (antenna1 != antenna2))[None, None, :]
        & correlations[:, None, :]
        & ~flags
        & np.isfinite(data.real)
        & np.isfinite(data.imag)
    )


def measure_constant_gain_norms(
    spec: ConstantGainSpec,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
) -> ConstantGainNorms:
    """Measure exact L2 norms from noiseless ``V`` in bounded chunks."""
    if not isinstance(spec, ConstantGainSpec):
        raise TypeError("spec must be ConstantGainSpec")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive integer, got {chunk_rows!r}")

    correlation_indices = _read_correlation_indices(spec.V_ms)
    tb = _new_table()
    tb.open(str(spec.V_ms / "ANTENNA"))
    try:
        antenna_count = int(tb.nrows())
    finally:
        tb.close()
    if spec.antenna_id >= antenna_count:
        raise ValueError(
            f"antenna_id {spec.antenna_id} does not exist; MS has "
            f"{antenna_count} antennas"
        )

    valid_count = affected_count = 0
    V_L2_sq = V_Ak_L2_sq = 0.0
    tb = _new_table()
    tb.open(str(spec.V_ms))
    try:
        columns = set(tb.colnames())
        required = {"DATA", "FLAG", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"MS is missing required columns: {', '.join(missing)}")
        total_rows = int(tb.nrows())
        for start in range(0, total_rows, chunk_rows):
            count = min(chunk_rows, total_rows - start)
            V = np.asarray(
                tb.getcol("DATA", startrow=start, nrow=count), dtype=np.complex128
            )
            flags = np.asarray(
                tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool
            )
            antenna1, antenna2, data_description_ids, flag_row = _chunk_metadata(
                tb, columns, start, count
            )
            valid = _valid_mask(
                V,
                flags,
                antenna1,
                antenna2,
                flag_row,
                data_description_ids,
                correlation_indices,
            )
            if not np.any(valid):
                continue
            power = V.real * V.real + V.imag * V.imag
            valid_count += int(np.count_nonzero(valid))
            V_L2_sq += float(np.sum(power[valid], dtype=np.float64))
            affected_rows = (antenna1 == spec.antenna_id) | (
                antenna2 == spec.antenna_id
            )
            affected = valid & affected_rows[None, None, :]
            affected_count += int(np.count_nonzero(affected))
            V_Ak_L2_sq += float(np.sum(power[affected], dtype=np.float64))
    finally:
        tb.close()

    if valid_count == 0 or not math.isfinite(V_L2_sq) or V_L2_sq <= 0.0:
        raise ValueError("No finite positive noiseless visibility norm remains")
    if affected_count == 0 or not math.isfinite(V_Ak_L2_sq) or V_Ak_L2_sq <= 0.0:
        raise ValueError(
            f"No finite positive visibility norm involves antenna_id {spec.antenna_id}"
        )
    V_L2 = math.sqrt(V_L2_sq)
    V_Ak_L2 = math.sqrt(V_Ak_L2_sq)
    return ConstantGainNorms(
        valid_sample_count=valid_count,
        affected_sample_count=affected_count,
        V_L2=V_L2,
        V_Ak_L2=V_Ak_L2,
        V_Ak_over_sigma_L2=V_Ak_L2 / spec.sigma,
    )


def solve_constant_gain(
    spec: ConstantGainSpec,
    norms: ConstantGainNorms | None = None,
) -> ConstantGainSolution:
    """Solve the note's constant one-antenna ``SNR_corr`` equation."""
    if not isinstance(spec, ConstantGainSpec):
        raise TypeError("spec must be ConstantGainSpec")
    measured = measure_constant_gain_norms(spec) if norms is None else norms
    if not isinstance(measured, ConstantGainNorms):
        raise TypeError("norms must be ConstantGainNorms")

    eps_g = spec.SNR_corr_target / measured.V_Ak_over_sigma_L2
    if not math.isfinite(eps_g) or eps_g <= 0.0:
        raise ValueError(f"eps_g must be finite and positive, got {eps_g!r}")

    g_amp = phi_rad = phi_deg = None
    if spec.corruption_type == "amp":
        g_amp = 1.0 + spec.sign * eps_g
        if not math.isfinite(g_amp) or g_amp <= 0.0:
            raise ValueError(
                "Requested amplitude branch is non-positive: "
                f"1 + sign*eps_g = {g_amp!r}"
            )
    else:
        if eps_g > 2.0:
            raise ValueError(f"Phase corruption requires eps_g <= 2, got {eps_g!r}")
        phi_rad = spec.sign * 2.0 * math.asin(eps_g / 2.0)
        phi_deg = math.degrees(phi_rad)

    return ConstantGainSolution(
        corruption_type=spec.corruption_type,
        antenna_id=spec.antenna_id,
        sign=spec.sign,
        sigma=spec.sigma,
        SNR_corr_target=spec.SNR_corr_target,
        SNR_corr_expected=eps_g * measured.V_Ak_over_sigma_L2,
        eps_g=eps_g,
        eps_vis_expected=eps_g * measured.V_Ak_L2 / measured.V_L2,
        g_amp=g_amp,
        phi_rad=phi_rad,
        phi_deg=phi_deg,
        norms=measured,
    )


def measure_corruption_metrics(
    V_ms: str | Path,
    V_corr_ms: str | Path,
    sigma: float,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
) -> CorruptionMetrics:
    """Measure ``Delta_V=V_corr-V`` before thermal noise is added."""
    V_path = _existing_ms(V_ms, name="V_ms")
    V_corr_path = _existing_ms(V_corr_ms, name="V_corr_ms")
    sigma_value = _finite_positive(sigma, name="sigma")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive integer, got {chunk_rows!r}")
    correlation_indices = _read_correlation_indices(V_path)

    V_tb = _new_table()
    V_corr_tb = _new_table()
    V_tb.open(str(V_path))
    V_corr_tb.open(str(V_corr_path))
    try:
        V_columns = set(V_tb.colnames())
        V_corr_columns = set(V_corr_tb.colnames())
        required = {"DATA", "FLAG", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID"}
        for name, columns in (("V_ms", V_columns), ("V_corr_ms", V_corr_columns)):
            missing = sorted(required - columns)
            if missing:
                raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
        total_rows = int(V_tb.nrows())
        if int(V_corr_tb.nrows()) != total_rows:
            raise ValueError("V_ms and V_corr_ms have different row counts")

        valid_count = 0
        V_L2_sq = Delta_V_L2_sq = 0.0
        for start in range(0, total_rows, chunk_rows):
            count = min(chunk_rows, total_rows - start)
            V = np.asarray(
                V_tb.getcol("DATA", startrow=start, nrow=count), dtype=np.complex128
            )
            V_corr = np.asarray(
                V_corr_tb.getcol("DATA", startrow=start, nrow=count),
                dtype=np.complex128,
            )
            if V_corr.shape != V.shape:
                raise ValueError(
                    f"V and V_corr DATA shapes differ: {V.shape} vs {V_corr.shape}"
                )
            V_flags = np.asarray(
                V_tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool
            )
            V_corr_flags = np.asarray(
                V_corr_tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool
            )
            antenna1, antenna2, data_description_ids, flag_row = _chunk_metadata(
                V_tb, V_columns, start, count
            )
            corr_ant1, corr_ant2, corr_ddid, corr_flag_row = _chunk_metadata(
                V_corr_tb, V_corr_columns, start, count
            )
            if not (
                np.array_equal(antenna1, corr_ant1)
                and np.array_equal(antenna2, corr_ant2)
                and np.array_equal(data_description_ids, corr_ddid)
            ):
                raise ValueError("V_ms and V_corr_ms row metadata differ")
            valid = _valid_mask(
                V,
                V_flags | V_corr_flags,
                antenna1,
                antenna2,
                flag_row | corr_flag_row,
                data_description_ids,
                correlation_indices,
            )
            valid &= np.isfinite(V_corr.real) & np.isfinite(V_corr.imag)
            if not np.any(valid):
                continue
            Delta_V = V_corr - V
            V_power = V.real * V.real + V.imag * V.imag
            Delta_V_power = (
                Delta_V.real * Delta_V.real + Delta_V.imag * Delta_V.imag
            )
            valid_count += int(np.count_nonzero(valid))
            V_L2_sq += float(np.sum(V_power[valid], dtype=np.float64))
            Delta_V_L2_sq += float(
                np.sum(Delta_V_power[valid], dtype=np.float64)
            )
    finally:
        V_corr_tb.close()
        V_tb.close()

    if valid_count == 0 or not math.isfinite(V_L2_sq) or V_L2_sq <= 0.0:
        raise ValueError("No finite positive noiseless visibility norm remains")
    if not math.isfinite(Delta_V_L2_sq) or Delta_V_L2_sq <= 0.0:
        raise ValueError("Delta_V has no finite positive L2 norm")
    V_L2 = math.sqrt(V_L2_sq)
    Delta_V_L2 = math.sqrt(Delta_V_L2_sq)
    Delta_V_over_sigma_L2 = Delta_V_L2 / sigma_value
    return CorruptionMetrics(
        valid_sample_count=valid_count,
        V_L2=V_L2,
        Delta_V_L2=Delta_V_L2,
        Delta_V_over_sigma_L2=Delta_V_over_sigma_L2,
        eps_vis=Delta_V_L2 / V_L2,
        SNR_corr=Delta_V_over_sigma_L2,
    )


__all__ = [
    "CORRUPTION_METRIC_DEFINITIONS",
    "ConstantGainNorms",
    "ConstantGainSolution",
    "ConstantGainSpec",
    "CorruptionMetricDefinition",
    "CorruptionMetrics",
    "corruption_metric_definitions",
    "measure_constant_gain_norms",
    "measure_corruption_metrics",
    "solve_constant_gain",
]
