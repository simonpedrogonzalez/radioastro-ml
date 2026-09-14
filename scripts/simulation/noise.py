"""Thermal-noise calculations and the small simulation noise selector."""

from __future__ import annotations

import json
import math
import operator
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple


VLA_OSS_2026A_SOURCE = (
    "https://science.nrao.edu/facilities/vla/docs/manuals/"
    "oss2026a/performance/sensitivity"
)
VLA_OSS_2026A_SEFD_JY = {
    "P": 2790.0,
    "L": 420.0,
    "S": 370.0,
    "C": 310.0,
    "X": 250.0,
    "KU": 320.0,
    "K": 500.0,
    "KA": 600.0,
    "Q": 1300.0,
}
VLA_OSS_2026A_ETA_C_8BIT = 0.93
VLA_OSS_2026A_ETA_C_3BIT_RANGE = (0.78, 0.83)

_VLA_BAND_RANGES_GHZ = {
    "P": (0.224, 0.480),
    "L": (1.0, 2.0),
    "S": (2.0, 4.0),
    "C": (4.0, 8.0),
    "X": (8.0, 12.0),
    "KU": (12.0, 18.0),
    "K": (18.0, 26.5),
    "KA": (26.5, 40.0),
    "Q": (40.0, 50.0),
}
_SUPPORTED_NOISE_MODELS = {"vla-thermal", "simplenoise"}
_RESERVED_NOISE_MODELS = {"tsys-atm", "tsys-manual"}


class _HomogeneousSampling(NamedTuple):
    spw_id: int
    effective_bw_hz: float
    exposure_s: float
    channel_frequencies_hz: tuple[float, ...]


def _new_table():
    try:
        from casatools import table
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatools is required to inspect Measurement Sets") from exc
    return table()


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "ravel"):
        value = value.ravel().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    result: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, (str, bytes)):
            result.append(item)
        elif isinstance(item, Iterable):
            for child in item:
                visit(child)
        else:
            result.append(item)

    visit(value)
    return result


def _positive_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _positive_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _one_constant_positive(values: Iterable[Any], *, name: str) -> float:
    numbers = [float(item) for item in values]
    if not numbers:
        raise RuntimeError(f"No {name} values were found")
    first = _positive_float(numbers[0], name=name)
    if any(not math.isfinite(item) or item <= 0 for item in numbers):
        raise RuntimeError(f"{name} contains non-positive or non-finite values")
    if any(not math.isclose(item, first, rel_tol=1e-12, abs_tol=0.0) for item in numbers[1:]):
        unique = sorted(set(numbers))
        preview = unique[:8]
        suffix = "..." if len(unique) > len(preview) else ""
        raise ValueError(f"{name} is not homogeneous: {preview}{suffix}")
    return first


def _frequency_scale(unit: str) -> float:
    scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    normalized = str(unit).strip().lower()
    if normalized not in scales:
        raise RuntimeError(f"Unsupported frequency unit {unit!r}")
    return scales[normalized]


def _time_scale(unit: str) -> float:
    scales = {"s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}
    normalized = str(unit).strip().lower()
    if normalized not in scales:
        raise RuntimeError(f"Unsupported exposure unit {unit!r}")
    return scales[normalized]


def _single_column_unit(tb: Any, column: str, *, expected: str) -> str:
    keywords = dict(tb.getcolkeywords(column) or {})
    units = [str(item) for item in _flat_values(keywords.get("QuantumUnits", []))]
    if len(units) != 1:
        raise RuntimeError(f"{column} must declare one {expected} unit, found {units!r}")
    return units[0]


def _inspect_homogeneous_sampling(ms: str | Path) -> _HomogeneousSampling:
    path = Path(ms).expanduser().resolve()
    tb = _new_table()
    tb.open(str(path), nomodify=True)
    try:
        columns = set(tb.colnames())
        required = {"DATA_DESC_ID", "EXPOSURE", "ANTENNA1", "ANTENNA2"}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"MS MAIN table is missing required columns: {missing}")
        data_description_ids = [int(item) for item in _flat_values(tb.getcol("DATA_DESC_ID"))]
        exposure_unit = _single_column_unit(tb, "EXPOSURE", expected="time")
        exposures = [
            float(item) * _time_scale(exposure_unit)
            for item in _flat_values(tb.getcol("EXPOSURE"))
        ]
        antenna1 = [int(item) for item in _flat_values(tb.getcol("ANTENNA1"))]
        antenna2 = [int(item) for item in _flat_values(tb.getcol("ANTENNA2"))]
    finally:
        tb.close()

    if not data_description_ids:
        raise RuntimeError(f"Measurement Set has no MAIN rows: {path}")
    row_count = len(data_description_ids)
    if any(len(values) != row_count for values in (exposures, antenna1, antenna2)):
        raise RuntimeError(
            "MS MAIN exposure/antenna/data-description columns have inconsistent lengths"
        )
    if any(left == right for left, right in zip(antenna1, antenna2)):
        raise ValueError("Autocorrelation rows are not supported by the VLA thermal formula")
    exposure_s = _one_constant_positive(exposures, name="MAIN.EXPOSURE")

    used_data_descriptions = sorted(set(data_description_ids))
    tb = _new_table()
    tb.open(str(path / "DATA_DESCRIPTION"), nomodify=True)
    try:
        nrows = int(tb.nrows())
        if any(item < 0 or item >= nrows for item in used_data_descriptions):
            raise RuntimeError(
                f"MAIN.DATA_DESC_ID refers outside DATA_DESCRIPTION ({nrows} rows)"
            )
        used_spws = sorted(
            {int(tb.getcell("SPECTRAL_WINDOW_ID", item)) for item in used_data_descriptions}
        )
    finally:
        tb.close()
    if len(used_spws) != 1:
        raise ValueError(f"Expected exactly one used spectral window, found {used_spws}")
    spw_id = used_spws[0]

    tb = _new_table()
    tb.open(str(path / "SPECTRAL_WINDOW"), nomodify=True)
    try:
        if spw_id < 0 or spw_id >= int(tb.nrows()):
            raise RuntimeError(
                f"DATA_DESCRIPTION refers to SPW {spw_id}, but SPECTRAL_WINDOW has "
                f"{int(tb.nrows())} rows"
            )
        bandwidth_unit = _single_column_unit(tb, "EFFECTIVE_BW", expected="frequency")
        frequency_unit = _single_column_unit(tb, "CHAN_FREQ", expected="frequency")
        effective_bandwidths = [
            float(item) * _frequency_scale(bandwidth_unit)
            for item in _flat_values(tb.getcell("EFFECTIVE_BW", spw_id))
        ]
        channel_frequencies = tuple(
            float(item) * _frequency_scale(frequency_unit)
            for item in _flat_values(tb.getcell("CHAN_FREQ", spw_id))
        )
    finally:
        tb.close()

    effective_bw_hz = _one_constant_positive(
        effective_bandwidths, name="SPECTRAL_WINDOW.EFFECTIVE_BW"
    )
    if not channel_frequencies or any(
        not math.isfinite(item) or item <= 0 for item in channel_frequencies
    ):
        raise RuntimeError("SPECTRAL_WINDOW.CHAN_FREQ contains no valid frequencies")
    return _HomogeneousSampling(
        spw_id,
        effective_bw_hz,
        exposure_s,
        channel_frequencies,
    )


def _theoretical_sigma(sefd_jy: float, eta_c: float, sampling: _HomogeneousSampling) -> float:
    return sefd_jy / (
        eta_c * math.sqrt(2.0 * sampling.effective_bw_hz * sampling.exposure_s)
    )


def theoretical_vla_simplenoise(
    ms: str | Path,
    *,
    sefd_jy: float,
    eta_c: float,
) -> float:
    """Return constant CASA ``simplenoise`` sigma for homogeneous VLA samples."""
    sefd = _positive_float(sefd_jy, name="sefd_jy")
    efficiency = _positive_float(eta_c, name="eta_c")
    return _theoretical_sigma(sefd, efficiency, _inspect_homogeneous_sampling(ms))


def simplenoise_from_image_rms(
    sigma_na_jy_per_beam: float,
    *,
    nchan: int,
    npol: int,
    nbaselines: int,
    nintegrations: int,
) -> float:
    """Convert ideal natural-image RMS to constant CASA ``simplenoise`` sigma."""
    image_sigma = _positive_float(sigma_na_jy_per_beam, name="sigma_na_jy_per_beam")
    sample_count = (
        _positive_count(nchan, name="nchan")
        * _positive_count(npol, name="npol")
        * _positive_count(nbaselines, name="nbaselines")
        * _positive_count(nintegrations, name="nintegrations")
    )
    return image_sigma * math.sqrt(sample_count)


def natural_image_rms_from_simplenoise(
    ms: str | Path,
    simplenoise_jy: float,
    *,
    chunk_rows: int = 4096,
) -> float:
    """Predict natural-weight Stokes-I RMS from flags and constant visibility noise.

    CASA ``simplenoise`` is the standard deviation of each real and imaginary
    visibility component.  With constant synthetic weights, the natural-image
    RMS is that value divided by the square root of the number of unflagged
    parallel-hand samples retained by the Measurement Set.
    """
    sigma = _positive_float(simplenoise_jy, name="simplenoise_jy")
    rows_per_chunk = _positive_count(chunk_rows, name="chunk_rows")
    path = Path(ms).expanduser().resolve()

    tb = _new_table()
    tb.open(str(path / "POLARIZATION"), nomodify=True)
    try:
        correlation_types = {
            row: tuple(int(item) for item in _flat_values(tb.getcell("CORR_TYPE", row)))
            for row in range(int(tb.nrows()))
        }
    finally:
        tb.close()

    # CASA Stokes enumeration: RR=5, LL=8, XX=9, YY=12.
    parallel_indices: dict[int, tuple[int, ...]] = {}
    for polarization_id, types in correlation_types.items():
        indices = tuple(index for index, code in enumerate(types) if code in {5, 8, 9, 12})
        codes = {types[index] for index in indices}
        if not ({5, 8} <= codes or {9, 12} <= codes):
            raise ValueError(
                "POLARIZATION.CORR_TYPE must contain RR/LL or XX/YY for "
                f"polarization row {polarization_id}; found {types}"
            )
        parallel_indices[polarization_id] = indices

    tb.open(str(path / "DATA_DESCRIPTION"), nomodify=True)
    try:
        data_description_to_polarization = {
            row: int(tb.getcell("POLARIZATION_ID", row))
            for row in range(int(tb.nrows()))
        }
    finally:
        tb.close()

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - CASA supplies NumPy
        raise RuntimeError("NumPy is required to count unflagged visibility samples") from exc

    usable_samples = 0
    tb.open(str(path), nomodify=True)
    try:
        columns = set(tb.colnames())
        required = {"DATA_DESC_ID", "FLAG"}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"MS MAIN table is missing required columns: {missing}")
        total_rows = int(tb.nrows())
        for start in range(0, total_rows, rows_per_chunk):
            count = min(rows_per_chunk, total_rows - start)
            flags = np.asarray(
                tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool
            )
            if flags.ndim != 3 or flags.shape[-1] != count:
                raise RuntimeError(
                    f"Unexpected FLAG shape {flags.shape}; expected correlations x channels x rows"
                )
            description_ids = np.asarray(
                tb.getcol("DATA_DESC_ID", startrow=start, nrow=count), dtype=int
            ).reshape(-1)
            if description_ids.size != count:
                raise RuntimeError("DATA_DESC_ID and FLAG row counts do not match")
            if "FLAG_ROW" in columns:
                row_flags = np.asarray(
                    tb.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
                ).reshape(-1)
                if row_flags.size != count:
                    raise RuntimeError("FLAG_ROW and FLAG row counts do not match")
            else:
                row_flags = np.zeros(count, dtype=bool)

            for description_id in np.unique(description_ids):
                try:
                    polarization_id = data_description_to_polarization[int(description_id)]
                    indices = parallel_indices[polarization_id]
                except KeyError as exc:
                    raise RuntimeError(
                        f"DATA_DESC_ID {int(description_id)} has no valid polarization mapping"
                    ) from exc
                selected_rows = np.flatnonzero(description_ids == description_id)
                selected_flags = flags[np.asarray(indices), :, :][:, :, selected_rows]
                usable = ~selected_flags
                usable &= ~row_flags[selected_rows].reshape(1, 1, -1)
                usable_samples += int(np.count_nonzero(usable))
    finally:
        tb.close()

    if usable_samples <= 0:
        raise ValueError(f"Measurement Set has no unflagged parallel-hand samples: {path}")
    return sigma / math.sqrt(usable_samples)


def _normalized_noise_request(
    noise_model: str,
    noise_parameters: Mapping[str, object] | None,
) -> tuple[str, dict[str, object]]:
    model = str(noise_model).strip().lower()
    if model in _RESERVED_NOISE_MODELS:
        raise NotImplementedError(
            f"noise_model={model!r} is reserved but not yet supported: CASA creates "
            "non-constant noise without updating SIGMA/WEIGHT, so a validated "
            "varying-weight policy is required"
        )
    if model not in _SUPPORTED_NOISE_MODELS:
        choices = ", ".join(sorted(_SUPPORTED_NOISE_MODELS))
        raise ValueError(f"Unknown noise_model {noise_model!r}; supported values: {choices}")
    if noise_parameters is None:
        parameters: dict[str, object] = {}
    elif not isinstance(noise_parameters, Mapping):
        raise TypeError("noise_parameters must be a mapping")
    else:
        parameters = dict(noise_parameters)
    if any(not isinstance(key, str) for key in parameters):
        raise TypeError("noise_parameters keys must be strings")
    try:
        json.dumps(parameters, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("noise_parameters must contain JSON-serializable finite values") from exc
    forbidden = {"mode", "seed"} & set(parameters)
    if forbidden:
        raise ValueError(
            f"noise_parameters cannot contain {sorted(forbidden)}; use the named arguments"
        )

    allowed = (
        {"band", "sampler", "sefd_jy", "eta_c"}
        if model == "vla-thermal"
        else {"simplenoise"}
    )
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown parameters for noise_model={model!r}: {unknown}")
    required = {"band", "sampler"} if model == "vla-thermal" else {"simplenoise"}
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"Missing parameters for noise_model={model!r}: {missing}")
    return model, parameters


def _normalize_band(value: object) -> str:
    band = str(value).strip().upper()
    if band not in VLA_OSS_2026A_SEFD_JY:
        raise ValueError(
            f"Unknown VLA band {value!r}; expected one of {sorted(VLA_OSS_2026A_SEFD_JY)}"
        )
    return band


def _normalize_sampler(value: object) -> str:
    sampler = str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if sampler not in {"8bit", "3bit"}:
        raise ValueError("sampler must be '8bit' or '3bit'")
    return sampler


def _validate_band_frequencies(band: str, frequencies_hz: Iterable[float]) -> None:
    lower, upper = _VLA_BAND_RANGES_GHZ[band]
    frequencies_ghz = tuple(float(item) / 1e9 for item in frequencies_hz)
    outside = [item for item in frequencies_ghz if not lower <= item <= upper]
    if outside:
        raise ValueError(
            f"Selected channels ({min(frequencies_ghz):.9g}--{max(frequencies_ghz):.9g} GHz) "
            f"do not all fall in VLA {band} band ({lower:g}--{upper:g} GHz)"
        )


def _quantity_in_jy(value: object) -> float:
    try:
        from casatools import quanta
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatools is required to parse simplenoise") from exc
    qa = quanta()
    try:
        quantity = qa.quantity(value)
        converted = qa.convert(quantity, "Jy")
        raw = converted.get("value") if isinstance(converted, dict) else None
        values = _flat_values(raw)
        if len(values) != 1:
            raise ValueError
        return _positive_float(values[0], name="simplenoise")
    except Exception as exc:
        raise ValueError(
            f"simplenoise must be a positive CASA flux-density quantity, got {value!r}"
        ) from exc


def _apply_noise(
    simulator: object,
    ms: str | Path,
    *,
    noise_model: str,
    noise_parameters: Mapping[str, object] | None,
    seed: int,
) -> dict[str, object]:
    """Apply a supported noise selection and return normalized run metadata."""
    model, parameters = _normalized_noise_request(noise_model, noise_parameters)
    random_seed = _positive_count(seed, name="seed")

    if model == "vla-thermal":
        sampling = _inspect_homogeneous_sampling(ms)
        band = _normalize_band(parameters["band"])
        sampler = _normalize_sampler(parameters["sampler"])
        _validate_band_frequencies(band, sampling.channel_frequencies_hz)
        sefd_source = "override" if "sefd_jy" in parameters else "VLA OSS 2026A fiducial"
        eta_source = "override" if "eta_c" in parameters else "VLA OSS 2026A 8-bit"
        sefd = _positive_float(
            parameters.get("sefd_jy", VLA_OSS_2026A_SEFD_JY[band]), name="sefd_jy"
        )
        if "eta_c" in parameters:
            efficiency = _positive_float(parameters["eta_c"], name="eta_c")
        elif sampler == "8bit":
            efficiency = VLA_OSS_2026A_ETA_C_8BIT
        else:
            raise ValueError(
                "eta_c is required for the 3-bit sampler because the OSS publishes a range"
            )
        sigma_jy = _theoretical_sigma(sefd, efficiency, sampling)
        resolved_parameters: dict[str, object] = {
            "band": band,
            "sampler": sampler,
            "sefd_jy": sefd,
            "sefd_source": sefd_source,
            "eta_c": efficiency,
            "eta_c_source": eta_source,
            "effective_bw_hz": sampling.effective_bw_hz,
            "exposure_s": sampling.exposure_s,
            "spw_id": sampling.spw_id,
            "reference": VLA_OSS_2026A_SOURCE,
        }
    else:
        sigma_jy = _quantity_in_jy(parameters["simplenoise"])
        resolved_parameters = {"simplenoise": parameters["simplenoise"]}

    casa_noise = f"{sigma_jy:.16g}Jy"
    if not simulator.setseed(random_seed):
        raise RuntimeError("CASA simulator.setseed() failed")
    if not simulator.setnoise(mode="simplenoise", simplenoise=casa_noise):
        raise RuntimeError("CASA simulator.setnoise() failed")
    if not simulator.corrupt():
        raise RuntimeError("CASA simulator.corrupt() failed")
    return {
        "noise_model": model,
        "requested_parameters": parameters,
        "resolved_parameters": resolved_parameters,
        "casa_mode": "simplenoise",
        "casa_parameters": {"simplenoise": casa_noise},
        "simplenoise_jy": sigma_jy,
        "seed": random_seed,
        "weight_initialization": "sigma",
    }


def _set_constant_sigma(ms: str | Path, sigma_jy: float, *, chunk_rows: int = 4096) -> None:
    sigma = _positive_float(sigma_jy, name="sigma_jy")
    rows_per_chunk = _positive_count(chunk_rows, name="chunk_rows")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - CASA supplies numpy
        raise RuntimeError("NumPy is required to initialize MS SIGMA") from exc

    path = Path(ms).expanduser().resolve()
    tb = _new_table()
    tb.open(str(path), nomodify=False)
    try:
        if "SIGMA" not in set(tb.colnames()):
            raise RuntimeError(f"Measurement Set has no SIGMA column: {path}")
        total_rows = int(tb.nrows())
        for start in range(0, total_rows, rows_per_chunk):
            count = min(rows_per_chunk, total_rows - start)
            values = np.asarray(tb.getcol("SIGMA", startrow=start, nrow=count))
            values.fill(sigma)
            tb.putcol("SIGMA", values, startrow=start, nrow=count)
    finally:
        tb.close()


__all__ = [
    "VLA_OSS_2026A_ETA_C_3BIT_RANGE",
    "VLA_OSS_2026A_ETA_C_8BIT",
    "VLA_OSS_2026A_SEFD_JY",
    "VLA_OSS_2026A_SOURCE",
    "natural_image_rms_from_simplenoise",
    "simplenoise_from_image_rms",
    "theoretical_vla_simplenoise",
]
