"""Linear CASA simulation flow on a copied Measurement Set."""

from __future__ import annotations

import json
import math
import operator
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from scripts.imaging.metadata import resolve_path

from .noise import _apply_noise, _normalized_noise_request, _set_constant_sigma
from .reporting import SIMULATION_REPORT_SCHEMA_VERSION, write_simulation_reports


@dataclass(frozen=True)
class SimulationResult:
    ms_path: Path
    component_list: Path | None
    metadata_json: Path
    simplenoise_jy: float | None
    seed: int | None
    metadata_text: Path | None = None


def _positive_count(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _output_paths(output_ms: str | Path) -> tuple[Path, Path, Path, Path]:
    ms_path = Path(output_ms).expanduser().resolve()
    if ms_path.suffix.lower() != ".ms":
        raise ValueError(f"output_ms must end in .ms: {ms_path}")
    base = ms_path.with_suffix("")
    return (
        ms_path,
        Path(f"{base}.components.cl"),
        Path(f"{base}.simulation.json"),
        Path(f"{base}.simulation.txt"),
    )


def _normalize_components(
    components: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("components must be a sequence of component mappings")
    result: list[dict[str, object]] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise TypeError(f"components[{index}] must be a mapping")
        normalized = dict(component)
        if not normalized:
            raise ValueError(f"components[{index}] cannot be empty")
        result.append(normalized)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("components must contain JSON-serializable finite values") from exc
    return result


def _zero_visibility_data(ms_path: Path, *, chunk_rows: int = 4096) -> None:
    rows_per_chunk = _positive_count(chunk_rows, name="chunk_rows")
    try:
        import numpy as np
        from casatools import table
    except ImportError as exc:  # pragma: no cover - CASA supplies these modules
        raise RuntimeError("CASA casatools and NumPy are required to zero MS data") from exc

    tb = table()
    tb.open(str(ms_path), nomodify=False)
    try:
        columns = set(tb.colnames())
        targets = [name for name in ("DATA", "CORRECTED_DATA") if name in columns]
        if not targets:
            raise RuntimeError(f"Measurement Set has neither DATA nor CORRECTED_DATA: {ms_path}")
        total_rows = int(tb.nrows())
        for column in targets:
            for start in range(0, total_rows, rows_per_chunk):
                count = min(rows_per_chunk, total_rows - start)
                values = np.asarray(tb.getcol(column, startrow=start, nrow=count))
                values.fill(0)
                tb.putcol(column, values, startrow=start, nrow=count)
    finally:
        tb.close()


def _write_component_list(
    components: Sequence[Mapping[str, object]], component_path: Path
) -> None:
    try:
        from casatools import componentlist
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatools is required to create component lists") from exc

    cl = componentlist()
    try:
        for index, component in enumerate(components):
            if not cl.addcomponent(**dict(component)):
                raise RuntimeError(f"CASA rejected component {index}")
        if not cl.rename(str(component_path)):
            raise RuntimeError(f"CASA could not create component list {component_path}")
    finally:
        cl.done()


def _casa_version() -> str | None:
    try:
        import casatools
    except ImportError:
        return None
    for name in ("version_string", "version"):
        function = getattr(casatools, name, None)
        if callable(function):
            try:
                value = function()
            except Exception:
                continue
            return str(value)
    return None


def _repository_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


SIMULATION_FLAG_POLICIES = ("unflag",)


def clear_flags_and_validate(ms_path: str | Path, *, chunk_rows: int = 4096) -> dict[str, object]:
    """Clear FLAG/FLAG_ROW of a simulation copy, then require usable sampling geometry.

    Every cross-correlation row must have finite, non-zero UVW; at least three antennas must
    occur. Returns counts of previously flagged samples/rows for the simulation metadata.
    """
    rows_per_chunk = _positive_count(chunk_rows, name="chunk_rows")
    try:
        import numpy as np
        from casatools import table
    except ImportError as exc:  # pragma: no cover - CASA supplies these modules
        raise RuntimeError("CASA casatools and NumPy are required to clear MS flags") from exc

    path = resolve_path(ms_path).path
    tb = table()
    tb.open(str(path), nomodify=False)
    flagged_samples = flagged_rows = total_samples = 0
    antennas: set[int] = set()
    try:
        columns = set(tb.colnames())
        if "FLAG" not in columns:
            raise RuntimeError(f"Measurement Set has no FLAG column: {path}")
        total_rows = int(tb.nrows())
        if total_rows == 0:
            raise ValueError(f"Measurement Set has no rows: {path}")
        for start in range(0, total_rows, rows_per_chunk):
            count = min(rows_per_chunk, total_rows - start)
            flags = np.asarray(tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool)
            flagged_samples += int(flags.sum())
            total_samples += int(flags.size)
            tb.putcol("FLAG", np.zeros_like(flags), startrow=start, nrow=count)
            if "FLAG_ROW" in columns:
                row_flags = np.asarray(tb.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool)
                flagged_rows += int(row_flags.sum())
                tb.putcol("FLAG_ROW", np.zeros_like(row_flags), startrow=start, nrow=count)
            uvw = np.asarray(tb.getcol("UVW", startrow=start, nrow=count), dtype=float)
            a1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=count), dtype=int)
            a2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=count), dtype=int)
            cross = a1 != a2
            if not np.all(np.isfinite(uvw[:, cross])):
                raise ValueError(f"Unflagged sampling has non-finite UVW: {path}")
            if np.any(np.sqrt((uvw[:, cross] ** 2).sum(axis=0)) == 0):
                raise ValueError(f"Unflagged sampling has zero-length cross-correlation baselines: {path}")
            antennas.update(a1[cross].tolist())
            antennas.update(a2[cross].tolist())
    finally:
        tb.close()
    if len(antennas) < 3:
        raise ValueError(f"Unflagged sampling has fewer than three antennas: {path}")
    return {
        "policy": "unflag",
        "cleared_flagged_samples": flagged_samples,
        "cleared_flagged_rows": flagged_rows,
        "total_samples": total_samples,
        "antennas": len(antennas),
    }


def _initialize_weights(ms_path: Path, sigma_jy: float | None) -> str:
    try:
        from casatasks import initweights
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatasks is required to initialize MS weights") from exc
    if sigma_jy is None:
        initweights(vis=str(ms_path), wtmode="ones", dowtsp=True)
        return "ones"
    if not math.isfinite(sigma_jy) or sigma_jy <= 0:
        raise RuntimeError(f"Resolved noise sigma is invalid: {sigma_jy!r}")
    _set_constant_sigma(ms_path, sigma_jy)
    initweights(vis=str(ms_path), wtmode="sigma", dowtsp=True)
    return "sigma"


def initialize_weights(ms: str | Path, sigma_jy: float | None) -> str:
    """Initialize homogeneous MS weights for a resolved per-component sigma."""
    return _initialize_weights(resolve_path(ms).path, sigma_jy)


def add_thermal_noise_inplace(
    ms: str | Path,
    *,
    noise_model: str,
    noise_parameters: Mapping[str, object] | None,
    seed: int,
) -> dict[str, object]:
    """Add thermal noise to an existing MS and initialize its noise weights."""
    path = resolve_path(ms).path
    model, parameters = _normalized_noise_request(noise_model, noise_parameters)
    random_seed = _positive_count(seed, name="seed")
    try:
        from casatools import simulator
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatools is required to add thermal noise") from exc
    sm = simulator()
    try:
        if not sm.openfromms(str(path)):
            raise RuntimeError(f"CASA simulator could not open {path}")
        metadata = _apply_noise(
            sm,
            path,
            noise_model=model,
            noise_parameters=parameters,
            seed=random_seed,
        )
    finally:
        sm.close()
    sigma = float(metadata["simplenoise_jy"])
    metadata = dict(metadata)
    metadata["weight_initialization"] = _initialize_weights(path, sigma)
    return metadata


def simulate_ms(
    ms: str | Path,
    components: Sequence[Mapping[str, object]],
    output_ms: str | Path,
    *,
    noise_model: str | None = None,
    noise_parameters: Mapping[str, object] | None = None,
    seed: int = 185349251,
    simulation_flags: str | None = None,
) -> SimulationResult:
    """Predict components and optional selected noise into a new copied MS.

    simulation_flags="unflag" clears the copy's inherited FLAG/FLAG_ROW before prediction
    (the input MS is never modified); None keeps the input flags.
    """
    if simulation_flags is not None and simulation_flags not in SIMULATION_FLAG_POLICIES:
        raise ValueError(f"Unsupported simulation_flags {simulation_flags!r}")
    source = resolve_path(ms).path
    component_records = _normalize_components(components)
    if noise_model is None:
        if noise_parameters:
            raise ValueError("noise_parameters requires noise_model")
        normalized_noise = None
        random_seed = None
    else:
        normalized_noise = _normalized_noise_request(noise_model, noise_parameters)
        random_seed = _positive_count(seed, name="seed")
    if not component_records and normalized_noise is None:
        raise ValueError("At least one component or a noise_model is required")

    destination, component_path, metadata_path, metadata_text_path = _output_paths(output_ms)
    if destination == source:
        raise ValueError("output_ms must not be the input Measurement Set")
    paths_to_check = [destination, metadata_path, metadata_text_path]
    if component_records:
        paths_to_check.append(component_path)
    existing = [path for path in paths_to_check if path.exists()]
    if existing:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(f"Refusing to overwrite simulation products:\n{rendered}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    noise_metadata: dict[str, object] | None = None
    try:
        shutil.copytree(source, destination)
        created.append(destination)
        flag_summary = None
        if simulation_flags == "unflag":
            flag_summary = clear_flags_and_validate(destination)
        if component_records:
            _write_component_list(component_records, component_path)
            created.append(component_path)
        else:
            _zero_visibility_data(destination)

        try:
            from casatools import simulator
        except ImportError as exc:  # pragma: no cover - exercised outside CASA
            raise RuntimeError("CASA casatools is required to simulate Measurement Sets") from exc
        sm = simulator()
        try:
            if not sm.openfromms(str(destination)):
                raise RuntimeError(f"CASA simulator could not open {destination}")
            if component_records and not sm.predict(
                complist=str(component_path), incremental=False
            ):
                raise RuntimeError(f"CASA simulator prediction failed for {destination}")
        finally:
            sm.close()

        if normalized_noise is not None:
            model, parameters = normalized_noise
            noise_metadata = add_thermal_noise_inplace(
                destination,
                noise_model=model,
                noise_parameters=parameters,
                seed=random_seed,
            )
        sigma = None if noise_metadata is None else float(noise_metadata["simplenoise_jy"])
        weight_mode = (
            _initialize_weights(destination, None)
            if noise_metadata is None
            else str(noise_metadata["weight_initialization"])
        )
        operations = ["copy"]
        if flag_summary is not None:
            operations.append("unflag")
        if component_records:
            operations.extend(("predict",))
        else:
            operations.append("zero_visibility_data")
        if noise_metadata is not None:
            operations.append("noise")
        operations.append("weights")
        payload = {
            "schema_version": SIMULATION_REPORT_SCHEMA_VERSION,
            "sample_id": destination.with_suffix("").name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": _repository_commit(),
            "input_ms": str(source),
            "output_ms": str(destination),
            "input_visibility": {
                "local_path": str(source),
                "archive_locator": None,
                "checksum": None,
            },
            "component_list": str(component_path) if component_records else None,
            "components": component_records,
            "noise": noise_metadata,
            "weight_initialization": weight_mode,
            "casa_version": _casa_version(),
            "operations": operations,
            "stages": [
                {"name": operation, "order": index}
                for index, operation in enumerate(operations, start=1)
            ],
            "generated_artifacts": [
                {
                    "role": "output_ms",
                    "path": str(destination),
                    "lifecycle": "temporary",
                },
                *(
                    [
                        {
                            "role": "component_list",
                            "path": str(component_path),
                            "lifecycle": "temporary",
                        }
                    ]
                    if component_records
                    else []
                ),
            ],
            "status": "success",
            "warnings": [],
            **({"simulation_flags": flag_summary} if flag_summary is not None else {}),
            "created_paths": [
                str(path)
                for path in (
                    destination,
                    component_path if component_records else None,
                    metadata_path,
                    metadata_text_path,
                )
                if path is not None
            ],
        }
        write_simulation_reports(payload, metadata_path, metadata_text_path)
        created.extend((metadata_path, metadata_text_path))
    except Exception as exc:
        if hasattr(exc, "add_note") and created:
            exc.add_note(
                "Partial simulation products created by this call:\n"
                + "\n".join(f"  - {path}" for path in created if path.exists())
            )
        raise

    return SimulationResult(
        ms_path=destination,
        component_list=component_path if component_records else None,
        metadata_json=metadata_path,
        simplenoise_jy=None if noise_metadata is None else float(noise_metadata["simplenoise_jy"]),
        seed=None if noise_metadata is None else int(noise_metadata["seed"]),
        metadata_text=metadata_text_path,
    )


__all__ = ["SIMULATION_FLAG_POLICIES", "SimulationResult", "add_thermal_noise_inplace",
           "clear_flags_and_validate", "simulate_ms"]
