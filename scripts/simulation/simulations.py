"""Linear CASA simulation flow on a copied Measurement Set."""

from __future__ import annotations

import json
import math
import operator
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts.imaging.metadata import resolve_path

from .noise import _apply_noise, _normalized_noise_request, _set_constant_sigma


@dataclass(frozen=True)
class SimulationResult:
    ms_path: Path
    component_list: Path | None
    metadata_json: Path
    simplenoise_jy: float | None
    seed: int | None


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


def _output_paths(output_ms: str | Path) -> tuple[Path, Path, Path]:
    ms_path = Path(output_ms).expanduser().resolve()
    if ms_path.suffix.lower() != ".ms":
        raise ValueError(f"output_ms must end in .ms: {ms_path}")
    base = ms_path.with_suffix("")
    return (
        ms_path,
        Path(f"{base}.components.cl"),
        Path(f"{base}.simulation.json"),
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


def simulate_ms(
    ms: str | Path,
    components: Sequence[Mapping[str, object]],
    output_ms: str | Path,
    *,
    noise_model: str | None = None,
    noise_parameters: Mapping[str, object] | None = None,
    seed: int = 185349251,
) -> SimulationResult:
    """Predict components and optional selected noise into a new copied MS."""
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

    destination, component_path, metadata_path = _output_paths(output_ms)
    if destination == source:
        raise ValueError("output_ms must not be the input Measurement Set")
    paths_to_check = [destination, metadata_path]
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
            if normalized_noise is not None:
                model, parameters = normalized_noise
                noise_metadata = _apply_noise(
                    sm,
                    destination,
                    noise_model=model,
                    noise_parameters=parameters,
                    seed=random_seed,
                )
        finally:
            sm.close()

        sigma = None if noise_metadata is None else float(noise_metadata["simplenoise_jy"])
        weight_mode = _initialize_weights(destination, sigma)
        payload = {
            "schema_version": 1,
            "input_ms": str(source),
            "output_ms": str(destination),
            "component_list": str(component_path) if component_records else None,
            "components": component_records,
            "noise": noise_metadata,
            "weight_initialization": weight_mode,
            "casa_version": _casa_version(),
            "created_paths": [
                str(path)
                for path in (
                    destination,
                    component_path if component_records else None,
                    metadata_path,
                )
                if path is not None
            ],
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        created.append(metadata_path)
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
    )


__all__ = ["SimulationResult", "simulate_ms"]
