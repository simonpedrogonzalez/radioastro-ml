"""Strict, versioned JSON and text reporting for simulations."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SIMULATION_REPORT_SCHEMA_VERSION = 3
SUPPORTED_SIMULATION_REPORT_SCHEMAS = (1, 2, 3)


def load_simulation_report(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read simulation report {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Simulation report must contain an object: {source}")
    version = payload.get("schema_version")
    if version not in SUPPORTED_SIMULATION_REPORT_SCHEMAS:
        raise ValueError(f"Unsupported simulation report schema_version {version!r}")
    return payload


def render_simulation_text(report: Mapping[str, Any]) -> str:
    components = report.get("components") or []
    noise = report.get("noise")
    lines = [
        "Simulation Report",
        "=" * 80,
        f"Schema version: {report.get('schema_version')}",
        f"Sample ID: {report.get('sample_id', 'unknown')}",
        f"Created: {report.get('created_at', 'unknown')}",
        f"Input MS: {report.get('input_ms', 'unknown')}",
        f"Output MS: {report.get('output_ms', 'unknown')}",
        f"CASA version: {report.get('casa_version') or 'unknown'}",
        f"Repository commit: {report.get('repository_commit') or 'unknown'}",
        f"Components: {len(components) if isinstance(components, list) else 'unknown'}",
        "Noise model: "
        + (
            str(noise.get("noise_model", "unknown"))
            if isinstance(noise, Mapping)
            else "none"
        ),
        f"Weight initialization: {report.get('weight_initialization', 'unknown')}",
        "",
        "Operation stages",
        "-" * 80,
    ]
    stages = report.get("stages") or []
    if isinstance(stages, list) and stages:
        for stage in stages:
            if isinstance(stage, Mapping):
                lines.append(f"- {stage.get('order')}: {stage.get('name')}")
    else:
        operations = report.get("operations") or []
        if isinstance(operations, list):
            lines.extend(f"- {operation}" for operation in operations)
    return "\n".join(lines).rstrip() + "\n"


def _temporary(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_simulation_reports(
    report: Mapping[str, Any],
    json_path: str | Path,
    text_path: str | Path,
) -> tuple[Path, Path]:
    json_destination = Path(json_path).expanduser().resolve()
    text_destination = Path(text_path).expanduser().resolve()
    if json_destination == text_destination:
        raise ValueError("Simulation JSON and text report paths must differ")
    existing = [path for path in (json_destination, text_destination) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite simulation reports:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    if report.get("schema_version") != SIMULATION_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"New simulation reports require schema_version {SIMULATION_REPORT_SCHEMA_VERSION}"
        )
    try:
        json_text = json.dumps(
            dict(report), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Simulation report is not strict JSON: {exc}") from exc
    text = render_simulation_text(report)
    json_temporary = text_temporary = None
    installed: list[Path] = []
    try:
        json_temporary = _temporary(json_destination, json_text)
        text_temporary = _temporary(text_destination, text)
        os.replace(json_temporary, json_destination)
        json_temporary = None
        installed.append(json_destination)
        os.replace(text_temporary, text_destination)
        text_temporary = None
        installed.append(text_destination)
    except Exception:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    finally:
        if json_temporary is not None:
            json_temporary.unlink(missing_ok=True)
        if text_temporary is not None:
            text_temporary.unlink(missing_ok=True)
    return json_destination, text_destination


__all__ = [
    "SIMULATION_REPORT_SCHEMA_VERSION",
    "SUPPORTED_SIMULATION_REPORT_SCHEMAS",
    "load_simulation_report",
    "render_simulation_text",
    "write_simulation_reports",
]
