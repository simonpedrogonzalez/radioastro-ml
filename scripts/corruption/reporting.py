"""Generic JSON and text reporting for self-describing corruption objects."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


CORRUPTION_REPORT_SCHEMA_VERSION = 1


class ReportableCorruption(Protocol):
    def to_report_dict(self) -> dict[str, object]: ...

    def to_report_text(self) -> str: ...


@dataclass(frozen=True)
class CorruptionReportPaths:
    json_path: Path
    text_path: Path


def _strict_json_value(value: Any, *, location: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string mapping key: {key!r}")
            result[key] = _strict_json_value(item, location=f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _strict_json_value(item_method(), location=location)
        except (TypeError, ValueError):
            raise
        except Exception:
            pass
    raise TypeError(f"{location} contains unsupported value {type(value).__name__}")


def _context_text(context: Mapping[str, Any]) -> list[str]:
    if not context:
        return []
    lines = ["", "Context", "-" * 80]
    for key in sorted(context):
        value = context[key]
        if isinstance(value, (dict, list)):
            displayed = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            displayed = str(value)
        lines.append(f"{key}: {displayed}")
    return lines


def _write_temporary(path: Path, text: str) -> Path:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_corruption_reports(
    corruption: ReportableCorruption,
    *,
    json_path: str | Path,
    text_path: str | Path,
    context: Mapping[str, Any] | None = None,
) -> CorruptionReportPaths:
    """Write reports without knowing the concrete corruption configuration."""
    dictionary_method = getattr(corruption, "to_report_dict", None)
    text_method = getattr(corruption, "to_report_text", None)
    if not callable(dictionary_method) or not callable(text_method):
        raise TypeError(
            "corruption must implement to_report_dict() and to_report_text()"
        )

    configuration = dictionary_method()
    if not isinstance(configuration, Mapping):
        raise TypeError("corruption.to_report_dict() must return a mapping")
    normalized_configuration = _strict_json_value(
        configuration, location="configuration"
    )
    normalized_context = _strict_json_value(context or {}, location="context")
    rendered_configuration = text_method()
    if not isinstance(rendered_configuration, str) or not rendered_configuration.strip():
        raise TypeError("corruption.to_report_text() must return non-empty text")

    metrics = getattr(corruption, "metrics", None)
    if metrics is None:
        normalized_metrics = None
        rendered_metrics = "Detectability metrics: not calculated"
    else:
        metrics_dictionary_method = getattr(metrics, "to_report_dict", None)
        metrics_text_method = getattr(metrics, "to_report_text", None)
        if not callable(metrics_dictionary_method) or not callable(metrics_text_method):
            raise TypeError(
                "corruption.metrics must implement to_report_dict() and to_report_text()"
            )
        metrics_dictionary = metrics_dictionary_method()
        if not isinstance(metrics_dictionary, Mapping):
            raise TypeError("corruption.metrics.to_report_dict() must return a mapping")
        normalized_metrics = _strict_json_value(metrics_dictionary, location="metrics")
        rendered_metrics = metrics_text_method()
        if not isinstance(rendered_metrics, str) or not rendered_metrics.strip():
            raise TypeError("corruption.metrics.to_report_text() must return non-empty text")

    payload = {
        "schema_version": CORRUPTION_REPORT_SCHEMA_VERSION,
        "context": normalized_context,
        "configuration": normalized_configuration,
        "metrics": normalized_metrics,
    }
    json_text = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
    ) + "\n"
    text_lines = [
        "Corruption Configuration",
        "=" * 80,
        *_context_text(normalized_context),
        "",
        "Configuration",
        "-" * 80,
        rendered_configuration.rstrip(),
        "",
        rendered_metrics.rstrip(),
    ]
    report_text = "\n".join(text_lines) + "\n"

    json_destination = Path(json_path).expanduser().resolve()
    text_destination = Path(text_path).expanduser().resolve()
    if json_destination == text_destination:
        raise ValueError("json_path and text_path must be different")
    existing = [path for path in (json_destination, text_destination) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite corruption reports:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    json_temporary: Path | None = None
    text_temporary: Path | None = None
    installed: list[Path] = []
    try:
        json_temporary = _write_temporary(json_destination, json_text)
        text_temporary = _write_temporary(text_destination, report_text)
        os.replace(json_temporary, json_destination)
        installed.append(json_destination)
        json_temporary = None
        os.replace(text_temporary, text_destination)
        installed.append(text_destination)
        text_temporary = None
    except Exception:
        for destination in installed:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if json_temporary is not None:
            json_temporary.unlink(missing_ok=True)
        if text_temporary is not None:
            text_temporary.unlink(missing_ok=True)

    return CorruptionReportPaths(json_destination, text_destination)


__all__ = [
    "CORRUPTION_REPORT_SCHEMA_VERSION",
    "CorruptionReportPaths",
    "ReportableCorruption",
    "write_corruption_reports",
]
