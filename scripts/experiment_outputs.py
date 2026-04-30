from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


def _sanitize_component(text: str, *, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def make_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime(TIMESTAMP_FORMAT)


@dataclass(frozen=True)
class ExperimentOutputLayout:
    extracted_dir: Path
    experiments_dir: Path
    experiment_name: str
    timestamp: str
    root_dir: Path

    def artifact_path(self, label: str, suffix: str) -> Path:
        safe_label = _sanitize_component(label, default="artifact")
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self.root_dir / f"{self.experiment_name}_{safe_label}_{self.timestamp}{safe_suffix}"


def setup_experiment_output_layout(
    extracted_dir: Path,
    experiment_name: str,
    *,
    timestamp: str | None = None,
) -> ExperimentOutputLayout:
    safe_name = _sanitize_component(experiment_name, default="experiment")
    ts = timestamp or make_timestamp()
    experiments_dir = extracted_dir.parent.parent / "experiments"
    root_dir = experiments_dir / f"{safe_name}_{ts}"
    root_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentOutputLayout(
        extracted_dir=extracted_dir,
        experiments_dir=experiments_dir,
        experiment_name=safe_name,
        timestamp=ts,
        root_dir=root_dir,
    )


def artifact_label_and_suffix(filename: str | None, *, default_label: str, default_suffix: str) -> tuple[str, str]:
    if not filename:
        return default_label, default_suffix
    path = Path(filename)
    label = _sanitize_component(path.stem or default_label, default=default_label)
    suffix = path.suffix or default_suffix
    return label, suffix


def find_latest_experiment_artifact(
    extracted_dir: Path,
    experiment_name: str,
    *,
    label: str,
    suffix: str,
) -> Path:
    safe_name = _sanitize_component(experiment_name, default="experiment")
    safe_label = _sanitize_component(label, default="artifact")
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    experiments_dir = extracted_dir.parent.parent / "experiments"
    pattern = f"{safe_name}_*/{safe_name}_{safe_label}_*{safe_suffix}"
    candidates = sorted(experiments_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No artifact found for experiment={safe_name!r}, label={safe_label!r} under {experiments_dir}"
        )
    return candidates[-1]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_ready(item())
        except Exception:
            pass

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value

    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    if numeric_value.is_integer():
        try:
            if value == int(numeric_value):
                return int(numeric_value)
        except Exception:
            pass
    return numeric_value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
