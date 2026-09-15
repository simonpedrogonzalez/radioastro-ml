"""High-level finalization of one completed simulation sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .cleanup import CleanupReport, cleanup_simulation_sample
from .schema import SampleManifest, add_sample_to_dataset, create_sample_manifest


@dataclass(frozen=True)
class FinalizedSample:
    manifest: SampleManifest
    cleanup: CleanupReport
    dataset_index: Path | None


def _corruption_pair(value: Any, index: int) -> tuple[Path, Path]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Path(value[0]), Path(value[1])
    json_path = getattr(value, "json_path", None)
    text_path = getattr(value, "text_path", None)
    if json_path is None or text_path is None:
        raise TypeError(
            f"corruption_reports[{index}] must be a (JSON, text) pair or expose "
            "json_path and text_path"
        )
    return Path(json_path), Path(text_path)


def finalize_simulation_sample(
    sample_dir: str | Path,
    *,
    sample_id: str,
    label_id: int,
    label_name: str,
    imaging_result: Any,
    simulation_result: Any,
    corruption_reports: Sequence[Any] = (),
    dataset_index: str | Path | None = None,
) -> FinalizedSample:
    """Write, validate, compact, and optionally index one completed sample."""
    root = Path(sample_dir).expanduser().resolve()
    metadata_text = getattr(simulation_result, "metadata_text", None)
    if metadata_text is None:
        raise ValueError("simulation_result has no metadata_text report")
    pairs = tuple(_corruption_pair(value, index) for index, value in enumerate(corruption_reports))
    manifest = create_sample_manifest(
        root / "sample.json",
        sample_id=sample_id,
        label_id=label_id,
        label_name=label_name,
        products={
            "dirty": imaging_result.dirty_fits,
            "clean": imaging_result.clean_fits,
            "residual": imaging_result.residual_fits,
        },
        imaging_qa=imaging_result.qa_json,
        imaging_text=imaging_result.qa_text,
        simulation=simulation_result.metadata_json,
        simulation_text=metadata_text,
        corruptions=pairs,
    )
    cleanup = cleanup_simulation_sample(root, dry_run=False)
    index_path = None
    if dataset_index is not None:
        index_path = Path(dataset_index).expanduser().resolve()
        add_sample_to_dataset(index_path, manifest.path)
    return FinalizedSample(manifest, cleanup, index_path)


__all__ = ["FinalizedSample", "finalize_simulation_sample"]
