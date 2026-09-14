#!/usr/bin/env python3
"""Batch VLA imaging with an incrementally rendered Quarto report.

Run from the repository root with pipeline-enabled CASA::

    '/Users/u1528314/Applications/CASA 2.app/Contents/MacOS/casa' \
        --pipeline --nogui --nologger -c scripts/compute_background_RMS.py
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_imaging_package_if_loaded() -> None:
    """Refresh imaging code when CASA execfile reuses its Python process."""
    if "scripts.imaging" not in sys.modules:
        return
    for module_name in (
        "scripts.imaging.models",
        "scripts.imaging.metadata",
        "scripts.imaging.config",
        "scripts.imaging.qa",
        "scripts.imaging.plot_utils",
        "scripts.imaging.imaging",
        "scripts.imaging.vla_pipeline",
        "scripts.imaging",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)
    print("Reloaded scripts.imaging package for this CASA session")


_reload_imaging_package_if_loaded()

from scripts.imaging import image_ms_VLA_pipe
from scripts.imaging.metadata import DEFAULT_EXTRACTED_MS_ROOT
from scripts.reporting import QuartoReporter


TITLE = "VLA Pipeline background RMS"
DESCRIPTION = "VLA Pipeline imaging parameters, products, and background-noise measurements."
REPORT_EVERY = 1
IMSIZE = (256, 256)
SAMPLE_IDS: list[str] | None = None
EXPERIMENT_DIR: str | Path | None = (
    ROOT / "collect" / "experiments" / "compute_background_RMS_20260903T133414"
)


def find_samples() -> list[Path]:
    folders = sorted(DEFAULT_EXTRACTED_MS_ROOT.iterdir())
    if SAMPLE_IDS is not None:
        wanted = set(SAMPLE_IDS)
        folders = [folder for folder in folders if folder.name in wanted]
    return [
        folder / folder.name / f"{folder.name}.ms"
        for folder in folders
        if (folder / folder.name / f"{folder.name}.ms").is_dir()
    ]


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def process_sample(ms_path: Path, experiment_dir: Path) -> dict:
    sample_id = ms_path.stem
    result = image_ms_VLA_pipe(
        ms_path,
        experiment_dir / sample_id / "vla_pipeline",
        imsize=IMSIZE,
    )
    return {
        "id": sample_id,
        "ms_path": str(result.ms_path),
        "result_dir": str(result.output_dir.relative_to(experiment_dir)),
    }


def main(experiment_dir: str | Path | None = None) -> Path:
    samples = find_samples()
    if experiment_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        experiment_dir = ROOT / "collect" / "experiments" / f"compute_background_RMS_{timestamp}"
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    report_path = experiment_dir / "report.qmd"
    if not report_path.exists():
        shutil.copyfile(ROOT / "scripts" / "reporting" / "vla_samples.qmd", report_path)

    manifest_path = experiment_dir / "report.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "title": TITLE,
            "description": DESCRIPTION,
            "samples": [],
            "failures": [],
        }
    manifest["total_samples"] = len(samples)
    manifest.setdefault("samples", [])
    manifest.setdefault("failures", [])
    write_manifest(manifest_path, manifest)
    reporter = QuartoReporter(report_path, every=REPORT_EVERY)

    attempted = {
        item["id"]
        for group in (manifest["samples"], manifest["failures"])
        for item in group
    }

    try:
        for ms_path in samples:
            sample_id = ms_path.stem
            if sample_id in attempted:
                print(f"Skipping already attempted sample: {sample_id}")
                continue
            try:
                manifest["samples"].append(process_sample(ms_path, experiment_dir))
            except Exception as exc:
                succeeded = False
                print(f"Sample failed, continuing: {sample_id}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                manifest["failures"].append(
                    {
                        "id": sample_id,
                        "ms_path": str(ms_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                succeeded = True
            attempted.add(sample_id)
            write_manifest(manifest_path, manifest)
            if succeeded:
                reporter.sample_completed()
    finally:
        reporter.finish()

    return experiment_dir


if __name__ in {"__main__", "<run_path>"}:
    print(f"Experiment complete: {main(EXPERIMENT_DIR)}")
