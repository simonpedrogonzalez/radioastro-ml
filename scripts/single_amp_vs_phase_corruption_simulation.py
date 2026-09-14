"""Compare one constant antenna-amplitude error with one constant phase error.

Run inside CASA from the repository root:

    casa --nogui --nologger \
        -c scripts/single_amp_vs_phase_corruption_simulation.py

The source is the already simulated and imaged 0012-399 sample from the newest
qualifying extracted-thermal-comparison experiment. The source MS is never
modified. Each corruption is applied to its own copy under a new experiment
directory, then imaged with the repository imaging package.

The report is a fixed three-row comparison for 0012-399 (thermal baseline,
amplitude-only, phase-only), with dirty, CLEAN, and residual image columns.
The diagnostic corruption defaults are intentionally strong: a 1.50 antenna
gain and a +45 degree antenna phase offset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "collect" / "experiments"
THERMAL_RUN_GLOB = "extracted_thermal_simulation_comparison_*"
REPORT_TEMPLATE = (
    ROOT
    / "scripts"
    / "reporting"
    / "single_amp_vs_phase_corruption_simulation.qmd"
)

SAMPLE_ID = "0012-399"
AMPLITUDE_GAIN = 1.50
PHASE_ERROR_DEGREES = 45.0
CORRUPTION_SOLINT = "10m"
BASE_SEED = 20260909
MIN_FREE_HEADROOM_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class SourceSimulation:
    experiment_dir: Path
    simulation_ms: Path
    simulation_result_dir: Path
    simulation_metadata: Path
    imsize: tuple[int, int]
    metric_min_radius_beams: float | None
    metric_max_radius_beams: float | None
    manifest_entry: dict[str, Any]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    amplitude_gain: float
    phase_radians: float
    seed: int


@dataclass(frozen=True)
class _ConstantCurve:
    """Duck-typed constant model for the current CorrFn calling convention."""

    value: float

    def sample(self, rng: Any, **kwargs: Any) -> "_ConstantCurve":
        del rng, kwargs
        return self

    def eval(self, times: Any) -> Any:
        import numpy as np

        values = np.asarray(times, dtype=float)
        return np.full(values.shape, self.value, dtype=float)


def _resolve_manifest_path(experiment_dir: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Thermal manifest has no valid {name}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = experiment_dir / candidate
    return candidate.resolve()


def _required_source_paths(source: SourceSimulation) -> tuple[Path, ...]:
    return (
        source.simulation_ms,
        source.simulation_metadata,
        source.simulation_result_dir / "qa.json",
        source.simulation_result_dir / "dirty.png",
        source.simulation_result_dir / "clean.png",
        source.simulation_result_dir / "residual.png",
    )


def _source_from_run(experiment_dir: Path) -> SourceSimulation:
    manifest_path = experiment_dir / "report.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read thermal manifest {manifest_path}: {exc}") from exc

    matches = [
        item
        for item in manifest.get("samples", [])
        if isinstance(item, dict) and item.get("id") == SAMPLE_ID
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one completed {SAMPLE_ID} sample in {manifest_path}, found {len(matches)}"
        )
    entry = dict(matches[0])
    configuration = manifest.get("configuration", {}) or {}
    imsize_value = configuration.get("imsize")
    if (
        not isinstance(imsize_value, list)
        or len(imsize_value) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in imsize_value
        )
    ):
        raise RuntimeError(f"Thermal manifest has an invalid imsize: {imsize_value!r}")

    region = entry.get("metric_region", {}) or {}
    minimum = region.get("min_radius_beams")
    maximum = region.get("max_radius_beams")
    for name, value in (("minimum", minimum), ("maximum", maximum)):
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise RuntimeError(
                f"Thermal manifest has invalid metric-region {name}: {value!r}"
            )

    source = SourceSimulation(
        experiment_dir=experiment_dir.resolve(),
        simulation_ms=_resolve_manifest_path(
            experiment_dir, entry.get("simulation_ms"), name="simulation_ms"
        ),
        simulation_result_dir=_resolve_manifest_path(
            experiment_dir,
            entry.get("simulation_result_dir"),
            name="simulation_result_dir",
        ),
        simulation_metadata=_resolve_manifest_path(
            experiment_dir,
            entry.get("simulation_metadata"),
            name="simulation_metadata",
        ),
        imsize=(int(imsize_value[0]), int(imsize_value[1])),
        metric_min_radius_beams=None if minimum is None else float(minimum),
        metric_max_radius_beams=None if maximum is None else float(maximum),
        manifest_entry=entry,
    )
    missing = [path for path in _required_source_paths(source) if not path.exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Thermal simulation is incomplete; missing:\n{rendered}")
    return source


def find_source_simulation(source_run: str | Path | None = None) -> SourceSimulation:
    """Resolve the requested run or newest run with a complete 0012-399 simulation."""
    if source_run is not None:
        run = Path(source_run).expanduser().resolve()
        if not run.is_dir():
            raise FileNotFoundError(f"Thermal experiment directory not found: {run}")
        return _source_from_run(run)

    failures: list[str] = []
    candidates = sorted(EXPERIMENTS_ROOT.glob(THERMAL_RUN_GLOB), reverse=True)
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "report.json").is_file():
            continue
        try:
            return _source_from_run(candidate)
        except RuntimeError as exc:
            failures.append(f"{candidate.name}: {exc}")
    detail = "\n".join(f"  - {item}" for item in failures[:8])
    raise RuntimeError(
        f"No completed {SAMPLE_ID} simulation was found under {EXPERIMENTS_ROOT}"
        + (f"\nChecked:\n{detail}" if detail else "")
    )


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _check_disk_space(source_ms: Path, destination_parent: Path) -> None:
    # Two MS copies plus conservative room for two direct-imaging product sets.
    required = 3 * _tree_size(source_ms) + MIN_FREE_HEADROOM_BYTES
    available = shutil.disk_usage(destination_parent).free
    if available < required:
        raise RuntimeError(
            "Insufficient disk space for two corrupted copies and images: "
            f"need approximately {required / 1024**3:.2f} GiB free, "
            f"found {available / 1024**3:.2f} GiB"
        )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(path: Path, experiment_dir: Path) -> str:
    return str(path.resolve().relative_to(experiment_dir.resolve()))


def _copy_baseline_products(source: SourceSimulation, experiment_dir: Path) -> Path:
    destination = experiment_dir / "baseline" / "default_imaging"
    destination.mkdir(parents=True)
    for name in ("dirty.png", "clean.png", "residual.png", "qa.json", "qa.txt"):
        origin = source.simulation_result_dir / name
        if origin.exists():
            shutil.copy2(origin, destination / name)
    return destination


def _select_antenna(ms_path: Path, requested: str | None) -> tuple[int, str]:
    from scripts.corrtab_utils import get_unflagged_antennas

    name_to_id, id_to_name = get_unflagged_antennas(str(ms_path))
    if not id_to_name:
        raise RuntimeError(f"No antenna occurs in an unflagged row of {ms_path}")
    if requested is None:
        antenna_id = min(id_to_name)
        return antenna_id, id_to_name[antenna_id]
    text = str(requested).strip()
    if text in name_to_id:
        return name_to_id[text], text
    try:
        antenna_id = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Unknown antenna {requested!r}; use one of {sorted(name_to_id)} or an ID"
        ) from exc
    if antenna_id not in id_to_name:
        raise ValueError(
            f"Antenna ID {antenna_id} has no unflagged rows; "
            f"available IDs are {sorted(id_to_name)}"
        )
    return antenna_id, id_to_name[antenna_id]


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_variant(
    source: SourceSimulation,
    experiment_dir: Path,
    antenna_id: int,
    antenna_name: str,
    spec: VariantSpec,
) -> dict[str, Any]:
    # Lazy imports let source discovery and fast tests run outside CASA.
    os.environ.setdefault("MPLBACKEND", "Agg")
    from scripts.corruption import AntennaGainCorruption
    from scripts.corrtab_utils import GCOLS, GTabQuery
    from scripts.imaging import BeamRegion, DefaultImagingConfig, image_ms
    from scripts.timegrid import TimeGrid

    variant_dir = experiment_dir / spec.name
    variant_dir.mkdir()
    copied_ms = variant_dir / f"{SAMPLE_ID}_{spec.name}.ms"
    gain_table = variant_dir / f"{SAMPLE_ID}_{spec.name}.G"
    images_dir = variant_dir / "images"
    images_dir.mkdir()

    print(f"[{spec.name}] Copying {source.simulation_ms} -> {copied_ms}")
    shutil.copytree(source.simulation_ms, copied_ms)

    query = (
        GTabQuery()
        .where_eq(GCOLS.ANTENNA1, antenna_id)
        .group_by([GCOLS.ANTENNA1])
    )
    amplitude = (
        None
        if math.isclose(spec.amplitude_gain, 1.0)
        else _ConstantCurve(spec.amplitude_gain)
    )
    phase = (
        None
        if math.isclose(spec.phase_radians, 0.0)
        else _ConstantCurve(spec.phase_radians)
    )
    corruption = AntennaGainCorruption(
        timegrid=TimeGrid(solint=CORRUPTION_SOLINT, interp="linear"),
        amp_fn=amplitude,
        phase_fn=phase,
        query=query,
    )
    # The current prototype writes images/corruption_function.png relative to
    # cwd. Confine that side effect to this variant's experiment directory.
    with _working_directory(variant_dir):
        corruption.build_corrtable(
            str(copied_ms.resolve()), str(gain_table.resolve()), seed=spec.seed
        ).apply_corrtable(
            str(copied_ms.resolve()), str(gain_table.resolve()), seed=spec.seed
        )

    region = BeamRegion(
        min_radius_beams=source.metric_min_radius_beams,
        max_radius_beams=source.metric_max_radius_beams,
    )
    result = image_ms(
        copied_ms,
        DefaultImagingConfig,
        variant_dir / "default_imaging",
        imsize=source.imsize,
        metric_region=region,
    )
    if result.qa.metrics.region != region:
        raise RuntimeError(
            f"{spec.name} image used the wrong metric region: "
            f"{result.qa.metrics.region!r}"
        )

    required = (
        gain_table,
        images_dir / "corruption_function.png",
        result.dirty_png,
        result.clean_png,
        result.residual_png,
        result.qa_json,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            f"{spec.name} did not produce all required artifacts:\n{rendered}"
        )

    return {
        "name": spec.name,
        "label": spec.label,
        "antenna_id": antenna_id,
        "antenna_name": antenna_name,
        "amplitude_gain": spec.amplitude_gain,
        "amplitude_error_fraction": spec.amplitude_gain - 1.0,
        "phase_error_degrees": math.degrees(spec.phase_radians),
        "phase_error_radians": spec.phase_radians,
        "seed": spec.seed,
        "weight_policy": "preserve; simulator.setapply(calwt=False)",
        "operation": "G(sky + thermal_noise)",
        "ms": _relative(copied_ms, experiment_dir),
        "gain_table": _relative(gain_table, experiment_dir),
        "corruption_plot": _relative(
            images_dir / "corruption_function.png", experiment_dir
        ),
        "result_dir": _relative(result.output_dir, experiment_dir),
        "qa_json": _relative(result.qa_json, experiment_dir),
    }


def _new_manifest(
    source: SourceSimulation,
    experiment_dir: Path,
    baseline_dir: Path,
    antenna_id: int,
    antenna_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "title": "0012-399 constant one-antenna amplitude versus phase corruption",
        "description": (
            "The same already-imaged thermal point-source simulation is copied twice. "
            "One copy receives only a significant constant 50% gain-amplitude error "
            "and the other only a significant constant +45 degree phase error on the "
            "same single antenna. The report compares dirty, clean, and residual "
            "images in exactly three rows: thermal baseline, amplitude-only, and "
            "phase-only."
        ),
        "sample_id": SAMPLE_ID,
        "source": {
            "thermal_experiment": str(source.experiment_dir),
            "simulation_ms": str(source.simulation_ms),
            "simulation_result_dir": str(source.simulation_result_dir),
            "simulation_metadata": str(source.simulation_metadata),
            "baseline_result_dir": _relative(baseline_dir, experiment_dir),
            "thermal_manifest_entry": source.manifest_entry,
        },
        "configuration": {
            "antenna_id": antenna_id,
            "antenna_name": antenna_name,
            "amplitude_gain": AMPLITUDE_GAIN,
            "amplitude_error_fraction": AMPLITUDE_GAIN - 1.0,
            "phase_error_degrees": PHASE_ERROR_DEGREES,
            "corruption_solint": CORRUPTION_SOLINT,
            "base_seed": BASE_SEED,
            "imaging_configuration": "DefaultImagingConfig",
            "imsize": list(source.imsize),
            "metric_region": {
                "min_radius_beams": source.metric_min_radius_beams,
                "max_radius_beams": source.metric_max_radius_beams,
            },
            "weight_policy": "preserve; simulator.setapply(calwt=False)",
            "operation_order": (
                "simulate sky plus thermal noise, then multiply by gains"
            ),
            "report_layout": {
                "rows": [
                    "clean simulation with thermal noise",
                    "one-antenna constant amplitude-only error",
                    "one-antenna constant phase-only error",
                ],
                "columns": ["dirty", "clean", "residual"],
                "final_clean_comparison": [
                    "thermal simulation",
                    "one-antenna amplitude-only error",
                    "one-antenna phase-only error",
                ],
            },
        },
        "variants": [],
        "failures": [],
    }


def run_experiment(
    *,
    source_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    antenna: str | None = None,
) -> Path:
    source = find_source_simulation(source_run)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output = (
            EXPERIMENTS_ROOT
            / f"single_amp_vs_phase_corruption_simulation_{timestamp}"
        )
    else:
        output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _check_disk_space(source.simulation_ms, output.parent)
    output.mkdir()

    print(f"Source thermal experiment: {source.experiment_dir}")
    print(f"Source simulated MS: {source.simulation_ms}")
    print(f"New experiment: {output}")

    shutil.copy2(REPORT_TEMPLATE, output / "report.qmd")
    baseline_dir = _copy_baseline_products(source, output)
    antenna_id, antenna_name = _select_antenna(source.simulation_ms, antenna)
    print(f"Selected antenna: {antenna_name} (ID {antenna_id})")

    manifest_path = output / "report.json"
    manifest = _new_manifest(
        source, output, baseline_dir, antenna_id, antenna_name
    )
    _atomic_write_json(manifest_path, manifest)

    variants = (
        VariantSpec(
            name="constant_amplitude",
            label="One-antenna constant +50% amplitude gain",
            amplitude_gain=AMPLITUDE_GAIN,
            phase_radians=0.0,
            seed=BASE_SEED + 1,
        ),
        VariantSpec(
            name="constant_phase",
            label="One-antenna constant +45 degree phase",
            amplitude_gain=1.0,
            phase_radians=math.radians(PHASE_ERROR_DEGREES),
            seed=BASE_SEED + 2,
        ),
    )

    from scripts.reporting import QuartoReporter

    reporter = QuartoReporter(output / "report.qmd", every=1)
    try:
        for spec in variants:
            try:
                entry = _run_variant(
                    source, output, antenna_id, antenna_name, spec
                )
            except Exception as exc:
                manifest["failures"].append(
                    {
                        "name": spec.name,
                        "stage": "corrupt_and_image",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                _atomic_write_json(manifest_path, manifest)
                print(f"[{spec.name}] FAILED: {type(exc).__name__}: {exc}")
                continue
            manifest["variants"].append(entry)
            _atomic_write_json(manifest_path, manifest)
            reporter.sample_completed()
            print(f"[{spec.name}] completed")
    finally:
        reporter.finish()

    print(f"Report: {output / 'report.html'}")
    print(
        f"Completed variants: {len(manifest['variants'])}; "
        f"failures: {len(manifest['failures'])}"
    )
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        help=(
            "Specific thermal-comparison experiment directory; default is "
            "the newest complete run"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "New experiment directory; default is timestamped under "
            "collect/experiments"
        ),
    )
    parser.add_argument(
        "--antenna",
        help=(
            "One unflagged antenna name or numeric ID; default is the lowest used ID"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    arguments = _parse_args(argv)
    return run_experiment(
        source_run=arguments.source_run,
        output_dir=arguments.output_dir,
        antenna=arguments.antenna,
    )


if __name__ in {"__main__", "<run_path>"}:
    main()
