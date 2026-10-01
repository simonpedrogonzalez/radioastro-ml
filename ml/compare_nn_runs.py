"""Build one neural-model comparison report from existing run directories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from ml.run_nn_experiments import FILES, write_report


def combine_runs(runs: Sequence[Path], output: Path, *, render: bool = True) -> Path:
    """Link completed jobs from multiple runs and render their joint report."""

    output = output.expanduser().resolve()
    sources = [run.expanduser().resolve() for run in runs]
    if len(sources) < 2:
        raise ValueError("At least two run directories are required")
    if output in sources:
        raise ValueError("Comparison output must differ from every source run")
    output.mkdir(parents=True, exist_ok=True)
    expected: list[dict[str, Any]] = []
    dataset_hashes = set()
    linked_sources = {}
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"Run directory not found: {source}")
        for directory in sorted(path for path in source.iterdir() if path.is_dir()):
            if not all((directory / name).is_file() for name in FILES):
                continue
            config = json.loads((directory / "config.json").read_text())
            required = {"experiment_id", "seed", "dataset_sha256"}
            if not required <= config.keys():
                raise ValueError(f"Incomplete NN config: {directory / 'config.json'}")
            job_id = directory.name
            destination = output / job_id
            if os.path.lexists(destination):
                if not destination.is_symlink() or destination.resolve() != directory:
                    raise FileExistsError(f"Comparison job name collision: {job_id}")
            else:
                destination.symlink_to(directory, target_is_directory=True)
            expected.append({"job_id": job_id, "experiment_id": config["experiment_id"],
                             "seed": int(config["seed"])})
            dataset_hashes.add(config["dataset_sha256"])
            linked_sources[job_id] = str(directory)
    if not expected:
        raise ValueError("No complete neural jobs found")
    if len(dataset_hashes) != 1:
        raise ValueError("Neural runs use different dataset fingerprints")
    expected.sort(key=lambda job: (job["experiment_id"], job["seed"]))
    (output / "comparison_sources.json").write_text(
        json.dumps({"runs": [str(path) for path in sources], "jobs": linked_sources}, indent=2) + "\n"
    )
    return write_report(output, expected, render=render, dataset_sha=dataset_hashes.pop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    combine_runs(args.runs, args.output, render=not args.no_render)


if __name__ == "__main__":
    main()
