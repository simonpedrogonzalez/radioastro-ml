"""CASA: thermal comparison for an explicit source list in its own experiment directory.

Runs compare_extracted_thermal_simulations.main(...) with the same settings as the
legacy thermal run, restricted to the given IDs. Rerunning the same command resumes.

    bin/run casa scripts/run_thermal_extension.py \
        --output collect/experiments/extracted_thermal_simulation_comparison_ext62_<date> \
        --ids-file new_sources.txt          # one ID per line, '#' comments allowed
    (or --ids 0001+002,0003-004)

The directory's report.json lists successful sources in "samples" and failures in "failures".
Refuses a directory whose report.json already holds sources outside the list
(e.g. the legacy thermal run), so old runs are never mixed into or extended.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source_ids(ids=None, ids_file=None):
    items = list(ids or [])
    if ids_file:
        for line in Path(ids_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                items.append(line)
    if not items:
        raise ValueError("No source IDs given")
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate source IDs: {sorted({i for i in items if items.count(i) > 1})}")
    return sorted(items)


def check_output(output, ids):
    report = Path(output) / "report.json"
    if not report.is_file():
        return
    manifest = json.loads(report.read_text())
    existing = {e["id"] for e in manifest.get("samples", []) + manifest.get("failures", [])}
    extra = sorted(existing - set(ids))
    if extra:
        raise ValueError(f"{report} already holds other sources ({len(extra)}, e.g. {extra[:3]}); "
                         "use a fresh directory")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh experiment directory (resumable)")
    p.add_argument("--ids", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=[])
    p.add_argument("--ids-file", type=Path)
    a = p.parse_args(argv)
    output = (a.output if a.output.is_absolute() else ROOT / a.output).resolve()
    ids = source_ids(a.ids, a.ids_file)
    check_output(output, ids)
    return output, ids


def main(argv=None):
    output, ids = parse(argv)
    print(f"THERMAL EXTENSION {len(ids)} sources -> {output}", flush=True)
    from scripts import compare_extracted_thermal_simulations as thermal

    thermal.main(output, sample_ids=ids)


if __name__ in {"__main__", "<run_path>"}:
    main()
