"""CASA: thermal comparison for an explicit source list in its own experiment directory.

Runs compare_extracted_thermal_simulations.main(...) with the legacy imaging/noise settings,
restricted to the given IDs. Rerunning the same command resumes (with the same policies).

    bin/run casa scripts/run_thermal_extension.py \
        --output collect/experiments/extracted_thermal_simulation_comparison_ext62_<date> \
        --ids-file new_sources.txt \
        --flux-policy nrao_then_clean --simulation-flags unflag \
        [--reuse-qa-from <earlier thermal dir> ...]

--ids-file: one ID per line, '#' comments allowed (or --ids a,b).

--flux-policy nrao_then_clean: simulate a point source at the valid NRAO pipeline flux (schema-2
flux_estimate.json beside the MS, evaluated at the simulated frequency); otherwise at the original
observation's CLEAN model flux (reused read-only from --reuse-qa-from when it is for the same input
and imaging settings, else imaged now with the original flags); otherwise the source fails.
--simulation-flags unflag: clear inherited FLAG/FLAG_ROW on the simulation copy only.
Without both options the legacy matched-dynamic-range simulation runs.

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
    p.add_argument("--flux-policy", choices=["nrao_then_clean"])
    p.add_argument("--simulation-flags", choices=["unflag"])
    p.add_argument("--reuse-qa-from", type=Path, action="append", default=[],
                   help="Earlier thermal directory whose original-image QA may be reused (repeatable)")
    p.add_argument("--extracted-root", type=Path, help="Default: the standard collect/extracted root")
    a = p.parse_args(argv)
    if a.simulation_flags and not a.flux_policy:
        p.error("--simulation-flags requires --flux-policy")
    if a.reuse_qa_from and not a.flux_policy:
        p.error("--reuse-qa-from requires --flux-policy")
    absolute = lambda path: (path if path.is_absolute() else ROOT / path).resolve()
    output = absolute(a.output)
    reuse = [absolute(d) for d in a.reuse_qa_from]
    if output in reuse:
        p.error("--reuse-qa-from must differ from --output")
    ids = source_ids(a.ids, a.ids_file)
    check_output(output, ids)
    options = dict(flux_policy=a.flux_policy, simulation_flags=a.simulation_flags, reuse_qa_from=reuse,
                   extracted_root=absolute(a.extracted_root) if a.extracted_root else None)
    return output, ids, options


def main(argv=None):
    output, ids, options = parse(argv)
    print(f"THERMAL EXTENSION {len(ids)} sources -> {output} {options}", flush=True)
    from scripts import compare_extracted_thermal_simulations as thermal

    thermal.main(output, sample_ids=ids, **options)


if __name__ in {"__main__", "<run_path>"}:
    main()
