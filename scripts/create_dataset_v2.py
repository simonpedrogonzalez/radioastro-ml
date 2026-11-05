"""Prepare, plan and execute compact paired Dataset v2 simulations.

Use CASA for --prepare/--execute/--resume; use the ML Python for --plan/--verify.
The default development sequence is train first, then val and test explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts import create_dataset_v1 as g
from scripts.dataset_v2 import DEFAULT, canonical_case, exact_counts, join_rows, plan_source
from scripts.injection_bounds_report import digest
from scripts.preprocessing.schema import atomic_write_json

DEFAULT_OUTPUT = Path("runs/dataset-v2-20261104")
PILOT = Path("runs/injection-bounds-pilot-20261102")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path):
    """Content fingerprint for the parent measurement set, independent of mtimes."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Parent measurement set is missing: {root}")
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Parent measurement set has no files: {root}")
    for item in files:
        h.update(str(item.relative_to(root)).encode())
        h.update(file_sha(item).encode())
    return h.hexdigest()


def ms_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def worker_sources(ids, sizes, workers, worker):
    """Deterministic size-balanced disjoint source partition (largest first)."""
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("Expected 0 <= worker < workers")
    loads, owner = [0] * workers, {}
    for sid in sorted(ids, key=lambda x: (-sizes[x], x)):
        target = min(range(workers), key=lambda w: (loads[w], w))
        owner[sid] = target
        loads[target] += sizes[sid]
    return [sid for sid in ids if owner[sid] == worker]


def require_free_disk(path, need, label, wait_seconds=0):
    """Fail, or wait up to wait_seconds for other workers to free scratch."""
    import time
    deadline = time.time() + wait_seconds
    while True:
        free = shutil.disk_usage(path).free
        if free >= need:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Insufficient free disk for {label}: need {need}, free {free}")
        print(f"WAIT DISK {label}: need {need}, free {free}", flush=True)
        time.sleep(60)


def configuration(path):
    overrides = json.loads(path.read_text()) if path else {}
    unknown = set(overrides) - (set(DEFAULT) | {"split", "source_ids"})
    if unknown:
        raise ValueError(f"Unknown v2 configuration fields: {sorted(unknown)}")
    config = dict(DEFAULT, **overrides)
    config["ranges"] = dict(DEFAULT["ranges"], **overrides.get("ranges", {}))
    config.setdefault("split", "train")
    if config["split"] not in ("train", "val", "test"):
        raise ValueError("Expected canonical train/val/test split")
    return config


def indexed_sources(config, smoke=False):
    """Read the Dataset 1 index and audit parent MS identity across splits."""
    from scripts.preprocessing import partition_for_sample, source_dataset_id

    index = g.DATASET_DIR / "dataset.json"
    members = json.loads(index.read_text())["samples"]
    ids = sorted({source_dataset_id(Path(x).parent.name) for x in members})
    thermal = json.loads((g.THERMAL_DIR / "report.json").read_text())
    entries = {x["id"]: x for x in thermal["samples"]}
    if set(ids) - set(entries):
        raise ValueError("Indexed parent is missing thermal provenance")
    parents = {}
    for sid in ids:
        parent = str(Path(entries[sid]["original_ms"]).resolve())
        if parent in parents and parents[parent] != sid:
            raise ValueError(f"Parent MS shared across source IDs: {parents[parent]}, {sid}")
        parents[parent] = sid
    selected = [sid for sid in ids if partition_for_sample(sid) == config["split"]]
    if "source_ids" in config:
        requested = list(config["source_ids"])
        if set(requested) - set(selected) or len(requested) != len(set(requested)):
            raise ValueError("Requested sources are outside the canonical split or duplicated")
        selected = requested
    if smoke:
        selected = ["0449+113"] if "0449+113" in selected else selected[:1]
    if not selected:
        raise ValueError("No indexed sources selected")
    audit = dict(index=str(index.resolve()), index_sha256=file_sha(index),
                 thermal_report=str((g.THERMAL_DIR / "report.json").resolve()),
                 thermal_report_sha256=file_sha(g.THERMAL_DIR / "report.json"),
                 all_parent_count=len(ids), split_counts={split: sum(partition_for_sample(s) == split for s in ids)
                                                         for split in ("train", "val", "test")},
                 prior_test_exposure="Dataset 1 test was used in a superseded earlier experiment; not pristine unseen test")
    return selected, entries, audit


def _pilot_contract(pilot):
    from scripts.imaging import DefaultImagingConfig
    from dataclasses import asdict

    plan = json.loads((pilot / "plan.json").read_text())
    current = json.loads(json.dumps(asdict(DefaultImagingConfig), default=str))
    if plan["configuration"] != current:
        raise ValueError("Current standard imaging config differs from the completed pilot")
    if plan["configuration"]["clean"]["dirty_peak_fraction"] != 1e-7 or plan["configuration"]["clean"]["nsigma"] != 3:
        raise ValueError("Pilot adaptive stopping contract changed")
    return plan


def original_flux_spec(source):
    """Freeze the pilot-v2 flux decision for this source."""
    qa_path = g.THERMAL_DIR / source.source_id / "original" / "default_imaging" / "qa.json"
    qa = json.loads(qa_path.read_text())
    flux = float(qa["tclean_summary"]["final_model_flux_jy"])
    peak = float(qa["metrics"]["clean_peak_jy_per_beam"])
    rms = float(source.predicted_image_rms_jy_per_beam)
    if not all(math.isfinite(v) and v > 0 for v in (flux, peak, rms)):
        raise ValueError(f"Invalid original flux/peak/nominal RMS: {source.source_id}")
    sky = g._simulation_payload(source)["components"]
    if len(sky) != 1 or sky[0]["shape"] != "point":
        raise ValueError(f"Expected one stored point source: {source.source_id}")
    sky = [dict(sky[0], flux=[flux] + list(sky[0]["flux"][1:]))]
    return dict(sky=sky, flux_jy=flux, nominal_source_snr=flux / rms,
                original_qa=str(qa_path.resolve()), original_qa_sha256=file_sha(qa_path),
                clean_peak_jy_per_beam=peak, flux_policy="original_clean_model_sum_v2_adjusted")


def import_prepared(output, source_dir, sid, config):
    """Reuse a smoke source preparation with an identical configuration and reference."""
    source_dir = Path(source_dir).resolve()
    prior_path = source_dir / "prepared" / f"{sid}.json"
    if not prior_path.is_file():
        return False
    prior = json.loads(prior_path.read_text())
    if prior["configuration_hash"] != digest(config) or prior["split"] != config["split"]:
        raise ValueError(f"Import source {sid} was prepared with a different configuration")
    if file_sha(source_dir / prior["reference_masks"]) != prior["reference_sha256"]:
        raise ValueError(f"Import source reference changed: {sid}")
    folder = output / "sources" / sid / "recovery"
    if folder.exists():
        raise ValueError(f"Refusing to overwrite partial preparation: {folder}")
    shutil.copytree(source_dir / "sources" / sid / "recovery", folder,
                    ignore=shutil.ignore_patterns("work", "scratch"))
    if file_sha(output / prior["reference_masks"]) != prior["reference_sha256"]:
        raise ValueError(f"Copied reference changed: {sid}")
    atomic_write_json(output / "prepared" / f"{sid}.json",
                      dict(prior, imported_from=str(prior_path), imported_sha256=file_sha(prior_path)))
    print("IMPORTED PREPARED", sid, "from", source_dir, flush=True)
    return True


def prepare(output, config, *, smoke=False, workers=1, worker=0, import_from=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pilot = Path(config["pilot_root"]).resolve()
    _pilot_contract(pilot)
    source_ids, thermal_entries, audit = indexed_sources(config, smoke)
    index_payload = dict(ids=source_ids, audit=audit, configuration=config, smoke=smoke)
    index_path = output / "source_index.json"
    index_payload = json.loads(json.dumps(index_payload))
    if index_path.exists() and json.loads(index_path.read_text()) != index_payload:
        raise ValueError("Source index changed for this output")
    atomic_write_json(index_path, index_payload)
    sizes = {sid: ms_bytes(thermal_entries[sid]["original_ms"]) for sid in source_ids}
    (output / "prepared").mkdir(exist_ok=True)
    for sid in worker_sources(source_ids, sizes, workers, worker):
        prepare_source(output, config, sid, thermal_entries[sid], sizes[sid], pilot, import_from)
    return source_ids


def prepare_source(output, config, sid, thermal_entry, size, pilot, import_from=None, retain_casa_tables=False):
    """CASA preparation of one source: noiseless V, antenna norms and fixed reference."""
    from scripts.corruption import ConstantGainSpec, measure_constant_gain_norms
    from scripts.recovery_products import recovery_products
    from scripts.simulation import initialize_weights

    target = output / "prepared" / f"{sid}.json"
    if not target.exists() and import_from and import_prepared(output, import_from, sid, config):
        pass
    if target.exists():
        existing = json.loads(target.read_text())
        if existing["configuration_hash"] != digest(config) or file_sha(output / existing["reference_masks"]) != existing["reference_sha256"]:
            raise ValueError(f"Prepared source changed: {sid}")
        return
    require_free_disk(output, 3 * size + 2_000_000_000, f"preparing {sid}")
    manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
    source = g._thermal_source(g.THERMAL_DIR, manifest, sid)
    if tuple(source.imsize) != (256, 256):
        raise ValueError(f"{sid}: combined8 requires native 256×256 images")
    flux_spec = original_flux_spec(source)
    sky = flux_spec["sky"]
    _, _, sigma = g._noise_request(source)
    folder = output / "sources" / sid / "recovery"
    folder.mkdir(parents=True, exist_ok=True)
    from scripts.injection_bounds_pilot import prepare_v
    v = prepare_v(source, folder, sky, v2=True)
    initialize_weights(v, sigma)
    norms = {}
    for ant, name in g._unflagged_antenna_choices(v):
        try:
            n = measure_constant_gain_norms(ConstantGainSpec(v, ant, "amp", 1, sigma, sign=1))
        except ValueError as exc:
            if "No finite positive visibility norm" in str(exc):
                continue
            raise
        norms[str(ant)] = dict(name=name, **asdict(n))
    if not norms:
        raise ValueError(f"{sid}: no eligible antennas")
    spec = dict(schema=1, source=sid, configuration_hash=digest(config),
                pilot_plan_sha256=file_sha(pilot / "plan.json"),
                thermal_metadata_sha256=file_sha(source.simulation_metadata),
                sky=sky, flux_spec=flux_spec, sigma0_jy=sigma)
    settings_hash = digest(spec)
    atomic_write_json(folder / "settings.json", spec)
    _, image_cell = recovery_products(output, folder, source, sky, sigma, settings_hash, v,
                                      retain_casa_tables=retain_casa_tables)
    reference_cell = folder / "noiseless"
    image_cell(reference_cell, v, 0., None, None)
    reference = json.loads((folder / "reference.json").read_text())
    masks_rel = str((folder / "reference_masks.npz").relative_to(output))
    # The normal preparation path owns its scratch MS; retained reference
    # FITS are the one-per-source fixed imaging infrastructure.
    work = folder / "work"
    if work.exists() and work.resolve().parent == folder.resolve():
        shutil.rmtree(work)
    data = dict(id=sid, split=config["split"], configuration_hash=digest(config),
                settings_hash=settings_hash, nominal_source_snr=flux_spec["nominal_source_snr"], flux_spec=flux_spec,
                nominal_rms=source.predicted_image_rms_jy_per_beam,
                sigma0_jy=sigma, norms=norms, reference_masks=masks_rel,
                reference_sha256=file_sha(output / masks_rel),
                reference_grid=str((folder / "reference_grid.json").relative_to(output)),
                reference_beam=reference["beam"],
                thermal_metadata_sha256=file_sha(source.simulation_metadata),
                parent_ms=str(source.simulation_ms), parent_original_ms=thermal_entry["original_ms"],
                parent_original_ms_sha256=tree_sha(thermal_entry["original_ms"]),
                grid_shape=list(source.imsize))
    atomic_write_json(target, data)
    print("PREPARED", sid, len(norms), flush=True)


def plan(output, config, *, smoke=False):
    output = output.resolve()
    ids, _, audit = indexed_sources(config, smoke)
    prepared = []
    for sid in ids:
        path = output / "prepared" / f"{sid}.json"
        if not path.is_file():
            raise ValueError(f"Source is not CASA-prepared: {sid}")
        item = json.loads(path.read_text())
        if item["configuration_hash"] != digest(config) or file_sha(output / item["reference_masks"]) != item["reference_sha256"]:
            raise ValueError(f"Prepared source changed: {sid}")
        prepared.append(item)
    plans = [plan_source(item, config, config["split"]) for item in prepared]
    preview_path = output / "sampling_preview.json"
    if config["split"] == "train" and not smoke and not preview_path.is_file():
        raise ValueError("Run the training sampling preview in this output directory before exact planning")
    if preview_path.is_file():
        preview = json.loads(preview_path.read_text())
        if preview["configuration"] != json.loads(json.dumps(config)):
            raise ValueError("Sampling preview configuration changed")
        preview_sources = {item["id"]: item for item in preview["sources"]}
        preview_vectors = {v["source"]: [] for v in preview["vectors"]}
        for vector in preview["vectors"]:
            preview_vectors[vector["source"]].append(vector)
        for item, source_plan in zip(prepared, plans, strict=True):
            prior = preview_sources.get(item["id"])
            if prior is None or prior["original_qa_sha256"] != item["flux_spec"]["original_qa_sha256"] or prior["nominal_source_snr"] != item["nominal_source_snr"]:
                raise ValueError(f"Sampling preview source metadata changed: {item['id']}")
            planned = json.loads(json.dumps([{k: v for k, v in vector.items() if k != "antenna"}
                                             for vector in source_plan["vectors"]]))
            if planned != preview_vectors.get(item["id"]):
                raise ValueError(f"Sampling preview vectors changed: {item['id']}")
    cases = {c["id"]: c for item in plans for c in item["cases"]}
    rows = [r for item in plans for r in item["rows"]]
    vectors = [v for item in plans for v in item["vectors"]]
    counts = exact_counts(plans)
    # Measured pilot case duration and source MS size anchor disk/time estimates.
    pilot = Path(config["pilot_root"]).resolve()
    # Seconds/image ~ 6 s + 0.085 s/MB of the original MS, fitted to the three
    # Parametrization v2 pilot sources and the adjusted smoke (about 12 s at 67 MB).
    # Scratch follows the noiseless V copy, which is as large as the matched-SNR
    # simulation MS (about 3x the original), so that MS sets peak scratch.
    copied = {item["id"]: ms_bytes(item["parent_original_ms"]) for item in prepared}
    seconds = sum((6 + .085 * copied[c["source"]] / 2**20) for c in cases.values())
    largest_ms = max(ms_bytes(item["parent_ms"]) for item in prepared)
    estimate = dict(unique_images=len(cases), imaging_seconds=int(math.ceil(seconds)),
                    final_bytes=int(len(cases) * 15_000 + len(rows) * 2_000 + len(prepared) * 20_000_000),
                    peak_scratch_bytes=int(4 * largest_ms + 500_000_000),
                    basis="serial seconds from size model 6 s + 0.085 s/MB of original MS; compact record allowance; "
                          "four simulation-MS-sized copies plus 0.5 GB imaging scratch per worker")
    free = shutil.disk_usage(output).free
    if (estimate["final_bytes"] > config["max_final_bytes"] or
            estimate["peak_scratch_bytes"] > config["max_scratch_bytes"] or
            estimate["peak_scratch_bytes"] + estimate["final_bytes"] > free * .8):
        raise RuntimeError(f"V2 disk preflight failed: {estimate}, free={free}")
    payload = dict(schema=1, split=config["split"], smoke=smoke, configuration=config,
                   configuration_hash=digest(config), source_index=audit,
                   prepared_sources=prepared, source_plans=plans,
                   cases=list(cases.values()), vectors=vectors, rows=rows,
                   counts=counts, estimate=estimate)
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Immutable v2 plan changed")
    atomic_write_json(path, payload)
    print("PLAN", counts, estimate, flush=True)
    return payload


def _gain_inject(v, observed, c, scratch):
    from casatools import table
    from scripts.corruption import AntennaGainCorruption, Constant, TimeGrid
    from scripts.corruption.tables import GTabQuery, GCOLS

    if c["antenna"] is None:
        return None
    gain = complex(c["A"] * math.cos(math.radians(c["phi_deg"])),
                   c["A"] * math.sin(math.radians(c["phi_deg"])))
    query = GTabQuery().where_eq(GCOLS.ANTENNA1, c["antenna"]).group_by([GCOLS.ANTENNA1])
    corr = AntennaGainCorruption(TimeGrid(g.CORRUPTION_SOLINT),
                                 amp_fn=Constant(c["A"]),
                                 phase_fn=Constant(math.radians(c["phi_deg"])), query=query)
    tab = scratch / "gain.G"
    corr.build_corrtable(str(observed), str(tab), seed=1, diagnostic_plot=scratch / "gain.png")
    tb = table()
    tb.open(str(tab))
    try:
        params = tb.getcol("CPARAM")
        ant = tb.getcol("ANTENNA1").reshape(-1)
        np.testing.assert_allclose(params[..., ant == c["antenna"]], gain, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(params[..., ant != c["antenna"]], 1, rtol=2e-6, atol=2e-7)
    finally:
        tb.close()
    corr.apply_corrtable(str(observed), str(tab), seed=1)
    a, b = table(), table()
    a.open(str(v)); b.open(str(observed))
    try:
        for start in range(0, a.nrows(), 4096):
            kw = dict(startrow=start, nrow=min(4096, a.nrows() - start))
            base = np.asarray(a.getcol("DATA", **kw), np.complex128)
            actual = np.asarray(b.getcol("DATA", **kw), np.complex128)
            ant1 = a.getcol("ANTENNA1", **kw).reshape(-1)
            ant2 = a.getcol("ANTENNA2", **kw).reshape(-1)
            g1 = np.where(ant1 == c["antenna"], gain, 1.)
            g2 = np.where(ant2 == c["antenna"], gain, 1.)
            expected = base * (g1 * np.conj(g2))[None, None, :]
            tolerance = 12 * np.finfo(np.float32).eps * max(float(np.max(np.abs(expected))), 1e-30)
            valid = ~a.getcol("FLAG", **kw) & ~a.getcol("FLAG_ROW", **kw)[None, None, :]
            if valid.any():
                np.testing.assert_allclose(actual[valid], expected[valid], rtol=0, atol=tolerance)
    finally:
        b.close(); a.close()
    return [gain.real, gain.imag]


def execute(output, *, resume=False, workers=1, worker=0, import_from=None):
    from scripts.injection_bounds_pilot import add_paired_noise
    from scripts.recovery_products import recovery_products
    from scripts.simulation import add_thermal_noise_inplace, initialize_weights

    output = output.resolve()
    plan_data = json.loads((output / "plan.json").read_text())
    plan_hash = file_sha(output / "plan.json")
    config = plan_data["configuration"]
    if plan_data["configuration_hash"] != digest(config):
        raise ValueError("Plan configuration hash changed")
    lock = output / (f"run.worker{worker}of{workers}.lock" if workers > 1 else "run.lock")
    others = [p for p in output.glob("run*.lock") if p != lock]
    if any(not p.name.endswith(f"of{workers}.lock") for p in others) or (workers == 1 and others):
        raise RuntimeError(f"Another execution layout owns this output: {others}")
    if resume and lock.exists():
        try:
            owner = int(lock.read_text().strip())
            os.kill(owner, 0)
        except ProcessLookupError:
            lock.unlink()
        except ValueError:
            raise RuntimeError(f"Invalid run owner marker: {lock}")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"V2 run is already owned: {lock}")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    try:
        source_manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
        sizes = {p["id"]: ms_bytes(p["parent_original_ms"]) for p in plan_data["prepared_sources"]}
        scratch_sizes = {p["id"]: ms_bytes(p["parent_ms"]) for p in plan_data["prepared_sources"]}
        mine = set(worker_sources([p["id"] for p in plan_data["prepared_sources"]], sizes, workers, worker))
        for prepared in plan_data["prepared_sources"]:
            sid = prepared["id"]
            if sid not in mine:
                continue
            if import_from:
                import_cases(output, import_from, plan_data, plan_hash, prepared)
            require_free_disk(output, 3 * scratch_sizes[sid] + 2_000_000_000, f"executing {sid}",
                              wait_seconds=8 * 3600)
            source = g._thermal_source(g.THERMAL_DIR, source_manifest, sid)
            if file_sha(source.simulation_metadata) != prepared["thermal_metadata_sha256"]:
                raise ValueError(f"Parent simulation changed: {sid}")
            if tree_sha(prepared["parent_original_ms"]) != prepared["parent_original_ms_sha256"]:
                raise ValueError(f"Parent measurement set changed: {sid}")
            flux_spec = original_flux_spec(source)
            if flux_spec != prepared["flux_spec"]:
                raise ValueError(f"Original flux provenance changed: {sid}")
            sky = flux_spec["sky"]
            model, params, sigma = g._noise_request(source)
            if sigma != prepared["sigma0_jy"]:
                raise ValueError(f"Nominal noise changed: {sid}")
            folder = output / "sources" / sid / "recovery"
            from scripts.injection_bounds_pilot import prepare_v
            v = prepare_v(source, folder, sky, v2=True)
            initialize_weights(v, sigma)
            _, image_cell = recovery_products(output, folder, source, sky, sigma,
                                              prepared["settings_hash"], v)
            source_plan = next(p for p in plan_data["source_plans"] if p["source"] == sid)
            for slot, casa_seed in enumerate(source_plan["noise_seeds"]):
                pending = []
                for c in source_plan["cases"]:
                    if c["slot"] != slot:
                        continue
                    committed = output / "cases" / f"{c['id']}.json"
                    if committed.exists():
                        rec = json.loads(committed.read_text())
                        if (rec.get("case") != c or rec.get("plan_sha256") != plan_hash or
                                rec.get("settings_hash") != prepared["settings_hash"] or
                                not math.isfinite(float(rec["y"])) or len(rec["X"]) != 8):
                            raise ValueError(f"Committed case changed: {committed}")
                        seal = committed.with_suffix(".seal.json")
                        if seal.exists() and json.loads(seal.read_text())["sha256"] != file_sha(committed):
                            raise ValueError(f"Committed case checksum changed: {committed}")
                        if not seal.exists():
                            atomic_write_json(seal, dict(sha256=file_sha(committed)))
                    else:
                        pending.append(c)
                if not pending:
                    continue
                scratch = folder / "scratch"
                if scratch.exists():
                    shutil.rmtree(scratch)
                scratch.mkdir()
                regular = scratch / "regular.ms"
                shutil.copytree(v, regular)
                add_thermal_noise_inplace(regular, noise_model=model, noise_parameters=params, seed=casa_seed)
                initialize_weights(regular, sigma)
                for c in sorted(pending, key=lambda x: (x["a"] != 1, x["antenna"] is not None, x["id"])):
                    cell = scratch / "case"
                    failure = output / "failures" / f"{c['id']}.json"
                    prior = json.loads(failure.read_text()) if failure.exists() else {}
                    try:
                        if cell.exists():
                            shutil.rmtree(cell)
                        cell.mkdir()
                        observed = cell / "observed.ms"
                        shutil.copytree(v, observed)
                        gain = _gain_inject(v, observed, c, cell)
                        coupling = add_paired_noise(v, regular, observed, scale=c["a"])
                        initialize_weights(observed, sigma)
                        image_cell(cell, observed, c["a"], slot, casa_seed,
                                   metadata=dict(case=c, gain_complex=gain, paired_noise=coupling))
                        raw = json.loads((cell / "record.json").read_text())
                        result_path = cell / "extracted.json"
                        command = [str(g.ROOT / "ml/.venv/bin/python"), "-m", "scripts.dataset_v2_extract",
                                   "--root", str(output), "--record", str(cell / "record.json"),
                                   "--masks", str(output / prepared["reference_masks"]),
                                   "--nominal-rms", str(prepared["nominal_rms"]),
                                   "--output", str(result_path)]
                        worker = subprocess.run(command, capture_output=True, text=True)
                        if worker.returncode:
                            raise RuntimeError(f"Feature/metric worker failed for {c['id']}: {worker.stdout[-1000:]} {worker.stderr[-3000:]}")
                        result = json.loads(result_path.read_text())
                        compact = dict(schema=1, case=c, plan_sha256=plan_hash,
                                       status="committed",
                                       settings_hash=prepared["settings_hash"],
                                       source_reference_sha256=prepared["reference_sha256"],
                                       y=result["y"], X=result["X"],
                                       feature_names=result["feature_names"],
                                       empty_raw_model=result["empty_raw_model"],
                                       convergence=result["iterations"],
                                       iteration_limit_reached=(result["iterations"].get("iterdone", 0) >=
                                                                result["iterations"].get("raw", {}).get("niter", 1_000_000)),
                                       warnings=result["warnings"],
                                       gain_complex=gain, paired_noise=coupling,
                                       product_sha256=raw["sha256"],
                                       product_digests_provenance="temporary products deleted after compact commit")
                        target = output / "cases" / f"{c['id']}.json"
                        atomic_write_json(target, compact)
                        atomic_write_json(target.with_suffix(".seal.json"), dict(sha256=file_sha(target)))
                        shutil.rmtree(cell)
                        failure.unlink(missing_ok=True)
                        print("V2 COMMITTED", sid, slot, c["id"], flush=True)
                    except Exception:
                        atomic_write_json(failure, dict(case=c, attempts=prior.get("attempts", 0) + 1,
                                                        traceback=traceback.format_exc()[-12000:]))
                        if cell.exists():
                            shutil.rmtree(cell)
                        raise
                shutil.rmtree(scratch)
            work = folder / "work"
            if work.exists():
                shutil.rmtree(work)
    finally:
        lock.unlink(missing_ok=True)


def import_cases(output, source_dir, plan_data, plan_hash, prepared):
    """Copy identical committed smoke cases into this plan, keeping provenance."""
    source_dir = Path(source_dir).resolve()
    count = 0
    for c in plan_data["cases"]:
        if c["source"] != prepared["id"]:
            continue
        target = output / "cases" / f"{c['id']}.json"
        origin = source_dir / "cases" / f"{c['id']}.json"
        if target.exists() or not origin.is_file():
            continue
        seal = origin.with_suffix(".seal.json")
        if not seal.is_file() or json.loads(seal.read_text())["sha256"] != file_sha(origin):
            raise ValueError(f"Import case seal missing or changed: {origin}")
        rec = json.loads(origin.read_text())
        if (rec["case"] != c or rec["settings_hash"] != prepared["settings_hash"] or
                rec["source_reference_sha256"] != prepared["reference_sha256"] or rec.get("status") != "committed"):
            raise ValueError(f"Import case does not match this plan: {origin}")
        rec.update(plan_sha256=plan_hash, imported_from=dict(path=str(origin), sha256=file_sha(origin),
                                                              plan_sha256=json.loads(origin.read_text())["plan_sha256"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, rec)
        atomic_write_json(target.with_suffix(".seal.json"), dict(sha256=file_sha(target)))
        count += 1
    if count:
        print("IMPORTED CASES", prepared["id"], count, "from", source_dir, flush=True)
    return count


def verify(output):
    output = output.resolve()
    plan_data = json.loads((output / "plan.json").read_text())
    plan_hash = file_sha(output / "plan.json")
    records = {}
    missing = []
    for c in plan_data["cases"]:
        path = output / "cases" / f"{c['id']}.json"
        if not path.is_file():
            missing.append(c["id"])
            continue
        rec = json.loads(path.read_text())
        seal = path.with_suffix(".seal.json")
        if not seal.exists() or json.loads(seal.read_text())["sha256"] != file_sha(path):
            raise ValueError(f"Compact case seal is missing or changed: {path}")
        if (rec["case"] != c or rec["plan_sha256"] != plan_hash or not np.isfinite(rec["y"]) or
                len(rec["X"]) != 8 or not np.isfinite(rec["X"]).all()):
            raise ValueError(f"Invalid compact case: {path}")
        records[c["id"]] = rec
    status = dict(planned=len(plan_data["cases"]), committed=len(records), missing=len(missing),
                  missing_ids=missing[:20])
    atomic_write_json(output / "status.json", status)
    if missing:
        print("INCOMPLETE", status, flush=True)
        return status
    rows = join_rows(plan_data, records)
    out = output / f"{plan_data['split']}.npz"
    np.savez_compressed(out, X=np.asarray([r["X"] for r in rows], np.float64),
                        b=np.asarray([r["b"] for r in rows], np.int8),
                        y_all=np.asarray([r["y_all"] for r in rows], np.float64),
                        d=np.asarray([[np.nan if x is None else x for x in r["d"]] for r in rows], np.float64),
                        o=np.asarray([r["o"] for r in rows], np.int8),
                        pi=np.asarray([r["pi"] for r in rows], np.float64),
                        row_ids=np.asarray([r["id"] for r in rows]),
                        source_ids=np.asarray([r["source"] for r in rows]))
    atomic_write_json(output / "rows.json", dict(rows=rows, plan_sha256=plan_hash))
    print("VERIFIED", status, out, flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Disjoint size-balanced source partitions")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--import-from", type=Path,
                        help="Reuse identical prepared sources/committed cases from a smoke run")
    args = parser.parse_args()
    config = configuration(args.config)
    if sum((args.prepare, args.plan, args.execute, args.resume, args.verify)) != 1:
        parser.error("Select one of --prepare, --plan, --execute, --resume, --verify")
    if args.prepare:
        prepare(args.output, config, smoke=args.smoke, workers=args.workers, worker=args.worker,
                import_from=args.import_from)
    elif args.plan:
        plan(args.output, config, smoke=args.smoke)
    elif args.execute or args.resume:
        execute(args.output, resume=args.resume, workers=args.workers, worker=args.worker,
                import_from=args.import_from)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
