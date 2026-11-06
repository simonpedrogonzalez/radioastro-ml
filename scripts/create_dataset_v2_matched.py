"""Matched-combination Dataset v2: prepare, plan, execute (sharded) and verify.

Stages (docs/dataset_v2_updated_plan.md §2), each restartable:
  ML:   --prepare --num-shards N --shard i --parallel P   (launches CASA --prepare-source)
  ML:   --plan --num-shards N                             (immutable plan + shard assignment)
  ML:   --execute --num-shards N --shard i --parallel P [--scratch-root DIR]
                                                          (launches CASA --unit per template)
  ML:   --backfill-features                               (add CLEAN2 to cases committed with 20 features)
  ML:   --verify                                          (join and tabular split)
Run each shard as at most one live attempt; sealed units are skipped on rerun.
The collected source data are only read; every simulation works on copies.
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
import time
import traceback
from pathlib import Path

import numpy as np

from scripts import create_dataset_v1 as g
from scripts.create_dataset_v2 import file_sha, indexed_sources, ms_bytes
from scripts.dataset_v2_matched import DEFAULT, assign_shards, join_rows, plan_source, unit_file
from scripts.injection_bounds_report import digest
from scripts.preprocessing.schema import atomic_write_json

ROOT = g.ROOT
SECONDS_PER_IMAGE = (6., .085)   # 6 s + 0.085 s per MB of the original MS (v2 production fit)


# ---------------------------------------------------------------- configuration

def configuration(path):
    overrides = json.loads(Path(path).read_text()) if path else {}
    unknown = set(overrides) - (set(DEFAULT) | {"split", "source_ids"})
    if unknown:
        raise ValueError(f"Unknown matched v2 configuration fields: {sorted(unknown)}")
    config = dict(DEFAULT, **overrides)
    for key in ("ranges", "weak_ranges"):
        config[key] = dict(DEFAULT[key], **overrides.get(key, {}))
    config.setdefault("split", "train")
    if config["split"] not in ("train", "val", "test"):
        raise ValueError("Expected canonical train/val/test split")
    return json.loads(json.dumps(config))


def load_index(output):
    path = output / "source_index.json"
    if not path.is_file():
        raise ValueError(f"Run --prepare first: {path} is missing")
    return json.loads(path.read_text())


def ml_python():
    custom = os.environ.get("RADIOASTRO_ML_PYTHON")
    if custom:
        return [custom]
    local = ROOT / "ml/.venv/bin/python"
    return [str(local)] if local.exists() else [str(ROOT / "bin/run"), "ml", "python"]


def tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def peak_rss_bytes():
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return dict(self=int(own), largest_child=int(children))


# ---------------------------------------------------------------- process pool

def run_pool(jobs, parallel, label):
    """Run (name, command, log) jobs with at most ``parallel`` live subprocesses."""
    if parallel < 1:
        raise ValueError("--parallel must be positive")
    queue, live, results = list(jobs), {}, []
    while queue or live:
        while queue and len(live) < parallel:
            name, command, log = queue.pop(0)
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a")
            handle.write(f"\n=== {time.strftime('%FT%T')} {' '.join(command)}\n"); handle.flush()
            live[name] = (subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT), handle)
            print(f"{time.strftime('%FT%T')} START {label} {name}", flush=True)
        for name, (process, handle) in list(live.items()):
            if process.poll() is not None:
                handle.close(); del live[name]
                results.append((name, process.returncode))
                print(f"{time.strftime('%FT%T')} {'DONE' if process.returncode == 0 else 'FAILED'} {label} {name}", flush=True)
        time.sleep(2)
    return results


def casa_command(*args):
    return [str(ROOT / "bin/run"), "casa", "scripts/create_dataset_v2_matched.py", *map(str, args)]


# ---------------------------------------------------------------- stage 1: prepare

def prepare_stage(output, config, *, smoke, num_shards, shard, parallel):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids, entries, audit = indexed_sources(config, smoke)
    payload = json.loads(json.dumps(dict(ids=ids, audit=audit, configuration=config, smoke=smoke)))
    path = output / "source_index.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Source index/configuration changed for this output")
    atomic_write_json(path, payload)
    (output / "prepared").mkdir(exist_ok=True)
    sizes = {sid: ms_bytes(entries[sid]["original_ms"]) for sid in ids}
    owner, _ = assign_shards([dict(id=s) for s in ids], sizes, num_shards)
    pending = [s for s in sorted(ids, key=lambda s: -sizes[s])
               if owner[s] == shard and not (output / "prepared" / f"{s}.json").exists()]
    jobs = [(sid, casa_command("--output", output, "--prepare-source", sid),
             output / "logs" / f"prepare_{sid}.log") for sid in pending]
    failed = [name for name, code in run_pool(jobs, parallel, "prepare") if code]
    if failed:
        raise RuntimeError(f"Preparation failed for {failed}; see {output / 'logs'}")
    print("PREPARE SHARD", shard, "of", num_shards, "done", len(pending), "sources", flush=True)


def prepare_one(output, sid):
    """CASA: prepare one source with the configuration committed in source_index.json."""
    from scripts.create_dataset_v2 import _pilot_contract, prepare_source

    output = output.resolve()
    index = load_index(output)
    config = index["configuration"]
    ids, entries, _ = indexed_sources(config, index["smoke"])
    if sid not in ids:
        raise ValueError(f"{sid} is not in this run's source index")
    pilot = Path(config["pilot_root"]).resolve()
    _pilot_contract(pilot)
    prepare_source(output, config, sid, entries[sid], ms_bytes(entries[sid]["original_ms"]), pilot,
                   retain_casa_tables=True)


# ---------------------------------------------------------------- stage 2: plan

def unit_costs(plan_data):
    mb = {p["id"]: ms_bytes(p["parent_original_ms"]) / 2**20 for p in plan_data["prepared_sources"]}
    return {u["id"]: len(u["cases"]) * (SECONDS_PER_IMAGE[0] + SECONDS_PER_IMAGE[1] * mb[u["source"]])
            for sp in plan_data["source_plans"] for u in sp["units"]}


def plan_stage(output, num_shards):
    output = output.resolve()
    index = load_index(output)
    config = index["configuration"]
    prepared = []
    for sid in index["ids"]:
        path = output / "prepared" / f"{sid}.json"
        if not path.is_file():
            raise ValueError(f"Source is not prepared yet: {sid}")
        item = json.loads(path.read_text())
        if item["configuration_hash"] != digest(config) or file_sha(output / item["reference_masks"]) != item["reference_sha256"]:
            raise ValueError(f"Prepared source changed: {sid}")
        prepared.append(item)
    plans = [plan_source(item, config, config["split"]) for item in prepared]
    payload = json.loads(json.dumps(dict(
        schema=3, split=config["split"], smoke=index["smoke"], configuration=config,
        configuration_hash=digest(config), source_index=index["audit"], prepared_sources=prepared,
        source_plans=plans,
        counts=dict(sources=len(plans), recipes=sum(p["counts"]["recipes"] for p in plans),
                    units=sum(p["counts"]["units"] for p in plans), rows=sum(p["counts"]["rows"] for p in plans),
                    cases=sum(p["counts"]["cases"] for p in plans)))))
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Immutable matched v2 plan changed")
    atomic_write_json(path, payload)
    costs = unit_costs(payload)
    units = [u for sp in plans for u in sp["units"]]
    owner, loads = assign_shards(units, costs, num_shards)
    assignment = dict(num_shards=num_shards, plan_sha256=file_sha(path), owner=owner,
                      estimated_serial_seconds=[round(x) for x in loads])
    shard_path = output / "shards" / f"N{num_shards}.json"
    if shard_path.exists() and json.loads(shard_path.read_text()) != assignment:
        raise ValueError("Shard assignment changed; stop all workers before repartitioning")
    shard_path.parent.mkdir(exist_ok=True)
    atomic_write_json(shard_path, assignment)
    print("PLAN", payload["counts"], "shards", num_shards, "estimated serial s per shard",
          assignment["estimated_serial_seconds"], flush=True)
    return payload


# ---------------------------------------------------------------- stage 3: execute

def execute_stage(output, *, num_shards, shard, parallel, scratch_root):
    output = output.resolve()
    plan_path = output / "plan.json"
    shard_path = output / "shards" / f"N{num_shards}.json"
    if not plan_path.is_file() or not shard_path.is_file():
        raise ValueError(f"Run --plan --num-shards {num_shards} first")
    assignment = json.loads(shard_path.read_text())
    if assignment["plan_sha256"] != file_sha(plan_path):
        raise ValueError("Shard assignment belongs to a different plan")
    if not 0 <= shard < num_shards:
        raise ValueError("Expected 0 <= shard < num_shards")
    plan_data = json.loads(plan_path.read_text())
    costs = unit_costs(plan_data)
    mine = sorted((u for sp in plan_data["source_plans"] for u in sp["units"]
                   if assignment["owner"][u["id"]] == shard), key=lambda u: (-costs[u["id"]], u["id"]))
    pending = [u for u in mine if not (output / "units" / f"{unit_file(u['id'])}.seal.json").exists()]
    scratch_root = Path(scratch_root).resolve()
    jobs = [(u["id"], casa_command("--output", output, "--unit", u["id"], "--scratch-root", scratch_root),
             output / "logs" / f"unit_{unit_file(u['id'])}.log") for u in pending]
    print(f"EXECUTE shard {shard}/{num_shards}: {len(mine)} units, {len(pending)} pending, parallel {parallel}", flush=True)
    failed = [name for name, code in run_pool(jobs, parallel, "unit") if code]
    if failed:
        raise RuntimeError(f"{len(failed)} units failed: {failed[:10]}; see {output / 'failures'}")


def column_hash(ms, minus=None):
    """SHA-256 of DATA (or DATA - minus.DATA) in complex128, row chunk order."""
    from casatools import table
    a, b = table(), table()
    a.open(str(ms))
    if minus is not None:
        b.open(str(minus))
    h = hashlib.sha256()
    try:
        for start in range(0, a.nrows(), 4096):
            kw = dict(startrow=start, nrow=min(4096, a.nrows() - start))
            data = np.asarray(a.getcol("DATA", **kw), np.complex128)
            if minus is not None:
                data = data - np.asarray(b.getcol("DATA", **kw), np.complex128)
            h.update(np.ascontiguousarray(data).tobytes())
    finally:
        if minus is not None:
            b.close()
        a.close()
    return h.hexdigest()


def _sealed(path):
    seal = path.with_suffix(".seal.json")
    return path.is_file() and seal.is_file() and json.loads(seal.read_text())["sha256"] == file_sha(path)


def _commit(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    atomic_write_json(path.with_suffix(".seal.json"), dict(sha256=file_sha(path)))


def execute_unit(output, unit_id, scratch_root):
    """CASA: simulate, image and extract every case of one sampling template."""
    from scripts.create_dataset_v2 import _gain_inject, original_flux_spec
    from scripts.injection_bounds_pilot import add_paired_noise, prepare_v
    from scripts.recovery_products import recovery_products
    from scripts.simulation import add_thermal_noise_inplace, initialize_weights

    started = time.time()
    output = output.resolve()
    plan_path = output / "plan.json"
    plan_data = json.loads(plan_path.read_text())
    plan_hash = file_sha(plan_path)
    if plan_data["configuration_hash"] != digest(plan_data["configuration"]):
        raise ValueError("Plan configuration hash changed")
    source_plan = next(sp for sp in plan_data["source_plans"] if any(u["id"] == unit_id for u in sp["units"]))
    unit = next(u for u in source_plan["units"] if u["id"] == unit_id)
    sid, slot = unit["source"], unit["slot"]
    unit_path = output / "units" / f"{unit_file(unit_id)}.json"
    if _sealed(unit_path):
        print("UNIT ALREADY SEALED", unit_id, flush=True)
        return
    prepared = next(p for p in plan_data["prepared_sources"] if p["id"] == sid)
    cases = {c["id"]: c for c in source_plan["cases"]}
    noise_seed = source_plan["noise_seeds"][slot]
    failure = output / "failures" / f"{unit_file(unit_id)}.json"
    scratch = scratch_root.resolve() / output.name / unit_file(unit_id)
    previous_cwd = Path.cwd()
    try:
        manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
        source = g._thermal_source(g.THERMAL_DIR, manifest, sid)
        if file_sha(source.simulation_metadata) != prepared["thermal_metadata_sha256"]:
            raise ValueError(f"Parent simulation changed: {sid}")
        flux_spec = original_flux_spec(source)
        if flux_spec != prepared["flux_spec"]:
            raise ValueError(f"Original flux provenance changed: {sid}")
        sky = flux_spec["sky"]
        noise_model, noise_params, sigma = g._noise_request(source)
        if sigma != prepared["sigma0_jy"]:
            raise ValueError(f"Nominal noise changed: {sid}")
        folder = output / "sources" / sid / "recovery"
        reference_record = folder / "noiseless" / "record.json"
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        os.chdir(scratch)   # CASA temporary lattices land in this unit's scratch
        # Each unit rebuilds the same noiseless V and nominal-noise realization; the
        # recorded hashes let verification prove they are identical across units.
        v = prepare_v(source, scratch, sky, v2=True)
        initialize_weights(v, sigma)
        regular = scratch / "regular.ms"
        shutil.copytree(v, regular)
        add_thermal_noise_inplace(regular, noise_model=noise_model, noise_parameters=noise_params, seed=noise_seed)
        initialize_weights(regular, sigma)
        v_hash, noise_hash = column_hash(v), column_hash(regular, minus=v)
        complete, image_cell = recovery_products(output, folder, source, sky, sigma, prepared["settings_hash"], v,
                                                 retain_casa_tables=True)
        peak_scratch, retained = tree_bytes(scratch), {}
        for case_id in unit["cases"]:
            c = cases[case_id]
            cell = output / "cases" / case_id
            target = cell / "case.json"
            if _sealed(target):
                rec = json.loads(target.read_text())
                if rec["case"] != c or rec["plan_sha256"] != plan_hash or rec["nominal_noise_sha256"] != noise_hash:
                    raise ValueError(f"Committed case changed: {target}")
                retained[case_id] = tree_bytes(cell)
                continue
            gain = coupling = None
            if not complete(cell):
                work = scratch / "case"
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir()
                observed = work / "observed.ms"
                shutil.copytree(v, observed)
                gain = _gain_inject(v, observed, c, work)
                coupling = add_paired_noise(v, regular, observed, scale=c["a"])
                initialize_weights(observed, sigma)
                image_cell(cell, observed, c["a"], slot, noise_seed,
                           metadata=dict(case=c, gain_complex=gain, paired_noise=coupling))
                peak_scratch = max(peak_scratch, tree_bytes(scratch))
                shutil.rmtree(work)
            record = json.loads((cell / "record.json").read_text())
            result_path = cell / "extracted.json"
            command = [*ml_python(), "-m", "scripts.dataset_v2_matched_extract", "--root", str(output),
                       "--record", str(cell / "record.json"), "--reference-record", str(reference_record),
                       "--masks", str(output / prepared["reference_masks"]),
                       "--nominal-rms", str(prepared["nominal_rms"]), "--output", str(result_path)]
            worker = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            if worker.returncode:
                raise RuntimeError(f"Feature worker failed for {case_id}: {worker.stdout[-1000:]} {worker.stderr[-3000:]}")
            result = json.loads(result_path.read_text())
            iterations = result["iterations"]
            retained[case_id] = tree_bytes(cell)
            _commit(target, dict(
                schema=3, case=c, unit=unit_id, plan_sha256=plan_hash, status="committed",
                settings_hash=prepared["settings_hash"], source_reference_sha256=prepared["reference_sha256"],
                eC=result["eC"], eD=result["eD"], X=result["X"], feature_names=result["feature_names"],
                model_flux=result["model_flux"], empty_raw_model=result["empty_raw_model"],
                convergence=iterations,
                iteration_limit_reached=iterations.get("iterdone", 0) >= iterations.get("raw", {}).get("niter", 1_000_000),
                warnings=result["warnings"], gain_complex=gain if gain is not None else record.get("gain_complex"),
                paired_noise=coupling if coupling is not None else record.get("paired_noise"),
                v_sha256=v_hash, nominal_noise_sha256=noise_hash, product_sha256=record["sha256"],
                products=record["products"], retained_bytes=retained[case_id]))
            print("MATCHED COMMITTED", unit_id, case_id, flush=True)
        _commit(unit_path, dict(
            schema=3, unit=unit, plan_sha256=plan_hash, v_sha256=v_hash, nominal_noise_sha256=noise_hash,
            noise_seed=noise_seed, seconds=round(time.time() - started, 1), peak_scratch_bytes=int(peak_scratch),
            peak_rss_bytes=peak_rss_bytes(), retained_bytes=retained, host=os.uname().nodename))
        failure.unlink(missing_ok=True)
        print("UNIT SEALED", unit_id, f"{time.time() - started:.0f}s", flush=True)
    except Exception:
        prior = json.loads(failure.read_text()) if failure.exists() else {}
        failure.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(failure, dict(unit=unit_id, attempts=prior.get("attempts", 0) + 1,
                                        traceback=traceback.format_exc()[-12000:]))
        raise
    finally:
        os.chdir(previous_cwd)
        if scratch.exists():
            shutil.rmtree(scratch)


# ---------------------------------------------------------------- stage 4: verify

def check_products(output, products, hashes, label):
    """Every retained product a label or feature was computed from must exist unchanged."""
    if set(products) != set(hashes) or not {"dirty", "clean", "residual", "psf", "model", "support"} <= set(products):
        raise ValueError(f"{label}: product list and stored hashes differ or are incomplete")
    for key, rel in products.items():
        path = output / rel
        if not path.is_file():
            raise ValueError(f"{label}: retained {key} product is missing: {path}")
        if file_sha(path) != hashes[key]:
            raise ValueError(f"{label}: retained {key} product changed: {path}")


def features_array(rows):
    """JSON null (missing feature) becomes NaN in the numeric dataset."""
    return np.asarray([[np.nan if x is None else float(x) for x in r["X"]] for r in rows], np.float64)


EXTRA = "clean2_features.json"


def backfill_stage(output):
    """Add CLEAN2 to committed 20-feature cases from their retained model FITS and stored convergence.

    Writes a sealed sidecar cases/<id>/clean2_features.json; case.json and every image are left unchanged.
    No simulation or imaging is rerun. Cases already holding all features, or a valid sidecar, are skipped.
    """
    from astropy.io import fits
    from scripts.dataset_v2_matched_extract import BASE20, CLEAN2, clean2

    output = output.resolve()
    done = skipped = 0
    for path in sorted((output / "cases").glob("*/case.json")):
        if not _sealed(path):
            raise ValueError(f"Case seal missing or changed: {path}")
        rec = json.loads(path.read_text())
        side = path.with_name(EXTRA)
        if rec["feature_names"] != list(BASE20) or _sealed(side):
            skipped += 1
            continue
        model_path = output / rec["products"]["model"]
        if file_sha(model_path) != rec["product_sha256"]["model"]:
            raise ValueError(f"Retained model changed: {model_path}")
        with fits.open(model_path, memmap=False) as hdul:
            model = np.asarray(np.squeeze(hdul[0].data), np.float64)
        _commit(side, dict(feature_names=list(CLEAN2), X=clean2(model, rec["convergence"]),
                           model_sha256=rec["product_sha256"]["model"], case_sha256=file_sha(path)))
        done += 1
    print("BACKFILLED", dict(written=done, skipped=skipped), flush=True)
    return done


def case_features(path, rec):
    """Feature names and values of one case: all of them, or the 20 committed plus the CLEAN2 sidecar."""
    from scripts.dataset_v2_matched_extract import BASE20, CLEAN2, FEATURES

    if rec["feature_names"] == list(FEATURES):
        return rec["X"]
    if rec["feature_names"] != list(BASE20):
        raise ValueError(f"Case has an unknown feature schema: {path}")
    side = path.with_name(EXTRA)
    if not _sealed(side):
        raise ValueError(f"{path.parent.name}: 20-feature case without CLEAN2; run --backfill-features first")
    extra = json.loads(side.read_text())
    if (extra["feature_names"] != list(CLEAN2) or extra["model_sha256"] != rec["product_sha256"]["model"]
            or extra["case_sha256"] != file_sha(path)):
        raise ValueError(f"CLEAN2 sidecar does not belong to this case: {side}")
    return rec["X"] + extra["X"]


def verify_stage(output):
    from astropy.io import fits
    from scripts.dataset_v2_matched_extract import FEATURES

    output = output.resolve()
    plan_path = output / "plan.json"
    plan_data = json.loads(plan_path.read_text())
    plan_hash = file_sha(plan_path)
    units = [u for sp in plan_data["source_plans"] for u in sp["units"]]
    missing = [u["id"] for u in units if not _sealed(output / "units" / f"{unit_file(u['id'])}.json")]
    status = dict(units=len(units), sealed=len(units) - len(missing), missing=missing[:20])
    atomic_write_json(output / "status.json", status)
    if missing:
        print("INCOMPLETE", status, flush=True)
        return status
    records, unit_records = {}, {}
    for u in units:
        unit_records[u["id"]] = json.loads((output / "units" / f"{unit_file(u['id'])}.json").read_text())
        for case_id in u["cases"]:
            path = output / "cases" / case_id / "case.json"
            if not _sealed(path):
                raise ValueError(f"Case seal missing or changed: {path}")
            rec = json.loads(path.read_text())
            if rec["plan_sha256"] != plan_hash:
                raise ValueError(f"Case belongs to another plan: {path}")
            rec["X"] = case_features(path, rec)
            check_products(output, rec["products"], rec["product_sha256"], f"case {case_id}")
            records[case_id] = rec
    for prepared in plan_data["prepared_sources"]:
        sid = prepared["id"]
        reference = json.loads((output / "sources" / sid / "recovery/noiseless/record.json").read_text())
        check_products(output, reference["products"], reference["sha256"], f"reference {sid}")
        if file_sha(output / prepared["reference_masks"]) != prepared["reference_sha256"]:
            raise ValueError(f"reference {sid}: fixed masks changed")
    # Pairing: every unit of a source/slot rebuilt bit-identical V and nominal noise.
    for sp in plan_data["source_plans"]:
        for slot in range(len(sp["noise_seeds"])):
            group = [unit_records[u["id"]] for u in sp["units"] if u["slot"] == slot]
            if len({(r["v_sha256"], r["nominal_noise_sha256"]) for r in group}) != 1:
                raise ValueError(f"Noiseless V or nominal noise differs across units: {sp['source']} slot {slot}")
    cache = {}

    def dirty(case_or_none, source):
        key = case_or_none or f"reference:{source}"
        if key not in cache:
            if case_or_none is None:
                rel = json.loads((output / "sources" / source / "recovery/noiseless/record.json").read_text())["products"]["dirty"]
            else:
                rel = records[case_or_none]["products"]["dirty"]
            with fits.open(output / rel, memmap=False) as hdul:
                cache[key] = np.asarray(np.squeeze(hdul[0].data), np.float64)
        return cache[key]

    rows = []
    for sp, prepared in zip(plan_data["source_plans"], plan_data["prepared_sources"], strict=True):
        with np.load(output / prepared["reference_masks"]) as masks:
            omega = np.asarray(masks["omega"], bool)
        d0 = dirty(None, sp["source"])
        norm = float(np.sqrt(np.square(d0[omega]).sum()))

        def change(a, b):
            return float(np.sqrt(np.square((dirty(a, None) - dirty(b, None))[omega]).sum())) / norm

        rows += join_rows(sp["rows"], records, change)
        cache.clear()
    recipes = {r["id"]: r for sp in plan_data["source_plans"] for r in sp["recipes"]}

    def params(row):
        r = recipes.get(row["recipe"])
        m = row["mask"]
        return [r["A"] if r and m & 1 else 1., r["phi_deg"] if r and m & 2 else 0., r["a"] if r and m & 4 else 1.]

    out = output / f"{plan_data['split']}.npz"
    np.savez_compressed(out, X=features_array(rows), feature_names=np.asarray(FEATURES),
                        b=np.asarray([r["b"] for r in rows], np.int8), mask=np.asarray([r["mask"] for r in rows], np.int8),
                        eC=np.asarray([r["eC"] for r in rows]), eD=np.asarray([r["eD"] for r in rows]),
                        d=np.asarray([r["d"] for r in rows]), p=np.asarray([r["p"] for r in rows]),
                        o=np.ones((len(rows), 3), np.int8), pi=np.ones((len(rows), 3)),
                        params=np.asarray([params(r) for r in rows]), slot=np.asarray([r["slot"] for r in rows]),
                        row_ids=np.asarray([r["id"] for r in rows]), case_ids=np.asarray([r["case"] for r in rows]),
                        recipe_ids=np.asarray([r["recipe"] or "" for r in rows]),
                        source_ids=np.asarray([r["source"] for r in rows]))
    atomic_write_json(output / "rows.json", dict(rows=rows, plan_sha256=plan_hash))
    resources = dict(units=len(unit_records),
                     unit_seconds=[r["seconds"] for r in unit_records.values()],
                     peak_scratch_bytes=max(r["peak_scratch_bytes"] for r in unit_records.values()),
                     peak_rss_bytes=max(max(r["peak_rss_bytes"].values()) for r in unit_records.values()),
                     retained_bytes_per_case=[b for r in unit_records.values() for b in r["retained_bytes"].values()])
    status.update(rows=len(rows), npz=str(out), resources=resources)
    atomic_write_json(output / "status.json", status)
    print("VERIFIED", {k: v for k, v in status.items() if k != "resources"}, flush=True)
    return status


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, help="JSON overrides of the matched DEFAULT (prepare only)")
    p.add_argument("--smoke", action="store_true", help="One training source (prepare only)")
    stage = p.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--prepare-source", metavar="SOURCE", help=argparse.SUPPRESS)
    stage.add_argument("--plan", action="store_true")
    stage.add_argument("--execute", action="store_true")
    stage.add_argument("--unit", metavar="UNIT", help=argparse.SUPPRESS)
    stage.add_argument("--backfill-features", action="store_true")
    stage.add_argument("--verify", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--parallel", type=int, default=1, help="Concurrent CASA processes inside this shard")
    p.add_argument("--scratch-root", type=Path, default=None,
                   help="Disposable visibility scratch (default <output>/scratch; use node-local storage on CHPC)")
    a = p.parse_args()
    if a.prepare:
        prepare_stage(a.output, configuration(a.config), smoke=a.smoke, num_shards=a.num_shards,
                      shard=a.shard, parallel=a.parallel)
    elif a.prepare_source:
        prepare_one(a.output, a.prepare_source)
    elif a.plan:
        plan_stage(a.output, a.num_shards)
    elif a.execute:
        execute_stage(a.output, num_shards=a.num_shards, shard=a.shard, parallel=a.parallel,
                      scratch_root=a.scratch_root or a.output / "scratch")
    elif a.unit:
        execute_unit(a.output, a.unit, a.scratch_root or a.output / "scratch")
    elif a.backfill_features:
        backfill_stage(a.output)
    else:
        verify_stage(a.output)


if __name__ == "__main__":
    main()
