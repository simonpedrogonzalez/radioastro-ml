"""VLA pipeline QA experiment: do gain calibration / automated self-calibration repair matched-v2 cases?

Design: docs/vla_pipeline_qa_experiment_draft.md. Two experiments, each a set of named recipes run on
regenerated matched-v2 cases, always evaluated with the unchanged matched-v2 imaging:

  A  known-model gain recovery: gaincal/applycal against the exact injected sky (MODEL_DATA = noiseless V)
  B  automated target self-calibration: NRAO imaging pipeline hif_selfcal on the case as a target field

Each gain recipe runs with flagging_off and flagging_on (A: applycal calonly/calflag. B: calonly and no
hifv_flagtargetsdata versus the documented calflag + hifv_flagtargetsdata). The source run is only read.

Stages (ML runtime; CASA work is launched as subprocesses, B uses bin/run casa-pipeline):
  python -m scripts.vla_pipeline_qa --output OUT --plan --source-run runs/dataset-v2-matched-smoke-20261105 \
        --experiment A --cases all
  python -m scripts.vla_pipeline_qa --output OUT --execute --parallel 3
  python -m scripts.vla_pipeline_qa --output OUT --report
Per case: regenerate observed.ms (hash-verified against the committed case) -> optional re-imaging check
of the committed before result -> every recipe -> common evaluation imaging (recipe flags/weights, fixed
grid, beam, mask) -> dirty reference I_r(V0) rebuilt with the recipe's flags/weights when they changed ->
eC, eD, features and optional frozen-model predictions. Work MSs are deleted after the case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# [USER] 2026-11-09: phase-only recipes dropped (amplitude correction is the sensible standard choice).
# B "default" equals phase-only in pipeline 2025.1.0.36 (amplitude_selfcal=False), so it is dropped too.
GAINS = {"A": ("phase_amp",), "B": ("phase_amp",)}
# Experiment A solver settings: provisional, to be frozen after the pilot (design: open decision).
A_SETTINGS = dict(
    phase=dict(calmode="p", solint="int", minsnr=3.0, minblperant=4, gaintype="G", solnorm=False),
    amp=dict(calmode="ap", solint="inf", combine="", minsnr=3.0, minblperant=4, gaintype="G", solnorm=False),
    interp="linear", calwt=False,
    model="MODEL_DATA = exact noiseless injected-sky visibilities V0",
    refant_rule="antenna with the most unflagged input visibilities, ties -> lowest antenna id (metadata only)",
    amplitude_normalization="solnorm=False: amplitudes solved against the known model flux")
# Experiment B: pinned to the pipeline that runs it (recorded per run).
B_SETTINGS = dict(maximsize=16384,
                  # [USER] 2026-11-09: the regcal hif_makeimages before hif_selfcal is never run. It makes the
                  # pipeline's deliverable images, which hif_selfcal and this experiment do not use.
                  makeimages_before_selfcal=False,
                  gain=dict(phase_amp=dict(amplitude_selfcal=True)),
                  flagging_off=dict(apply_cal_mode_default="calonly", flagtargetsdata=False),
                  flagging_on=dict(apply_cal_mode_default=None, flagtargetsdata=True))
FLAG_COLUMNS = ("FLAG", "FLAG_ROW")
WEIGHT_COLUMNS = ("WEIGHT", "WEIGHT_SPECTRUM")          # what imaging uses. SIGMA* are recorded, not compared
SIGMA_COLUMNS = ("SIGMA", "SIGMA_SPECTRUM")


def recipe_names(experiment):
    return [f"{experiment}:{g}:{f}" for g in GAINS[experiment] for f in ("flagging_off", "flagging_on")]


def parse(recipe):
    experiment, gain, flagging = recipe.split(":")
    return experiment, gain, flagging == "flagging_on"


def safe(name):
    return name.replace(":", "__")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- plan (ML)

def spread_sources(source_plan, n):
    """n sources evenly spaced in nominal source S/N (all when n is None)."""
    snr = {x["id"]: x["nominal_source_snr"] for x in source_plan["prepared_sources"]}
    ids = sorted(snr, key=snr.get)
    if n is None or n >= len(ids):
        return set(ids)
    return {ids[round(i * (len(ids) - 1) / max(1, n - 1))] for i in range(n)}


def select_cases(source_plan, committed, spec):
    """'all', 'representative[:N]' (control + masks 1..7 of the first recipe), 'controls[:N]'
    (N sources evenly spread in S/N) or comma-separated case ids."""
    if spec == "all":
        return sorted(committed)
    kind, _, count = spec.partition(":")
    if kind in ("representative", "controls"):
        sources = spread_sources(source_plan, int(count) if count else None)
        picked = []
        for sp in source_plan["source_plans"]:
            if sp["source"] not in sources:
                continue
            if kind == "controls":
                picked.append(next(r["case"] for r in sp["rows"] if r["mask"] == 0))
                continue
            control = next(r["case"] for r in sp["rows"] if r["mask"] == 0)
            first = sp["recipes"][0]["id"]
            by_mask = {r["mask"]: r["case"] for r in sp["rows"] if r["recipe"] == first}
            picked += [control] + [by_mask[m] for m in (1, 2, 4, 3, 5, 6, 7) if m in by_mask]
        return [c for c in picked if c in committed]
    ids = [x.strip() for x in spec.split(",") if x.strip()]
    missing = set(ids) - set(committed)
    if missing:
        raise ValueError(f"Cases not committed in the source run: {sorted(missing)}")
    return ids


def plan(output, source_run, experiment, recipes, cases, reproduce, model, model_names):
    output, source_run = output.resolve(), source_run.resolve()
    sp = json.loads((source_run / "plan.json").read_text())
    committed = {p.parent.name for p in (source_run / "cases").glob("*/case.seal.json")}
    selected = select_cases(sp, committed, cases)
    recipes = recipes or recipe_names(experiment)
    if any(parse(r)[0] != experiment for r in recipes):
        raise ValueError("All recipes must belong to the chosen experiment")
    repro = set(select_cases(sp, committed, reproduce)) & set(selected) if reproduce != "none" else set()
    payload = dict(schema=1, design="docs/vla_pipeline_qa_experiment_draft.md", experiment=experiment,
                   source_run=str(source_run), source_plan_sha256=file_sha(source_run / "plan.json"),
                   recipes=recipes, settings=dict(A=A_SETTINGS, B=B_SETTINGS)[experiment],
                   model=None if model is None else dict(path=str(Path(model).resolve()), names=model_names),
                   units=[dict(case=c, reproduce=c in repro) for c in selected])
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(payload, default=str)):
        raise ValueError("Plan changed; use a fresh output directory")
    write_json(path, payload)
    print("PLAN", experiment, len(selected), "cases x", len(recipes), "recipes;", len(repro), "reproduction checks")


# ---------------------------------------------------------------- execute (ML orchestration)

def runtime_command(runtime, *args):
    return [str(ROOT / "bin/run"), runtime, "scripts/vla_pipeline_qa.py", *map(str, args)]


def execute(output, parallel):
    from scripts.create_dataset_v2_matched import ml_python, run_pool
    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    pending = [u["case"] for u in p["units"] if not (output / "units" / f"{u['case']}.json").exists()]
    jobs = [(c, [*ml_python(), "-m", "scripts.vla_pipeline_qa", "--output", str(output), "--unit", c],
             output / "logs" / f"unit_{c}.log") for c in pending]
    print(f"EXECUTE {len(p['units'])} cases, {len(pending)} pending, parallel {parallel}", flush=True)
    failed = [n for n, code in run_pool(jobs, parallel, "case") if code]
    if failed:
        raise RuntimeError(f"{len(failed)} cases failed: {failed[:10]}; see {output / 'logs'}")


def unit(output, case_id):
    """ML: one case, all recipes. Science outcomes (incl. recipe failures) are results, not exceptions."""
    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    u = next(x for x in p["units"] if x["case"] == case_id)
    log = output / "logs" / f"unit_{case_id}"
    started, work = time.time(), output / "work" / case_id

    def casa(runtime, *args, label):
        res = subprocess.run(runtime_command(runtime, "--output", output, *args), cwd=ROOT,
                             capture_output=True, text=True)
        Path(f"{log}_{label}.log").write_text(res.stdout[-200000:] + "\n--- stderr ---\n" + res.stderr[-50000:])
        return res.returncode

    if casa("casa", "--casa-regen", case_id, label="regen"):
        raise RuntimeError(f"Regeneration failed: {case_id}")
    regen = json.loads((work / "regen.json").read_text())
    summary = dict(case=case_id, regen=regen, recipes={})
    if u["reproduce"]:
        code = casa("casa", "--casa-image", case_id, "before_repro", work / "observed.ms", label="repro")
        summary["reproduction"] = reproduction(output, p, case_id) if code == 0 else dict(status="imaging failed")
    for recipe in p["recipes"]:
        target = output / "results" / f"{case_id}__{safe(recipe)}.json"
        if target.exists():
            continue
        experiment, gain, flagging = parse(recipe)
        result = dict(case=case_id, recipe=recipe, experiment=experiment, gain=gain, flagging=flagging)
        runtime = "casa" if experiment == "A" else "casa-pipeline"
        code = casa(runtime, "--casa-recipe", case_id, recipe, label=safe(recipe))
        rpath = work / safe(recipe) / "recipe.json"
        rec = json.loads(rpath.read_text()) if rpath.exists() else dict(
            disposition="runtime_failure", reason=f"recipe process exit {code}, no record")
        if experiment == "B":
            rec["pipeline_notes"] = pipeline_notes(Path(f"{log}_{safe(recipe)}.log"))
        result["recipe_record"] = rec
        if rec.get("output_ms") and rec["disposition"] not in ("runtime_failure", "unsupported", "no_usable_output"):
            code = casa("casa", "--casa-image", case_id, safe(recipe), rec["output_ms"], label=safe(recipe) + "_image")
            if code:
                result["evaluation"] = dict(status="unavailable", reason=f"evaluation imaging failed (exit {code})")
            else:
                result["evaluation"] = evaluate(output, p, case_id, safe(recipe))
        else:
            result["evaluation"] = dict(status="unavailable", reason=rec.get("reason", rec["disposition"]))
        write_json(target, result)
        shutil.rmtree(work / safe(recipe), ignore_errors=True)
    summary["seconds"] = round(time.time() - started, 1)
    write_json(output / "units" / f"{case_id}.json", summary)
    shutil.rmtree(work, ignore_errors=True)


NOTE_PATTERNS = ("Estimated SNR per solint", r"^\s*(inf_EB|inf|int|[0-9.]+s): [0-9.]+$", "No regions found by aut",
                 "mask file .* is empty", "Peak SNR", "Final solint", "Final SNR", "Starting with solint",
                 "[Ss]top", "[Pp]assed", "[Ff]ailed")


def pipeline_notes(log, limit=40):
    """Self-calibration decision lines from the pipeline log (the stated reason for a skip or rejection)."""
    import re
    if not log.exists():
        return []
    notes = []
    for line in log.read_text(errors="replace").splitlines():
        text = line.split("INFO: ", 1)[-1].split("\t")[-1].strip()
        if any(re.search(p, text) for p in NOTE_PATTERNS) and "update path" not in text and text not in notes:
            notes.append(text[:200])
    return notes[:limit]


def source_paths(p, case_id):
    source_run = Path(p["source_run"])
    sp = json.loads((source_run / "plan.json").read_text())
    source_plan = next(s for s in sp["source_plans"] if any(c["id"] == case_id for c in s["cases"]))
    prepared = next(x for x in sp["prepared_sources"] if x["id"] == source_plan["source"])
    return source_run, sp, source_plan, prepared


def reproduction(output, p, case_id):
    """Re-imaged regenerated case versus the committed before result."""
    from scripts.dataset_v2_matched_extract import extract
    source_run, _, _, prepared = source_paths(p, case_id)
    committed = json.loads((source_run / "cases" / case_id / "case.json").read_text())
    record = json.loads((output / "cases" / case_id / "before_repro" / "record.json").read_text())
    noiseless = json.loads((source_run / "sources" / prepared["id"] / "recovery" / "noiseless" / "record.json").read_text())
    ref = dict(noiseless, products={k: str(source_run / v) for k, v in noiseless["products"].items()})
    got = extract(output, record, source_run / prepared["reference_masks"], prepared["nominal_rms"], ref)
    same_images = all(file_sha(output / record["products"][k]) == committed["product_sha256"].get(k)
                      for k in ("dirty", "model")) if "product_sha256" in committed else None
    return dict(status="agree" if abs(got["eC"] - committed["eC"]) < 1e-6 and abs(got["eD"] - committed["eD"]) < 1e-6
                else "DISAGREE", eC=got["eC"], eC_committed=committed["eC"], eD=got["eD"], eD_committed=committed["eD"],
                identical_product_files=same_images)


def evaluate(output, p, case_id, label):
    """ML: after metrics against the recipe-specific dirty reference, plus optional frozen-model predictions."""
    from scripts.dataset_v2_matched_extract import FEATURES, extract
    source_run, _, _, prepared = source_paths(p, case_id)
    cell = output / "cases" / case_id / label
    record = json.loads((cell / "record.json").read_text())
    samples = json.loads((cell / "samples.json").read_text())
    if samples["dirty_reference"] == "source_noiseless":
        noiseless = json.loads((source_run / "sources" / prepared["id"] / "recovery" / "noiseless" / "record.json").read_text())
        ref = dict(noiseless, products={k: str(source_run / v) for k, v in noiseless["products"].items()})
    else:
        ref = json.loads((cell / "dirty_reference" / "record.json").read_text())
    masks = source_run / prepared["reference_masks"]
    after = extract(output, record, masks, prepared["nominal_rms"], ref)
    committed = json.loads((source_run / "cases" / case_id / "case.json").read_text())
    flux = json.loads((source_run / "sources" / prepared["id"] / "recovery" / "reference.json").read_text())["flux_jy"]
    rms = FEATURES.index("metrics.residual.scaled_mad_jy_per_beam")

    def summary(eC, eD, X, model_flux, iterations):
        return dict(eC=eC, eD=eD, flux_error=model_flux / flux - 1, residual_rms_jy=X[rms],
                    clean_iterations=iterations.get("iterdone"), X=X)

    out = dict(status="measured", dirty_reference=samples["dirty_reference"], samples=samples,
               before=summary(committed["eC"], committed["eD"], committed["X"], committed["model_flux"]["sum_all_pixels"],
                              committed["convergence"]),
               after=summary(after["eC"], after["eD"], after["X"], after["model_flux"]["sum_all_pixels"],
                             after["iterations"]),
               feature_names=list(FEATURES))
    if p.get("model"):
        out["predictions"] = predictions(p["model"], [out["before"]["X"], out["after"]["X"]])
    return out


def predictions(model, rows):
    """Frozen multitask models (exploratory): decoded outputs for the before and after feature rows."""
    from ml import multitask as mt
    from scripts.dataset_v2_matched_extract import FEATURES, CLEAN2, DIRTY6, MODEL6
    from ml.logreg import COMBINED8_FEATURES
    path = Path(model["path"])
    summary = json.loads((path / "results.json").read_text())
    view = summary.get("feature_view", "combined8")
    selected = (tuple(COMBINED8_FEATURES) + (DIRTY6 if view in ("combined8_dirty6", "all20", "all22") else ())
                + (MODEL6 if view in ("combined8_model6", "all20", "all22") else ()) + (CLEAN2 if view == "all22" else ()))
    x = np.asarray([[np.nan if r[FEATURES.index(n)] is None else r[FEATURES.index(n)] for n in selected] for r in rows], float)
    names = model["names"] or ["xgb_xgb"]
    needed = set(names) | ({"xgb_independent", "xgb_shared"} if set(names) & set(mt.STACKED) else set())
    fitted = {n: mt.load_frozen(path, n) for n in needed}
    out = dict(model=str(path), feature_view=view, outputs=list(mt.outputs(mt.has_dirty(fitted[names[0]]["scale"]))))
    for n in names:
        obj = fitted[n]
        xi = np.column_stack((x, mt._expert_matrix(fitted["xgb_independent"], fitted["xgb_shared"], x))) if n in mt.STACKED else x
        raw = mt.predict_mlp(obj, xi) if n in mt.MLP_NAMES else mt.predict_xgb(obj["model"], xi)
        decoded = mt.decode(raw, obj["scale"])
        out[n] = dict(before=decoded[0].tolist(), after=decoded[1].tolist())
    return out


# ---------------------------------------------------------------- CASA steps

def table_hash(ms, columns):
    from casatools import table
    tb = table(); tb.open(str(ms))
    try:
        out = {}
        for key in columns:
            if key not in tb.colnames():
                continue
            h = hashlib.sha256()
            for start in range(0, tb.nrows(), 4096):
                h.update(np.ascontiguousarray(tb.getcol(key, startrow=start, nrow=min(4096, tb.nrows() - start))).tobytes())
            out[key] = h.hexdigest()
        return out
    finally:
        tb.close()


def flag_stats(ms):
    from casatools import table
    tb = table(); tb.open(str(ms))
    try:
        flagged = total = 0
        weight = 0.
        for start in range(0, tb.nrows(), 4096):
            n = min(4096, tb.nrows() - start)
            f = tb.getcol("FLAG", startrow=start, nrow=n) | tb.getcol("FLAG_ROW", startrow=start, nrow=n)[None, None, :]
            flagged += int(f.sum()); total += f.size
            weight += float(tb.getcol("WEIGHT", startrow=start, nrow=n).sum())
        return dict(rows=tb.nrows(), flagged_fraction=flagged / total, weight_sum=weight)
    finally:
        tb.close()


def same_rows(a, b):
    from casatools import table
    ta, tb = table(), table()
    ta.open(str(a)); tb.open(str(b))
    try:
        if ta.nrows() != tb.nrows():
            return False
        return all(np.array_equal(ta.getcol(k), tb.getcol(k)) for k in ("TIME", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID", "FIELD_ID"))
    finally:
        ta.close(); tb.close()


def case_context(output, case_id):
    from scripts import create_dataset_v1 as g
    p = json.loads((output / "plan.json").read_text())
    source_run, sp, source_plan, prepared = source_paths(p, case_id)
    c = next(x for x in source_plan["cases"] if x["id"] == case_id)
    manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
    source = g._thermal_source(g.THERMAL_DIR, manifest, prepared["id"])
    return p, source_run, source_plan, prepared, c, source


def casa_regen(output, case_id):
    """CASA: rebuild observed.ms exactly as create_dataset_v2_matched.execute_unit and verify it."""
    import casatasks
    from scripts import create_dataset_v1 as g
    from scripts.create_dataset_v2 import _gain_inject, original_flux_spec
    from scripts.create_dataset_v2_matched import column_hash
    from scripts.injection_bounds_pilot import add_paired_noise, prepare_v
    from scripts.simulation import add_thermal_noise_inplace, initialize_weights

    output = output.resolve()
    p, source_run, source_plan, prepared, c, source = case_context(output, case_id)
    committed = json.loads((source_run / "cases" / case_id / "case.json").read_text())
    if committed["case"] != c:
        raise ValueError("Committed case parameters differ from the source plan")
    if file_sha(source.simulation_metadata) != prepared["thermal_metadata_sha256"]:
        raise ValueError("Parent simulation changed")
    flux_spec = original_flux_spec(source)
    if flux_spec != prepared["flux_spec"]:
        raise ValueError("Original flux provenance changed")
    noise_model, noise_params, sigma = g._noise_request(source)
    if sigma != prepared["sigma0_jy"]:
        raise ValueError("Nominal noise changed")
    seed = source_plan["noise_seeds"][c["slot"]]
    work = output / "work" / case_id
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    os.chdir(work)
    v = prepare_v(source, work, flux_spec["sky"], v2=True)
    initialize_weights(v, sigma)
    regular = work / "regular.ms"
    shutil.copytree(v, regular)
    add_thermal_noise_inplace(regular, noise_model=noise_model, noise_parameters=noise_params, seed=seed)
    initialize_weights(regular, sigma)
    v_hash, noise_hash = column_hash(v), column_hash(regular, minus=v)
    if v_hash != committed["v_sha256"] or noise_hash != committed["nominal_noise_sha256"]:
        raise ValueError("Regenerated noiseless/nominal-noise visibilities differ from the committed case")
    observed = work / "observed.ms"
    shutil.copytree(v, observed)
    (work / "gain").mkdir()
    gain = _gain_inject(v, observed, c, work / "gain")
    coupling = add_paired_noise(v, regular, observed, scale=c["a"])
    initialize_weights(observed, sigma)
    stored_gain, stored_noise = committed.get("gain_complex"), committed.get("paired_noise") or {}
    if json.loads(json.dumps(gain)) != json.loads(json.dumps(stored_gain)):
        raise ValueError(f"Gain application differs: {gain} vs {stored_gain}")
    if stored_noise.get("noise_delta_sha256") and coupling["noise_delta_sha256"] != stored_noise["noise_delta_sha256"]:
        raise ValueError("Paired noise realization differs from the committed case")
    shutil.rmtree(regular)
    write_json(work / "regen.json", dict(
        case=c, v_ms=str(v), observed_ms=str(observed), v_sha256=v_hash, nominal_noise_sha256=noise_hash,
        gain_complex=gain, paired_noise=coupling, verified=["case parameters", "v_sha256", "nominal_noise_sha256",
                                                             "gain_complex", "noise_delta_sha256"],
        casa=casatasks.version_string(), pipeline_default_amplitude_selfcal=pipeline_default(output)))


def pipeline_default(output):
    """Cached probe of the pinned pipeline's hif_selfcal amplitude_selfcal default (B only)."""
    p = json.loads((output / "plan.json").read_text())
    if p["experiment"] != "B":
        return None
    path = output / "pipeline_defaults.json"
    if not path.exists():
        res = subprocess.run(runtime_command("casa-pipeline", "--output", output, "--probe-pipeline"), cwd=ROOT,
                             capture_output=True, text=True)
        if not path.exists():
            raise RuntimeError("Pipeline probe failed: " + res.stdout[-2000:] + res.stderr[-2000:])
    return json.loads(path.read_text())["amplitude_selfcal"]


def probe_pipeline(output):
    import pipeline
    from pipeline.hif.tasks.selfcal import selfcal as sc
    defaults = {k: sc.SelfcalInputs.__dict__[k].default for k in
                ("amplitude_selfcal", "apply_cal_mode_default", "gaincal_minsnr", "minsnr_to_proceed", "n_solints")}
    import casatasks
    write_json(output / "pipeline_defaults.json", dict(defaults, pipeline=pipeline.revision,
                                                        casa=casatasks.version_string()))


def refant(ms):
    """Fixed metadata rule: most unflagged visibilities, ties -> lowest antenna id. Never the injection."""
    from casatools import table
    tb = table(); tb.open(str(ms))
    try:
        counts = {}
        for start in range(0, tb.nrows(), 4096):
            n = min(4096, tb.nrows() - start)
            good = (~tb.getcol("FLAG", startrow=start, nrow=n)).sum(axis=(0, 1)) * ~tb.getcol("FLAG_ROW", startrow=start, nrow=n)
            for key in ("ANTENNA1", "ANTENNA2"):
                for ant, k in zip(tb.getcol(key, startrow=start, nrow=n), good):
                    counts[int(ant)] = counts.get(int(ant), 0) + int(k)
    finally:
        tb.close()
    best = min(counts, key=lambda a: (-counts[a], a))
    tb = table(); tb.open(str(Path(ms) / "ANTENNA"))
    try:
        return best, str(tb.getcol("NAME")[best])
    finally:
        tb.close()


def solution_flags(caltable):
    from casatools import table
    tb = table(); tb.open(str(caltable))
    try:
        f = tb.getcol("FLAG")
        return dict(solutions=int(f.size), flagged_fraction=float(f.mean()))
    finally:
        tb.close()


def casa_recipe(output, case_id, recipe):
    """CASA (A) or CASA pipeline (B): run one recipe on a fresh copy of observed.ms -> out.ms (DATA = output)."""
    output = output.resolve()
    work_case = output / "work" / case_id
    regen = json.loads((work_case / "regen.json").read_text())
    experiment, gain, flagging = parse(recipe)
    work = work_case / safe(recipe)
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    os.chdir(work)
    record = dict(recipe=recipe, flagging=flagging)
    started = time.time()
    try:
        record.update((recipe_a if experiment == "A" else recipe_b)(work, regen, gain, flagging))
    except Exception:
        record.update(disposition=record.get("disposition", "runtime_failure"), reason=traceback.format_exc()[-6000:])
    record["seconds"] = round(time.time() - started, 1)
    if record.get("output_ms") and Path(record["output_ms"]).exists():
        observed = regen["observed_ms"]
        record["rows_match_input"] = same_rows(observed, record["output_ms"])
        record["input_samples"], record["output_samples"] = flag_stats(observed), flag_stats(record["output_ms"])
        record["new_flag_fraction"] = record["output_samples"]["flagged_fraction"] - record["input_samples"]["flagged_fraction"]
        before = table_hash(observed, FLAG_COLUMNS + WEIGHT_COLUMNS + SIGMA_COLUMNS)
        after = table_hash(record["output_ms"], FLAG_COLUMNS + WEIGHT_COLUMNS + SIGMA_COLUMNS)
        record["changed_columns"] = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        record["flags_changed"] = any(k in FLAG_COLUMNS for k in record["changed_columns"])
        record["weights_changed"] = any(k in WEIGHT_COLUMNS for k in record["changed_columns"])
        if not flagging and record["flags_changed"]:
            record["flagging_off_violated"] = True     # labelled, never silently accepted
        if not record["rows_match_input"]:
            record.update(disposition="unsupported", reason="output rows cannot be matched to the input sampling")
        elif record["output_samples"]["flagged_fraction"] >= 1:
            record.update(disposition="no_usable_output", reason="all output visibilities flagged")
    write_json(work / "recipe.json", record)


def solve_a(work_case, regen, gain):
    """Solve the A gains once per case and gain recipe (flagging_off and flagging_on only differ in applymode)."""
    from casatasks import clearcal, gaincal
    from casatools import table
    solve = work_case / f"A_solve_{gain}"
    done = solve / "solve.json"
    if done.exists():
        return json.loads(done.read_text())
    shutil.rmtree(solve, ignore_errors=True); solve.mkdir(parents=True)
    ms = solve / "solve.ms"
    shutil.copytree(regen["observed_ms"], ms)
    clearcal(vis=str(ms), addmodel=True)
    a, b = table(), table()
    a.open(regen["v_ms"]); b.open(str(ms), nomodify=False)
    try:                                           # exact injected sky as the model (row order is a copy)
        for start in range(0, a.nrows(), 4096):
            n = min(4096, a.nrows() - start)
            b.putcol("MODEL_DATA", a.getcol("DATA", startrow=start, nrow=n), startrow=start, nrow=n)
    finally:
        a.close(); b.close()
    ref_id, ref_name = refant(ms)
    tables = [solve / "phase.G"]
    gaincal(vis=str(ms), caltable=str(tables[0]), refant=ref_name, **A_SETTINGS["phase"])
    if gain == "phase_amp":
        tables.append(solve / "amp.G")
        gaincal(vis=str(ms), caltable=str(tables[1]), refant=ref_name, gaintable=[str(tables[0])], **A_SETTINGS["amp"])
    shutil.rmtree(ms)
    missing = [t.name for t in tables if not t.exists()]
    out = dict(tables=[str(t) for t in tables], refant=dict(id=ref_id, name=ref_name), missing=missing,
               solutions={t.name: solution_flags(t) for t in tables if t.exists()})
    write_json(done, out)
    return out


def recipe_a(work, regen, gain, flagging):
    from casatasks import applycal, split
    solved = solve_a(work.parent, regen, gain)
    solutions, refant_info = solved["solutions"], solved["refant"]
    if solved["missing"]:
        return dict(disposition="no_valid_solutions", reason=f"gaincal wrote no table {solved['missing']}",
                    refant=refant_info)
    inp = work / "in.ms"
    shutil.copytree(regen["observed_ms"], inp)
    tables = solved["tables"]
    applycal(vis=str(inp), gaintable=tables, interp=[A_SETTINGS["interp"]] * len(tables),
             calwt=[A_SETTINGS["calwt"]] * len(tables), applymode="calflag" if flagging else "calonly", flagbackup=False)
    out = work / "out.ms"
    split(vis=str(inp), outputvis=str(out), datacolumn="corrected", keepflags=True)
    valid = any(x["flagged_fraction"] < 1 for x in solutions.values())
    return dict(output_ms=str(out), refant=refant_info, solutions=solutions,
                applymode="calflag" if flagging else "calonly", correction_applied=dict(
                    phase=solutions["phase.G"]["flagged_fraction"] < 1,
                    amplitude=gain == "phase_amp" and solutions.get("amp.G", {}).get("flagged_fraction", 1) < 1),
                disposition="applied" if valid else "no_valid_solutions",
                reason=None if valid else "every gain solution flagged (minsnr/minblperant)")


def recipe_b(work, regen, gain, flagging):
    import pipeline
    from casatasks import split
    from casatools import table
    pipeline.initcli()
    g = sys.modules["__main__"].__dict__
    task = lambda n: g.get(n) or getattr(pipeline.hif.cli, n, None) or getattr(pipeline.hifv.cli, n, None) \
        or getattr(pipeline.h.cli, n)
    inp = work / "pipe_in.ms"
    split(vis=regen["observed_ms"], outputvis=str(inp), datacolumn="data")   # explicit: DATA holds the corruption
    tb = table(); tb.open(str(inp))
    cols = tb.colnames(); tb.close()
    if "MODEL_DATA" in cols or "CORRECTED_DATA" in cols:
        raise ValueError(f"Pipeline input must hold only DATA, has {cols}")
    tb.open(str(inp / "STATE"), nomodify=False)
    try:                                           # register the simulated field as a science target
        tb.putcol("OBS_MODE", np.array(["OBSERVE_TARGET#ON_SOURCE"] * tb.nrows()))
    finally:
        tb.close()
    tb.open(str(inp / "FIELD")); field = str(tb.getcol("NAME")[0]); tb.close()
    planned = json.loads((work.parents[2] / "plan.json").read_text())["settings"]
    settings = planned["flagging_on" if flagging else "flagging_off"]
    kwargs = dict(planned["gain"][gain])
    if settings["apply_cal_mode_default"]:
        kwargs["apply_cal_mode_default"] = settings["apply_cal_mode_default"]
    record = dict(pipeline=pipeline.revision, field=field, selfcal_arguments=dict(kwargs, apply=True),
                  flagtargetsdata=settings["flagtargetsdata"],
                  modified_recipe=True,
                  modification=", ".join(["no regcal hif_makeimages"] + (["calonly + no hifv_flagtargetsdata"] if not flagging else [])))
    context = task("h_init")()
    context.set_state("ProjectSummary", "observatory", "Karl G. Jansky Very Large Array")
    context.set_state("ProjectSummary", "telescope", "EVLA")
    try:
        try:
            task("hifv_importdata")(vis=[str(inp)], datacolumns={"data": "regcal_contline_science"}, specline_spws="none")
        except Exception:
            return dict(record, disposition="unsupported", reason="hifv_importdata: " + traceback.format_exc()[-3000:])
        if settings["flagtargetsdata"]:
            task("hifv_flagtargetsdata")()
        task("hif_checkproductsize")(maximsize=planned["maximsize"])
        task("hif_makeimlist")(specmode="cont", datatype="regcal", field=field)   # no hif_makeimages (decision)
        try:
            result = task("hif_selfcal")(field=field, apply=True, **kwargs)
        except Exception:
            return dict(record, disposition="runtime_failure", reason="hif_selfcal: " + traceback.format_exc()[-3000:])
        record["selfcal_result"] = summarize_selfcal(result)
    finally:
        try:
            task("h_save")()
        except Exception:
            pass
    # Which MS did hif_selfcal correct? Check every MS in the work directory, never assume.
    changed = {}
    for ms in sorted(work.glob("*.ms")):
        tb = table(); tb.open(str(ms))
        try:
            if "CORRECTED_DATA" in tb.colnames():
                n = min(20000, tb.nrows())
                changed[ms.name] = bool(np.any(tb.getcol("CORRECTED_DATA", 0, n) != tb.getcol("DATA", 0, n)))
        finally:
            tb.close()
    record["corrected_columns"] = changed
    try:
        record["native_crops"] = save_native(work, Path(regen["observed_ms"]).parents[2] / "native" /
                                             f"{Path(regen['observed_ms']).parent.name}__{safe(record['recipe'])}.npz")
    except Exception:
        record["native_crops"] = None
        record["native_crops_error"] = traceback.format_exc()[-1500:]
    weblogs = sorted(work.glob("pipeline-*"))
    if weblogs:
        record["weblog"] = str(weblogs[-1])
    applied = [name for name, diff in changed.items() if diff]
    source = work / applied[0] if applied else inp
    out = work / "out.ms"
    split(vis=str(source), outputvis=str(out), datacolumn="corrected" if applied else "data", keepflags=True)
    record.update(output_ms=str(out), output_from=f"{source.name}:{'CORRECTED_DATA' if applied else 'DATA'}",
                  correction_applied=dict(any=bool(applied)),
                  disposition="applied" if applied else "not_applied",
                  reason=None if applied else "hif_selfcal applied no solutions (see selfcal_result)")
    return record


NATIVE = ("dirty.image.tt0", "initial.image.tt0", "initial.residual.tt0", "initial.psf.tt0", "initial.mask",
          "final.image.tt0", "final.mask")


def save_native(work, target, half=128):
    """Central crops of hif_selfcal's own images (secondary diagnostics), plus whole-image stats."""
    from casatools import image
    arrays, meta = {}, {}
    for d in sorted(work.glob("sc_workdir_*")):
        for path in sorted(d.glob("sc.*")):
            key = path.name.rsplit("_", 1)[-1]
            if key not in NATIVE or not path.is_dir():
                continue
            ia = image(); ia.open(str(path))
            try:
                shape = ia.shape()
                cx, cy = shape[0] // 2, shape[1] // 2
                blc, trc = [cx - half, cy - half] + [0] * (len(shape) - 2), [cx + half - 1, cy + half - 1] + [0] * (len(shape) - 2)
                arrays[key] = np.squeeze(ia.getchunk(blc=blc, trc=trc)).T.astype(np.float32)
                stats = ia.statistics(verbose=False)
                cell = ia.coordsys().increment()["numeric"][:2]
                beam = ia.restoringbeam() if key.endswith("image.tt0") else {}
                meta[key] = dict(shape=[int(x) for x in shape], cell_arcsec=[float(abs(c)) * 206264.806 for c in cell],
                                 full_max=float(stats["max"][0]), full_min=float(stats["min"][0]),
                                 full_sum=float(stats["sum"][0]), full_rms=float(stats["rms"][0]),
                                 beam=dict(major_arcsec=float(beam["major"]["value"]) if beam else None,
                                           minor_arcsec=float(beam["minor"]["value"]) if beam else None))
            finally:
                ia.close()
    if not arrays:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, meta=json.dumps(meta), **{k.replace(".", "_"): v for k, v in arrays.items()})
    return str(target)


def summarize_selfcal(result):
    """JSON-safe scalars from the hif_selfcal result (acceptance flags, final solints, reasons)."""
    out = dict(type=type(result).__name__)
    items = result if isinstance(result, (list, tuple)) else getattr(result, "results", None) or [result]
    targets = []
    for r in items:
        for t in (getattr(r, "targets", None) or []):
            if isinstance(t, dict):
                targets.append({k: v for k, v in t.items() if isinstance(v, (str, int, float, bool, type(None)))})
    out["targets"] = targets
    out["repr"] = repr(result)[:2000]
    return out


def casa_image(output, case_id, label, ms):
    """CASA: unchanged matched-v2 evaluation imaging of one MS; rebuild I_r(V0) if flags/weights changed."""
    from scripts.recovery_products import recovery_products
    output = output.resolve()
    p, source_run, source_plan, prepared, c, source = case_context(output, case_id)
    from scripts.create_dataset_v2 import original_flux_spec
    from scripts import create_dataset_v1 as g
    _, _, sigma = g._noise_request(source)
    regen = json.loads((output / "work" / case_id / "regen.json").read_text())
    v = Path(regen["v_ms"])
    folder = source_run / "sources" / prepared["id"] / "recovery"
    scratch = output / "work" / case_id / f"img_{label}"
    shutil.rmtree(scratch, ignore_errors=True); scratch.mkdir(parents=True)
    os.chdir(scratch)
    complete, image_cell = recovery_products(output, folder, source, original_flux_spec(source)["sky"], sigma,
                                             prepared["settings_hash"], v)
    cell = output / "cases" / case_id / label
    if cell.exists():
        shutil.rmtree(cell)
    seed = source_plan["noise_seeds"][c["slot"]]
    image_cell(cell, Path(ms), c["a"], c["slot"], seed, metadata=dict(case=c, evaluated_ms=str(ms)),
               require_same_samples=False)
    if label == "before_repro":
        os.chdir(output)
        shutil.rmtree(scratch, ignore_errors=True)
        return
    changed = table_hash(v, FLAG_COLUMNS + WEIGHT_COLUMNS) != table_hash(ms, FLAG_COLUMNS + WEIGHT_COLUMNS)
    samples = dict(flags_or_weights_changed=changed, dirty_reference="source_noiseless")
    if changed:
        if not same_rows(v, ms):
            raise ValueError("Cannot rebuild the dirty reference: output rows do not match V0")
        from casatools import table
        ref = scratch / "reference.ms"
        shutil.copytree(ms, ref)
        a, b = table(), table()
        a.open(str(v)); b.open(str(ref), nomodify=False)
        try:                                       # same flags/weights/sampling, noiseless sky, no gains applied
            for start in range(0, a.nrows(), 4096):
                n = min(4096, a.nrows() - start)
                b.putcol("DATA", a.getcol("DATA", startrow=start, nrow=n), startrow=start, nrow=n)
        finally:
            a.close(); b.close()
        image_cell(cell / "dirty_reference", ref, 0., c["slot"], seed, metadata=dict(reference="I_r(V0)"),
                   require_same_samples=False)
        samples["dirty_reference"] = "recipe_flags_weights"
    write_json(cell / "samples.json", samples)
    os.chdir(output)                               # CASA aborts at exit if its cwd was deleted
    shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- report (ML)

def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    results = [json.loads(x.read_text()) for x in sorted((output / "results").glob("*.json"))]
    units = {u["case"]: json.loads((output / "units" / f"{u['case']}.json").read_text())
             for u in p["units"] if (output / "units" / f"{u['case']}.json").exists()}
    source_run = Path(p["source_run"])
    case = {r["case"]: json.loads((source_run / "cases" / r["case"] / "case.json").read_text())["case"] for r in results}

    def mask(c):
        return (c["A"] != 1) + 2 * (c["phi_deg"] != 0) + 4 * (c["a"] != 1)
    names = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise", 7: "all"}
    by_recipe = {r: [x for x in results if x["recipe"] == r] for r in p["recipes"]}
    html = ["<!doctype html><meta charset=utf-8><title>VLA pipeline QA</title><style>body{font-family:sans-serif;"
            "max-width:1300px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;margin:.6em 0}td,th{border:1px "
            "solid #ccc;padding:3px 7px;text-align:right}th{background:#f3f3f3}td:first-child{text-align:left}"
            "img{max-width:100%}</style>",
            f"<h1>VLA pipeline QA, experiment {p['experiment']}</h1><p>{len(units)}/{len(p['units'])} cases from "
            f"{source_run.name}. eC: CLEAN recovery error (0 = perfect). eD: dirty error vs I_r(V0). "
            "Medians over measured cases. Same evaluation imaging for every recipe.</p>"]
    repro = [u.get("reproduction") for u in units.values() if u.get("reproduction")]
    if repro:
        html.append(f"<p>Reproduction of committed before results: {sum(r['status'] == 'agree' for r in repro)}/{len(repro)} agree.</p>")
    html.append("<h2>Outcomes per recipe</h2><table><tr><th>recipe</th><th>attempted</th><th>measured</th><th>applied</th>"
                "<th>not applied</th><th>no valid sol.</th><th>no usable output</th><th>unsupported</th><th>failure</th>"
                "<th>eC before</th><th>eC after</th><th>eD before</th><th>eD after</th><th>flux err after</th>"
                "<th>new flags</th><th>flag-off violated</th></tr>")
    med = lambda xs: "–" if not xs else f"{np.median(xs):.3f}"
    for r, xs in by_recipe.items():
        ev = [x["evaluation"] for x in xs if x["evaluation"].get("status") == "measured"]
        disp = [x["recipe_record"].get("disposition") for x in xs]
        rec = [x["recipe_record"] for x in xs]
        html.append(f"<tr><td>{r}</td><td>{len(xs)}</td><td>{len(ev)}</td>" +
                    "".join(f"<td>{disp.count(k)}</td>" for k in ("applied", "not_applied", "no_valid_solutions",
                                                                  "no_usable_output", "unsupported", "runtime_failure")) +
                    f"<td>{med([e['before']['eC'] for e in ev])}</td><td>{med([e['after']['eC'] for e in ev])}</td>"
                    f"<td>{med([e['before']['eD'] for e in ev])}</td><td>{med([e['after']['eD'] for e in ev])}</td>"
                    f"<td>{med([e['after']['flux_error'] for e in ev])}</td>"
                    f"<td>{med([q.get('new_flag_fraction', 0) for q in rec if 'new_flag_fraction' in q])}</td>"
                    f"<td>{sum(bool(q.get('flagging_off_violated')) for q in rec)}</td></tr>")
    html.append("</table><h2>Median eC before → after by corruption</h2><table><tr><th>corruption</th>" +
                "".join(f"<th>{r}</th>" for r in p["recipes"]) + "</tr>")
    for m in sorted(names):
        row = []
        for r in p["recipes"]:
            ev = [x["evaluation"] for x in by_recipe[r] if mask(case[x["case"]]) == m and x["evaluation"].get("status") == "measured"]
            row.append("–" if not ev else f"{np.median([e['before']['eC'] for e in ev]):.2f} → {np.median([e['after']['eC'] for e in ev]):.2f} (n={len(ev)})")
        if any(v != "–" for v in row):
            html.append(f"<tr><td>{names[m]}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    html.append("</table>")
    fig, axes = plt.subplots(1, len(p["recipes"]), figsize=(3.3 * len(p["recipes"]), 3.4), squeeze=False, layout="constrained")
    colors = {0: "#000", 1: "#d62728", 2: "#1f77b4", 4: "#2ca02c"}
    for ax, r in zip(axes[0], p["recipes"]):
        for x in by_recipe[r]:
            e = x["evaluation"]
            if e.get("status") != "measured":
                continue
            m = mask(case[x["case"]])
            ax.scatter(e["before"]["eC"], e["after"]["eC"], s=14, color=colors.get(m, "#9467bd"), alpha=.8)
        ax.plot([0, 1.2], [0, 1.2], color="#888", lw=.8); ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.2)
        ax.set_title(r, fontsize=8); ax.set_xlabel("eC before"); ax.set_ylabel("eC after")
    fig.suptitle("eC before vs after (red amp, blue phase, green noise, purple mixtures, black control)", fontsize=9)
    fig.savefig(output / "eC_before_after.png", dpi=130); plt.close(fig)
    html.append("<img src='eC_before_after.png'><h2>Unavailable results</h2><table><tr><th>case</th><th>recipe</th>"
                "<th>disposition</th><th>reason</th></tr>")
    for x in results:
        if x["evaluation"].get("status") != "measured":
            reason = str(x["recipe_record"].get("reason") or x["evaluation"].get("reason"))[-300:]
            html.append(f"<tr><td>{x['case']}</td><td>{x['recipe']}</td><td>{x['recipe_record'].get('disposition')}</td>"
                        f"<td style=text-align:left>{reason}</td></tr>")
    html.append("</table>")
    skipped = [x for x in results if x["recipe_record"].get("disposition") == "not_applied"]
    if skipped:
        html.append("<h2>Corrections not applied: pipeline's stated reason</h2><table><tr><th>case</th><th>recipe</th>"
                    "<th>pipeline log</th></tr>")
        for x in skipped:
            notes = "<br>".join(x["recipe_record"].get("pipeline_notes", [])[:12]) or "–"
            html.append(f"<tr><td>{x['case']}</td><td>{x['recipe']}</td><td style=text-align:left>{notes}</td></tr>")
        html.append("</table>")
    (output / "report.html").write_text("\n".join(html))
    write_json(output / "summary.json", dict(recipes={r: [dict(case=x["case"], disposition=x["recipe_record"].get("disposition"),
                                                                eC_before=x["evaluation"].get("before", {}).get("eC"),
                                                                eC_after=x["evaluation"].get("after", {}).get("eC"))
                                                           for x in xs] for r, xs in by_recipe.items()}))
    print("REPORT", output / "report.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    st = ap.add_mutually_exclusive_group(required=True)
    st.add_argument("--plan", action="store_true")
    st.add_argument("--execute", action="store_true")
    st.add_argument("--report", action="store_true")
    st.add_argument("--unit", metavar="CASE", help=argparse.SUPPRESS)
    st.add_argument("--casa-regen", metavar="CASE", help=argparse.SUPPRESS)
    st.add_argument("--casa-recipe", nargs=2, metavar=("CASE", "RECIPE"), help=argparse.SUPPRESS)
    st.add_argument("--casa-image", nargs=3, metavar=("CASE", "LABEL", "MS"), help=argparse.SUPPRESS)
    st.add_argument("--probe-pipeline", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--source-run", type=Path)
    ap.add_argument("--experiment", choices=("A", "B"))
    ap.add_argument("--recipes", nargs="*", help="Subset of recipe names (default: all of the experiment)")
    ap.add_argument("--cases", default="representative", help="all | representative | comma-separated case ids")
    ap.add_argument("--reproduce", default="representative",
                    help="Cases whose committed before result is re-imaged and checked (same syntax, or none)")
    ap.add_argument("--model", type=Path, help="Frozen ml.multitask output folder for exploratory predictions")
    ap.add_argument("--model-names", nargs="*", default=None)
    ap.add_argument("--parallel", type=int, default=1)
    a = ap.parse_args()
    if a.plan:
        if not a.source_run or not a.experiment:
            ap.error("--plan needs --source-run and --experiment")
        plan(a.output, a.source_run, a.experiment, a.recipes, a.cases, a.reproduce, a.model, a.model_names)
    elif a.execute:
        execute(a.output, a.parallel)
    elif a.report:
        report(a.output)
    elif a.unit:
        unit(a.output, a.unit)
    elif a.casa_regen:
        casa_regen(a.output, a.casa_regen)
    elif a.casa_recipe:
        casa_recipe(a.output, *a.casa_recipe)
    elif a.casa_image:
        casa_image(a.output, *a.casa_image)
    else:
        probe_pipeline(a.output.resolve())


if __name__ == "__main__":
    main()
