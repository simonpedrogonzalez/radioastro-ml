"""Visualize the summary-table injection values with paired pilot imaging."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.injection_bounds_report import case, cell, digest
from scripts.preprocessing.schema import atomic_write_json

PILOT = Path("runs/injection-bounds-pilot-20261102")
DEFAULT_OUTPUT = Path("runs/injection-range-visual-20261104")
SOURCE_ORDER = ("0449+113", "1415+133", "1044+809")
PHASE = {
    "negative": {"low": ("-60", "-20", "-33.54"), "high": ("-180", "-180", "-180")},
    "positive": {"low": ("20", "60", "32.5"), "high": ("180", "180", "180")},
}
AMP = {
    "attenuation": {"low": ("0.3536", "0.8216", "0.6384"), "high": ("0.01", "0.01", "0.01")},
    "amplification": {"low": ("1.308", "2.378", "1.767"), "high": ("6.727", "32", "14.45")},
}
NOISE = {
    "a-summary": {"low": ("1.375", "150", "61.73"), "high": ("20", "4000", "1304")},
    "r-summary": {"low": ("0.01133", "0.01934", "0.01597"), "high": ("0.1812", "0.3624", "0.3018")},
}
STATS = ("minimum", "maximum", "mean")
CHANNELS = ("truth", "recovered", "error", "dirty", "clean", "residual")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def specification(pilot):
    plan = json.loads((pilot / "plan.json").read_text())
    if plan.get("smoke") or [s["id"] for s in plan["sources"]] != list(SOURCE_ORDER) or plan["seeds"] != [0, 1, 2]:
        raise ValueError("Expected the completed three-template, three-seed pilot")
    if json.loads((pilot / "status.json").read_text()).get("status") != "complete":
        raise ValueError("The source pilot is incomplete")
    return plan


def requested_stops(source):
    """One stop per physical slider value, with every summary badge retained."""
    snr0 = float(source["nominal_source_snr"])
    result = {kind: {} for kind in ("phase", "amp", "noise")}

    def add(kind, physical, badge, *, from_summary=None):
        key = f"{physical:.14g}"
        stop = result[kind].setdefault(key, dict(value=float(key), badges=[], admissible=True))
        stop["badges"].append(badge)
        if kind == "noise":
            stop["a"] = float(key)
            stop["r"] = float(key) / snr0
            if from_summary is not None:
                stop.setdefault("sources", []).append(from_summary)
            if physical < 1:
                stop.update(admissible=False, reason="a < 1 is outside the requested extra-noise domain")

    for kind, control in (("phase", 0.), ("amp", 1.), ("noise", 1.)):
        add(kind, control, "ordinary control")
    for branch, columns in PHASE.items():
        for column, values in columns.items():
            for stat, value in zip(STATS, values, strict=True):
                add("phase", float(value), f"{branch} · {column} parameter · {stat}")
    for branch, columns in AMP.items():
        for column, values in columns.items():
            for stat, value in zip(STATS, values, strict=True):
                add("amp", float(value), f"{branch} · {column} parameter · {stat}")
    for origin, columns in NOISE.items():
        for column, values in columns.items():
            for stat, value in zip(STATS, values, strict=True):
                requested = float(value)
                actual = requested if origin == "a-summary" else requested * snr0
                add("noise", actual, f"{origin} · {column} parameter · {stat}", from_summary=origin)
    for kind in result:
        result[kind] = sorted(result[kind].values(), key=lambda stop: stop["value"])
    for stop in result["phase"]:
        if abs(stop["value"]) == 180.:
            stop["badges"].append("equivalent ±180° physical case")
    return result


def requested_cases(plan):
    cases = {}
    sliders = {}
    for source in plan["sources"]:
        sid = source["id"]
        stops = requested_stops(source)
        sliders[sid] = stops
        for kind, items in stops.items():
            antennas = [None] if kind == "noise" else source["selected_antennas"]
            for stop in items:
                stop["cases"] = {}
                if not stop["admissible"]:
                    continue
                for antenna in antennas:
                    for seed in plan["seeds"]:
                        # The pilot case helper canonicalizes ±180° to one
                        # physical case while the two slider labels remain.
                        c = case(sid, kind, stop["value"], seed, antenna)
                        cases[c["id"]] = c
                        stop["cases"][f"{antenna if antenna is not None else 'noise'}:{seed}"] = c["id"]
    return sliders, cases


def verify_case(root, c, pilot_plan, *, full_hashes=True):
    """Confirm a checkpoint matches physical, imaging and reference contracts."""
    folder = root / "sources" / c["source"] / "recovery"
    record_path = cell(root, c) / "record.json"
    if not record_path.is_file():
        return None
    rec = json.loads(record_path.read_text())
    if rec.get("case") != c:
        raise ValueError(f"Case mismatch: {record_path}")
    source = next(s for s in pilot_plan["sources"] if s["id"] == c["source"])
    specification = dict(schema=1, source=source, configuration=pilot_plan["configuration"],
                         seeds=pilot_plan["seeds"], software=pilot_plan["software"],
                         protocol=pilot_plan["protocol"],
                         seed_map={str(seed): __import__("scripts.create_dataset_v1", fromlist=["_stable_seed"])._stable_seed(
                             c["source"], f"injection_bounds:noise_seed:{seed}") for seed in pilot_plan["seeds"]})
    settings_hash = digest(specification)
    if rec.get("settings_hash") != settings_hash or rec.get("case_hash") != digest(dict(case=c, settings=settings_hash)):
        raise ValueError(f"Settings/case hash mismatch: {record_path}")
    if json.loads((folder / "settings.json").read_text()) != specification:
        raise ValueError(f"Source settings differ: {folder}")
    reference = json.loads((folder / "reference.json").read_text())
    grid = json.loads((folder / "reference_grid.json").read_text())
    if rec["resolved_configuration"]["grid"]["image_grid"] != grid["image_grid"]:
        raise ValueError(f"Image grid differs: {record_path}")
    if not np.allclose(list(rec["beam"].values()), list(reference["beam"].values()), rtol=1e-6, atol=1e-6):
        raise ValueError(f"Restoring beam differs: {record_path}")
    for name, expected in rec["reference_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Reference/mask hash differs: {record_path}: {name}")
    if full_hashes:
        for key, expected in rec["sha256"].items():
            if sha(root / rec["products"][key]) != expected:
                raise ValueError(f"Imaging product hash differs: {record_path}: {key}")
    return rec


def prepare(output, pilot):
    output, pilot = output.resolve(), pilot.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = specification(pilot)
    pilot_plan_hash = sha(pilot / "plan.json")
    sliders, cases = requested_cases(plan)
    pilot_cases = {c["id"]: c for c in json.loads((pilot / "cases.json").read_text())["cases"]}
    ownership = {}
    for cid, c in cases.items():
        if cid in pilot_cases and pilot_cases[cid] == c and verify_case(pilot, c, plan):
            ownership[cid] = "pilot"
        elif verify_case(output, c, plan):
            ownership[cid] = "report"
        else:
            ownership[cid] = "missing"
    # The report-owned source specification preserves the exact fixed reference.
    for source in plan["sources"]:
        sid = source["id"]
        source_folder = pilot / "sources" / sid / "recovery"
        target = output / "sources" / sid / "recovery"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("settings.json", "reference.json", "reference_grid.json", "reference_masks.npz"):
            src, dst = source_folder / name, target / name
            if dst.exists():
                if sha(src) != sha(dst):
                    raise ValueError(f"Report-owned fixed reference changed: {dst}")
            else:
                shutil.copy2(src, dst)
    existing = output / "case_manifest.json"
    payload = dict(schema=1, pilot_root=str(pilot), pilot_plan_sha256=pilot_plan_hash,
                   sources=[dict(id=s["id"], nominal_source_snr=s["nominal_source_snr"],
                                 selected_antennas=s["selected_antennas"]) for s in plan["sources"]],
                   antenna_roles={s["id"]: {str(ant): ("low contribution" if ant == min(
                       s["selected_antennas"], key=lambda a: s["norms"][str(a)]["V_Ak_over_sigma_L2"]) else
                       "high contribution") for ant in s["selected_antennas"]} for s in plan["sources"]},
                   seeds=plan["seeds"], sliders=sliders,
                   cases=[dict(case=c, origin=ownership[cid]) for cid, c in sorted(cases.items())])
    if existing.exists():
        old = json.loads(existing.read_text())
        # Migrate a preflight manifest that listed the equivalent +180° run
        # separately. Its old case records remain untouched on disk.
        old_sliders = old["sliders"]
        for sid in SOURCE_ORDER:
            for stop in old_sliders[sid]["phase"]:
                if stop["value"] == 180.:
                    stop["cases"] = next(s["cases"] for s in old_sliders[sid]["phase"] if s["value"] == -180.)
                if abs(stop["value"]) == 180. and "equivalent ±180° physical case" not in stop["badges"]:
                    stop["badges"].append("equivalent ±180° physical case")
        if {k: old[k] for k in ("schema", "pilot_root", "pilot_plan_sha256", "sources", "seeds")} != {
                k: payload[k] for k in ("schema", "pilot_root", "pilot_plan_sha256", "sources", "seeds")} or old_sliders != sliders:
            raise ValueError("Explicit case manifest scope changed")
        old_cases = {entry["case"]["id"]: entry for entry in old["cases"]}
        for entry in payload["cases"]:
            prior = old_cases.get(entry["case"]["id"], {})
            if entry["origin"] == prior.get("origin") and prior.get("record") and prior.get("record_sha256"):
                entry.update(record=prior["record"], record_sha256=prior["record_sha256"])
    atomic_write_json(existing, payload)
    missing = sum(origin == "missing" for origin in ownership.values())
    invalid = [(sid, stop["a"], stop["r"]) for sid, kinds in sliders.items()
               for stop in kinds["noise"] if not stop["admissible"]]
    free = shutil.disk_usage(output).free
    pilot_size = sum((pilot / json.loads((cell(pilot, c) / "record.json").read_text())["products"]["clean"]).stat().st_size
                     for cid, c in cases.items() if ownership[cid] == "pilot")
    reused = sum(origin == "pilot" for origin in ownership.values())
    estimated_case_bytes = max(4_000_000, int(pilot_size / max(reused, 1) * 8))
    estimated_disk = missing * estimated_case_bytes + len(cases) * 6 * 500_000
    preflight = dict(unique_cases=len(cases), reused_pilot=reused,
                     reused_report=sum(origin == "report" for origin in ownership.values()), missing=missing,
                     inadmissible_noise_stops=invalid, estimated_disk_bytes=estimated_disk,
                     available_disk_bytes=free, estimate_rule="8x mean reused clean FITS plus six PNGs per case")
    if estimated_disk > free * .8:
        raise RuntimeError("Insufficient free disk for report preflight")
    atomic_write_json(output / "preflight.json", preflight)
    return payload, preflight


def simulate(output, pilot, manifest):
    """Execute only absent cases using the pilot's injection/imaging implementation."""
    from scripts import injection_bounds_pilot as runner
    output, pilot = output.resolve(), pilot.resolve()
    plan = specification(pilot)
    missing = [entry["case"] for entry in manifest["cases"] if entry["origin"] == "missing"]
    if not missing:
        return
    atomic_write_json(output / "plan.json", plan)
    status = dict(started_epoch=time.time(), new_images=0, failures=0, reused_images=0)
    atomic_write_json(output / "status.json", status)
    original_render = runner.render_report
    runner.render_report = lambda root: None
    try:
        runner.run_batch(output, plan, {"cases": missing}, status)
    finally:
        runner.render_report = original_render
    failures = [str(cell(output, c) / "failure.json") for c in missing if (cell(output, c) / "failure.json").exists()]
    if failures:
        raise RuntimeError(f"{len(failures)} explicit cases failed; first: {failures[0]}")
    for c in missing:
        verify_case(output, c, plan)


def _derived_fits(path, array, header):
    from astropy.io import fits
    header = header.copy()
    header["BUNIT"] = "Jy/beam"
    fits.PrimaryHDU(np.asarray(array, dtype=np.float64), header=header).writeto(path)
    return path


def _render_assets(output, pilot, manifest, *, plan=None, peak_anchor=False):
    from tempfile import TemporaryDirectory
    from astropy.io import fits
    from scripts.clean_recovery_noise_sweep import convolve_model, load_products, measure
    from scripts.imaging.plot_utils import casa_image_to_png, shared_fits_display_limits

    plan = specification(pilot) if plan is None else plan
    entries = {entry["case"]["id"]: entry for entry in manifest["cases"]}
    images = output / "images"
    images.mkdir(exist_ok=True)
    recipes = {}
    data = {}
    color_limits = {}
    with TemporaryDirectory(prefix="derived-", dir=output) as temporary:
        temp = Path(temporary)
        for source in manifest["sources"]:
            sid = source["id"]
            source_cases = [entry for entry in manifest["cases"] if entry["case"]["source"] == sid]
            source_temp = temp / sid
            source_temp.mkdir()
            masks_path = pilot / "sources" / sid / "recovery" / "reference_masks.npz"
            with np.load(masks_path) as saved:
                masks = {key: saved[key] for key in saved.files}
            truth = np.asarray(masks["truth"], dtype=float)
            omega = masks["omega"]
            control_cache = {}
            source_frames = {}
            paths_by_group = defaultdict(list)
            truth_path = None
            for n, entry in enumerate(source_cases, 1):
                c = entry["case"]
                root = pilot if entry["origin"] == "pilot" else output
                rec = verify_case(root, c, plan, full_hashes=False)
                if rec is None:
                    raise ValueError(f"Missing visual case {c['id']}")
                with np.load(root / rec["products"]["support"]) as support:
                    if not support["valid"][omega].all() or not np.array_equal(support["clean_search"], masks["clean_search"]):
                        raise ValueError(f"Fixed mask/support mismatch for {c['id']}")
                raw = load_products(root, rec)
                clean_path = root / rec["products"]["clean"]
                header = fits.getheader(clean_path)
                recovered = convolve_model(raw["model"], header)
                error = recovered - truth
                arrays = dict(truth=truth, recovered=recovered, error=error,
                              dirty=raw["dirty"], clean=raw["clean"], residual=raw["residual"])
                if any(a.shape != truth.shape or not np.isfinite(a[omega]).all() for a in arrays.values()):
                    raise ValueError(f"Invalid six-channel image row for {c['id']}")
                if truth_path is None:
                    truth_path = _derived_fits(source_temp / "truth.fits", truth, header)
                recovered_path = _derived_fits(source_temp / f"{c['id']}_recovered.fits", recovered, header)
                error_path = _derived_fits(source_temp / f"{c['id']}_error.fits", error, header)
                control_id = case(sid, "noise", 1, c["seed"] if c["seed"] is not None else manifest["seeds"][0])["id"]
                if control_id not in control_cache:
                    control_entry = entries[control_id]
                    control_root = pilot if control_entry["origin"] == "pilot" else output
                    control_rec = verify_case(control_root, control_entry["case"], plan, full_hashes=False)
                    control_raw = load_products(control_root, control_rec)
                    control_C = convolve_model(control_raw["model"], fits.getheader(control_root / control_rec["products"]["clean"]))
                    control_cache[control_id] = (control_raw, control_C)
                control_raw, control_C = control_cache[control_id]
                metrics = measure(raw["clean"], raw["residual"], raw["model"], truth,
                                  masks["rho"], omega, masks["annulus"], rec["flux_jy"],
                                  control_raw["clean"], control_raw["residual"],
                                  smoothed_model=recovered, regular_smoothed_model=control_C,
                                  compute_position=False)
                y = float(np.sqrt(np.square(error[omega]).sum() / np.square(truth[omega]).sum()))
                np.testing.assert_allclose(metrics["e_TM_2"], y, rtol=1e-12, atol=1e-12)
                product_paths = dict(truth=truth_path, recovered=recovered_path, error=error_path,
                                     dirty=root / rec["products"]["dirty"], clean=clean_path,
                                     residual=root / rec["products"]["residual"])
                source_frames[c["id"]] = product_paths
                paths_by_group["truth_recovered"].extend((truth_path, recovered_path))
                paths_by_group["dirty_clean"].extend((product_paths["dirty"], product_paths["clean"]))
                paths_by_group["error"].append(error_path)
                paths_by_group["residual"].append(product_paths["residual"])
                data[c["id"]] = dict(case=c, origin=entry["origin"], y=y,
                                     record=str((cell(root, c) / "record.json").resolve()),
                                     record_sha256=sha(cell(root, c) / "record.json"), frames={})
                if n % 30 == 0:
                    print(f"Validated six arrays and y for {sid}: {n}/{len(source_cases)}", flush=True)
            if peak_anchor:
                peak_mjy=float(np.max(truth[omega]))*1e3
                if not np.isfinite(peak_mjy) or peak_mjy<=0:raise ValueError('Invalid source peak for display')
                limits=dict(truth_recovered=(0.,1.15*peak_mjy),
                    dirty_clean=(-.15*peak_mjy,1.15*peak_mjy),
                    error=(-.2*peak_mjy,.2*peak_mjy),
                    residual=shared_fits_display_limits(paths_by_group['residual'],symmetric=True))
            else:
                limits = {
                    "truth_recovered": shared_fits_display_limits(paths_by_group["truth_recovered"], symmetric=False),
                    "dirty_clean": shared_fits_display_limits(paths_by_group["dirty_clean"], symmetric=False),
                    "error": shared_fits_display_limits(paths_by_group["error"], symmetric=True),
                    "residual": shared_fits_display_limits(paths_by_group["residual"], symmetric=True),
                }
            color_limits[sid] = {key: list(pair) for key, pair in limits.items()}
            truth_png = images / sid / "truth.png"
            truth_png.parent.mkdir(parents=True, exist_ok=True)
            truth_recipe = casa_image_to_png(truth_path, truth_png, title="True convolved sky T", draw_beam=True,
                                             symmetric=False, display_limits_mjy_per_beam=limits["truth_recovered"])
            for n, entry in enumerate(source_cases, 1):
                cid = entry["case"]["id"]
                recipes[cid] = {"truth": truth_recipe}
                data[cid]["frames"]["truth"] = str(truth_png.relative_to(output))
                for channel in CHANNELS[1:]:
                    group = ("truth_recovered" if channel == "recovered" else
                             "dirty_clean" if channel in ("dirty", "clean") else channel)
                    png = images / sid / f"{cid}_{channel}.png"
                    recipe = casa_image_to_png(source_frames[cid][channel], png,
                        title={"recovered": "Recovered CLEAN model C", "error": "Signed error C−T",
                               "dirty": "Dirty", "clean": "Restored CLEAN", "residual": "Residual"}[channel],
                        draw_beam=True, symmetric=channel in ("error", "residual"),
                        display_limits_mjy_per_beam=limits[group])
                    if not np.array_equal([recipe["vmin"], recipe["vmax"]], limits[group]):
                        raise ValueError(f"Rendered color limit drift for {cid}/{channel}")
                    recipes[cid][channel] = recipe
                    data[cid]["frames"][channel] = str(png.relative_to(output))
                if n % 20 == 0:
                    print(f"Rendered {sid}: {n}/{len(source_cases)} unique cases", flush=True)
    display = dict(source_limits_mjy_per_beam=color_limits, recipes=recipes,
                   pool_rule="source-peak anchored truth/recovered, dirty/clean and error; pooled residual; fixed across all cases" if peak_anchor else "all unique physical cases and all three seeds per template; duplicate summary badges share images",
                   channels=list(CHANNELS))
    atomic_write_json(output / "display.json", display)
    atomic_write_json(output / "render_data.json", data)
    # Provenance remains in the manifest; the page only carries viewer data.
    for entry in manifest["cases"]:
        rec = data[entry["case"]["id"]]
        entry.update(origin=rec["origin"], record=rec["record"], record_sha256=rec["record_sha256"])
    atomic_write_json(output / "case_manifest.json", manifest)
    return data, display


def write_html(output, manifest, data, display, *, overview_html='', conclusion_boundaries=None, by_kind=False):
    """Write a self-contained viewer; PNG color scales are already fixed."""
    sources = {s["id"]: {"snr0": s["nominal_source_snr"],
                            "snr_role": ('low', 'mid', 'high')[rank] if len(manifest['sources'])==3 else 'middle',
                            "antennas": s["selected_antennas"],
                            "roles": manifest["antenna_roles"][s["id"]],
                            "sliders": manifest["sliders"][s["id"]]} for rank,s in enumerate(manifest["sources"])}
    viewer_cases = {cid: {"y": item["y"], "frames": item["frames"]} for cid, item in data.items()}
    payload = json.dumps({"sources": sources, "cases": viewer_cases,
                          "seeds": manifest["seeds"], "channels": CHANNELS,
                          "boundaries": conclusion_boundaries or {}, "byKind": by_kind},
                         separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Injection range visual check</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151d;color:#e7edf5}
*{box-sizing:border-box}body{margin:0;padding:14px 20px 50px}h1{font-size:1.3rem;margin:0 0 16px}h2{font-size:1.12rem;margin:24px 0 8px}h3{font-size:1rem;margin:0}
.slider-card{border:1px solid #425267;background:#19232f;border-radius:8px;padding:12px;margin-bottom:14px}
.top{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.chooser{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.chooser label{font-size:.83rem;color:#b8c8d9;margin-right:3px}
button{background:#29394c;color:#e7edf5;border:1px solid #556982;border-radius:5px;padding:5px 9px;cursor:pointer}button:hover{background:#36516e}button.active{background:#b8d9f1;color:#13202e;border-color:#b8d9f1}
.range{display:flex;align-items:center;gap:9px;margin:9px 0 5px}.range input{width:min(100%,700px);flex:1;accent-color:#a7d6f1}.range .step{font-variant-numeric:tabular-nums;color:#b6c9d9;min-width:48px;text-align:right}
.caption{min-height:45px;font-variant-numeric:tabular-nums}.caption strong{color:#fff}.badge{display:inline-block;margin:2px 4px 2px 0;padding:2px 6px;border:1px solid #75879b;border-radius:12px;font-size:.72rem;color:#d9e7f3}.warning{color:#ffd28b}
.viewport{overflow-x:auto;overflow-y:hidden;width:100%;padding-bottom:6px}.images{display:flex;gap:8px;width:max-content;min-height:245px}.panel{margin:0;width:295px;flex:none}.panel img{display:block;width:295px;aspect-ratio:4/3;height:auto;cursor:zoom-in;border:1px solid #4d5d6c}.panel figcaption{font-size:.75rem;color:#bbcad9;margin:3px 0}.missing{height:221px;display:grid;place-items:center;background:#27323c;color:#ffd28b}
.plot{display:block;width:100%;height:200px;margin-top:8px;background:#141d28;border:1px solid #35485d;border-radius:4px}.plot text{fill:#c7d7e7;font:11px system-ui,sans-serif}.plot .grid{stroke:#334458;stroke-width:1}.plot .axis{stroke:#8da4ba;stroke-width:1}
#modal{display:none;position:fixed;inset:0;z-index:10;background:#071018ed;flex-direction:column}#modal.open{display:flex}.modalbar{display:flex;gap:7px;align-items:center;padding:8px 12px;background:#1d2d3d}.modalbar span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.modalbar button{min-width:38px}
#modalview{flex:1;overflow:auto;touch-action:pan-x pan-y;cursor:grab}#modalview.dragging{cursor:grabbing}#modalimage{display:block;max-width:none;height:auto;margin:12px auto;user-select:none;-webkit-user-drag:none}
@media(max-width:650px){body{padding:10px}.panel,.panel img{width:260px}.top{gap:8px}}
</style></head><body><h1>Injection range visual check</h1>__OVERVIEW__<main id="app"></main>
<div id="modal" role="dialog" aria-modal="true" aria-label="Full resolution image viewer"><div class="modalbar"><span id="modal-title"></span><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-reset">Reset</button><button id="modal-close" aria-label="Close">×</button></div><div id="modalview"><img id="modalimage" alt="Full resolution science image"></div></div>
<script>const DATA=__PAYLOAD__;
const LABELS={phase:'Phase φ (degrees)',amp:'Amplitude A',noise:'Thermal noise a',truth:'True convolved sky T',recovered:'Recovered CLEAN model C',error:'Signed error C−T',dirty:'Dirty',clean:'Restored CLEAN',residual:'Residual'};
const app=document.getElementById('app'),modal=document.getElementById('modal'),view=document.getElementById('modalview'),modalImage=document.getElementById('modalimage');
let modalState=null,zoom=1,drag=null,touchDistance=null;
const fmt=x=>Number.parseFloat(Number(x).toPrecision(5)).toString();
function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function valueText(kind,stop){if(kind==='phase')return `φ = ${fmt(stop.value)}°`;if(kind==='amp')return `A = ${fmt(stop.value)}`;return `a = ${fmt(stop.a)}, r = ${fmt(stop.r)}`}
function branch(kind,stop){if(kind==='phase')return stop.value<0?'negative':stop.value>0?'positive':'control';if(kind==='amp')return stop.value<1?'attenuation':stop.value>1?'amplification':'control';if(stop.sources?.includes('sweep'))return 'noise';return stop.sources?.includes('a-summary')?'a-summary':stop.sources?.includes('r-summary')?'r-summary':'control'}
function caseId(kind,stop,antenna,seed){return stop.cases[`${kind==='noise'?'noise':antenna}:${seed}`]}
function svgEl(tag,attrs){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e}
function plot(svg,kind,stops,antenna,seed,selected,sid){svg.replaceChildren();const W=1000,H=200,L=64,R=20,T=13,B=38,valid=stops.map((stop,i)=>({stop,i,id:caseId(kind,stop,antenna,seed)})).filter(p=>p.id&&DATA.cases[p.id]&&(kind==='phase'||p.stop.value>0));
if(!valid.length)return;const xs=valid.map(p=>kind==='phase'?p.stop.value:Math.log10(p.stop.value)),ys=valid.map(p=>DATA.cases[p.id].y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(0,...ys),ymax=Math.max(...ys)*1.05||1;
const x=v=>L+(W-L-R)*((kind==='phase'?v:Math.log10(v))-xmin)/(xmax-xmin||1),y=v=>H-B-(H-T-B)*(v-ymin)/(ymax-ymin||1);
for(let i=0;i<=4;i++){const yy=T+i*(H-T-B)/4;svg.append(svgEl('line',{x1:L,x2:W-R,y1:yy,y2:yy,class:'grid'}));const label=svgEl('text',{x:L-8,y:yy+4,'text-anchor':'end'});label.textContent=fmt(ymax-(ymax-ymin)*i/4);svg.append(label)}
for(let i=0;i<=4;i++){const xx=L+i*(W-L-R)/4;svg.append(svgEl('line',{x1:xx,x2:xx,y1:T,y2:H-B,class:'grid'}));const val=kind==='phase'?xmin+(xmax-xmin)*i/4:10**(xmin+(xmax-xmin)*i/4);const label=svgEl('text',{x:xx,y:H-B+17,'text-anchor':'middle'});label.textContent=fmt(val);svg.append(label)}
svg.append(svgEl('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'axis'}));svg.append(svgEl('line',{x1:L,x2:L,y1:T,y2:H-B,class:'axis'}));
const xlabel=svgEl('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle'});xlabel.textContent=kind==='phase'?'φ (degrees)':kind==='amp'?'A':'a';svg.append(xlabel);
const ylabel=svgEl('text',{x:15,y:H/2,transform:`rotate(-90 15 ${H/2})`,'text-anchor':'middle'});ylabel.textContent='e_TM,2';svg.append(ylabel);
for(const marker of DATA.boundaries?.[sid]?.[kind]||[]){const value=marker.value;if(value<=0&&kind!=='phase')continue;const xx=x(value);if(xx<L||xx>W-R)continue;const low=marker.role==='low',color=low?'#49a8d5':'#e68181';const line=svgEl('line',{x1:xx,x2:xx,y1:T,y2:H-B,stroke:color,'stroke-dasharray':low?'5 4':'3 3','stroke-width':1.5,opacity:.9,'data-bound-value':value,'data-bound-role':marker.role});const title=svgEl('title',{});title.textContent=marker.purpose;line.append(title);svg.append(line);const label=kind==='phase'?`φ=${fmt(value)}°`:kind==='amp'?`A=${fmt(value)}`:`a=${fmt(value)}, r=${fmt(marker.r)}`;const side=marker.branch==='phase_negative'||marker.branch==='amp_attenuation'?(low?'end':'start'):(low?'start':'end');const text=svgEl('text',{x:xx+(side==='start'?5:-5),y:T+(low?17:32),'text-anchor':side,style:`fill:${color};font:10px system-ui,sans-serif`});text.textContent=`${label} · ${low?'0.05':marker.reached?'0.9':'max tested'}`;svg.append(text)}
const colors={negative:'#77c4ff',positive:'#ffaa88',attenuation:'#77c4ff',amplification:'#ffaa88',noise:'#8ed49f','a-summary':'#8ed49f','r-summary':'#ddb5f7',control:'#ffffff'};
if(DATA.byKind){const points=[...valid].sort((a,z)=>a.stop.value-z.stop.value);if(points.length>1)svg.append(svgEl('polyline',{points:points.map(p=>`${x(p.stop.value)},${y(DATA.cases[p.id].y)}`).join(' '),fill:'none',stroke:'#77c4ff','stroke-width':2}))}
else for(const b of [...new Set(valid.map(p=>branch(kind,p.stop)))].filter(b=>b!=='control')){const points=valid.filter(p=>branch(kind,p.stop)===b).sort((a,z)=>a.stop.value-z.stop.value);if(points.length>1)svg.append(svgEl('polyline',{points:points.map(p=>`${x(p.stop.value)},${y(DATA.cases[p.id].y)}`).join(' '),fill:'none',stroke:colors[b], 'stroke-width':2}))}
for(const p of valid){const selectedPoint=p.i===selected,c=svgEl('circle',{cx:x(p.stop.value),cy:y(DATA.cases[p.id].y),r:selectedPoint?6:3.7,fill:colors[branch(kind,p.stop)],stroke:selectedPoint?'#fff':'#15202d','stroke-width':selectedPoint?2:1});const tip=svgEl('title',{});tip.textContent=`${valueText(kind,p.stop)} · e_TM,2 = ${fmt(DATA.cases[p.id].y)} · ${p.stop.badges.join('; ')}`;c.append(tip);svg.append(c)}}
function makeCard(sid,source,kind){const allStops=source.sliders[kind],card=node('section','slider-card'),top=node('div','top'),title=node('h3','',LABELS[kind]);top.append(title);const state={sid,source,kind,stops:allStops,index:allStops.findIndex(s=>s.value===(kind==='phase'?0:1)),antenna:source.antennas[0],seed:DATA.seeds[0]};let renderVersion=0;
function buttons(name,items,value,labelFn,select){const box=node('div','chooser');box.append(node('label','',name));for(const item of items){const button=node('button','',labelFn(item));button.type='button';button.setAttribute('aria-pressed',item===value?'true':'false');button.onclick=()=>{select(item);for(const b of box.querySelectorAll('button')){b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',b===button?'true':'false')}update()};if(item===value)button.classList.add('active');box.append(button)}top.append(box)}
if(kind!=='noise')buttons('Antenna',source.antennas,state.antenna,a=>`${a} · ${source.roles[a]}`,a=>state.antenna=a);
buttons('Seed',DATA.seeds,state.seed,s=>String(s),s=>state.seed=s);card.append(top);
const range=node('div','range'),slider=node('input'),step=node('span','step');slider.type='range';slider.min='0';slider.max=String(allStops.length-1);slider.step='1';slider.value=String(state.index);slider.setAttribute('aria-label',`${sid} ${LABELS[kind]} stop`);slider.oninput=()=>{state.index=Number(slider.value);update()};range.append(slider,step);card.append(range);
const caption=node('div','caption'),viewport=node('div','viewport'),images=node('div','images');viewport.append(images);card.append(caption,viewport);const svg=svgEl('svg',{class:'plot',viewBox:'0 0 1000 200',role:'img','aria-label':`e_TM,2 versus ${LABELS[kind]}`});card.append(svg);
function update(){const previous=state.stops[state.index]?.value;state.stops=DATA.byKind?allStops.filter(stop=>{const id=caseId(kind,stop,state.antenna,state.seed);return id&&DATA.cases[id]}):allStops;
if(!state.stops.length)throw new Error(`No measured ${kind} cases for ${sid}, antenna ${state.antenna}, seed ${state.seed}`);
const exact=state.stops.findIndex(stop=>stop.value===previous);state.index=exact>=0?exact:previous===undefined?0:state.stops.reduce((best,stop,i)=>Math.abs(stop.value-previous)<Math.abs(state.stops[best].value-previous)?i:best,0);slider.max=String(state.stops.length-1);slider.value=String(state.index);
const stop=state.stops[state.index],id=caseId(kind,stop,state.antenna,state.seed),result=DATA.cases[id],version=++renderVersion;step.textContent=`${state.index+1}/${state.stops.length}`;caption.replaceChildren();caption.append(node('strong','',valueText(kind,stop)+' · '));caption.append(node('span','',result?`e_TM,2 = ${fmt(result.y)}`:'No measured case for this selection'));for(const badge of stop.badges)caption.append(node('span','badge',badge));if(!stop.admissible)caption.classList.add('warning');else caption.classList.remove('warning');const pending=node('span','loading',' · loading images…');if(result)caption.append(pending);
const paths=result?DATA.channels.map(channel=>result.frames[channel]):DATA.channels.map(()=>null);Promise.all(paths.map(path=>new Promise(resolve=>{if(!path){resolve(null);return}const image=new Image();image.onload=()=>{if(image.decode)image.decode().then(()=>resolve(image),()=>resolve(image));else resolve(image)};image.onerror=()=>resolve(null);image.src=path}))).then(loaded=>{if(version!==renderVersion)return;const fragment=document.createDocumentFragment();for(let i=0;i<DATA.channels.length;i++){const channel=DATA.channels[i],fig=node('figure','panel'),image=loaded[i];if(image){image.alt=`${sid} ${LABELS[channel]} at ${valueText(kind,stop)}`;image.onclick=()=>openModal(state,channel);fig.append(image)}else fig.append(node('div','missing',!stop.admissible&&kind==='noise'?'Outside a ≥ 1 domain':'Image unavailable'));fig.append(node('figcaption','',LABELS[channel]));fragment.append(fig)}images.replaceChildren(fragment);pending.remove()});
plot(svg,kind,state.stops,state.antenna,state.seed,state.index,state.sid);if(modalState?.state===state)syncModal()}
update();return card}
if(DATA.byKind){for(const kind of ['amp','phase','noise']){const section=node('section'),h=node('h2','',LABELS[kind]);section.append(h);const img=node('img');img.src=`plots/overview_${kind}_interpolated.svg`;img.style.width='100%';section.append(img);for(const [sid,source] of Object.entries(DATA.sources)){const sh=node('h3','',`${source.snr_role} SNR · ${sid} · SNR₀ ${fmt(source.snr0)}`);section.append(sh,makeCard(sid,source,kind))}app.append(section)}}
else for(const [sid,source] of Object.entries(DATA.sources)){const section=node('section'),h=node('h2','',`${sid} · SNR₀ ${fmt(source.snr0)}`);section.append(h);for(const kind of ['phase','amp','noise'])section.append(makeCard(sid,source,kind));app.append(section)}
function modalPath(){const {state,channel}=modalState,stop=state.stops[state.index],id=caseId(state.kind,stop,state.antenna,state.seed);return DATA.cases[id]?.frames[channel]}
function syncModal(){const path=modalPath();if(!path)return;const px=view.scrollLeft/(view.scrollWidth-view.clientWidth||1),py=view.scrollTop/(view.scrollHeight-view.clientHeight||1);modalImage.src=path;document.getElementById('modal-title').textContent=`${modalState.state.sid} · ${LABELS[modalState.channel]} · ${valueText(modalState.state.kind,modalState.state.stops[modalState.state.index])}`;modalImage.onload=()=>{modalImage.style.width=`${modalImage.naturalWidth*zoom}px`;requestAnimationFrame(()=>{view.scrollLeft=px*(view.scrollWidth-view.clientWidth);view.scrollTop=py*(view.scrollHeight-view.clientHeight)})}}
function openModal(state,channel){modalState={state,channel};zoom=1;modal.classList.add('open');syncModal()}
function setZoom(next,clientX=view.clientWidth/2,clientY=view.clientHeight/2){const old=zoom;zoom=Math.max(.25,Math.min(8,next));if(!modalImage.naturalWidth)return;const px=(view.scrollLeft+clientX)/old,py=(view.scrollTop+clientY)/old;modalImage.style.width=`${modalImage.naturalWidth*zoom}px`;requestAnimationFrame(()=>{view.scrollLeft=px*zoom-clientX;view.scrollTop=py*zoom-clientY})}
document.getElementById('zoom-in').onclick=()=>setZoom(zoom*1.3);document.getElementById('zoom-out').onclick=()=>setZoom(zoom/1.3);document.getElementById('zoom-reset').onclick=()=>setZoom(1);document.getElementById('modal-close').onclick=()=>{modal.classList.remove('open');modalState=null};document.addEventListener('keydown',e=>{if(e.key==='Escape'&&modalState)document.getElementById('modal-close').click()});
view.addEventListener('wheel',e=>{if(!modalState)return;e.preventDefault();const box=view.getBoundingClientRect();setZoom(zoom*(e.deltaY<0?1.13:1/1.13),e.clientX-box.left,e.clientY-box.top)},{passive:false});
view.addEventListener('pointerdown',e=>{if(e.pointerType==='touch')return;drag={x:e.clientX,y:e.clientY,left:view.scrollLeft,top:view.scrollTop};view.setPointerCapture(e.pointerId);view.classList.add('dragging')});view.addEventListener('pointermove',e=>{if(drag){view.scrollLeft=drag.left+drag.x-e.clientX;view.scrollTop=drag.top+drag.y-e.clientY}});view.addEventListener('pointerup',()=>{drag=null;view.classList.remove('dragging')});
view.addEventListener('touchmove',e=>{if(e.touches.length!==2)return;e.preventDefault();const a=e.touches[0],b=e.touches[1],d=Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY);if(touchDistance)setZoom(zoom*d/touchDistance,(a.clientX+b.clientX)/2-view.getBoundingClientRect().left,(a.clientY+b.clientY)/2-view.getBoundingClientRect().top);touchDistance=d},{passive:false});view.addEventListener('touchend',()=>touchDistance=null);
</script></body></html>'''
    page=page.replace("__PAYLOAD__",payload).replace("__OVERVIEW__",overview_html)
    if by_kind:page=page.replace('Injection range visual check','Simulation parametrization v2')
    (output / "report.html").write_text(page, encoding="utf-8")


def verify_display(manifest, data, display):
    """Audit the saved, fixed scale attached to every image recipe."""
    groups = {"truth": "truth_recovered", "recovered": "truth_recovered",
              "dirty": "dirty_clean", "clean": "dirty_clean",
              "error": "error", "residual": "residual"}
    if set(data) != {entry["case"]["id"] for entry in manifest["cases"]}:
        raise ValueError("Report data has the wrong case set")
    for cid, item in data.items():
        sid = item["case"]["source"]
        recipes = display["recipes"][cid]
        for channel in CHANNELS:
            expected = display["source_limits_mjy_per_beam"][sid][groups[channel]]
            recipe = recipes[channel]
            if [recipe["vmin"], recipe["vmax"]] != expected:
                raise ValueError(f"Display scale drift: {cid}/{channel}")


def render(output, pilot):
    output, pilot = output.resolve(), pilot.resolve()
    manifest, preflight = prepare(output, pilot)
    if preflight["missing"]:
        raise ValueError(f"{preflight['missing']} explicit cases are not complete")
    cached = output / "render_data.json"
    display_path = output / "display.json"
    if cached.exists() and display_path.exists():
        data = json.loads(cached.read_text())
        display = json.loads(display_path.read_text())
        if set(data) != {entry["case"]["id"] for entry in manifest["cases"]}:
            raise ValueError("Report cache has the wrong case set")
        for entry in manifest["cases"]:
            cid = entry["case"]["id"]
            if sha(data[cid]["record"]) != data[cid]["record_sha256"]:
                raise ValueError(f"Case record changed since rendering: {cid}")
            for channel in CHANNELS:
                if not (output / data[cid]["frames"][channel]).is_file():
                    raise ValueError(f"Missing rendered frame: {cid}/{channel}")
    else:
        data, display = _render_assets(output, pilot, manifest)
    verify_display(manifest, data, display)
    write_html(output, manifest, data, display)
    print("Report:", output / "report.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--simulate-one", action="store_true")
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        render(args.output, args.pilot)
        return
    manifest, preflight = prepare(args.output, args.pilot)
    print("Preflight:", preflight, flush=True)
    if args.prepare_only:
        return
    if args.simulate_one:
        first = next((entry for entry in manifest["cases"] if entry["origin"] == "missing"), None)
        if first is not None:
            simulate(args.output, args.pilot, dict(manifest, cases=[first]))
        return
    simulate(args.output, args.pilot, manifest)
    if not args.simulate_only:
        render(args.output, args.pilot)


if __name__ == "__main__":
    main()

