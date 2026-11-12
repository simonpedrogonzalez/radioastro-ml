"""Compare several VLA pipeline QA runs (one experiment/variant each) in one report.

Each run of scripts/vla_pipeline_qa.py holds one experiment (A or B) with one set of self-calibration
options (image size, masking), and both flagging recipes. This reads the per-case results of several such
runs and writes (design doc, "Requested report redesign and My Criteria"):

  1. one aggregate row per run configuration and flagging mode: coverage, a My Criteria category-count bar
     chart on a shared scale, and measured recipe runtime;
  2. the full per-case table (every planned case, pending ones included) with Corruption (per-kind removal damage,
     total and solvable eC), Pipeline action, My Criteria, eC before/after, CLEAN images before and after the
     recipe on one shared color scale per case, matched reference, damage reduction/remaining excess, runtime and
     Description.

Retained per-case evidence is written to descriptions/, the image thumbnails to images/ (rendered from the FITS
the eC evaluation used; existing PNGs are reused). Run folders are only read.

  bin/run ml scripts/vla_pipeline_qa_compare.py --output runs/vla-pipeline-qa-compare RUN [RUN ...]
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
from pathlib import Path

import numpy as np

from scripts.vla_pipeline_qa import (ACCOUNTING, CATEGORIES, CRITERIA_VERSION, EPSILON, EPSILON_RMS, EPSILON_SNR,
                                     EXPECTATIONS, IMAGE, IMAGE_METRIC_VERSION, PIPELINE, RECOVERY, TOLERANCE,
                                     V4_VERSION, describe, expectation, gain_mask, image_pair, matched_reference,
                                     measured_recovery, my_criteria, pipeline_outcome, safe, write_json)

CORRUPTIONS = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise",
               7: "amp+phase+noise"}
COLORS = dict(preserved="#009E73", solved="#0072B2", improved="#56B4E9", degraded="#D55E00",
              blocked="#E69F00", rejected="#CC79A7", unchanged="#777777")
NAMES = {}
name = lambda k: NAMES.get(k, k)
# v4 dimensions: categories, colors (design doc legends) and accounting states shown as "other"
BLUE, AMBER, PURPLE, VERMILION, GRAY = "#0072B2", "#E69F00", "#CC79A7", "#D55E00", "#777777"
DIMENSIONS = {"pipeline": (PIPELINE, dict(applied=BLUE, blocked=AMBER, rejected=PURPLE, error=VERMILION)),
              "recovery": (RECOVERY, dict(improved=BLUE, similar=GRAY, degraded=VERMILION)),
              "image": (IMAGE, dict(cleaner=BLUE, similar=GRAY, worse=VERMILION, mixed=PURPLE))}
EXPECTATION_NAMES = dict(no_improvement_expected="No improvement expected", improvement_expected="Improvement expected",
                         expectation_unavailable="Expectation unavailable")


def configuration(p, recipe):
    """Label and sort key, e.g. 'B · 512 px · center · flags off'."""
    flags = "flags off" if recipe.endswith("flagging_off") else "flags on"
    gain = recipe.split(":")[1]
    flags = flags if gain == "phase_amp" else f"{gain} · {flags}"          # older runs had several gain recipes
    if p["experiment"] == "A":
        return f"A · true model · {flags}", ("A", 0, "", recipe)
    o = p.get("options") or {}
    size = o.get("selfcal_imsize")
    parts = ["B", f"{size} px" if size else "pipeline size", o.get("selfcal_masking", "auto")]
    if o.get("calonly_model_fix") is False:
        parts.append("calonly fix OFF")
    return " · ".join(parts + [flags]), ("B", size or 10 ** 9, " ".join(parts[2:]), recipe)


def clean_image(root, case_dir):
    path = root / "cases" / case_dir / "imaging" / "clean.fits.gz"
    return str(path) if path.exists() else None


def render_pair(before, after, out_before, out_after, label):
    """Before/after CLEAN images through the canonical imaging plotter (inferno, WCS axes, colorbar, beam), on one
    shared symmetric scale: +-99.5th percentile of |pixel| over both images. Returns (vmin, vmax) in mJy/beam."""
    from scripts.imaging.plot_utils import casa_image_to_png, shared_fits_display_limits
    limits = shared_fits_display_limits([Path(before), Path(after)])
    for path, out, when in ((before, out_before, "before"), (after, out_after, "after")):
        casa_image_to_png(Path(path), Path(out), title=f"{label} clean {when}", draw_beam=True,
                          display_limits_mjy_per_beam=limits, dpi=90)
    return list(limits)


def superseded(result):
    """B flagging_off results without the calonly model-write fix carry the known flux-scale error."""
    if not result["recipe"].endswith("flagging_off") or not result["recipe"].startswith("B:"):
        return False
    fix = result["recipe_record"].get("calonly_model_fix")
    return not (isinstance(fix, dict) and fix.get("applied"))


def load(run):
    run = run.resolve()
    p = json.loads((run / "plan.json").read_text())
    source_run = Path(p["source_run"])
    sp = json.loads((source_run / "plan.json").read_text())
    plan_of = {c["id"]: s for s in sp["source_plans"] for c in s["cases"]}
    masks_of = {x["id"]: str(source_run / x["reference_masks"]) for x in sp["prepared_sources"]}
    rows_path = source_run / "rows.json"                   # per-kind removal damage d = eC(case) - eC(kind removed)
    damage_of = ({r["case"]: r for r in json.loads(rows_path.read_text())["rows"]} if rows_path.exists() else {})
    archived = {}

    def eC_of(case_id):
        if case_id not in archived:
            path = source_run / "cases" / case_id / "case.json"
            archived[case_id] = json.loads(path.read_text()).get("eC") if path.exists() else None
        return archived[case_id]

    unit_files = list((run / "units").glob("*.json"))
    units = {x.stem: json.loads(x.read_text()) for x in unit_files}
    unit_times = sorted(x.stat().st_mtime for x in unit_files)     # case completion times (elapsed of killed calls)
    log = run / "execute.json"
    calls = json.loads(log.read_text()) if log.exists() else []
    rows = []
    for recipe in p["recipes"]:
        label, key = configuration(p, recipe)
        cases = []
        for u in p["units"]:
            cid = u["case"]
            splan = plan_of[cid]
            params = next(c for c in splan["cases"] if c["id"] == cid)
            dmg = damage_of.get(cid, {})
            ref = matched_reference(splan, cid, eC_of)
            item = dict(case=cid, label=next(r["id"] for r in splan["rows"] if r["case"] == cid),
                        corruption=CORRUPTIONS[gain_mask(params)], mask=gain_mask(params), action="–",
                        d=dmg.get("d"), eC_total=dmg.get("eC", eC_of(cid)),
                        clean_before=clean_image(source_run, cid), masks=masks_of.get(splan["source"]),
                        id=f"{run.name}|{cid}|{recipe}")
            # Expectation: original case and matched reference only (never the recipe outcome)
            item.update(expectation(gain_mask(params), eC_of(cid), ref.get("eC")))
            if ref.get("status") == "unavailable":     # e.g. invalid matched counterpart: keep its reason
                item.update(expectation="expectation_unavailable", D=None, expectation_reason=ref.get("reason"))
            path = run / "results" / f"{cid}__{safe(recipe)}.json"
            if not path.exists():
                missing = "error" if cid in units else "pending"
                item.update(category=missing, pipeline=missing, recovery="unavailable" if cid in units else "pending",
                            image="unavailable" if cid in units else "pending",
                            description="No result record although the case finished." if cid in units else
                            "Not run yet (no result record).")
            else:
                result = json.loads(path.read_text())
                crit = my_criteria(result, gain_mask(params) & 3 != 0, ref)
                reused = result.get("evaluation", {}).get("evaluation_reused")
                ev = result.get("evaluation", {})
                item.update(pipeline=pipeline_outcome(crit["action"]),
                            # independent of the legacy classifier (and of the reference): measured eC only
                            recovery=measured_recovery(ev),
                            eC_before_measured=(ev.get("before") or {}).get("eC") if ev.get("status") == "measured" else None,
                            eC_after_measured=(ev.get("after") or {}).get("eC") if ev.get("status") == "measured" else None,
                            flux_before=(ev.get("before") or {}).get("flux_error"),
                            flux_after=(ev.get("after") or {}).get("flux_error"))
                item.update({k: v for k, v in crit.items() if k != "reason"}, criteria_reason=crit.get("reason"),
                            reference_id=ref.get("id"), seconds=result["recipe_record"].get("seconds"),
                            evaluation_reused=bool(reused), clean_after=(
                                item["clean_before"] if reused else clean_image(run, f"{cid}/{safe(recipe)}")),
                            superseded=superseded(result), description=describe(result, crit),
                            notes=result["recipe_record"].get("pipeline_notes", []),
                            evidence=dict(recipe_record=result["recipe_record"], evaluation={
                                k: v for k, v in result["evaluation"].items() if k not in ("before", "after")},
                                reference=ref))
            cases.append(item)
        rows.append(dict(run=run.name, config=label, key=key, recipe=recipe, cases=cases, calls=calls,
                         unit_times=unit_times,
                         superseded=any(c.get("superseded") for c in cases)))
    return rows


V4_FIELDS = ("expectation", "expectation_reason", "D", "near_zero", "pipeline", "recovery", "eC_before_measured",
             "eC_after_measured", "image", "image_reason", "epsilon_rms", "epsilon_snr",
             "rms_before", "rms_after", "mean_before", "mean_after", "peak_before", "peak_after", "snr_before",
             "snr_after", "r_rms", "r_snr", "off_pixels", "source_pixels", "off_beams", "units", "flux_before",
             "flux_after", "image_metric_version")


def add_image_metrics(rows, cache, eps_rms=EPSILON_RMS, eps_snr=EPSILON_SNR):
    """v4 Image outcome per case from the retained restored CLEAN FITS (before: archived case, after: evaluation
    image) and the source's fixed reference masks. Results are cached by product identity and metric version.
    Before is reused as after only when the QA evaluation established an unchanged input (evaluation_reused)."""
    import hashlib
    cache.mkdir(parents=True, exist_ok=True)
    masks_cache, done, reused = {}, 0, 0
    for row in rows:
        for c in row["cases"]:
            if c.get("image") == "pending" or c["category"] in ("pending", "error"):
                continue
            before, after, masks = c.get("clean_before"), c.get("clean_after"), c.get("masks")
            if not (before and after and masks and Path(masks).exists()):
                c.update(image="unavailable", image_reason="missing restored CLEAN FITS or reference masks "
                         f"(before={bool(before)}, after={bool(after)}, masks={bool(masks)})")
                continue
            ident = [IMAGE_METRIC_VERSION, "strict-meta-v2", eps_rms, eps_snr] + [
                (x, Path(x).stat().st_size, Path(x).stat().st_mtime) for x in (before, after, masks)]
            path = cache / (hashlib.sha256(json.dumps(ident).encode()).hexdigest()[:24] + ".json")
            if path.exists():
                c.update(json.loads(path.read_text()))
                reused += 1
                continue
            if masks not in masks_cache:
                with np.load(masks, allow_pickle=False) as z:
                    masks_cache[masks] = dict(annulus=np.asarray(z["annulus"], bool), source=np.asarray(z["source"], bool))
            out = image_pair(before, after, masks_cache[masks], eps_rms, eps_snr)
            path.write_text(json.dumps(out))
            c.update(out)
            done += 1
    print(f"IMAGE METRICS computed {done}, cached {reused}")


def summarize_v4(rows):
    """One summary per (configuration, expectation group): coverage, per-dimension counts and runtime."""
    out = []
    for row in rows:
        for group in EXPECTATIONS:
            cases = [c for c in row["cases"] if c["expectation"] == group]
            if not cases:
                continue
            counts = {}
            for dim, (cats, _) in DIMENSIONS.items():
                counts[dim] = {k: sum(c.get(dim) == k for c in cases) for k in cats}
                counts[dim]["other"] = len(cases) - sum(counts[dim][k] for k in cats)
            secs = [c["seconds"] for c in cases if c.get("seconds") is not None]
            out.append(dict(run=row["run"], config=row["config"], expectation=group, planned=len(cases),
                            finished=sum(c["category"] != "pending" for c in cases),
                            image_available=sum(c.get("image") in IMAGE for c in cases), counts=counts,
                            runtime=dict(timed=len(secs), median_min=float(np.median(secs)) / 60,
                                         min_min=min(secs) / 60, max_min=max(secs) / 60,
                                         recipe_hours=sum(secs) / 3600) if secs else dict(timed=0)))
    return out


def elapsed_hours(calls, unit_times):
    """Summed wall time of every --execute call. A call killed before it recorded its end (preemption, time limit)
    ends at the last case it completed (unit file time before the next call started)."""
    total = 0.
    starts = [c["started"] for c in calls if "started" in c]
    for i, c in enumerate(calls):
        if "started" not in c:
            continue
        end = c.get("finished")
        if end is None:
            limit = starts[i + 1] if i + 1 < len(starts) else float("inf")
            inside = [t for t in unit_times if c["started"] <= t < limit]
            end = max(inside) if inside else c["started"]
        total += end - c["started"]
    return total / 3600 if calls else None


def summarize(row):
    cases = row["cases"]
    counts = {k: sum(c["category"] == k for c in cases) for k in CATEGORIES + ACCOUNTING}
    secs = [c["seconds"] for c in cases if c.get("seconds") is not None]
    runtime = dict(timed=len(secs), median_min=float(np.median(secs)) / 60, min_min=min(secs) / 60,
                   max_min=max(secs) / 60, recipe_hours=sum(secs) / 3600) if secs else dict(timed=0)
    elapsed = elapsed_hours(row["calls"], row.get("unit_times", []))
    actions = {a: sum(c.get("action") == a for c in cases) for a in
               ("Applied", "Left unchanged", "Could not start", "Error", "Unknown")}
    description = "; ".join(f"{n} {label}" for a, label in (
        ("Could not start", "blocked before correction trials"), ("Applied", "applied corrections"),
        ("Left unchanged", "retained no correction after trials"), ("Error", "execution errors"),
        ("Unknown", "unknown decisions")) if (n := actions[a]))
    initial_known = [c for c in cases if c.get("before") is not None and c.get("boundary") is not None]
    return dict(run=row["run"], config=row["config"], recipe=row["recipe"], superseded=row["superseded"],
                planned=len(cases), finished=len(cases) - counts["pending"],
                classified=sum(counts[k] for k in CATEGORIES), counts=counts, runtime=runtime,
                description=description, initially_above=sum(c["before"] > c["boundary"] for c in initial_known),
                initial_known=len(initial_known),
                elapsed_hours=elapsed,
                parallel=sorted({c["parallel"] for c in row["calls"]}),
                cpus=sorted({str(c.get("slurm_cpus") or c.get("cpu_count")) for c in row["calls"]}))


def bar(label, n, scale, color, flt):
    """One clickable count bar; flt is the exact filter state it selects (replaces previous outcome filters)."""
    esc = htmllib.escape
    return (f"<div class=bar tabindex=0 role=button title='{esc(label)}: {n}' data-filter='{esc(json.dumps(flt))}'>"
            f"<span class=lab>{esc(label)}</span><span class=fill style='width:{100 * n / scale:.0f}px;"
            f"background:{color}'></span><span>{n}</span></div>")


def bars(counts, scale, run, config):
    return "".join(bar(name(k), counts[k], scale, COLORS[k], dict(run=run, configuration=config, legacy=k))
                   for k in CATEGORIES)


def v4_bars(dim, counts, scale, run, config, group):
    """Each bar selects run AND configuration AND expectation group AND this outcome (run: identical labels of
    different runs stay separate)."""
    cats, colors = DIMENSIONS[dim]
    base = dict(run=run, configuration=config, expectation=group)
    out = [bar(k, counts[k], scale, colors[k], {**base, dim: k}) for k in cats]
    if counts["other"]:
        out.append(bar("pending/unavailable", counts["other"], scale, "#bbb", {**base, dim: "other"}))
    return "".join(out)


def v4_badge(dim, k):
    color = DIMENSIONS[dim][1].get(k)
    style = f"background:{color};color:#fff" if color else "border:1px dashed #999"
    return f"<span class=badge style='{style}'>{k}</span>"


def badge(k):
    style = f"background:{COLORS[k]};color:#fff" if k in COLORS else "border:1px dashed #999"
    return f"<span class=badge style='{style}'>{name(k)}</span>"


def corruption_cell(c):
    """Kind, per-kind removal damage of the injected kinds, total eC and eC_solvable: the gain (amp+phase) damage,
    eC before − matched reference eC, which a perfect gain solution would remove."""
    out = [c["corruption"]]
    d, mask = c.get("d"), c.get("mask", 0)
    if d is not None and mask:
        out.append(" / ".join(f"d_{k} {d[i]:.3f}" for i, k in enumerate(("amp", "phase", "noise")) if mask >> i & 1))
    total = c.get("eC_total", c.get("before"))
    line = f"eC={total:.3f}" if total is not None else ""
    if mask & 3 and c.get("damage") is not None:
        line += f" (eC_solvable={c['damage']:.3f})"
    if line:
        out.append(line)
    return "<br>".join(out)


def image_cell(c, key):
    path = c.get(f"{key}_png")
    if not path:
        return "<td><small>–</small></td>"
    note = f"<br><small>±{c['png_scale'][1]:.3g} mJy/beam</small>" if key == "after" else ""
    if key == "after" and c.get("evaluation_reused"):
        note += "<br><small>input unchanged</small>"
    return f"<td><a href='{path}'><img src='{path}' loading=lazy width=160></a>{note}</td>"


def write_report(output, summaries, rows, runs, filename, v4, eps_rms=EPSILON_RMS, eps_snr=EPSILON_SNR):
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    esc = htmllib.escape
    scale = max([1] + [s["counts"][k] for s in summaries for k in CATEGORIES])
    out = ["<!doctype html><meta charset=utf-8><title>VLA pipeline QA comparison</title><style>body{font-family:"
           "sans-serif;max-width:1600px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;margin:0}.scroll{overflow-x:auto;margin:.6em 0}"
           "td,th{border:1px solid #ccc;padding:3px 7px;text-align:left;vertical-align:top}th{background:#f3f3f3}"
           "td.n{text-align:right}tr.old td{color:#999}.bar{display:flex;align-items:center;gap:4px;font-size:12px;"
           "cursor:pointer;line-height:15px}.bar .lab{width:135px}.fill{display:inline-block;height:11px}"
           ".badge{padding:1px 6px;border-radius:3px;font-size:12px;white-space:nowrap}.partial{color:#D55E00;"
           "font-weight:bold}details{font-size:12px}small{color:#555}.grp{cursor:pointer;text-decoration:underline dotted}"
           ".filters label{margin-right:.6em;white-space:nowrap}</style>",
           f"<h1>VLA pipeline QA comparison</h1><p>{len(runs)} run(s): {', '.join(esc(r.name) for r in runs)}. "
           f"Outcomes: {V4_VERSION}; legacy My Criteria: {CRITERIA_VERSION}.</p>",
           "<h3>v4 outcomes</h3><ul>"
           "<li><b>Expectation group</b> (fixed before processing, same in every configuration), from the joint gain damage "
           "D = eC before − matched reference eC: <i>No improvement expected</i> = control, noise only, or gain-containing "
           "with D ≤ 0; <i>Improvement expected</i> = gain-containing with D &gt; 0 (any size). Not proof of recoverability.</li>"
           "<li><b>Pipeline</b>: applied (a gain correction was retained), blocked (a prerequisite stopped the trials), "
           "rejected (trials ran, nothing retained), error.</li>"
           f"<li><b>Recovery</b> from Δ = eC before − eC after: improved (Δ &gt; {EPSILON}), similar, degraded (Δ &lt; −{EPSILON}). "
           "ε PROVISIONAL.</li>"
           "<li><b>Image</b> from the restored CLEAN images on fixed regions of the reference geometry: RMS = standard "
           "deviation over the off-source annulus (3 beams to 1 beam from the edge), SNR = (peak in the true-source "
           "ellipse − annulus mean) / RMS. r_RMS = (RMS before − after) / before, r_SNR = (SNR after − before) / before, "
           f"both positive = better, tolerances {eps_rms} / {eps_snr} (PROVISIONAL, configurable). Cleaner: one better, none "
           "worse; Worse: one worse, none better; Mixed: one of each; Similar: neither. Cleaner is not proof of correct "
           "flux or morphology.</li>"
           "<li>The three outcomes are independent: none overrides another.</li></ul>"
           "<details><summary><b>Legacy v3 My Criteria</b> (kept for continuity)</summary><ul>"
           "<li><b>eC</b>: CLEAN recovery error at the common restoring beam (0 = perfect).</li>"
           "<li><b>Reference eC</b>: what eC should be once the gain errors are solved; archived eC of the same source case with amplitude and phase injections "
           "removed, keeping the noise seed and multiplier (nominal-noise control for gain-only cases, matched "
           "noise-only case otherwise).</li>"
           "<li><b>Quantities</b>: damage (eC_solvable) = before − reference; remaining excess = after − reference; "
           "change = before − after.</li>"
           f"<li><b>Boundary</b> = reference + {TOLERANCE} (acceptable remaining gain damage).</li>"
           f"<li><b>ε = {EPSILON}</b>: meaningful-change tolerance (PROVISIONAL, not yet justified).</li></ul>"
           "<p>Categories, checked in this order (after errors and missing measurements):</p><ol>"
           "<li><b>blocked</b>: a prerequisite prevented correction trials, regardless of initial quality.</li>"
           "<li><b>degraded</b>: change &lt; −ε (even inside the boundary).</li>"
           "<li><b>preserved</b>: before ≤ boundary and |change| ≤ ε (after a valid processing decision, never an "
           "empty-model block).</li>"
           "<li><b>solved</b>: amplitude/phase injected, before &gt; boundary, after ≤ boundary, change &gt; ε, and a "
           "correction was applied.</li>"
           "<li><b>improved</b>: change &gt; ε otherwise.</li>"
           "<li>before &gt; boundary and |change| ≤ ε: <b>rejected</b> if trials ran but no correction was retained, "
           "<b>unchanged</b> if a correction was applied (unknown action: unavailable).</li></ol>"
           "<p>Notes:</p><ul>"
           "<li>Applying calibration earns no success label by itself.</li>"
           "<li>The reference is a benchmark, not a proven attainable bound: gains act before noise injection, so "
           "inverting amplitude errors can change the output noise.</li>"
           "<li>Meeting it does not prove the true antenna gains were recovered.</li>"
           "<li>Pending, error and unavailable are accounting states, not scientific outcomes.</li></ul></details>"]
    if any(s["superseded"] for s in summaries):
        out.append("<p><b>Grey rows</b>: B flagging_off without the calonly model-write fix (superseded, reference "
                   "only).</p>")
    out.append("<h2>Configurations (legacy v3 My Criteria)</h2><p>Legend: " + " ".join(badge(k) for k in CATEGORIES) + ". Bars share one "
               "count scale; click a bar to list its cases. Runtime: measured wall time of the recipe's own CASA "
               "process (median, range) and summed recipe-process hours, which is not elapsed run time. Elapsed: "
               "summed duration of all the run's recorded --execute calls (whole cases: regeneration, both flagging "
               "recipes, evaluation), shared by both flagging rows; a call killed before recording its end counts until "
               "its last completed case. Calls of runs executed before 2026-11-12 were only recorded when they finished, "
               "so their elapsed time can miss killed calls.</p><div class=scroll><table><tr><th>Configuration</th><th>Coverage</th>"
               "<th>My Criteria</th><th>Runtime</th><th>Description</th></tr>")
    for s in summaries:
        c, r = s["counts"], s["runtime"]
        partial = " <span class=partial>partial</span>" if s["finished"] < s["planned"] else ""
        cover = (f"{s['finished']}/{s['planned']} finished, {s['classified']} classified{partial}<br><small>pending "
                 f"{c['pending']} · error {c['error']} · unavailable {c['unavailable']}</small>")
        runtime = (f"{f(r['median_min'], 1)} min (range {f(r['min_min'], 1)}–{f(r['max_min'], 1)}), "
                   f"{f(r['recipe_hours'], 2)} recipe-h, {r['timed']} timed" if r["timed"] else "–")
        runtime += (f"<br><small>elapsed {f(s['elapsed_hours'], 2)} h" if s["elapsed_hours"] is not None
                    else "<br><small>elapsed not recorded") + (
            f", parallel {'/'.join(map(str, s['parallel']))}, CPUs {'/'.join(s['cpus'])}"
            if s["parallel"] else "") + f"; run {esc(s['run'])}</small>"
        out.append(f"<tr{' class=old' if s['superseded'] else ''}><td>{esc(s['config'])}</td><td>{cover}</td>"
                   f"<td>{bars(c, scale, s['run'], s['config'])}</td><td>{runtime}</td>"
                   f"<td>{esc(s['description'])}<br><b>{s['initially_above']}/{s['initial_known']} with known baseline "
                   "initially exceed the gain-damage boundary.</b></td></tr>")
    # v4 summary: one section per expectation group, one row per configuration
    v4scale = {dim: max([1] + [x["counts"][dim][k] for x in v4 for k in x["counts"][dim]]) for dim in DIMENSIONS}
    out.append("</table></div><h2>Outcomes by expectation group (v4)</h2><p>Click a bar to list exactly its cases "
               "(configuration AND expectation group AND that outcome); click a group/configuration label for the whole "
               "subset of that run, or a group heading for the group across all configurations. Bars share one count scale per dimension. Coverage: planned / finished / with Image measurement.</p>"
               "<div class=scroll><table><tr><th>Expectation group</th><th>Configuration</th><th>Coverage</th>"
               "<th>Pipeline</th><th>Recovery</th><th>Image</th><th>Runtime</th></tr>")
    for group in EXPECTATIONS:
        items = [x for x in v4 if x["expectation"] == group]
        for n, x in enumerate(items):
            flt = esc(json.dumps(dict(run=x["run"], configuration=x["config"], expectation=group)))
            gflt = esc(json.dumps(dict(expectation=group)))
            label = (f"<td rowspan={len(items)}><b class=grp tabindex=0 role=button data-filter='{gflt}'>"
                     f"{EXPECTATION_NAMES[group]}</b><br><small>all configurations</small></td>" if n == 0 else "")
            r = x["runtime"]
            runtime = (f"{f(r['median_min'], 1)} min (range {f(r['min_min'], 1)}–{f(r['max_min'], 1)}), "
                       f"{f(r['recipe_hours'], 2)} recipe-h" if r["timed"] else "–")
            out.append(f"<tr>{label}<td><span class=grp tabindex=0 role=button data-filter='{flt}'>{esc(x['config'])}"
                       f"</span><br><small>{esc(x['run'])}</small></td><td>{x['planned']} / {x['finished']} / {x['image_available']}</td>"
                       + "".join(f"<td>{v4_bars(dim, x['counts'][dim], v4scale[dim], x['run'], x['config'], group)}</td>"
                                 for dim in DIMENSIONS) + f"<td>{runtime}</td></tr>")
    out.append("</table></div>")

    # Cases: filters and table
    def select(key, label, values):
        return (f"<label>{label} <select data-key={key}><option value=''>all</option>"
                + "".join(f"<option value='{esc(v)}'>{esc(t)}</option>" for v, t in values) + "</select></label> ")
    configs = list(dict.fromkeys(s["config"] for s in summaries))
    out.append("<h2>Cases</h2><ul>"
               "<li><b>Expectation</b>: group, reason and signed joint gain damage D (≈0: |D| &lt; 0.0005, sign kept).</li>"
               "<li><b>Reference eC</b>: eC of the same case without the amp/phase injections (same noise).</li>"
               "<li><b>eC_solvable</b> = eC − reference eC: joint gain damage, i.e. excess over the matched reference, "
               "before → after (negative after: below the reference). The reference is a benchmark, not a physical bound.</li>"
               "<li><b>reduction</b> = before − after (positive = better); the same number for eC and eC_solvable.</li>"
               "<li><b>RMS</b> (Jy/beam, off-source annulus) and <b>SNR</b> (peak image SNR) of the restored images, with "
               "r_RMS / r_SNR (positive = better).</li>"
               "<li><b>Flux error</b>: model flux / true flux − 1, before → after.</li></ul>"
               "<p class=filters>" + select("run", "Run", [(r, r) for r in dict.fromkeys(s["run"] for s in summaries)])
               + select("configuration", "Configuration", [(c, c) for c in configs])
               + select("expectation", "Expectation", [(g, EXPECTATION_NAMES[g]) for g in EXPECTATIONS])
               + "".join(select(dim, dim.capitalize(), [(k, k) for k in cats] + [("other", "pending/unavailable")])
                         for dim, (cats, _) in DIMENSIONS.items())
               + select("legacy", "Legacy v3", [(k, name(k)) for k in CATEGORIES + ACCOUNTING])
               + "<button id=reset>reset</button> <b id=count></b> <span id=active></span></p>"
               "<p><b>Corruption</b>: injected kinds; d_k = eC(case) − eC(same case with kind k removed), archived removal "
               "damage (per-kind values need not add up); eC = archived total. <b>Images</b>: restored CLEAN image used for "
               "eC, before (archived case) and after the recipe, canonical plot style with one shared linear scale per case "
               "(±99.5th percentile of |pixel| over both images, shown under the after image); click to enlarge.</p>"
               "<div class=scroll><table id=cases><tr><th>Configuration</th><th>Source / case</th><th>Corruption</th>"
               "<th>Expectation</th><th>Pipeline</th><th>Recovery</th><th>Image</th><th>eC before → after</th>"
               "<th>CLEAN before</th><th>CLEAN after</th><th>eC_solvable before → after</th><th>RMS before → after</th>"
               "<th>SNR before → after</th><th>Flux error</th><th>Legacy v3</th><th>Recipe runtime (min)</th>"
               "<th>Description</th></tr>")
    g = lambda x, d=4: "–" if x is None else f"{x:.{d}g}"
    for row in rows:
        for c in row["cases"]:
            link = c.get("evidence_file")
            notes = "<br>".join(esc(n) for n in c.get("notes", []))
            details = (f"<details><summary>evidence</summary>{notes or 'no pipeline log lines'}"
                       + (f"<br><a href='{esc(link)}'>{esc(link)}</a>" if link else "") + "</details>"
                       if notes or link else "")
            D = c.get("D")
            exp = (f"{EXPECTATION_NAMES[c['expectation']]}<br><small>{esc(c.get('expectation_reason') or '')}"
                   + (f"<br>D = {D:+.4f}{' (≈0)' if c.get('near_zero') else ''}" if D is not None else "") + "</small>")
            image_note = (f"<br><small>{esc(c['image_reason'])}</small>" if c.get("image_reason") else
                          f"<br><small>r_RMS {c['r_rms']:+.3f}<br>r_SNR {c['r_snr']:+.3f}</small>"
                          if c.get("r_rms") is not None and c.get("r_snr") is not None else "")
            attrs = dict(run=row["run"], configuration=row["config"], expectation=c["expectation"], pipeline=c.get("pipeline"),
                         recovery=c.get("recovery"), image=c.get("image"), legacy=c["category"], id=c["id"])
            out.append(
                "<tr " + " ".join(f"data-{k}='{esc(str(v))}'" for k, v in attrs.items())
                + f"{' class=old' if c.get('superseded') else ''}><td>{esc(row['config'])}</td>"
                f"<td>{esc(c['label'])}<br><small>{c['case']}</small></td><td>{corruption_cell(c)}</td>"
                f"<td>{exp}</td><td>{v4_badge('pipeline', c.get('pipeline'))}<br><small>{esc(c['action'])}</small></td>"
                f"<td>{v4_badge('recovery', c.get('recovery'))}</td><td>{v4_badge('image', c.get('image'))}{image_note}</td>"
                f"<td class=n>{f(c.get('eC_before_measured', c.get('before')))} → "
                f"{f(c.get('eC_after_measured', c.get('after')))}<br>reduction {f(c.get('reduction'))}</td>"
                f"{image_cell(c, 'before')}{image_cell(c, 'after')}"
                f"<td class=n>{f(c.get('damage'))} → {f(c.get('remaining'))}<br>reduction {f(c.get('reduction'))}"
                f"<br><small>reference eC {f(c.get('reference'))} ({esc(c.get('reference_id') or '–')})</small></td>"
                f"<td class=n>{g(c.get('rms_before'))} → {g(c.get('rms_after'))}</td>"
                f"<td class=n>{g(c.get('snr_before'))} → {g(c.get('snr_after'))}</td>"
                f"<td class=n>{f(c.get('flux_before'))} → {f(c.get('flux_after'))}</td>"
                f"<td>{badge(c['category'])}</td>"
                f"<td class=n>{f(c['seconds'] / 60 if c.get('seconds') is not None else None, 1)}</td>"
                f"<td>{esc(c['description'])}{details}</td></tr>")
    known = {dim: list(cats) for dim, (cats, _) in DIMENSIONS.items()}
    out.append("</table></div><script>const KNOWN=" + json.dumps(known) + ";"
               "const sel=[...document.querySelectorAll('select[data-key]')],rows=[...document.querySelectorAll("
               "'#cases tr[data-id]')];"
               "function match(tr,k,v){const x=tr.dataset[k];return !v||(v==='other'&&KNOWN[k]?!KNOWN[k].includes(x):x===v)}"
               "function apply(){let n=0;for(const tr of rows){const ok=sel.every(s=>match(tr,s.dataset.key,s.value));"
               "tr.style.display=ok?'':'none';n+=ok}document.getElementById('count').textContent=n+' / '+rows.length+"
               "' cases';document.getElementById('active').textContent=sel.filter(s=>s.value).map(s=>s.dataset.key+"
               "'='+s.value).join(', ')}"
               "function setFilter(f){for(const s of sel)s.value=f[s.dataset.key]||'';apply();"
               "document.getElementById('cases').scrollIntoView()}"
               "for(const s of sel)s.onchange=apply;document.getElementById('reset').onclick=()=>setFilter({});"
               "for(const b of document.querySelectorAll('[data-filter]')){const go=()=>setFilter(JSON.parse(b.dataset.filter));"
               "b.onclick=go;b.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}}}"
               "apply()</script>")
    (output / filename).write_text("\n".join(out))


def build(runs, output, filename="comparison.html", eps_rms=EPSILON_RMS, eps_snr=EPSILON_SNR):
    rows = [row for run in runs for row in load(run)]
    rows.sort(key=lambda r: r["key"])
    output.mkdir(parents=True, exist_ok=True)
    add_image_metrics(rows, output / "image_metrics", eps_rms, eps_snr)
    for row in rows:      # retained evidence per case (after the image metrics), linked from Description
        for c in row["cases"]:
            if "evidence" in c:
                c["evidence_file"] = f"descriptions/{row['run']}__{c['case']}__{safe(row['recipe'])}.json"
                write_json(output / c["evidence_file"], dict(
                    run=row["run"], configuration=row["config"], case=c["case"], label=c["label"],
                    description=c["description"], my_criteria={k: c.get(k) for k in (
                        "category", "action", "before", "after", "reference", "boundary", "reference_id", "damage", "remaining",
                        "reduction", "nonpositive_damage", "criteria_reason")}, v4={k: c.get(k) for k in V4_FIELDS},
                    **c.pop("evidence")))

    images = output / "images"
    images.mkdir(exist_ok=True)
    rendered = reused = 0
    for row in rows:                        # before/after CLEAN thumbnails, one shared color scale per case
        for c in row["cases"]:
            before, after = c.pop("clean_before", None), c.pop("clean_after", None)
            if not (before and after):
                continue
            stem = f"images/{row['run']}__{c['case']}__{safe(row['recipe'])}"
            pb, pa, scale = output / f"{stem}__before.png", output / f"{stem}__after.png", output / f"{stem}.json"
            newest = max(Path(before).stat().st_mtime, Path(after).stat().st_mtime)
            if pb.exists() and pa.exists() and scale.exists() and scale.stat().st_mtime >= newest:
                c["png_scale"] = json.loads(scale.read_text())
                reused += 1
            else:
                try:
                    c["png_scale"] = render_pair(before, after, pb, pa, c["label"])
                except Exception as exc:    # a broken FITS must not stop the report
                    print("IMAGE FAILED", c["case"], row["recipe"], exc)
                    continue
                scale.write_text(json.dumps(c["png_scale"]))
                rendered += 1
            c["before_png"], c["after_png"] = f"{stem}__before.png", f"{stem}__after.png"
    print(f"IMAGES rendered {rendered}, reused {reused}")
    summaries = [summarize(r) for r in rows]
    v4 = summarize_v4(rows)
    write_json(output / filename.replace(".html", ".json"), dict(
        criteria_version=CRITERIA_VERSION, tolerance=TOLERANCE, epsilon=EPSILON, epsilon_status="provisional",
        v4_version=V4_VERSION, epsilon_rms=eps_rms, epsilon_snr=eps_snr, image_metric_version=IMAGE_METRIC_VERSION,
        v4_thresholds_status="provisional", v4_summary=v4,
        runs=[str(r.resolve()) for r in runs], rows=summaries,
        cases={f"{r['run']} | {r['config']}": [{k: v for k, v in c.items() if k != "notes"} for c in r["cases"]]
               for r in rows}))
    write_report(output, summaries, rows, runs, filename, v4, eps_rms, eps_snr)
    print("REPORT", (output / filename).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="vla_pipeline_qa output folders")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epsilon-rms", type=float, default=EPSILON_RMS,
                    help="relative RMS change tolerance of the v4 Image outcome (PROVISIONAL default)")
    ap.add_argument("--epsilon-snr", type=float, default=EPSILON_SNR,
                    help="relative SNR change tolerance of the v4 Image outcome (PROVISIONAL default)")
    args = ap.parse_args()
    build(args.runs, args.output, eps_rms=args.epsilon_rms, eps_snr=args.epsilon_snr)


if __name__ == "__main__":
    main()
