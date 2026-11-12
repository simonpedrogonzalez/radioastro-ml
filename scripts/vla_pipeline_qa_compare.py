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

from scripts.vla_pipeline_qa import (ACCOUNTING, CATEGORIES, CRITERIA_VERSION, EPSILON, TOLERANCE, describe,
                                     gain_mask, matched_reference, my_criteria, safe, write_json)

CORRUPTIONS = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise",
               7: "amp+phase+noise"}
COLORS = dict(preserved="#009E73", solved="#0072B2", improved="#56B4E9", degraded="#D55E00",
              blocked="#E69F00", rejected="#CC79A7", unchanged="#777777")
NAMES = {}
name = lambda k: NAMES.get(k, k)


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
    rows_path = source_run / "rows.json"                   # per-kind removal damage d = eC(case) - eC(kind removed)
    damage_of = ({r["case"]: r for r in json.loads(rows_path.read_text())["rows"]} if rows_path.exists() else {})
    archived = {}

    def eC_of(case_id):
        if case_id not in archived:
            path = source_run / "cases" / case_id / "case.json"
            archived[case_id] = json.loads(path.read_text()).get("eC") if path.exists() else None
        return archived[case_id]

    units = {x.stem: json.loads(x.read_text()) for x in (run / "units").glob("*.json")}
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
            item = dict(case=cid, label=next(r["id"] for r in splan["rows"] if r["case"] == cid),
                        corruption=CORRUPTIONS[gain_mask(params)], mask=gain_mask(params), action="–",
                        d=dmg.get("d"), eC_total=dmg.get("eC", eC_of(cid)),
                        clean_before=clean_image(source_run, cid))
            path = run / "results" / f"{cid}__{safe(recipe)}.json"
            if not path.exists():
                item.update(category="error" if cid in units else "pending",
                            description="No result record although the case finished." if cid in units else
                            "Not run yet (no result record).")
            else:
                result = json.loads(path.read_text())
                ref = matched_reference(splan, cid, eC_of)
                crit = my_criteria(result, gain_mask(params) & 3 != 0, ref)
                reused = result.get("evaluation", {}).get("evaluation_reused")
                item.update(crit, reference_id=ref.get("id"), seconds=result["recipe_record"].get("seconds"),
                            evaluation_reused=bool(reused), clean_after=(
                                item["clean_before"] if reused else clean_image(run, f"{cid}/{safe(recipe)}")),
                            superseded=superseded(result), description=describe(result, crit),
                            notes=result["recipe_record"].get("pipeline_notes", []),
                            evidence=dict(recipe_record=result["recipe_record"], evaluation={
                                k: v for k, v in result["evaluation"].items() if k not in ("before", "after")},
                                reference=ref))
            cases.append(item)
        rows.append(dict(run=run.name, config=label, key=key, recipe=recipe, cases=cases, calls=calls,
                         superseded=any(c.get("superseded") for c in cases)))
    return rows


def summarize(row):
    cases = row["cases"]
    counts = {k: sum(c["category"] == k for c in cases) for k in CATEGORIES + ACCOUNTING}
    secs = [c["seconds"] for c in cases if c.get("seconds") is not None]
    runtime = dict(timed=len(secs), median_min=float(np.median(secs)) / 60, min_min=min(secs) / 60,
                   max_min=max(secs) / 60, recipe_hours=sum(secs) / 3600) if secs else dict(timed=0)
    done = [c for c in row["calls"] if "finished" in c]
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
                elapsed_hours=sum(c["finished"] - c["started"] for c in done) / 3600 if done else None,
                parallel=sorted({c["parallel"] for c in row["calls"]}),
                cpus=sorted({str(c.get("slurm_cpus") or c.get("cpu_count")) for c in row["calls"]}))


def bars(counts, scale, config):
    esc = htmllib.escape
    out = []
    for k in CATEGORIES:
        n = counts[k]
        out.append(f"<div class=bar title='{name(k)}: {n}' data-config='{esc(config)}' data-cat={k}>"
                   f"<span class=lab>{name(k)}</span><span class=fill style='width:{100 * n / scale:.0f}px;"
                   f"background:{COLORS[k]}'></span><span>{n}</span></div>")
    return "".join(out)


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


def write_report(output, summaries, rows, runs, filename):
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    esc = htmllib.escape
    scale = max([1] + [s["counts"][k] for s in summaries for k in CATEGORIES])
    out = ["<!doctype html><meta charset=utf-8><title>VLA pipeline QA comparison</title><style>body{font-family:"
           "sans-serif;max-width:1600px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;margin:0}.scroll{overflow-x:auto;margin:.6em 0}"
           "td,th{border:1px solid #ccc;padding:3px 7px;text-align:left;vertical-align:top}th{background:#f3f3f3}"
           "td.n{text-align:right}tr.old td{color:#999}.bar{display:flex;align-items:center;gap:4px;font-size:12px;"
           "cursor:pointer;line-height:15px}.bar .lab{width:135px}.fill{display:inline-block;height:11px}"
           ".badge{padding:1px 6px;border-radius:3px;font-size:12px;white-space:nowrap}.partial{color:#D55E00;"
           "font-weight:bold}details{font-size:12px}small{color:#555}</style>",
           f"<h1>VLA pipeline QA comparison</h1><p>{len(runs)} run(s): {', '.join(esc(r.name) for r in runs)}. "
           f"Criteria: {CRITERIA_VERSION}.</p>",
           "<h3>My Criteria</h3><ul>"
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
           "<li>Pending, error and unavailable are accounting states, not scientific outcomes.</li></ul>"]
    if any(s["superseded"] for s in summaries):
        out.append("<p><b>Grey rows</b>: B flagging_off without the calonly model-write fix (superseded, reference "
                   "only).</p>")
    out.append("<h2>Configurations</h2><p>Legend: " + " ".join(badge(k) for k in CATEGORIES) + ". Bars share one "
               "count scale; click a bar to list its cases. Runtime: measured wall time of the recipe's own CASA "
               "process (median, range) and summed recipe-process hours, which is not elapsed run time. Elapsed: "
               "summed duration of the run's --execute calls (whole cases: regeneration, both flagging recipes, "
               "evaluation), shared by both flagging rows.</p><div class=scroll><table><tr><th>Configuration</th><th>Coverage</th>"
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
                   f"<td>{bars(c, scale, s['config'])}</td><td>{runtime}</td>"
                   f"<td>{esc(s['description'])}<br><b>{s['initially_above']}/{s['initial_known']} with known baseline "
                   "initially exceed the gain-damage boundary.</b></td></tr>")
    out.append("</table></div><h2>Cases</h2><ul>"
               "<li><b>Reference eC</b>: eC of the same case without the amp/phase errors, i.e. what eC should be once "
               "the gains are solved.</li>"
               "<li><b>eC_solvable</b> = eC − reference eC: the part of eC a perfect gain solution removes; shown before → "
               "after the recipe (negative after: below the reference).</li>"
               "<li><b>reduction</b> = before − after (positive = better, negative = worse); the same number for eC and "
               "eC_solvable, since both share the reference.</li>"
               "<li><b>Pipeline action</b>: <i>Applied</i> (a correction was applied), <i>Left unchanged</i> (trials ran, "
               "no correction kept), <i>Could not start</i> (blocked, e.g. empty mask or S/N too low), <i>Error</i> "
               "(the recipe crashed), <i>Unknown</i> (no decision recorded).</li></ul>"
               "<p>Every planned case. Filter: <select id=fc><option value=''>all "
               "configurations</option>" + "".join(f"<option>{esc(s['config'])}</option>" for s in summaries) +
               "</select> <select id=fk><option value=''>all categories</option>" +
               "".join(f"<option value={k}>{name(k)}</option>" for k in CATEGORIES + ACCOUNTING) + "</select></p>"
               "<p><b>Corruption</b>: injected kinds; d_k = eC(case) − eC(same case with kind k removed), archived removal damage "
               "(per-kind values need not add up); eC = archived total. <b>Images</b>: CLEAN image used for eC, before "
               "(archived case) and after the recipe, canonical imaging plot style with one shared linear scale per case "
               "(±99.5th percentile of |pixel| over both images, shown under the after image); click to enlarge.</p>"
               "<div class=scroll><table id=cases><tr><th>Configuration</th><th>Source / case</th><th>Corruption</th>"
               "<th>Pipeline action</th><th>My Criteria</th><th>eC before → after</th><th>CLEAN before</th>"
               "<th>CLEAN after</th><th>eC_solvable before → after</th>"
               "<th>Recipe runtime (min)</th><th>Description</th></tr>")
    for row in rows:
        for c in row["cases"]:
            link = c.get("evidence_file")
            notes = "<br>".join(esc(n) for n in c.get("notes", []))
            details = (f"<details><summary>evidence</summary>{notes or 'no pipeline log lines'}"
                       + (f"<br><a href='{esc(link)}'>{esc(link)}</a>" if link else "") + "</details>"
                       if notes or link else "")
            out.append(
                f"<tr data-config='{esc(row['config'])}' data-cat={c['category']}"
                f"{' class=old' if c.get('superseded') else ''}><td>{esc(row['config'])}</td>"
                f"<td>{esc(c['label'])}<br><small>{c['case']}</small></td><td>{corruption_cell(c)}</td>"
                f"<td>{esc(c['action'])}</td><td>{badge(c['category'])}</td>"
                f"<td class=n>{f(c.get('before'))} → {f(c.get('after'))}<br>reduction {f(c.get('reduction'))}</td>"
                f"{image_cell(c, 'before')}{image_cell(c, 'after')}"
                f"<td class=n>{f(c.get('damage'))} → {f(c.get('remaining'))}<br>reduction {f(c.get('reduction'))}"
                f"<br><small>reference eC {f(c.get('reference'))} ({esc(c.get('reference_id') or '–')})</small></td>"
                f"<td class=n>{f(c['seconds'] / 60 if c.get('seconds') is not None else None, 1)}</td>"
                f"<td>{esc(c['description'])}{details}</td></tr>")
    out.append("</table></div><script>const fc=document.getElementById('fc'),fk=document.getElementById('fk');"
               "function apply(){for(const tr of document.querySelectorAll('#cases tr[data-cat]'))tr.style.display="
               "(!fc.value||tr.dataset.config===fc.value)&&(!fk.value||tr.dataset.cat===fk.value)?'':'none'}"
               "fc.onchange=fk.onchange=apply;for(const b of document.querySelectorAll('.bar'))b.onclick=()=>{"
               "fc.value=b.dataset.config;fk.value=b.dataset.cat;apply();"
               "document.getElementById('cases').scrollIntoView()}</script>")
    (output / filename).write_text("\n".join(out))


def build(runs, output, filename="comparison.html"):
    rows = [row for run in runs for row in load(run)]
    rows.sort(key=lambda r: r["key"])
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:                        # retained evidence per case, linked from Description
        for c in row["cases"]:
            if "evidence" in c:
                c["evidence_file"] = f"descriptions/{row['run']}__{c['case']}__{safe(row['recipe'])}.json"
                write_json(output / c["evidence_file"], dict(
                    run=row["run"], configuration=row["config"], case=c["case"], label=c["label"],
                    description=c["description"], my_criteria={k: c.get(k) for k in (
                        "category", "action", "before", "after", "reference", "boundary", "reference_id", "damage", "remaining",
                        "reduction", "nonpositive_damage", "reason")}, **c.pop("evidence")))
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
    write_json(output / filename.replace(".html", ".json"), dict(
        criteria_version=CRITERIA_VERSION, tolerance=TOLERANCE, epsilon=EPSILON, epsilon_status="provisional",
        runs=[str(r.resolve()) for r in runs], rows=summaries,
        cases={f"{r['run']} | {r['config']}": [{k: v for k, v in c.items() if k != "notes"} for c in r["cases"]]
               for r in rows}))
    write_report(output, summaries, rows, runs, filename)
    print("REPORT", (output / filename).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="vla_pipeline_qa output folders")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    build(args.runs, args.output)


if __name__ == "__main__":
    main()
