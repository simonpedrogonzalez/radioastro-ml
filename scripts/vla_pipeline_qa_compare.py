"""Report of one or several VLA pipeline QA runs (one experiment/variant each), v5 evaluation.

Each run of scripts/vla_pipeline_qa.py holds one experiment (A or B) with one set of self-calibration options (image
size, masking) and its flagging recipes. This reads the retained per-case results, labels every case with the v5
evaluation (design doc, "Requested v5 evaluation setup", 2026-11-16) and writes:

  Table 0  glossary: one phrase per dimension/category, thresholds and the reference definition;
  then per run and recipe configuration (never pooled across runs):
  Table 1  one ungrouped row: Config (coverage, runtime) | Pipeline Result | Recovery | Image | Run;
  Table 2  the same columns for Improvement expected / No improvement expected (+ exception/unavailable accounting);
  Table 3  the same columns for the No improvement expected subgroups (Uncorrupted, Noise only, Cancelling, Small);
  Cases    filters and one row per planned case and recipe (pending ones included).

Every bar selects exactly its own cases (run, configuration, group/subgroup, dimension, category, blocked reason).
Retained per-case evidence (all raw metrics, thresholds, reasons, version, legacy v3 labels as history) is written to
descriptions/, image thumbnails to images/ (rendered from the FITS the eC evaluation used; existing PNGs are reused).
Restored-image metrics are cached in image_metrics/ without labels. Run folders are only read; nothing is executed.

  bin/run ml scripts/vla_pipeline_qa_compare.py --output runs/vla-pipeline-qa-compare RUN [RUN ...]
"""
from __future__ import annotations

import argparse
import hashlib
import html as htmllib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.vla_pipeline_qa import (ACCOUNTING_STATES, BLOCKED_REASONS, CRITERIA_VERSION, EPSILON_RMS, EPSILON_SNR,
                                     EXCEPTION, GROUPS, IMAGE, IMAGE_METRIC_VERSION, RECOVERY, RESULTS, SUBGROUPS,
                                     TOL_CHANGE, TOL_REFERENCE, V5_VERSION, check_tolerance, describe, gain_mask,
                                     image_label, image_measure, matched_reference, my_criteria, safe, v5_labels,
                                     write_json)

CORRUPTIONS = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise",
               7: "amp+phase+noise"}
TEAL, BLUE, VERMILION, GRAY, PURPLE, AMBER, LIGHT = "#009E73", "#0072B2", "#D55E00", "#777777", "#CC79A7", "#E69F00", "#bbbbbb"
NAMES = dict(success="Success", partial_success="Partial success", failure="Failure", solved="Solved",
             improved="Improved", similar="Similar", degraded="Degraded", cleaner="Cleaner", worse="Worse", mixed="Mixed",
             applied="Applied", blocked="Blocked", unavailable="unavailable", error="error", unknown="unknown",
             pending="pending", improvement_expected="Improvement expected",
             no_improvement_expected="No improvement expected", exception="Exception: gain-only negative D",
             uncorrupted="Uncorrupted", noise_only="Noise only", cancelling="Cancelling Corruptions", small="Small",
             low_initial_snr="Low Initial SNR", solution_lowers_snr="Solution Lowers SNR", empty_model="Empty Model/Mask",
             other="Other", reason_unknown="Reason unknown")
COLORS = dict(success=TEAL, partial_success=BLUE, failure=VERMILION, solved=TEAL, improved=BLUE, similar=GRAY,
              degraded=VERMILION, cleaner=BLUE, worse=VERMILION, mixed=PURPLE, applied=BLUE, blocked=AMBER)
# Report dimensions in column order: key in the case rows / filters, heading, scientific categories
DIMENSIONS = (("result", "Pipeline Result", RESULTS), ("recovery", "Recovery", RECOVERY), ("image", "Image", IMAGE),
              ("application", "Run", ("applied", "blocked")))
NAMED_REASONS = ("low_initial_snr", "solution_lowers_snr")     # always shown; other reasons only when present
FILTER_KEYS = ("run", "config", "group", "subgroup", "result", "recovery", "image", "application", "reason")
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


def load(run, tol_change=TOL_CHANGE, tol_ref=TOL_REFERENCE):
    """One row per recipe of the run, holding every planned case with its v5 labels (Image is added later)."""
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
            mask = gain_mask(params)
            ref = matched_reference(splan, cid, eC_of)
            path = run / "results" / f"{cid}__{safe(recipe)}.json"
            result = json.loads(path.read_text()) if path.exists() else None
            labels = v5_labels(mask, eC_of(cid), ref.get("eC"), result, cid in units, tol_change, tol_ref)
            if labels["group"] == "unavailable" and ref.get("status") == "unavailable":
                labels["group_reason"] = ref.get("reason")
            item = dict(case=cid, label=next(r["id"] for r in splan["rows"] if r["case"] == cid),
                        corruption=CORRUPTIONS[mask], mask=mask, d=damage_of.get(cid, {}).get("d"),
                        eC_archived=eC_of(cid), reference_id=ref.get("id"), reference_reason=ref.get("reason"),
                        clean_before=clean_image(source_run, cid), masks=masks_of.get(splan["source"]),
                        id=f"{run}|{cid}|{recipe}", **labels)
            if result is None:
                item.update(action="–", legacy=dict(category=labels["application"]),
                            description="No result record although the case finished." if cid in units else
                            "Not run yet (no result record).")
            else:
                ev = result.get("evaluation", {})
                crit = my_criteria(result, mask & 3 != 0, ref)          # legacy v3: historical metadata only
                reused = bool(ev.get("evaluation_reused"))
                item.update(action=crit["action"], legacy=crit, seconds=result["recipe_record"].get("seconds"),
                            evaluation_reused=reused,
                            clean_after=item["clean_before"] if reused else clean_image(run, f"{cid}/{safe(recipe)}"),
                            flux_before=(ev.get("before") or {}).get("flux_error"),
                            flux_after=(ev.get("after") or {}).get("flux_error"),
                            superseded=superseded(result), description=describe(result, labels),
                            notes=result["recipe_record"].get("pipeline_notes", []),
                            evidence=dict(recipe_record=result["recipe_record"], evaluation={
                                k: v for k, v in ev.items() if k not in ("before", "after")}, reference=ref))
            cases.append(item)
        rows.append(dict(run=run.name, run_id=str(run), tag=run.name, config=label, key=key, recipe=recipe, cases=cases, calls=calls,
                         unit_times=unit_times, superseded=any(c.get("superseded") for c in cases)))
    return rows


V5_FIELDS = ("group", "subgroup", "group_reason", "D", "reference", "reference_id", "reference_reason", "eC_archived",
             "eC_before", "eC_after", "change", "remaining", "recovery", "recovery_reason", "result", "application",
             "blocked_reason", "stop_reasons", "image", "image_reason", "epsilon_rms", "epsilon_snr", "rms_before",
             "rms_after", "mean_before", "mean_after", "peak_before", "peak_after", "snr_before", "snr_after", "r_rms",
             "r_snr", "off_pixels", "source_pixels", "off_beams", "units", "flux_before", "flux_after",
             "image_metric_version", "evaluation_reused", "progress_status", "progress_pct", "progress_required_pct",
             "progress_small_D")


def add_image_metrics(rows, cache, eps_rms=EPSILON_RMS, eps_snr=EPSILON_SNR):
    """Image dimension per case from the retained restored CLEAN FITS (before: archived case, after: evaluation image)
    and the source's fixed reference masks. Raw metrics are cached by product identity and metric version only; the
    label is applied at the requested tolerances every time, so changing a tolerance never reuses old categories.
    Before is reused as after only when the QA evaluation established an unchanged input (evaluation_reused)."""
    cache.mkdir(parents=True, exist_ok=True)
    masks_cache, done, reused = {}, 0, 0
    for row in rows:
        for c in row["cases"]:
            if c["application"] in ("pending", "error"):
                c.update(image=c["application"], image_reason="no after image")
                continue
            before, after, masks = c.get("clean_before"), c.get("clean_after"), c.get("masks")
            if not (before and after and masks and Path(masks).exists()):
                c.update(image="unavailable", image_reason="missing restored CLEAN FITS or reference masks "
                         f"(before={bool(before)}, after={bool(after)}, masks={bool(masks)})")
                continue
            ident = [IMAGE_METRIC_VERSION, "strict-meta-v2", "raw-v1"] + [
                (x, Path(x).stat().st_size, Path(x).stat().st_mtime) for x in (before, after, masks)]
            path = cache / (hashlib.sha256(json.dumps(ident).encode()).hexdigest()[:24] + ".json")
            if path.exists():
                raw = json.loads(path.read_text())
                reused += 1
            else:
                if masks not in masks_cache:
                    with np.load(masks, allow_pickle=False) as z:
                        masks_cache[masks] = dict(annulus=np.asarray(z["annulus"], bool),
                                                  source=np.asarray(z["source"], bool))
                raw = image_measure(before, after, masks_cache[masks])
                path.write_text(json.dumps(raw))
                done += 1
            c.update(image_label(raw, eps_rms, eps_snr))
    print(f"IMAGE METRICS computed {done}, cached {reused}")


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


def stats(cases):
    """Coverage, per-dimension counts (blocked split by reason) and recipe runtime of a set of cases."""
    secs = [c["seconds"] for c in cases if c.get("seconds") is not None]
    return dict(ids=[c["id"] for c in cases], planned=len(cases),
                finished=sum(c["application"] != "pending" for c in cases),
                available=sum(c["recovery"] in RECOVERY for c in cases),
                image_available=sum(c.get("image") in IMAGE for c in cases),
                counts={key: dict(Counter(str(c.get(key)) for c in cases)) for key, _, _ in DIMENSIONS},
                reasons=dict(Counter(c["blocked_reason"] for c in cases if c["application"] == "blocked")),
                runtime=dict(timed=len(secs), median_min=float(np.median(secs)) / 60, min_min=min(secs) / 60,
                             max_min=max(secs) / 60, recipe_hours=sum(secs) / 3600) if secs else dict(timed=0),
                progress=progress_stats(cases))


# ---------------------------------------------------------------- recovery progress plots
# [USER] 2026-11-18: per summary row a neutral boxplot and two swarms (colored by Pipeline Result, by Recovery) of
# progress = 100 (before - after) / D for the cases of that row with D > 0. Same cases, values and positions in all
# three; nothing clamped, sampled or aggregated.

POINT_FIELDS = ("id", "label", "case", "result", "recovery", "eC_before", "eC_after", "reference", "D", "remaining",
                "progress_pct", "progress_required_pct", "progress_small_D")
PROGRESS_ZOOM = (-100., 200.)           # optional readability zoom; the full range is the default view
PLOT = dict(height=260, top=16, bottom=8, axis=38, box=40, swarm=160, gap=6, radius=1.7)
NO_PROGRESS = "Not applicable: no positive initial joint gain damage"


def progress_stats(cases):
    """Eligible points (progress defined: finite metrics and D > 0) sorted by (progress, id), with the denominators."""
    points = sorted(({k: c.get(k) for k in POINT_FIELDS} for c in cases if c.get("progress_status") == "applicable"),
                    key=lambda p: (p["progress_pct"], p["id"]))
    na = sum(c.get("progress_status") == "not_applicable" for c in cases)
    return dict(points=points, total=len(cases), plotted=len(points), not_applicable=na,
                missing=len(cases) - len(points) - na)


def box_stats(values):
    """Tukey boxplot: quartiles with numpy's linear interpolation (Hyndman-Fan type 7), whiskers to the most extreme
    observations within 1.5 IQR of the box, outliers beyond."""
    v = np.sort(np.asarray(values, float))
    q1, median, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    inside = v[(v >= lo) & (v <= hi)]
    return dict(q1=q1, median=median, q3=q3, whisker_low=float(inside.min()), whisker_high=float(inside.max()),
                outliers=[float(x) for x in v[(v < lo) | (v > hi)]], n=int(v.size))


def swarm_offsets(ys, spacing):
    """Deterministic dot-swarm: points in the given order (ascending value) take the first column 0, +1, -1, +2, ...
    whose last point is at least `spacing` px away along the value axis. Returns column indices."""
    last, out = {}, []
    for y in ys:
        k = 0
        for j in range(len(ys) + 1):
            k = (j + 1) // 2 * (1 if j % 2 else -1)
            if k not in last or abs(y - last[k]) >= spacing:
                break
        last[k] = y
        out.append(k)
    return out


def nice_ticks(lo, hi, n=5):
    step = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(step))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= step)
    return [t * step for t in range(math.ceil(lo / step), math.floor(hi / step) + 1)]


def progress_layout(points, lo, hi):
    """Pixel positions shared by both swarms: (index into points, y px, x offset px) of the points inside [lo, hi].
    Dense swarms are compressed to the panel width (spacing shrinks, no point is dropped)."""
    h = PLOT["height"] - PLOT["top"] - PLOT["bottom"]
    ypx = lambda v: PLOT["top"] + (hi - v) / (hi - lo) * h
    keep = [i for i, p in enumerate(points) if lo <= p["progress_pct"] <= hi]
    ys = [ypx(points[i]["progress_pct"]) for i in keep]
    spacing = 2 * PLOT["radius"] + .4
    cols = swarm_offsets(ys, spacing)
    half = PLOT["swarm"] / 2 - PLOT["radius"] - 1
    widest = max([abs(k) for k in cols] + [0]) * spacing
    squeeze = min(1., half / widest) if widest else 1.
    return [(i, y, k * spacing * squeeze) for i, y, k in zip(keep, ys, cols)], ypx, squeeze


def point_color(dim, value):
    color = COLORS.get(value) if value in (RESULTS if dim == "result" else RECOVERY) else None
    return f"fill='{color}'" if color else "fill='none' stroke='#555' stroke-width='.8'"   # accounting: hollow


def progress_svg(st, lo, hi, key_of):
    """One SVG: shared progress axis (0% and 100% marked), boxplot, swarm by Pipeline Result, swarm by Recovery."""
    P = PLOT
    points = st["points"]
    layout, ypx, squeeze = progress_layout(points, lo, hi)
    x_box = P["axis"]
    x_res = x_box + P["box"] + P["gap"]
    x_rec = x_res + P["swarm"] + P["gap"]
    width = x_rec + P["swarm"] + 2
    out = [f"<svg class=prog width={width} height={P['height']} role=img aria-label='recovery progress: "
           f"{st['plotted']} cases'>"]
    for t in nice_ticks(lo, hi):
        out.append(f"<text x={P['axis'] - 3} y={ypx(t) + 3:.1f} text-anchor=end font-size=9>{t:g}</text>"
                   f"<line x1={P['axis']} x2={width} y1={ypx(t):.1f} y2={ypx(t):.1f} stroke='#eee'/>")
    for t in (0, 100):
        if lo <= t <= hi:
            out.append(f"<line x1={P['axis']} x2={width} y1={ypx(t):.1f} y2={ypx(t):.1f} stroke='#333' "
                       f"stroke-dasharray='3 2'/><text x={width - 2} y={ypx(t) - 2:.1f} text-anchor=end font-size=9>"
                       f"{t}%</text>")
    for x, title in ((x_box, "box"), (x_res, "by Pipeline Result"), (x_rec, "by Recovery")):
        out.append(f"<text x={x + 2} y=10 font-size=9 font-weight=bold>{title}</text>")
    b = box_stats([p["progress_pct"] for p in points])
    clip = lambda v: min(max(v, lo), hi)
    cx, w = x_box + P["box"] / 2, P["box"] * .55
    y1, y3, ym = ypx(clip(b["q3"])), ypx(clip(b["q1"])), ypx(clip(b["median"]))
    out.append(f"<g class=box data-n={b['n']}><line x1={cx} x2={cx} y1={ypx(clip(b['whisker_high'])):.1f} "
               f"y2={ypx(clip(b['whisker_low'])):.1f} stroke='#444'/><rect x={cx - w / 2:.1f} y={y1:.1f} width={w:.1f} "
               f"height={max(y3 - y1, .8):.1f} fill='#ddd' stroke='#444'/><line x1={cx - w / 2:.1f} x2={cx + w / 2:.1f} "
               f"y1={ym:.1f} y2={ym:.1f} stroke='#000' stroke-width=1.6/>")
    for v in b["outliers"]:
        if lo <= v <= hi:
            out.append(f"<circle cx={cx} cy={ypx(v):.1f} r=1.6 fill='none' stroke='#444' stroke-width=.7/>")
    out.append("</g>")
    for dim, x0 in (("result", x_res), ("recovery", x_rec)):
        c0 = x0 + P["swarm"] / 2
        out.append(f"<g class=swarm data-dim={dim}>" + "".join(
            f"<circle cx={c0 + dx:.1f} cy={y:.1f} r={P['radius']} {point_color(dim, points[i][dim])} "
            f"data-k={key_of[points[i]['id']]} data-c={esc_attr(points[i][dim])}/>" for i, y, dx in layout) + "</g>")
    out.append("</svg>")
    off = (sum(p["progress_pct"] < lo for p in points), sum(p["progress_pct"] > hi for p in points))
    return "".join(out), off, squeeze


def esc_attr(value):
    return htmllib.escape(str(value), quote=True)


def progress_cell(st, full, key_of):
    """Summary cell: coverage line, the full-range SVG and the zoomed SVG (one shown at a time)."""
    head = (f"<small>plotted {st['plotted']} / {st['total']} · not applicable {st['not_applicable']} · "
            f"missing/invalid {st['missing']}</small>")
    if not st["plotted"]:
        why = (NO_PROGRESS if st["not_applicable"] and not st["missing"] else
               f"No plotted case: {st['missing']} with missing/invalid metrics"
               + (f"; {st['not_applicable']} not applicable (no positive initial joint gain damage)"
                  if st["not_applicable"] else "") if st["total"] else "No cases in this row")
        return f"<td class=progress>{head}<br><i>{htmllib.escape(why)}</i></td>"
    parts = [f"<td class=progress>{head}"]
    for cls, (lo, hi) in (("pfull", full), ("pzoom", PROGRESS_ZOOM)):
        svg, (below, above), squeeze = progress_svg(st, lo, hi, key_of)
        notes = []
        if below or above:
            notes.append(f"off-axis: {below} below {lo:g}%, {above} above {hi:g}% (full range: reset zoom)")
        if squeeze < 1:
            notes.append(f"dense swarm compressed ×{squeeze:.2f}")
        parts.append(f"<div class={cls}>{svg}" + (f"<br><small>{'; '.join(notes)}</small>" if notes else "") + "</div>")
    return "".join(parts) + "</td>"


def progress_range(summaries):
    """One common linear full-range axis over every summary row (0 and 100 always included, nothing clipped)."""
    values = [p["progress_pct"] for s in summaries for st in [s["table1"]] for p in st["progress"]["points"]]
    lo, hi = min(values + [0.]), max(values + [100.])
    pad = .04 * (hi - lo)
    return lo - pad, hi + pad


def summarize(row):
    """Tables 1-3 of one run configuration. Group/subgroup partitions must reconcile with the ungrouped total."""
    cases = row["cases"]
    groups = list(GROUPS) + [g for g in (EXCEPTION, "unavailable") if any(c["group"] == g for c in cases)]
    table2 = {g: stats([c for c in cases if c["group"] == g]) for g in groups}
    table3 = {s: stats([c for c in cases if c["subgroup"] == s]) for s in SUBGROUPS}
    if sum(t["planned"] for t in table2.values()) != len(cases):
        raise ValueError(f"{row['run']} {row['config']}: groups do not reconcile with the total")
    if sum(t["planned"] for t in table3.values()) != table2["no_improvement_expected"]["planned"]:
        raise ValueError(f"{row['run']} {row['config']}: subgroups do not reconcile with No improvement expected")
    for st in [stats(cases)] + list(table2.values()) + list(table3.values()):   # every case in exactly one bar
        for key, _, _ in DIMENSIONS:
            if sum(n for _, _, n, _ in entries(key, st)) != st["planned"]:
                raise ValueError(f"{row['run']} {row['config']}: {key} bars do not cover every case")
    return dict(run=row["run"], run_id=row["run_id"], config=row["config"], recipe=row["recipe"], superseded=row["superseded"],
                table1=stats(cases), table2=table2, table3=table3,
                elapsed_hours=elapsed_hours(row["calls"], row.get("unit_times", [])),
                parallel=sorted({c["parallel"] for c in row["calls"]}),
                cpus=sorted({str(c.get("slurm_cpus") or c.get("cpu_count")) for c in row["calls"]}))


def entries(key, st, no_partial=False):
    """(label, filter part, count, color) of every bar of one dimension: all scientific categories (zero included),
    blocked split by reason (the two named reasons always), then the accounting states that occur."""
    counts = st["counts"][key]
    if key == "application":
        out = [("Applied", dict(application="applied"), counts.get("applied", 0), COLORS["applied"])]
        out += [(f"Blocked — {name(r)}", dict(application="blocked", reason=r), st["reasons"].get(r, 0), AMBER)
                for r in BLOCKED_REASONS if r in NAMED_REASONS or st["reasons"].get(r)]
    else:
        cats = next(c for k, _, c in DIMENSIONS if k == key)
        out = [(name(k), {key: k}, counts.get(k, 0), COLORS[k]) for k in cats if not (no_partial and k == "partial_success")]
    return out + [(name(a), {key: a}, counts[a], LIGHT) for a in ACCOUNTING_STATES if counts.get(a)]


def bar(label, n, scale, color, flt):
    """One clickable count bar; flt is the exact filter state it selects (replaces all previous filters)."""
    esc = htmllib.escape
    return (f"<div class=bar tabindex=0 role=button title='{esc(label)}: {n}' data-filter='{esc(json.dumps(flt))}'>"
            f"<span class=lab>{esc(label)}</span><span class=fill style='width:{100 * n / scale:.0f}px;"
            f"background:{color}'></span><span>{n}</span></div>")


def badge(k):
    style = f"background:{COLORS[k]};color:#fff" if k in COLORS else "border:1px dashed #999"
    return f"<span class=badge style='{style}'>{htmllib.escape(name(k))}</span>"


def corruption_cell(c):
    """Kind, per-kind removal damage of the injected kinds and the archived total eC."""
    out = [c["corruption"]]
    d, mask = c.get("d"), c.get("mask", 0)
    if d is not None and mask:
        out.append(" / ".join(f"d_{k} {d[i]:.3f}" for i, k in enumerate(("amp", "phase", "noise")) if mask >> i & 1))
    if c.get("eC_archived") is not None:
        out.append(f"eC={c['eC_archived']:.3f}")
    return "<br>".join(out)


def image_cell(c, key):
    path = c.get(f"{key}_png")
    if not path:
        return "<td><small>–</small></td>"
    note = f"<br><small>±{c['png_scale'][1]:.3g} mJy/beam</small>" if key == "after" else ""
    if key == "after" and c.get("evaluation_reused"):
        note += "<br><small>input unchanged</small>"
    return f"<td><a href='{path}'><img src='{path}' loading=lazy width=160></a>{note}</td>"


def glossary(eps_rms, eps_snr):
    """Table 0: one phrase per dimension/category, thresholds and the reference definition."""
    t, r = TOL_CHANGE, TOL_REFERENCE
    rows = [
        ("Quantities", "eC, reference, D, change, remaining",
         "eC: relative L2 error of the recovered CLEAN model at the evaluation beam against the known truth (0 = perfect). "
         "Reference: archived eC of the same case with both amplitude and phase injections removed (same source, noise "
         "realization, noise multiplier and evaluation settings; match validated). D = before − reference (signed joint "
         "gain damage); change = after − before (positive = worse); remaining = after − reference."),
        ("Thresholds", f"eC {t} / {r} absolute, image {eps_rms:.0%} / {eps_snr:.0%} relative",
         f"Recovery change counts only if |change| > {t}; Solved if after ≤ reference + {r}; RMS/SNR change counts only "
         f"if > {eps_rms:.0%} / {eps_snr:.0%}. A value exactly at a tolerance is Similar / within tolerance. Practical "
         "tolerances, not significance tests."),
        ("Group", name("improvement_expected"), f"amplitude/phase injected and D > {r}."),
        ("Group", name("no_improvement_expected"), "no expected benefit in this benchmark (RMS/SNR may still change):"),
        ("Subgroup", name("uncorrupted"), "nothing injected."),
        ("Subgroup", name("noise_only"), "only noise injected, any amount."),
        ("Subgroup", name("cancelling"), "noise and gains injected, D < 0: removing the gains alone makes recovery worse."),
        ("Subgroup", name("small"), f"gains injected, 0 ≤ D ≤ {r}."),
        ("Group", name(EXCEPTION), "gain only with D < 0: unexpected, kept visible pending investigation."),
        ("Pipeline Result", "Success / Partial success / Failure",
         "evaluation label, not the pipeline's QA score. Improvement expected: Solved → Success, Improved → Partial "
         "success, Similar/Degraded → Failure. No improvement expected: Degraded → Failure, otherwise Success."),
        ("Recovery", name("solved"), f"Improvement expected: remaining ≤ {r}, whatever the reduction (checked first). "
                                     f"Other groups: change < −{t} and remaining ≤ {r}."),
        ("Recovery", name("improved"), f"change < −{t} and remaining > {r}."),
        ("Recovery", name("similar"), f"|change| ≤ {t}."),
        ("Recovery", name("degraded"), f"change > {t}."),
        ("Recovery progress (%)", "100 (before − after) / D = 100 (1 − remaining / D)",
         "progress toward the matched reference for cases with D > 0: 0% unchanged, 100% after = reference, negative = "
         "worse, above 100% = better than the reference (never clamped). Not defined for D ≤ 0 (controls, noise only, "
         f"Cancelling, exceptions): counted as not applicable. An Improvement expected case is Solved from "
         f"100 (1 − {r} / D)% on, so the cutoff depends on D (shown per point). A benchmark, not proof of recovered "
         "antenna gains. Small D (< 0.01) amplifies the percentage and is flagged."),
        ("Recovery progress (%)", "boxplot, two swarms",
         "same cases, values and point positions in all three; one shared linear axis (dashed lines at 0% and 100%). "
         "Box: median, quartiles (linear interpolation), whiskers to the most extreme values within 1.5 IQR, hollow "
         "outliers. Swarms: one point per case, colored by Pipeline Result (Success teal, Partial success blue, Failure "
         "vermilion) and by Recovery (Solved teal, Improved blue, Similar gray, Degraded vermilion); hollow = "
         "accounting state. Hover for details, click a point to show that case below. Zoom −100…200% is optional; the "
         "full range is the default."),
        ("Image", "RMS, SNR",
         "restored CLEAN images on fixed regions of the reference geometry: RMS = standard deviation over the off-source "
         "annulus (Jy/beam), SNR = (peak in the true-source ellipse − annulus mean) / RMS; same units/beam/geometry required."),
        ("Image", name("cleaner"), "both better, or one better and the other similar."),
        ("Image", name("similar"), "both similar."),
        ("Image", name("worse"), "both worse, or one worse and the other similar."),
        ("Image", name("mixed"), "one better, one worse."),
        ("Run", name("applied"), "a final gain correction was retained (a later failed trial does not override it)."),
        ("Run", f"Blocked — {name('low_initial_snr')}", "initial S/N checks prevented correction trials."),
        ("Run", f"Blocked — {name('solution_lowers_snr')}",
         "trials failed the S/N checks and no correction remains (incl. a revoked provisional first interval)."),
        ("Run", f"Blocked — {name('empty_model')} / {name('other')} / {name('reason_unknown')}",
         "other evidence (e.g. RMS increase only, no valid solutions), shown only when present."),
        ("Accounting", "unavailable, error, unknown, pending",
         "missing measurement, failed execution, unknown decision, not run yet: never scientific outcomes."),
    ]
    return ("<h2>Table 0. Glossary</h2><div class=scroll><table><tr><th>Dimension</th><th>Category</th><th>Meaning</th></tr>"
            + "".join(f"<tr><td>{a}</td><td>{htmllib.escape(b)}</td><td>{htmllib.escape(c)}</td></tr>" for a, b, c in rows)
            + "</table></div>")


def progress_text(c):
    """Numeric progress of the case table (and why there is none)."""
    if c.get("progress_status") == "applicable":
        return (f"{c['progress_pct']:.1f}%<br><small>Solved from {c['progress_required_pct']:.1f}%"
                + ("<br>small D: amplified" if c.get("progress_small_D") else "") + "</small>")
    return "<small>n/a (D ≤ 0)</small>" if c.get("progress_status") == "not_applicable" else "<small>–</small>"


def write_report(output, summaries, rows, runs, filename, eps_rms=EPSILON_RMS, eps_snr=EPSILON_SNR):
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    esc = htmllib.escape
    out = ["<!doctype html><meta charset=utf-8><title>VLA pipeline QA report</title><style>body{font-family:"
           "sans-serif;max-width:1600px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;margin:0}"
           ".scroll{overflow-x:auto;margin:.6em 0}td,th{border:1px solid #ccc;padding:3px 7px;text-align:left;"
           "vertical-align:top}th{background:#f3f3f3}td.n{text-align:right}tr.old td{color:#999}.bar{display:flex;"
           "align-items:center;gap:4px;font-size:12px;cursor:pointer;line-height:15px}.bar .lab{width:170px}"
           ".fill{display:inline-block;height:11px}.badge{padding:1px 6px;border-radius:3px;font-size:12px;"
           "white-space:nowrap}.partial{color:#D55E00;font-weight:bold}details{font-size:12px}small{color:#555}"
           ".grp{cursor:pointer;text-decoration:underline dotted}.filters label{margin-right:.6em;white-space:nowrap}"
           ".pzoom{display:none}body.zoomed .pzoom{display:block}body.zoomed .pfull{display:none}svg.prog circle[data-k]"
           "{cursor:pointer}#tip{position:fixed;display:none;background:#fff;border:1px solid #888;padding:4px 6px;"
           "font-size:12px;max-width:360px;pointer-events:none;z-index:9}.lg{display:inline-block;width:9px;height:9px;"
           "border-radius:50%;margin:0 2px 0 6px}.lg.o{border:1px solid #555}"
           "</style><div id=tip></div>",
           f"<h1>VLA pipeline QA report</h1><p>{len(runs)} run(s): {', '.join(esc(r.name) for r in runs)}. "
           f"Labels: {V5_VERSION}. Target self-calibration on simulated residual corruptions, not the full "
           "calibrator-transfer VLA pipeline. Legacy v3 My Criteria labels are kept only in the exported evidence.</p>",
           glossary(eps_rms, eps_snr)]
    if any(s["superseded"] for s in summaries):
        out.append("<p><b>Grey rows</b>: B flagging_off without the calonly model-write fix (superseded, reference "
                   "only).</p>")
    # one count scale per dimension over all tables
    scale = {key: max([1] + [n for s in summaries for st in [s["table1"]] for _, _, n, _ in entries(key, st)])
             for key, _, _ in DIMENSIONS}
    full = progress_range(summaries)
    cases_js = list({p["id"]: p for s in summaries for p in s["table1"]["progress"]["points"]}.values())
    key_of = {p["id"]: k for k, p in enumerate(cases_js)}
    dot = lambda k, hollow=False: f"<span class='lg{' o' if hollow else ''}' style='background:{COLORS.get(k, 'none')}'></span>"
    legend = ("<br><small>by Pipeline Result:" + "".join(dot(k) + name(k) for k in RESULTS) + dot("", True) + "other"
              + "<br>by Recovery:" + "".join(dot(k) + name(k) for k in RECOVERY) + dot("", True) + "other</small>")
    head = "".join(f"<th>{h}</th>" + (f"<th>Recovery progress (%){legend}</th>" if key == "recovery" else "")
                   for key, h, _ in DIMENSIONS)

    def cells(st, base, no_partial=False):
        return "".join("<td>" + "".join(bar(label, n, scale[key], color, {**base, **part})
                                        for label, part, n, color in entries(key, st, no_partial)) + "</td>"
                       + (progress_cell(st["progress"], full, key_of) if key == "recovery" else "")
                       for key, _, _ in DIMENSIONS)

    def runtime(st):
        r = st["runtime"]
        return (f"{f(r['median_min'], 1)} min (range {f(r['min_min'], 1)}–{f(r['max_min'], 1)}), "
                f"{f(r['recipe_hours'], 2)} recipe-h, {r['timed']} timed" if r["timed"] else "no timed recipe")

    def cover(st):
        partial = " <span class=partial>partial</span>" if st["finished"] < st["planned"] else ""
        return (f"{st['planned']} planned / {st['finished']} finished / {st['available']} eC available / "
                f"{st['image_available']} image available{partial}")

    def label(text, flt, st):
        return (f"<b class=grp tabindex=0 role=button data-filter='{esc(json.dumps(flt))}'>{esc(text)}</b><br>"
                f"<small>{cover(st)}<br>{runtime(st)}</small>")

    out.append(f"<p><button id=zoom>Recovery progress axis: zoom to {PROGRESS_ZOOM[0]:g}…{PROGRESS_ZOOM[1]:g}%</button> "
               f"<small>Default: full range {full[0]:.0f}…{full[1]:.0f}% shared by every summary row.</small></p>")
    for s in summaries:
        base = dict(run=s["run_id"], config=s["config"])
        old = " class=old" if s["superseded"] else ""
        elapsed = (f"known elapsed {f(s['elapsed_hours'], 2)} h (sum of the run's recorded --execute calls, whole "
                   "cases and all recipes of the run" + (f", parallel {'/'.join(map(str, s['parallel']))}, CPUs "
                                                          f"{'/'.join(s['cpus'])}" if s["parallel"] else "") + ")"
                   if s["elapsed_hours"] is not None else "elapsed not recorded")
        out.append(f"<h2>{esc(s['config'])}</h2><p><small>run {esc(s['run_id'])} · {elapsed}. Runtime per row: the "
                   "recipe's own CASA process (median, range) and summed recipe-hours, not elapsed time. Click a bar "
                   "to list exactly its cases, a row label for the whole row.</small></p>")
        out.append(f"<h3>Table 1. All cases</h3><div class=scroll><table><tr><th>Config</th>{head}</tr>"
                   f"<tr{old}><td>{label(s['config'], base, s['table1'])}</td>{cells(s['table1'], base)}</tr>"
                   "</table></div>")
        out.append(f"<h3>Table 2. By initial group</h3><div class=scroll><table><tr><th>Group</th>{head}</tr>")
        for g, st in s["table2"].items():
            flt = {**base, "group": g}
            out.append(f"<tr{old}><td>{label(name(g), flt, st)}</td>"
                       f"{cells(st, flt, no_partial=g != 'improvement_expected')}</tr>")
        out.append("</table></div><p><small>Partial success applies only to Improvement expected.</small></p>"
                   f"<h3>Table 3. No improvement expected, by subgroup</h3><div class=scroll><table><tr><th>Subgroup</th>"
                   f"{head}</tr>")
        for sg, st in s["table3"].items():
            flt = {**base, "group": "no_improvement_expected", "subgroup": sg}
            out.append(f"<tr{old}><td>{label(name(sg), flt, st)}</td>{cells(st, flt, no_partial=True)}</tr>")
        out.append("</table></div>")

    # Cases: filters and table
    def select(key, text, values):
        return (f"<label>{text} <select data-key={key}><option value=''>all</option>"
                + "".join(f"<option value='{esc(v)}'>{esc(t)}</option>" for v, t in values) + "</select></label> ")
    present = lambda key: list(dict.fromkeys(str(c.get(key)) for r in rows for c in r["cases"] if c.get(key) is not None))
    ordered = lambda key, cats: [(k, name(k)) for k in list(cats) + [x for x in present(key) if x not in cats]]
    out.append("<h2>Cases</h2><p class=filters>"
               + select("run", "Run", list(dict.fromkeys((s["run_id"], s["run"]) for s in summaries)))
               + select("config", "Configuration", [(c, c) for c in dict.fromkeys(s["config"] for s in summaries)])
               + select("group", "Group", ordered("group", GROUPS + (EXCEPTION, "unavailable")))
               + select("subgroup", "Subgroup", ordered("subgroup", SUBGROUPS))
               + select("result", "Pipeline Result", ordered("result", RESULTS))
               + select("recovery", "Recovery", ordered("recovery", RECOVERY))
               + select("image", "Image", ordered("image", IMAGE))
               + select("application", "Run outcome", ordered("application", ("applied", "blocked")))
               + select("reason", "Blocked reason", ordered("blocked_reason", BLOCKED_REASONS))
               + "<button id=reset>reset</button> <b id=count></b> <span id=active></span></p>"
               "<p><small><b>Corruption</b>: injected kinds; d_k = eC(case) − eC(same case with kind k removed), archived "
               "removal damage (per-kind values do not add up to D). <b>Images</b>: restored CLEAN image used for eC, "
               "before (archived case) and after the recipe, one shared linear scale per case (±99.5th percentile of "
               "|pixel| over both, shown under the after image); click to enlarge. r_RMS = (RMS before − after) / before, "
               "r_SNR = (SNR after − before) / before: positive = better.</small></p>"
               "<div class=scroll><table id=cases><tr><th>Configuration</th><th>Source / case</th><th>Corruption</th>"
               "<th>Group</th><th>Pipeline Result</th><th>Recovery</th><th>Recovery progress (%)</th><th>Image</th>"
               "<th>Run</th>"
               "<th>eC before → after</th><th>Reference eC / D / remaining</th><th>CLEAN before</th><th>CLEAN after</th>"
               "<th>RMS before → after</th><th>SNR before → after</th><th>Flux error</th><th>Recipe runtime (min)</th>"
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
            group = (f"{esc(name(c['group']))}" + (f"<br>{esc(name(c['subgroup']))}" if c.get("subgroup") else "")
                     + f"<br><small>{esc(c.get('group_reason') or '')}</small>")
            run_cell = badge(c["application"]) + (f"<br><small>{esc(name(c['blocked_reason']))}</small>"
                                                  if c.get("blocked_reason") else "")
            rec_note = f"<br><small>{esc(c['recovery_reason'])}</small>" if c.get("recovery_reason") else ""
            image_note = (f"<br><small>{esc(c['image_reason'])}</small>" if c.get("image_reason") else
                          f"<br><small>r_RMS {c['r_rms']:+.3f}<br>r_SNR {c['r_snr']:+.3f}</small>"
                          if c.get("r_rms") is not None and c.get("r_snr") is not None else "")
            attrs = dict(run=row["run_id"], config=row["config"], group=c["group"], subgroup=c.get("subgroup") or "",
                         result=c["result"], recovery=c["recovery"], image=c.get("image"), application=c["application"],
                         reason=c.get("blocked_reason") or "", id=c["id"])
            out.append(
                "<tr " + " ".join(f"data-{k}='{esc(str(v))}'" for k, v in attrs.items())
                + f"{' class=old' if c.get('superseded') else ''}><td>{esc(row['config'])}</td>"
                f"<td>{esc(c['label'])}<br><small>{c['case']}</small></td><td>{corruption_cell(c)}</td>"
                f"<td>{group}</td><td>{badge(c['result'])}</td><td>{badge(c['recovery'])}{rec_note}</td>"
                f"<td class=n>{progress_text(c)}</td>"
                f"<td>{badge(c.get('image'))}{image_note}</td><td>{run_cell}</td>"
                f"<td class=n>{f(c.get('eC_before'))} → {f(c.get('eC_after'))}<br>change {f(c.get('change'))}</td>"
                f"<td class=n>{f(c.get('reference'))}<br>D {f(D, 4)}<br>remaining {f(c.get('remaining'))}"
                f"<br><small>{esc(c.get('reference_id') or c.get('reference_reason') or '–')}</small></td>"
                f"{image_cell(c, 'before')}{image_cell(c, 'after')}"
                f"<td class=n>{g(c.get('rms_before'))} → {g(c.get('rms_after'))}<br><small>r_RMS {f(c.get('r_rms'))}</small></td>"
                f"<td class=n>{g(c.get('snr_before'))} → {g(c.get('snr_after'))}<br><small>r_SNR {f(c.get('r_snr'))}</small></td>"
                f"<td class=n>{f(c.get('flux_before'))} → {f(c.get('flux_after'))}</td>"
                f"<td class=n>{f(c['seconds'] / 60 if c.get('seconds') is not None else None, 1)}</td>"
                f"<td>{esc(c['description'])}{details}</td></tr>")
    out.append("</table></div><script>"
               "const sel=[...document.querySelectorAll('select[data-key]')],rows=[...document.querySelectorAll("
               "'#cases tr[data-id]')];"
               "const CASES=" + json.dumps([{k: p[k] for k in POINT_FIELDS} for p in cases_js]).replace("</", "<\\/")
               + ";let pinned=null;"
               "function apply(){let n=0;for(const tr of rows){const ok=sel.every(s=>!s.value||tr.dataset[s.dataset.key]"
               "===s.value)&&(!pinned||tr.dataset.id===pinned);tr.style.display=ok?'':'none';n+=ok}"
               "document.getElementById('count').textContent=n+' / '+rows.length+' cases';"
               "document.getElementById('active').textContent=pinned?'case='+pinned:sel.filter(s=>s.value).map(s=>"
               "s.dataset.key+'='+s.value).join(', ')}"
               "function setFilter(f){pinned=null;for(const s of sel){const v=f[s.dataset.key]||'';if(v&&![...s.options]"
               ".some(o=>o.value===v))s.add(new Option(v,v));s.value=v}apply();document.getElementById('cases')"
               ".scrollIntoView()}"
               "function pick(k){for(const s of sel)s.value='';pinned=CASES[k].id;apply();"
               "document.getElementById('cases').scrollIntoView()}"
               "const fx=(x,d=4)=>x==null?'–':(+x).toFixed(d),tip=document.getElementById('tip');"
               "function detail(p){return p.label+' ('+p.case+')\\nPipeline Result: '+p.result+' · Recovery: '+p.recovery+"
               "'\\neC before '+fx(p.eC_before)+' → after '+fx(p.eC_after)+', reference '+fx(p.reference)+"
               "'\\nD '+fx(p.D)+(p.progress_small_D?' (small D: progress amplified)':'')+', remaining '+fx(p.remaining)+"
               "'\\nprogress '+fx(p.progress_pct,1)+'% (Solved from '+fx(p.progress_required_pct,1)+'%)'}"
               "document.addEventListener('mouseover',e=>{const k=e.target.dataset&&e.target.dataset.k;if(k===undefined)"
               "{tip.style.display='none';return}tip.innerText=detail(CASES[+k]);tip.style.display='block';"
               "tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'});"
               "document.addEventListener('click',e=>{const k=e.target.dataset&&e.target.dataset.k;if(k!==undefined)pick(+k)});"
               "const zb=document.getElementById('zoom'),zt=zb.textContent;zb.onclick=()=>{const z=document.body.classList"
               ".toggle('zoomed');zb.textContent=z?'Recovery progress axis: reset to full range':zt};"
               "for(const s of sel)s.onchange=()=>{pinned=null;apply()};document.getElementById('reset').onclick=()=>setFilter({});"
               "for(const b of document.querySelectorAll('[data-filter]')){const go=()=>setFilter(JSON.parse(b.dataset.filter));"
               "b.onclick=go;b.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}}}"
               "apply()</script>")
    (output / filename).write_text("\n".join(out))


def build(runs, output, filename="comparison.html", eps_rms=EPSILON_RMS, eps_snr=EPSILON_SNR,
          tol_change=TOL_CHANGE, tol_ref=TOL_REFERENCE):
    eps_rms, eps_snr = check_tolerance("epsilon_rms", eps_rms), check_tolerance("epsilon_snr", eps_snr)
    tol_change, tol_ref = check_tolerance("tol_change", tol_change), check_tolerance("tol_ref", tol_ref)
    rows = [row for run in runs for row in load(run, tol_change, tol_ref)]
    rows.sort(key=lambda r: r["key"])
    paths_of = {}
    for row in rows:                        # identity = absolute run path; file names stay readable when unique
        paths_of.setdefault(row["run"], set()).add(row["run_id"])
    for row in rows:
        if len(paths_of[row["run"]]) > 1:
            row["tag"] = f"{row['run']}-{hashlib.sha256(row['run_id'].encode()).hexdigest()[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    add_image_metrics(rows, output / "image_metrics", eps_rms, eps_snr)
    thresholds = dict(tol_change=tol_change, tol_reference=tol_ref, epsilon_rms=eps_rms, epsilon_snr=eps_snr)
    for row in rows:      # retained evidence per case (after the image metrics), linked from Description
        for c in row["cases"]:
            if "evidence" in c:
                c["evidence_file"] = f"descriptions/{row['tag']}__{c['case']}__{safe(row['recipe'])}.json"
                write_json(output / c["evidence_file"], dict(
                    run=row["run"], run_path=row["run_id"], configuration=row["config"], case=c["case"], label=c["label"],
                    description=c["description"], v5_version=V5_VERSION, thresholds=thresholds,
                    v5={k: c.get(k) for k in V5_FIELDS},
                    legacy_v3=dict(c["legacy"], criteria_version=CRITERIA_VERSION), **c.pop("evidence")))

    images = output / "images"
    images.mkdir(exist_ok=True)
    rendered = reused = 0
    for row in rows:                        # before/after CLEAN thumbnails, one shared color scale per case
        for c in row["cases"]:
            before, after = c.pop("clean_before", None), c.pop("clean_after", None)
            if not (before and after):
                continue
            stem = f"images/{row['tag']}__{c['case']}__{safe(row['recipe'])}"
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
        v5_version=V5_VERSION, thresholds=thresholds, image_metric_version=IMAGE_METRIC_VERSION,
        legacy_criteria_version=CRITERIA_VERSION, runs=[str(r.resolve()) for r in runs], summaries=summaries,
        cases={f"{r['run_id']} | {r['config']}": [{k: v for k, v in c.items() if k != "notes"} for c in r["cases"]]
               for r in rows}))
    write_report(output, summaries, rows, runs, filename, eps_rms, eps_snr)
    print("REPORT", (output / filename).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="vla_pipeline_qa output folders")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epsilon-rms", type=float, default=EPSILON_RMS,
                    help="relative RMS change tolerance of the Image dimension (default 0.05 = 5%%)")
    ap.add_argument("--epsilon-snr", type=float, default=EPSILON_SNR,
                    help="relative SNR change tolerance of the Image dimension (default 0.05 = 5%%)")
    args = ap.parse_args()
    build(args.runs, args.output, eps_rms=args.epsilon_rms, eps_snr=args.epsilon_snr)


if __name__ == "__main__":
    main()
