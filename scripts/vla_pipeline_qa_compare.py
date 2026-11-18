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
                progress=progress_stats(cases), ec=ec_stats(cases))


# ---------------------------------------------------------------- recovery progress plots
# [USER] 2026-11-18: per summary row, progress = 100 (before - after) / D for the cases of that row with D > 0.
# [USER] 2026-11-18 (later): static matplotlib images instead of the hand-drawn SVG swarms. One PNG per row with two
# panels (by Pipeline Result, by Recovery): per category a Tukey box and one jittered point per case. Every case is
# drawn; symlog axis (linear within +-100%) shared by all rows, so extreme small-D values stay on the plot.

POINT_FIELDS = ("id", "label", "case", "result", "recovery", "eC_before", "eC_after", "reference", "D", "remaining",
                "progress_pct", "progress_required_pct", "progress_small_D")
NO_PROGRESS = "Not applicable: no positive initial joint gain damage"
LINTHRESH = 100.                        # symlog: linear within +-100 %, logarithmic beyond


def progress_stats(cases):
    """Eligible points (progress defined: finite metrics and D > 0) sorted by (progress, id), with the denominators."""
    points = sorted(({k: c.get(k) for k in POINT_FIELDS} for c in cases if c.get("progress_status") == "applicable"),
                    key=lambda p: (p["progress_pct"], p["id"]))
    na = sum(c.get("progress_status") == "not_applicable" for c in cases)
    return dict(points=points, total=len(cases), plotted=len(points), not_applicable=na,
                missing=len(cases) - len(points) - na)


def progress_range(summaries):
    """One common y range over every summary row (0 and 100 always included, nothing clipped)."""
    values = [p["progress_pct"] for s in summaries for p in s["table1"]["progress"]["points"]]
    lo, hi = min(values + [-LINTHRESH]), max(values + [2 * LINTHRESH])
    pad = lambda v: v * 1.5 if abs(v) > LINTHRESH else v * 1.15
    return pad(lo), pad(hi)


def progress_groups(points, dim, cats):
    """[(category, points)] for one panel: All first, then the categories present in the given order."""
    present = [k for k in cats if any(p[dim] == k for p in points)]
    present += sorted({str(p[dim]) for p in points} - set(present))
    return [("all", points)] + [(k, [p for p in points if str(p[dim]) == k]) for k in present]


def progress_png(st, path, ylim):
    """Two panels (by Pipeline Result, by Recovery); per category a box (1.5 IQR whiskers) and every case as a point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    points = st["points"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=110, sharey=True)
    for ax, (dim, title, cats) in zip(axes, (("result", "by Pipeline Result", RESULTS),
                                             ("recovery", "by Recovery", RECOVERY))):
        groups = progress_groups(points, dim, cats)
        for x, (k, pts) in enumerate(groups):
            v = np.array([p["progress_pct"] for p in pts], float)
            jitter = np.random.default_rng(x).uniform(-.28, .28, v.size)     # deterministic
            if k == "all":
                ax.scatter(x + jitter, v, s=6, color="#888", alpha=.35, lw=0, zorder=2)
            elif k in cats and k in COLORS:
                ax.scatter(x + jitter, v, s=8, color=COLORS[k], alpha=.6, lw=0, zorder=2)
            else:                                                            # accounting state: hollow
                ax.scatter(x + jitter, v, s=10, facecolors="none", edgecolors="#444", lw=.6, zorder=2)
            ax.boxplot(v, positions=[x], widths=.6, whis=1.5, showfliers=False, patch_artist=True, zorder=3,
                       boxprops=dict(facecolor="none", edgecolor="#222", lw=1.1),
                       medianprops=dict(color="#000", lw=1.8), whiskerprops=dict(color="#222", lw=1),
                       capprops=dict(color="#222", lw=1))
        ax.set_xticks(range(len(groups)), [f"{'All' if k == 'all' else name(k)}\nn={len(p)}" for k, p in groups],
                      fontsize=8)
        ax.set_xlim(-.6, len(groups) - .4)
        for y in (0, 100):
            ax.axhline(y, color="#333", lw=.8, ls="--", zorder=1)
        ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=2.5)
        ax.set_ylim(*ylim)
        ticks = [t for t in (-1e5, -1e4, -1e3, -100, -50, 0, 50, 100, 1e3, 1e4, 1e5, 1e6) if ylim[0] <= t <= ylim[1]]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks], fontsize=8)
        ax.grid(axis="y", color="#e6e6e6", lw=.6, zorder=0)
        ax.set_title(title, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("recovery progress (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


EC_FIELDS = ("id", "result", "recovery", "eC_before", "eC_after")


def finite(x):
    return x is not None and math.isfinite(x)


def ec_stats(cases):
    """Cases with finite measured eC before and after (any group), for the before/after scatter."""
    points = [{k: c.get(k) for k in EC_FIELDS} for c in cases if finite(c.get("eC_before")) and finite(c.get("eC_after"))]
    return dict(points=points, total=len(cases), plotted=len(points))


def ec_range(summaries):
    """One common square log range over every summary row (eC spans decades; a few cases reach eC > 1)."""
    values = [v for s in summaries for p in s["table1"]["ec"]["points"] for v in (p["eC_before"], p["eC_after"]) if v > 0]
    return min(values + [.1]) / 1.3, max(values + [1.]) * 1.3


def ec_png(st, path, lim, tol=TOL_CHANGE):
    """eC before (x) vs after (y) on log axes, panels colored by Pipeline Result and by Recovery. y = x and the
    +-tol band (absolute, so it widens toward small eC on log axes). Larger categories are drawn first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = st["points"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1), dpi=110, sharex=True, sharey=True, layout="constrained")
    grid = np.geomspace(*lim, 400)
    for ax, (dim, title, cats) in zip(axes, (("result", "by Pipeline Result", RESULTS),
                                             ("recovery", "by Recovery", RECOVERY))):
        ax.plot(grid, grid, color="#222", lw=.9, zorder=1)
        for sign in (-1, 1):
            band = grid + sign * tol
            ax.plot(grid[band > 0], band[band > 0], color="#555", lw=.7, ls="--", zorder=1)
        groups = progress_groups(points, dim, cats)[1:]
        for k, pts in sorted(groups, key=lambda g: -len(g[1])):
            x = [p["eC_before"] for p in pts]
            y = [p["eC_after"] for p in pts]
            label = f"{name(k)} ({len(pts)})"
            if k in cats and k in COLORS:
                ax.scatter(x, y, s=8, color=COLORS[k], alpha=.55, lw=0, zorder=2, label=label)
            else:                                                            # accounting state: hollow
                ax.scatter(x, y, s=10, facecolors="none", edgecolors="#444", lw=.6, zorder=2, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(f"{name(k)} ({len(p)})") for k, p in groups]          # legend in category order
        ax.set_xlabel("eC before (archived)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(color="#eee", lw=.6, zorder=0)
        ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7, loc="upper left",
                  frameon=False, markerscale=1.5, handletextpad=.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("eC after recipe", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def ec_cell(st, img):
    head = f"<small>plotted {st['plotted']} / {st['total']} (finite eC before and after)</small>"
    if not st["plotted"]:
        return f"<td class=progress>{head}<br><i>No case with measured eC before and after</i></td>"
    return (f"<td class=progress>{head}<br><a href='{htmllib.escape(img)}'><img class=prog "
            f"src='{htmllib.escape(img)}' alt='eC before vs after, {st['plotted']} cases'></a></td>")


def progress_cell(st, img):
    """Summary cell: coverage line and the row's progress image (or why there is none)."""
    head = (f"<small>plotted {st['plotted']} / {st['total']} · not applicable {st['not_applicable']} · "
            f"missing/invalid {st['missing']}</small>")
    if not st["plotted"]:
        why = (NO_PROGRESS if st["not_applicable"] and not st["missing"] else
               f"No plotted case: {st['missing']} with missing/invalid metrics"
               + (f"; {st['not_applicable']} not applicable (no positive initial joint gain damage)"
                  if st["not_applicable"] else "") if st["total"] else "No cases in this row")
        return f"<td class=progress>{head}<br><i>{htmllib.escape(why)}</i></td>"
    return (f"<td class=progress>{head}<br><a href='{htmllib.escape(img)}'><img class=prog "
            f"src='{htmllib.escape(img)}' alt='recovery progress, {st['plotted']} cases'></a></td>")


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
        ("Recovery progress (%)", "plot per summary row",
         "left panel by Pipeline Result, right panel by Recovery, each with an All column first. Per column a box "
         "(median, quartiles, whiskers to the most extreme values within 1.5 IQR) and every case as one jittered point "
         "(Success/Solved teal, Partial success/Improved blue, Failure/Degraded vermilion, Similar gray, All gray; "
         "hollow = accounting state). Same y axis in every row: linear within ±100%, logarithmic beyond, so the few "
         "extreme small-D values stay visible without crushing 0…100%. Dashed lines at 0% and 100%. Click to enlarge; "
         "per-case values are in the case table."),
        ("eC before vs after", "plot per summary row",
         "every case with measured eC before (archived) and after the recipe, any group; same two colorings. Solid "
         f"line y = x (unchanged); dashed lines y = x ± {t} (inside the band: Similar; the band is absolute, so it looks "
         "wider at small eC on the log axes). Below the band = eC reduced, above = worse. Same square log axis in "
         "every row; larger categories drawn first."),
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
           "img.prog{width:560px;display:block}"
           "</style>",
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
    ylim, eclim = progress_range(summaries), ec_range(summaries)
    (output / "images").mkdir(parents=True, exist_ok=True)
    head = "".join(f"<th>{h}</th>" + ("<th>Recovery progress (%)</th><th>eC before vs after</th>"
                                      if key == "recovery" else "") for key, h, _ in DIMENSIONS)

    def image(kind, st, base, draw):
        """One image per summary row and plot kind, named by the row's exact filter."""
        if not st["plotted"]:
            return None
        ident = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:10]
        img = f"images/{kind}__{safe(base.get('subgroup') or base.get('group') or 'all')}__{ident}.png"
        draw(st, output / img)
        return img

    def cells(st, base, no_partial=False):
        prog = image("progress", st["progress"], base, lambda s, path: progress_png(s, path, ylim))
        ec = image("ec", st["ec"], base, lambda s, path: ec_png(s, path, eclim))
        return "".join("<td>" + "".join(bar(label, n, scale[key], color, {**base, **part})
                                        for label, part, n, color in entries(key, st, no_partial)) + "</td>"
                       + (progress_cell(st["progress"], prog) + ec_cell(st["ec"], ec) if key == "recovery" else "")
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
               "function apply(){let n=0;for(const tr of rows){const ok=sel.every(s=>!s.value||tr.dataset[s.dataset.key]"
               "===s.value);tr.style.display=ok?'':'none';n+=ok}"
               "document.getElementById('count').textContent=n+' / '+rows.length+' cases';"
               "document.getElementById('active').textContent=sel.filter(s=>s.value).map(s=>"
               "s.dataset.key+'='+s.value).join(', ')}"
               "function setFilter(f){for(const s of sel){const v=f[s.dataset.key]||'';if(v&&![...s.options]"
               ".some(o=>o.value===v))s.add(new Option(v,v));s.value=v}apply();document.getElementById('cases')"
               ".scrollIntoView()}"
               "for(const s of sel)s.onchange=apply;document.getElementById('reset').onclick=()=>setFilter({});"
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
