"""Training-only Dataset v2 sampling and recovery coverage report."""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.create_dataset_v2 import file_sha, configuration, indexed_sources, original_flux_spec
from scripts import create_dataset_v1 as g
from scripts.dataset_v2 import TYPES, join_rows, sample_vectors
from scripts.preprocessing.schema import atomic_write_json

COLORS = ("#0b6fa4", "#d46b39", "#8a58a3")
COLORS_MASK = ("#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7B61A8", "#C49A00", "#222222")
MASK_LABELS = ("control", "amp", "phase", "amp+phase", "noise", "amp+noise", "phase+noise", "all three")


def _weighted_curve(values, weights):
    order = np.argsort(values)
    x = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    return x, np.cumsum(w) / w.sum()


def report(dataset, partial=False):
    dataset = dataset.resolve()
    plan = json.loads((dataset / "plan.json").read_text())
    plan_hash = file_sha(dataset / "plan.json")
    if plan["split"] != "train":
        raise ValueError("Coverage report reads training plans only")
    records = {}
    failed = []
    for case in plan["cases"]:
        p = dataset / "cases" / f"{case['id']}.json"
        if p.exists():
            record = json.loads(p.read_text())
            if record["case"] != case or record["plan_sha256"] != plan_hash:
                raise ValueError(f"Case/plan mismatch: {p}")
            records[case["id"]] = record
        elif (dataset / "failures" / f"{case['id']}.json").exists():
            failed.append(case["id"])
    complete = [row for row in plan["rows"] if row["full"] in records and
                (row["removal"] is None or row["removal"] in records)]
    if len(complete) != len(plan["rows"]) and not partial:
        raise ValueError("Training coverage is incomplete; pass --partial for read-only refresh")
    rows = join_rows(dict(rows=complete), records)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate supervised row IDs")
    assets = dataset / "coverage_assets"
    assets.mkdir(exist_ok=True)
    planned_masks = Counter(r["mask"] for r in plan["rows"])
    committed_masks = Counter(r["mask"] for r in rows)
    with (dataset / "coverage_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dimension", "value", "planned_rows", "committed_rows", "failed_rows"))
        for field in ("mask", "source", "slot"):
            planned = Counter(r[field] for r in plan["rows"])
            done = Counter(r[field] for r in rows)
            for key in sorted(planned):
                relevant = [r for r in plan["rows"] if r[field] == key]
                failures = sum(r["full"] in failed or r["removal"] in failed for r in relevant)
                writer.writerow((field, key, planned[key], done[key], failures))
        vectors = {v["id"]: v for v in plan["vectors"]}
        ranks = {}
        for source in plan["prepared_sources"]:
            eligible = sorted(source["norms"], key=lambda a: (source["norms"][a]["V_Ak_over_sigma_L2"], int(a)))
            ranks[source["id"]] = {int(a): ("low" if i < len(eligible) / 3 else
                                               "middle" if i < 2 * len(eligible) / 3 else "high")
                                    for i, a in enumerate(eligible)}
        for vector in vectors.values():
            vector["antenna_contribution"] = (ranks[vector["source"]][vector["antenna"]]
                                              if vector["antenna"] is not None else "none")
        for field in ("amp_branch", "phase_sign", "antenna"):
            planned = Counter(v[field] for v in vectors.values())
            done_ids = {r["vector"] for r in rows if r["vector"]}
            done = Counter(vectors[vid][field] for vid in done_ids)
            for key in sorted(planned, key=str):
                writer.writerow((field, key, planned[key], done[key], ""))
        for field in ("antenna_contribution",):
            planned = Counter(v[field] for v in vectors.values())
            done_ids = {r["vector"] for r in rows if r["vector"]}
            done = Counter(vectors[vid][field] for vid in done_ids)
            for key in sorted(planned):
                writer.writerow((field, key, planned[key], done[key], ""))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(8)
    ax.bar(x - .18, [planned_masks[m] for m in x], .36, label="planned")
    ax.bar(x + .18, [committed_masks[m] for m in x], .36, label="committed")
    ax.set_xticks(x, [format(m, "03b") for m in x]); ax.set_xlabel("Presence mask: noise phase amp bits")
    ax.set_ylabel("Rows"); ax.legend(); fig.tight_layout(); fig.savefig(assets / "masks.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    y = np.asarray([r["y_all"] for r in rows])
    if y.size:
        axes[0].hist(y, bins=30, color=COLORS[0]); axes[1].step(*_weighted_curve(y, np.ones_like(y)), where="post", color=COLORS[0])
    for ax in axes:
        for guide in (.05, .9, 1.): ax.axvline(guide, color="#777", ls=":")
        ax.set_xlabel("y_all")
    axes[0].set_ylabel("Rows"); axes[1].set_ylabel("Empirical CDF")
    fig.tight_layout(); fig.savefig(assets / "total_error.png", dpi=150); plt.close(fig)
    unique_vectors = plan["vectors"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].hist([v["phi_deg"] for v in unique_vectors if v["mask"] & 2], bins=22, color=COLORS[1])
    axes[0, 0].axvline(-180, ls=":", color="#555"); axes[0, 0].axvline(180, ls=":", color="#555")
    axes[0, 0].set_xlabel("Active φ (degrees)")
    def _tidy_log(ax):
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())
    attenuation = [v["A"] for v in unique_vectors if v["amp_branch"] == "attenuation"]
    amplification = [v["A"] for v in unique_vectors if v["amp_branch"] == "amplification"]
    axes[0, 1].hist(attenuation, bins=20, color=COLORS[0]); axes[0, 1].set_xlabel("Attenuation A (linear)")
    ranges = plan["configuration"]["ranges"]
    axes[0, 1].set_xlim(*ranges["amp_attenuation"])
    axes[0, 2].hist(amplification, bins=np.geomspace(*ranges["amp_amplification"], 21), color=COLORS[2]); axes[0, 2].set_xscale("log"); _tidy_log(axes[0, 2]); axes[0, 2].set_xlabel("Amplification A (log)")
    active_noise = [v for v in unique_vectors if v["mask"] & 4]
    noise_lo = min(min(v["sample_bounds"]["noise"]) for v in active_noise) if active_noise else ranges["noise_r"][0]
    axes[1, 0].hist([v["r"] for v in active_noise], bins=np.geomspace(noise_lo, ranges["noise_r"][1], 21), color=COLORS[2]); axes[1, 0].set_xscale("log"); axes[1, 0].set_xlabel("Active r (log)")
    for column, branch in ((1, "attenuation"), (2, "amplification")):
        pair = [v for v in unique_vectors if v["mask"] & 3 == 3 and v["amp_branch"] == branch]
        axes[1, column].scatter([v["phi_deg"] for v in pair], [v["A"] for v in pair], s=10, alpha=.5)
        axes[1, column].set_xlabel("φ (degrees)"); axes[1, column].set_ylabel(f"{branch} A")
    axes[1, 2].set_yscale("log")
    for ax in axes.flat: ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(assets / "parameters.png", dpi=150); plt.close(fig)
    damage_summary = {}
    for k, kind in enumerate(TYPES):
        active = [r for r in rows if r["b"][k]]
        observed = [r for r in active if r["o"][k]]
        d = np.asarray([r["d"][k] for r in observed], float)
        w = np.asarray([1 / r["pi"][k] for r in observed], float)
        damage_summary[kind] = dict(known_active=len(active), observed_active=len(observed),
                                    weight_sum=float(w.sum()), effective_n=float(w.sum()**2 / np.square(w).sum()) if len(w) else 0)
        vector_lookup = {v["id"]: v for v in plan["vectors"]}
        # Amp scatter is split by branch: attenuation is linear, amplification log.
        panels = ([("attenuation", "Attenuation A (linear)", False), ("amplification", "Amplification A (log)", True)]
                  if kind == "amp" else [(None, "φ (degrees)" if kind == "phase" else "r=a/SNR₀ (log)", kind == "noise")])
        fig, axes = plt.subplots(1, 2 + len(panels), figsize=(4.7 * (2 + len(panels)), 3.5))
        if len(d):
            axes[0].hist(d, bins=25, weights=w / w.sum(), color=COLORS[k])
            axes[1].step(*_weighted_curve(d, w), where="post", color=COLORS[k])
            key = {"amp": "A", "phase": "phi_deg", "noise": "r"}[kind]
            for ax, (branch, label, logarithmic) in zip(axes[2:], panels):
                keep = [i for i, r in enumerate(observed) if branch is None or vector_lookup[r["vector"]]["amp_branch"] == branch]
                ax.scatter([vector_lookup[observed[i]["vector"]][key] for i in keep], d[keep],
                           c=[COLORS_MASK[observed[i]["mask"]] for i in keep], s=12, alpha=.75)
                if logarithmic: ax.set_xscale("log"); _tidy_log(ax)
                ax.set_xlabel(label); ax.axhline(0, color="#777", ls=":")
        for ax in axes[:2]: ax.axvline(0, color="#777", ls=":")
        axes[0].set_ylabel("Weighted proportion"); axes[1].set_ylabel("Weighted CDF"); axes[2].set_ylabel(f"Observed d_{kind} (color = mask)")
        fig.tight_layout(); fig.savefig(assets / f"damage_{kind}.png", dpi=150); plt.close(fig)
    # Each original LHS branch group is checked before pooling seeds/sources.
    lhs = []
    for source_plan in plan["source_plans"]:
        groups = defaultdict(list)
        for v in source_plan["vectors"]:
            groups[v["group"]].append(v)
        for group, items in groups.items():
            n = len(items)
            for kind in items[0]["unit_cube"]:
                strata = sorted(int(v["unit_cube"][kind] * n) for v in items)
                lhs.append(dict(source=source_plan["source"], group=group, kind=kind,
                                count=n, occupied=strata, one_per_stratum=strata == list(range(n))))
    if not all(x["one_per_stratum"] for x in lhs):
        raise ValueError("LHS marginal stratum occupancy failed")
    # Achieved total error against each active injected parameter, by mask:
    # shows where in each sampled range rows are small, moderate or saturated.
    vector_lookup = {v["id"]: v for v in plan["vectors"]}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    specs = (("phase", "phi_deg", lambda v: v["mask"] & 2, "φ (degrees)", False, None),
             ("attenuation", "A", lambda v: v["amp_branch"] == "attenuation", "Attenuation A", False, ranges["amp_attenuation"]),
             ("amplification", "A", lambda v: v["amp_branch"] == "amplification", "Amplification A (log)", True, ranges["amp_amplification"]),
             ("noise", "r", lambda v: v["mask"] & 4, "r=a/SNR₀ (log)", True, (noise_lo, ranges["noise_r"][1])))
    for ax, (name, key, keep, label, logarithmic, bounds) in zip(axes, specs):
        for mask in range(1, 8):
            pts = [r for r in rows if r["vector"] and r["mask"] == mask and keep(vector_lookup[r["vector"]])]
            if pts:
                ax.scatter([vector_lookup[r["vector"]][key] for r in pts], [r["y_all"] for r in pts],
                           s=12, alpha=.7, c=COLORS_MASK[mask], label=MASK_LABELS[mask])
        for guide in (.05, .9): ax.axhline(guide, color="#777", ls=":")
        if bounds:
            for bound in bounds: ax.axvline(bound, color="#bd4c34", ls=":", lw=.8)
        if logarithmic: ax.set_xscale("log"); _tidy_log(ax)
        ax.set_xlabel(label); ax.set_title(name); ax.grid(alpha=.15)
    axes[0].set_ylabel("y_all (e_TM,2)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(assets / "y_by_parameter.png", dpi=150); plt.close(fig)
    summary = dict(split="train", partial=len(complete) < len(plan["rows"]),
                   plan_sha256=plan_hash, planned_rows=len(plan["rows"]),
                   committed_rows=len(rows), failed_cases=len(failed),
                   planned_vectors=len(plan["vectors"]),
                   committed_vectors=len({r["vector"] for r in rows if r["vector"]}),
                   mask_counts={str(m): dict(planned=planned_masks[m], committed=committed_masks[m]) for m in range(8)},
                   marginal_presence={kind: sum(r["b"][k] for r in rows) for k, kind in enumerate(TYPES)},
                   observed_damage={kind: sum(r["o"][k] and r["b"][k] for r in rows) for k, kind in enumerate(TYPES)},
                   y_guides={str(g): dict(at_or_below=int((y <= g).sum()), proportion=float((y <= g).mean()) if len(y) else None)
                             for g in (.05, .9, 1.)},
                   damage=damage_summary, lhs=lhs,
                   unit_transform="linear phase/attenuation; logarithmic amplification and r; per source/mask/branch group")
    atomic_write_json(dataset / "coverage_data.json", summary)
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Dataset v2 training coverage</title>",
             "<style>body{font:15px system-ui;max-width:1150px;margin:auto;padding:1.5rem;color:#1d2a33}img{max-width:100%}table{border-collapse:collapse}td,th{padding:.35rem .7rem;border:1px solid #ccd;text-align:right}th:first-child,td:first-child{text-align:left}</style></head><body>",
             "<h1>Dataset v2 training coverage" + (" — partial" if summary["partial"] else "") + "</h1>",
             f"<p>{len(rows)} / {len(plan['rows'])} planned training rows; {summary['committed_vectors']} / {summary['planned_vectors']} parameter vectors. ",
             "Two noise slots share each vector; rows and vectors have different denominators. ",
             "Active damage distributions use observed labels with inverse-observation weights.</p>",
             "<p><a href='coverage_data.json'>Plotted data</a> · <a href='coverage_counts.csv'>Counts CSV</a></p>",
             "<h2>Presence masks</h2><table><tr><th>Mask</th><th>Planned rows</th><th>Committed rows</th><th>Proportion of committed rows</th></tr>"]
    for mask in range(8):
        parts.append(f"<tr><td>{mask:03b}</td><td>{planned_masks[mask]}</td><td>{committed_masks[mask]}</td><td>{committed_masks[mask]/len(rows):.3f}</td></tr>" if rows else
                     f"<tr><td>{mask:03b}</td><td>{planned_masks[mask]}</td><td>0</td><td>—</td></tr>")
    parts.append("</table><img src='coverage_assets/masks.png' alt='Planned and committed mask counts'>")
    parts.append("<h2>Sampled physical parameter vectors</h2><img src='coverage_assets/parameters.png' alt='Parameter distributions and pair projections'>")
    parts.append("<h2>Total error against injected parameters</h2><p>Each point is a committed row, placed at its active parameter "
                 "and colored by injection mask. Dotted horizontal lines mark y=0.05 and 0.9; red dotted vertical lines mark the sampled bounds. "
                 f"Latin-hypercube check: all {len(lhs)} branch-group marginals have one point per stratum (by construction).</p>"
                 "<img src='coverage_assets/y_by_parameter.png' alt='Total error against each injected parameter'>")
    parts.append("<h2>Fully observed total error</h2><img src='coverage_assets/total_error.png' alt='Training total-error distribution and CDF'>")
    for kind in TYPES:
        s = damage_summary[kind]
        parts.append(f"<h2>{html.escape(kind)} damage, active only</h2><p>Observed {s['observed_active']} of {s['known_active']} active rows; "
                     f"weight sum {s['weight_sum']:.1f}; effective n {s['effective_n']:.1f}. Unknown active labels are excluded; absent known zeros are separate.</p>")
        parts.append(f"<img src='coverage_assets/damage_{kind}.png' alt='{kind} weighted damage distribution and parameter scatter'>")
    parts.append(f"<p>Original LHS groups: {len(lhs)} marginal checks; all one point per planned stratum. "
                 "The marginal design does not guarantee coverage of multidimensional cells or damage ranges.</p></body></html>")
    (dataset / "coverage_train.html").write_text("".join(parts))
    print("Coverage:", dataset / "coverage_train.html", flush=True)
    return summary


def preview(config, output, *, smoke=False):
    """Render the training parameter design without CASA preparation."""
    if config["split"] != "train":
        raise ValueError("Sampling preview uses training sources only")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids, _, audit = indexed_sources(config, smoke)
    manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
    sources = [dict(id=sid, **{k: v for k, v in original_flux_spec(
        g._thermal_source(g.THERMAL_DIR, manifest, sid)).items() if k != "sky"}) for sid in ids]
    vectors = [v for src in sources for v in sample_vectors(src["id"], src["nominal_source_snr"], config, "train")]
    groups = defaultdict(list)
    for v in vectors:
        groups[v["source"], v["group"]].append(v)
    for items in groups.values():
        for kind in items[0]["unit_cube"]:
            if sorted(int(v["unit_cube"][kind] * len(items)) for v in items) != list(range(len(items))):
                raise ValueError("LHS marginal stratum mismatch")
    data = dict(configuration=config, source_index=audit, sources=sources, vectors=vectors,
                counts={str(mask): dict(vectors=sum(v["mask"] == mask for v in vectors),
                                        rows=(len(ids) * config["noise_slots"] if mask == 0 else
                                              sum(v["mask"] == mask for v in vectors) * config["noise_slots"]))
                        for mask in range(8)})
    atomic_write_json(output / "sampling_preview.json", data)
    assets = output / "sampling_preview_assets"
    assets.mkdir(exist_ok=True)
    labels = ["control", "amp", "phase", "amp+phase", "noise", "amp+noise", "phase+noise", "all three"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(8), [data["counts"][str(m)]["vectors"] for m in range(8)], color="#0b6fa4")
    ax.set_xticks(range(8), labels, rotation=30, ha="right"); ax.set_ylabel("Unique parameter vectors")
    fig.tight_layout(); fig.savefig(assets / "masks.png", dpi=160); plt.close(fig)
    ranges = config["ranges"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    specs = [("phase", [v["phi_deg"] for v in vectors if v["mask"] & 2],
              [*ranges["phase_negative"], *ranges["phase_positive"]], False),
             ("attenuation", [v["A"] for v in vectors if v["amp_branch"] == "attenuation"], ranges["amp_attenuation"], False),
             ("amplification", [v["A"] for v in vectors if v["amp_branch"] == "amplification"], ranges["amp_amplification"], True),
             ("thermal r", [v["r"] for v in vectors if v["mask"] & 4], ranges["noise_r"], True)]
    for ax, (name, values, bounds, logarithmic) in zip(axes.flat, specs):
        bins = np.geomspace(min(bounds), max(bounds), 25) if logarithmic else 25
        ax.hist(values, bins=bins, color="#0b6fa4")
        for bound in bounds: ax.axvline(bound, color="#bd4c34", ls=":")
        if logarithmic: ax.set_xscale("log")
        ax.set_title(name); ax.set_ylabel("Vectors")
    fig.tight_layout(); fig.savefig(assets / "marginals.png", dpi=160); plt.close(fig)
    for branch in ("attenuation", "amplification"):
        subset = [v for v in vectors if v["amp_branch"] == branch or v["amp_branch"] == "inactive"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        pairs = [("phi_deg", "A", "phase", "A"), ("A", "r", "A", "r"), ("phi_deg", "r", "phase", "r")]
        for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
            for mask in range(1, 8):
                pts = [v for v in subset if v["mask"] == mask]
                ax.scatter([v[xkey] for v in pts], [v[ykey] for v in pts], s=12, alpha=.5,
                           c=COLORS_MASK[mask], label=MASK_LABELS[mask])
            if branch == "amplification" and "A" in (xkey,ykey): (ax.set_xscale if xkey == "A" else ax.set_yscale)("log")
            if "r" in (xkey,ykey): (ax.set_xscale if xkey == "r" else ax.set_yscale)("log")
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        fig.suptitle(f"{branch} branch (inactive A=1, phase=0, r=1/SNR0 shown at their fixed values)", fontsize=10)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="center right", fontsize=8)
        fig.tight_layout(rect=(0, 0, .9, 1)); fig.savefig(assets / f"pairs_{branch}.png", dpi=160); plt.close(fig)
    # A representative original group, rather than pooled points from different cubes.
    example = next(items for items in groups.values() if len(items[0]["unit_cube"]) >= 2)
    kinds = list(example[0]["unit_cube"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for kind in kinds[:2]:
        axes[0].scatter([v["unit_cube"][kind] for v in example], [kind] * len(example), label=kind)
    axes[0].set_xlim(0, 1); axes[0].set_title("Marginal strata")
    axes[1].scatter([v["unit_cube"][kinds[0]] for v in example], [v["unit_cube"][kinds[1]] for v in example])
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel=kinds[0], ylabel=kinds[1], title="Pairwise projection")
    fig.tight_layout(); fig.savefig(assets / "unit_cube.png", dpi=160); plt.close(fig)
    rows = "".join(f"<tr><td>{labels[m]}</td><td>{data['counts'][str(m)]['vectors']}</td><td>{data['counts'][str(m)]['rows']}</td></tr>" for m in range(8))
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Dataset v2 sampling preview</title>",
             "<style>body{font:15px system-ui;max-width:1200px;margin:auto;padding:1.5rem;color:#24323a}img{max-width:100%}table{border-collapse:collapse}td,th{padding:.4rem .8rem;border:1px solid #ccd}</style></head><body>",
             f"<h1>Dataset v2 sampling preview</h1><p>{len(ids)} training sources; {len(vectors)} active parameter vectors. Two noise seeds repeat each vector at the same coordinates. Controls are fixed points, not LHS draws.</p>",
             "<p><a href='sampling_preview.json'>Parameters and provenance</a></p><table><tr><th>Injection combination</th><th>Vectors</th><th>Planned rows</th></tr>", rows, "</table>",
             "<img src='sampling_preview_assets/masks.png'><h2>Physical parameters</h2><p>Dotted lines mark active bounds. Inactive values are A=1, phase=0 and r=1/SNR0 for each source.</p>",
             "<img src='sampling_preview_assets/marginals.png'><h2>Pairwise distributions</h2>",
             "<img src='sampling_preview_assets/pairs_attenuation.png'><img src='sampling_preview_assets/pairs_amplification.png'>",
             "<h2>Inspect one source</h2><p>Each point is one parameter vector. Select a source and a pair to see its sparse design.</p>",
             "<label>Source <select id='source'></select></label> <label>Parameters <select id='pair'><option value='A,phi_deg'>A / phase</option><option value='A,r'>A / r</option><option value='phi_deg,r'>phase / r</option></select></label>",
             "<p>" + " ".join(f"<span style='color:{c}'>&#9679;</span> {l}" for c, l in zip(COLORS_MASK[1:], MASK_LABELS[1:])) + "</p>",
             "<svg id='sourcePlot' viewBox='0 0 720 400' style='width:100%;max-width:720px;border:1px solid #ccd'></svg>",
             "<h2>One original LHS group in unit coordinates</h2><img src='sampling_preview_assets/unit_cube.png'>",
             "<p>LHS spreads each marginal. It does not fill every pairwise cell or ensure a recovery-error range. Both-gain mixtures have only two points per branch combination per source at this budget.</p></body></html>"]
    source_points = {sid: [[v["A"], v["phi_deg"], v["r"], v["mask"]] for v in vectors if v["source"] == sid] for sid in ids}
    parts.append("<script>const points=" + json.dumps(source_points, separators=(",", ":")) + ";const MASK_COLORS=" + json.dumps(COLORS_MASK) + ";" + r"""
const source=document.getElementById('source'), pair=document.getElementById('pair'), svg=document.getElementById('sourcePlot');
for(const name of Object.keys(points)){let o=document.createElement('option');o.value=o.textContent=name;source.appendChild(o)}
const ix={A:0,phi_deg:1,r:2};
function draw(){let [xk,yk]=pair.value.split(','), pts=points[source.value], x=pts.map(p=>xk==='A'&&p[0]>1?Math.log(p[0]):xk==='r'?Math.log(p[ix[xk]]):p[ix[xk]]), y=pts.map(p=>yk==='r'?Math.log(p[2]):p[ix[yk]]), xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...y),ymax=Math.max(...y);
let sx=v=>55+610*(v-xmin)/(xmax-xmin||1),sy=v=>345-290*(v-ymin)/(ymax-ymin||1);
svg.innerHTML='<rect x="55" y="55" width="610" height="290" fill="#fff" stroke="#777"/>'+pts.map((p,i)=>'<circle cx="'+sx(x[i])+'" cy="'+sy(y[i])+'" r="4" opacity=".7" fill="'+MASK_COLORS[p[3]]+'"><title>mask '+p[3].toString(2).padStart(3,'0')+'; A='+p[0].toPrecision(5)+'; phase='+p[1].toPrecision(5)+'; r='+p[2].toPrecision(5)+'</title></circle>').join('')+'<text x="360" y="382" text-anchor="middle">'+xk+' (A>1 and r shown in log coordinate)</text><text x="13" y="40">'+yk+'</text>';
}
source.onchange=pair.onchange=draw;draw();
""" + "</script>")
    (output / "sampling_preview.html").write_text("".join(parts))
    print("Sampling preview:", output / "sampling_preview.html", flush=True)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    if a.preview:
        if not a.output: p.error("--preview requires --output")
        preview(configuration(a.config), a.output, smoke=a.smoke)
    else:
        if not a.dataset: p.error("Coverage report requires --dataset")
        report(a.dataset, a.partial)


if __name__ == "__main__":
    main()
