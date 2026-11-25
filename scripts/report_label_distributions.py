"""Label-distribution report for matched Dataset v2 / v2.1 training and validation sets (read-only).

Compares, across datasets, the distributions of
  eC   CLEAN recovery error of the row's image,
  eD   dirty-image error ||D - D0|| / ||D0||,
  d_k  recovery damage of corruption k (eC minus eC of the matched case without k), and
  p_k  dirty change of corruption k (||D - D_without_k|| / ||D0||),
with d_k and p_k shown separately for each mask that contains k: k alone, k with one other
corruption, and k with both others. It splits the eC distribution by number of corruptions per row
(0-3; the matched design makes 3:3:1 rows per recipe) to show whether its shape comes from that
construction rather than from the parameter sampling. It also counts rows per eC bin (with the ratio to a
reference split, normally validation) and splits each eC bin by origin (weak band, amplification
above the v2.1 cap, strong in-range), so it shows where training coverage is thin.
All eC-scale axes are linear with dotted lines at eC = 0.05 and 1.

Each dataset is a matched run folder (reads <split>.npz and plan.json) or a v2.1 manifest
(.json, joined and filtered exactly as training loads it). Nothing is written outside --output.

  bin/submit ml-cpu python -m scripts.report_label_distributions --output runs/label-distributions-v2-v21
  # same as the defaults, spelled out:
  #   --dataset v2-train=runs/dataset-v2-matched-train
  #   --dataset v2.1-train=runs/dataset-v2.1-train/manifest.json
  #   --dataset v2.1-nocap-train=runs/dataset-v2.1-noampcap-train/manifest.json
  #   --dataset v2.2-train=runs/dataset-v2.2-train/manifest.json
  #   --dataset v2.3-train=runs/dataset-v2.3-train/manifest.json
  #   --dataset v2-val=runs/dataset-v2-matched-val --reference v2-val
Rows are also counted per eC bin when an active corruption lies in a v2.2 tail
(amplification A >= 13.2, |phase| >= 159 deg, noise ratio r >= 0.141).
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.dataset_v21 import _resolve, load_manifest, read_part

DEFAULT_DATASETS = (("v2-train", "runs/dataset-v2-matched-train"),
                    ("v2.1-train", "runs/dataset-v2.1-train/manifest.json"),
                    ("v2.1-nocap-train", "runs/dataset-v2.1-noampcap-train/manifest.json"),
                    ("v2.2-train", "runs/dataset-v2.2-train/manifest.json"),
                    ("v2.3-train", "runs/dataset-v2.3-train/manifest.json"),
                    ("v2-val", "runs/dataset-v2-matched-val"))
AMP_CAP = 8.5468   # v2.1 amplification cap, only used to label rows
TYPES = ("amp", "phase", "noise")
MASK_LABELS = ("control", "amp", "phase", "amp+phase", "noise", "amp+noise", "phase+noise", "amp+phase+noise")
EC_BINS = (0, .05, .2, .5, 1, 1.5, 2, np.inf)
ORIGINS = ("control", "noise only", "weak amp/phase", f"A > {AMP_CAP:g}", "strong, in range")
MARKS = (.05, 1.)   # eC thresholds marked on every eC-scale axis
NCORR_LABELS = ("0 (control)", "1 corruption", "2 corruptions", "3 corruptions")
NCORR_COLORS = ("#c9c9c4", "#a9cbf2", "#4f8fdc", "#173f78")   # ordinal: one blue hue, light to dark
TAIL = dict(amp=13.2, phase=159., noise=.141)   # v2.2 tails: A >= 13.2, |phase| >= 159 deg, r >= 0.141
TAILS = ("any tail", "amp tail", "phase tail", "noise tail")
# Validated categorical order (dataviz reference palette, light mode), as in report_dataset_v2_matched.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
STYLE = ("<style>body{font:15px system-ui;max-width:1250px;margin:auto;padding:1.5rem;color:#24323a;background:#fcfcfb}"
         "img{max-width:100%}table{border-collapse:collapse;margin:.5rem 0 1.2rem}td,th{padding:.3rem .6rem;"
         "border:1px solid #ccd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f6}"
         ".note{color:#4a5a63}code{font-size:90%}</style>")


# ---------------------------------------------------------------- loading

def load_dataset(path):
    """Arrays of a run folder or a v2.1 manifest, plus per-row origin labels."""
    path = _resolve(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        data = load_manifest(path)
        split = manifest["split"]
        recipes = {}
        for e in manifest["parts"]:
            plan = json.loads((_resolve(e["run"]) / "plan.json").read_text())
            recipes.update({f"{e['tag']}/{r['id']}": r for sp in plan["source_plans"] for r in sp["recipes"]})
    else:
        plan = json.loads((path / "plan.json").read_text())
        split = plan["split"]
        data, plan = read_part(path, split)
        recipes = {r["id"]: r for sp in plan["source_plans"] for r in sp["recipes"]}
        snr = {p["id"]: p["nominal_source_snr"] for p in plan["prepared_sources"]}
        data["noise_ratio"] = data["params"][:, 2] / np.asarray([snr[str(s)] for s in data["source_ids"]], float)
    b = data["b"].astype(bool)
    origin = np.empty(len(b), object)
    for i, rid in enumerate(map(str, data["recipe_ids"])):
        r = recipes.get(rid)
        if r is None:
            origin[i] = ORIGINS[0]
        elif not (b[i, 0] or b[i, 1]):
            origin[i] = ORIGINS[1]
        elif (b[i, 0] and r["amp_band"] == "weak") or (b[i, 1] and r["phase_band"] == "weak"):
            origin[i] = ORIGINS[2]
        elif b[i, 0] and r["A"] > AMP_CAP:
            origin[i] = ORIGINS[3]
        else:
            origin[i] = ORIGINS[4]
    A, phi, r = data["params"][:, 0], np.abs(data["params"][:, 1]), np.asarray(data["noise_ratio"], float)
    each = np.column_stack((b[:, 0] & (A >= TAIL["amp"]), b[:, 1] & (phi >= TAIL["phase"]), b[:, 2] & (r >= TAIL["noise"])))
    tail = dict(zip(TAILS, (each.any(1), each[:, 0], each[:, 1], each[:, 2])))
    return dict(split=split, path=str(path), eC=np.asarray(data["eC"], float), eD=np.asarray(data["eD"], float), tail=tail,
                d=np.asarray(data["d"], float), p=np.asarray(data["p"], float), b=b,
                mask=np.asarray(data["mask"], int), origin=origin,
                sources=len(set(map(str, data["source_ids"]))),
                recipes=len({x for x in map(str, data["recipe_ids"]) if x}))


def contexts(k):
    """Masks containing corruption k: alone, with each other type, with both others."""
    others = [j for j in range(3) if j != k]
    return [1 << k, (1 << k) | (1 << others[0]), (1 << k) | (1 << others[1]), 7]


# ---------------------------------------------------------------- statistics

def quantiles(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(n=0)
    q = np.quantile(x, [.01, .1, .5, .9, .99])
    return dict(n=int(len(x)), mean=float(x.mean()), p1=float(q[0]), p10=float(q[1]), median=float(q[2]),
                p90=float(q[3]), p99=float(q[4]), max=float(x.max()), min=float(x.min()),
                frac_negative=float((x < 0).mean()), frac_gt_0_5=float((x > .5).mean()), frac_gt_1=float((x > 1).mean()))


def bin_counts(x):
    return np.histogram(np.asarray(x, float), bins=np.asarray(EC_BINS, float))[0]


def bin_labels():
    edges = EC_BINS
    return [f"< {edges[1]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(edges[1:-2], edges[2:-1])] + [f"> {edges[-2]:g}"]


def summarize(datasets):
    out = {}
    for name, ds in datasets.items():
        active = {}
        for k, t in enumerate(TYPES):
            for m in contexts(k):
                sel = ds["mask"] == m
                active[f"{t}|{MASK_LABELS[m]}"] = dict(d=quantiles(ds["d"][sel, k]), p=quantiles(ds["p"][sel, k]))
        out[name] = dict(path=ds["path"], split=ds["split"], rows=int(len(ds["eC"])), sources=ds["sources"],
                         recipes=ds["recipes"], masks={MASK_LABELS[m]: int((ds["mask"] == m).sum()) for m in range(8)},
                         eC=quantiles(ds["eC"]), eD=quantiles(ds["eD"]),
                         eC_by_mask={MASK_LABELS[m]: quantiles(ds["eC"][ds["mask"] == m]) for m in range(8)},
                         eD_by_mask={MASK_LABELS[m]: quantiles(ds["eD"][ds["mask"] == m]) for m in range(8)},
                         eC_bins=dict(zip(bin_labels(), bin_counts(ds["eC"]).tolist())),
                         eD_bins=dict(zip(bin_labels(), bin_counts(ds["eD"]).tolist())),
                         eC_bins_by_origin={o: bin_counts(ds["eC"][ds["origin"] == o]).tolist() for o in ORIGINS},
                         eC_bins_by_ncorr={NCORR_LABELS[j]: bin_counts(ds["eC"][ds["b"].sum(1) == j]).tolist()
                                           for j in range(4)},
                         eC_bins_by_tail={k: bin_counts(ds["eC"][v]).tolist() for k, v in ds["tail"].items()},
                         damages=active)
    return out


# ---------------------------------------------------------------- plots

def _tidy(ax):
    ax.grid(color="#d9dee1", lw=.5, alpha=.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mark(ax, x=True, y=False):
    """Dotted reference lines at the eC thresholds 0.05 and 1 (labelled on x)."""
    for v in MARKS:
        if x:
            ax.axvline(v, color="#24323a", ls=":", lw=1, zorder=1)
            ax.text(v, 1.0, f" {v:g}", transform=ax.get_xaxis_transform(), fontsize=7, color="#24323a",
                    va="top", ha="left")
        if y:
            ax.axhline(v, color="#24323a", ls=":", lw=1, zorder=1)


def lin_bins(values, n=120):
    """Linear bins from min(0, data) to the data maximum."""
    v = np.concatenate([np.asarray(x, float) for x in values])
    v = v[np.isfinite(v)]
    lo = min(0., float(v.min())) if len(v) else 0.
    hi = float(v.max()) if len(v) else 1.
    return np.linspace(lo, hi if hi > lo else lo + 1, n + 1)


def hist_fraction(ax, x, bins, color, label):
    """Step histogram as fraction of the panel's rows per bin (comparable across dataset sizes)."""
    x = np.asarray(x, float)
    if not len(x):
        return
    ax.hist(x, bins=bins, weights=np.full(len(x), 1 / len(x)), histtype="step", lw=1.6, color=color,
            label=f"{label} (n={len(x)})")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def plot_overall(datasets, assets):
    names = list(datasets)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, key, label in ((0, "eC", "eC (CLEAN recovery error)"), (1, "eD", "eD (dirty-image error)")):
        bins = lin_bins([ds[key] for ds in datasets.values()], 150)
        ax = axes[0, col]
        for i, n in enumerate(names):
            hist_fraction(ax, datasets[n][key], bins, COLORS[i], n)
        ax.set(xlabel=label, ylabel="fraction of rows per bin", title=f"{key}: distribution")
        mark(ax); _tidy(ax); ax.legend(frameon=False, fontsize=8)
        ax = axes[1, col]
        for i, n in enumerate(names):
            x = np.sort(datasets[n][key])
            surv = 1 - np.arange(len(x)) / len(x)
            ax.step(x, surv, where="post", color=COLORS[i], lw=1.6, label=n)
        ax.set(yscale="log", xlabel=label, ylabel=f"fraction of rows with {key} ≥ x (log)",
               title=f"{key}: tail (survival function)")
        mark(ax); _tidy(ax); ax.legend(frameon=False, fontsize=8)
    return save(fig, assets / "overall.png")


def plot_counts(datasets, assets):
    """Absolute rows per eC bin: what the model actually sees, not the shape."""
    names = list(datasets)
    labels = bin_labels()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    width = .8 / len(names)
    x = np.arange(len(labels))
    for i, n in enumerate(names):
        c = bin_counts(datasets[n]["eC"])
        ax.bar(x + (i - (len(names) - 1) / 2) * width, c, width * .92, color=COLORS[i], label=n)
    ax.set(yscale="log", xticks=x, xticklabels=labels, xlabel="eC bin", ylabel="rows (log)", title="Rows per eC bin")
    _tidy(ax); ax.legend(frameon=False, fontsize=8)
    return save(fig, assets / "eC_bin_counts.png")


def plot_composition(datasets, assets):
    """eC distribution split by the number of corruptions in the row, on one shared linear eC axis.

    Left: stacked row counts (the stack is the dataset's eC histogram). Right: share of each bin.
    """
    names = list(datasets)
    bins = lin_bins([ds["eC"] for ds in datasets.values()], 100)
    share_bins = lin_bins([ds["eC"] for ds in datasets.values()], 40)   # coarser: shares need rows per bin
    fig, axes = plt.subplots(len(names), 2, figsize=(15, 3.3 * len(names)), sharex=True, squeeze=False)
    for r, n in enumerate(names):
        ds = datasets[n]
        k = ds["b"].sum(1)
        groups = [ds["eC"][k == j] for j in range(4)]
        ax = axes[r, 0]
        ax.hist(groups, bins=bins, stacked=True, color=NCORR_COLORS, label=NCORR_LABELS)
        ax.set(ylabel="rows", title=f"{n}: rows per eC bin, stacked by number of corruptions")
        mark(ax); _tidy(ax)
        ax = axes[r, 1]
        counts = np.array([np.histogram(g, bins=share_bins)[0] for g in groups], float)
        total = counts.sum(0)
        share = np.where(total >= 5, counts / np.maximum(total, 1), np.nan)   # bins with < 5 rows left blank
        centers, width = (share_bins[:-1] + share_bins[1:]) / 2, np.diff(share_bins)
        bottom = np.zeros(len(centers))
        for j in range(4):
            ax.bar(centers, share[j], width, bottom=bottom, color=NCORR_COLORS[j], lw=0)
            bottom = bottom + np.nan_to_num(share[j])
        ax.set(ylim=(0, 1), ylabel="share of the bin", title=f"{n}: share of each eC bin (bins with < 5 rows blank)")
        mark(ax); _tidy(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("eC")
    return save(fig, assets / "eC_by_ncorr.png")


def plot_by_mask(datasets, key, assets):
    names = list(datasets)
    bins = lin_bins([ds[key] for ds in datasets.values()], 80)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for m, ax in enumerate(axes.flat):
        for i, n in enumerate(names):
            ds = datasets[n]
            hist_fraction(ax, ds[key][ds["mask"] == m], bins, COLORS[i], n)
        ax.set(title=MASK_LABELS[m])
        mark(ax); _tidy(ax); ax.legend(frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel(key)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of the mask's rows")
    return save(fig, assets / f"{key}_by_mask.png")


def plot_damages(datasets, which, assets):
    """3 corruption types x 4 contexts; d (signed) or p (positive), linear axes."""
    names = list(datasets)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
    for k, t in enumerate(TYPES):
        values = [datasets[n][which][datasets[n]["mask"] == m, k] for n in names for m in contexts(k)]
        bins = lin_bins(values, 80)
        for c, m in enumerate(contexts(k)):
            ax = axes[k, c]
            for i, n in enumerate(names):
                ds = datasets[n]
                hist_fraction(ax, ds[which][ds["mask"] == m, k], bins, COLORS[i], n)
            if which == "d":
                ax.axvline(0, color="#8a8a85", lw=.8)
            ax.set_title(f"{t}: row mask {MASK_LABELS[m]}", fontsize=10)
            mark(ax); _tidy(ax); ax.legend(frameon=False, fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{which}_{t}: fraction of rows")
            if k == 2:
                ax.set_xlabel(f"{'recovery damage d' if which == 'd' else 'dirty change p'}_{t}")
    return save(fig, assets / f"{which}_by_context.png")


def plot_ec_vs_ed(datasets, assets):
    names = list(datasets)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharex=True, sharey=True, squeeze=False)
    for ax, n in zip(axes[0], names):
        ds = datasets[n]
        ax.hexbin(ds["eD"], ds["eC"], gridsize=50, bins="log", cmap="Blues", mincnt=1)
        ax.set(title=n, xlabel="eD")
        mark(ax, x=True, y=True); _tidy(ax)
    axes[0, 0].set_ylabel("eC")
    return save(fig, assets / "eC_vs_eD.png")


# ---------------------------------------------------------------- html

def fmt(x, digits=3):
    if isinstance(x, (int, np.integer)):
        return str(x)
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}g}"


def table(header, rows):
    return ("<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>")


def quantile_row(label, q, keys=("n", "p1", "p10", "median", "p90", "p99", "max", "frac_gt_0_5", "frac_gt_1")):
    return [label] + [fmt(q.get(k)) for k in keys]


def report(datasets, summary, reference, output, images):
    names = list(datasets)
    labels = bin_labels()
    ref = summary.get(reference)
    parts = [f"<!doctype html><meta charset='utf-8'><title>Label distributions</title>{STYLE}",
             "<h1>Label distributions: eC, eD and per-corruption damages</h1>",
             "<p class='note'>Read-only comparison of the training and validation label sets. v2.1 manifests are "
             "loaded exactly as training loads them (filter applied). Histograms show the <b>fraction</b> of the "
             "panel's rows per bin so datasets of different size can be compared; absolute counts are in the "
             "tables. All eC-scale axes are linear, with dotted lines at eC = 0.05 and 1.</p>",
             "<h2>1. Datasets</h2>",
             table(["Dataset", "Path", "Split", "Rows", "Sources", "Recipes"] + list(MASK_LABELS),
                   [[html.escape(n), f"<code>{html.escape(s['path'])}</code>", s["split"], s["rows"], s["sources"],
                     s["recipes"], *[s["masks"][m] for m in MASK_LABELS]] for n, s in summary.items()]),
             "<h2>2. eC and eD overall</h2>", f"<img src='{images['overall']}' alt='eC and eD distributions and tails'>",
             table(["eC", "n", "p1", "p10", "median", "p90", "p99", "max", "frac > 0.5", "frac > 1"],
                   [quantile_row(n, s["eC"]) for n, s in summary.items()]),
             table(["eD", "n", "p1", "p10", "median", "p90", "p99", "max", "frac > 0.5", "frac > 1"],
                   [quantile_row(n, s["eD"]) for n, s in summary.items()]),
             "<h2>3. Rows per eC bin</h2>",
             f"<p class='note'>Absolute coverage. 'per {html.escape(reference)} row' divides each dataset's count by "
             "the reference's count in the same bin: how many training examples the model has for each validation "
             "example of that severity.</p>" if ref else "",
             f"<img src='{images['counts']}' alt='Rows per eC bin'>"]
    rows = []
    for n, s in summary.items():
        counts = list(s["eC_bins"].values())
        rows.append([html.escape(n)] + [f"{c} ({100 * c / max(s['rows'], 1):.1f}%)" for c in counts])
        if ref and n != reference:
            rc = list(ref["eC_bins"].values())
            rows.append([f"&nbsp;&nbsp;per {html.escape(reference)} row"] + [fmt(c / r if r else np.nan) for c, r in zip(counts, rc)])
    parts.append(table(["eC bin"] + labels, rows))
    rows = [[html.escape(n)] + [f"{c} ({100 * c / max(s['rows'], 1):.1f}%)" for c in s["eD_bins"].values()]
            for n, s in summary.items()]
    parts += ["<p>Same bins for eD:</p>", table(["eD bin"] + labels, rows),
              "<h2>4. What each eC bin is made of</h2>",
              f"<p class='note'>Origin of each row: control; noise only; contains a weak-band amp or phase corruption; "
              f"contains amplification A &gt; {AMP_CAP:g} (the v2.1 cap; weak takes precedence); everything else.</p>"]
    for n, s in summary.items():
        parts.append(f"<h3>{html.escape(n)}</h3>")
        parts.append(table(["origin"] + labels + ["total"],
                           [[o] + s["eC_bins_by_origin"][o] + [sum(s["eC_bins_by_origin"][o])] for o in ORIGINS]))
    parts += ["<h2>4b. Rows in the v2.2 tails</h2>",
              f"<p class='note'>A row counts when one of its active corruptions lies in a v2.2 tail: amplification "
              f"A &ge; {TAIL['amp']:g}, |phase| &ge; {TAIL['phase']:g}&deg;, noise ratio r &ge; {TAIL['noise']:g}. "
              "'any tail' counts each row once; a row can be in several per-corruption tails.</p>"]
    rows = []
    for n, s in summary.items():
        for k in TAILS:
            c = s["eC_bins_by_tail"][k]
            rows.append([f"{html.escape(n)} · {k}"] + c + [f"{sum(c)} ({100 * sum(c) / max(s['rows'], 1):.1f}%)"])
    parts.append(table(["dataset · tail"] + labels + ["total"], rows))
    parts += ["<h2>5. Is the eC distribution set by the number of corruptions?</h2>",
              "<p class='note'>The matched design builds 7 rows from every sampled recipe: 3 with one corruption, "
              "3 with two and 1 with all three (plus one control per source). If eC mostly grows with the number of "
              "corruptions, the shape of the eC distribution is set by this construction, not by the parameter "
              "sampling. Left: rows per eC bin stacked by number of corruptions (the stack is the dataset's "
              "histogram). Right: each bin's share. Filtered sets (v2.1) no longer have the 3:3:1 ratio.</p>",
              f"<img src='{images['ncorr']}' alt='eC distribution stacked by number of corruptions'>"]
    header = ["dataset · corruptions"] + bin_labels() + ["total"]
    rows = []
    for n, s in summary.items():
        totals = np.array(list(s["eC_bins"].values()), float)
        for j, lab in enumerate(NCORR_LABELS):
            c = np.array(s["eC_bins_by_ncorr"][lab])
            rows.append([f"{html.escape(n)} · {lab}"] +
                        [f"{int(v)} ({100 * v / t:.0f}%)" if t else "0" for v, t in zip(c, totals)] + [int(c.sum())])
    parts += [table(header, rows), "<h2>6. eC and eD by injection mask</h2>",
              f"<img src='{images['eC_mask']}' alt='eC by mask'>", f"<img src='{images['eD_mask']}' alt='eD by mask'>"]
    rows = [quantile_row(f"{m} · {n}", summary[n]["eC_by_mask"][m]) for m in MASK_LABELS for n in names]
    parts.append(table(["eC: mask · dataset", "n", "p1", "p10", "median", "p90", "p99", "max", "frac > 0.5", "frac > 1"], rows))
    rows = [quantile_row(f"{m} · {n}", summary[n]["eD_by_mask"][m]) for m in MASK_LABELS for n in names]
    parts.append(table(["eD: mask · dataset", "n", "p1", "p10", "median", "p90", "p99", "max", "frac > 0.5", "frac > 1"], rows))
    parts += ["<h2>7. Per-corruption damages by context</h2>",
              "<p class='note'>For corruption k, each column is a row mask containing k: k alone (all others removed), "
              "k with one other corruption, and k with both others. <b>Recovery damage</b> d_k = eC(row) − eC(same "
              "recipe without k), signed. <b>Dirty change</b> p_k = ‖D(row) − "
              "D(without k)‖ / ‖D0‖, always ≥ 0.</p>",
              f"<h3>Recovery damage d_k</h3><img src='{images['d']}' alt='Recovery damage by corruption and context'>",
              f"<h3>Dirty change p_k</h3><img src='{images['p']}' alt='Dirty change by corruption and context'>"]
    for which, title in (("d", "Recovery damage d_k"), ("p", "Dirty change p_k")):
        rows = []
        for k, t in enumerate(TYPES):
            for m in contexts(k):
                for n in names:
                    q = summary[n]["damages"][f"{t}|{MASK_LABELS[m]}"][which]
                    rows.append([f"{t} · {MASK_LABELS[m]} · {html.escape(n)}"] +
                                [fmt(q.get(c)) for c in ("n", "frac_negative", "p1", "p10", "median", "p90", "p99", "max",
                                                         "frac_gt_0_5")])
        parts += [f"<h3>{title}: quantiles</h3>",
                  table(["type · mask · dataset", "n", "frac < 0", "p1", "p10", "median", "p90", "p99", "max", "frac > 0.5"], rows)]
    parts += ["<h2>8. eC against eD</h2>", f"<img src='{images['ec_ed']}' alt='eC against eD per dataset'>",
              "<p class='note'>Numbers behind every table: <code>summary.json</code>.</p>"]
    (output / "report.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", action="append", metavar="NAME=PATH",
                   help="Run folder or v2.1 manifest; repeatable. Default: v2 train, v2.1 train, v2.1 no-cap train, v2 val.")
    p.add_argument("--reference", default="v2-val", help="Dataset the per-bin count ratios divide by (default v2-val).")
    a = p.parse_args()
    specs = [tuple(x.split("=", 1)) for x in a.dataset] if a.dataset else list(DEFAULT_DATASETS)
    if any(len(s) != 2 for s in specs) or len({s[0] for s in specs}) != len(specs):
        p.error("--dataset must be NAME=PATH with unique names")
    if len(specs) > len(COLORS):
        p.error(f"At most {len(COLORS)} datasets")
    output = _resolve(a.output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, path in specs:
        datasets[name] = load_dataset(path)
        print(f"{name}: {len(datasets[name]['eC'])} rows from {datasets[name]['path']}", flush=True)
    summary = summarize(datasets)
    (output / "summary.json").write_text(json.dumps(dict(datasets=summary, eC_bins=bin_labels(), amp_cap=AMP_CAP,
                                                         reference=a.reference), indent=1) + "\n")
    images = {k: f"assets/{v}" for k, v in dict(
        overall=plot_overall(datasets, assets), counts=plot_counts(datasets, assets),
        ncorr=plot_composition(datasets, assets), eC_mask=plot_by_mask(datasets, "eC", assets), eD_mask=plot_by_mask(datasets, "eD", assets),
        d=plot_damages(datasets, "d", assets), p=plot_damages(datasets, "p", assets),
        ec_ed=plot_ec_vs_ed(datasets, assets)).items()}
    report(datasets, summary, a.reference if a.reference in datasets else None, output, images)
    print(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
    main()
