"""Report on the validation cases that carry most of one model's eC squared error.

Read-only. Takes one multitask run's saved predictions_val_<model>.npz (decoded column 3 = eC), selects the
top --fraction of validation rows by squared eC error, and writes an HTML report with:

  1. one table row per source (sources ordered by their selected cases' share of the total squared error),
     one column per selected case (cases ordered by squared error, left to right). Each cell shows the CLEAN
     image, the true and recovered models (both convolved with the restoring beam, as in eC) and their
     difference, plus true/predicted eC, the error, the corruption recipe and all model input features;
  2. scatter plots of every input feature vs true eC: training rows grey, validation rows yellow, selected red.

eC is recomputed from the images (scripts/dataset_v2_matched_extract.py) and must match the stored value.

  bin/submit ml-cpu python -m ml.val_error_cases --output ml/runs/val-error-top10-xgb_mlp-v2
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from astropy.io import fits

from scripts.clean_recovery_noise_sweep import convolve_model

ROOT = Path(__file__).resolve().parents[1]
EC_COLUMN = 3   # decoded multitask output order: d_amp, d_phase, d_noise, eC, ...
MASK_NAMES = {0: "control", 1: "amp", 2: "phase", 3: "amp+phase", 4: "noise", 5: "amp+noise", 6: "phase+noise",
              7: "amp+phase+noise"}
STYLE = ("<style>body{font-family:system-ui,sans-serif;margin:16px;color:#1d2a31}table{border-collapse:collapse}"
         "td,th{border:1px solid #c9d3d8;padding:4px;vertical-align:top;font-size:11px}"
         "th{background:#eef2f4;position:sticky;left:0}.src{background:#f6f8f9;min-width:130px;position:sticky;left:0;z-index:1}"
         ".wrap{overflow-x:auto;max-width:100%}.feat td{border:none;padding:0 4px;font-size:10.5px}"
         ".big{font-size:12px;font-weight:600}.over{color:#b3261e}.under{color:#1f5fa8}img{display:block}"
         ".note{color:#4a5a63;max-width:1100px}</style>")


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def fmt(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.4g}"


def plane(path):
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(np.squeeze(hdul[0].data), np.float64), hdul[0].header.copy()


def case_images(val_root, row):
    """Clean image, true and recovered convolved models, and the eC recomputed exactly as the extractor does."""
    record = json.loads((val_root / "cases" / row["case"] / "record.json").read_text())
    model, header = plane(val_root / record["products"]["model"])
    clean, _ = plane(val_root / record["products"]["clean"])
    with np.load(val_root / "sources" / row["source"] / "recovery" / "reference_masks.npz") as m:
        truth, omega = np.asarray(m["truth"], np.float64), np.asarray(m["omega"], bool)
    recovered = convolve_model(model, header)
    eC = float(np.sqrt(np.square((recovered - truth)[omega]).sum() / np.square(truth[omega]).sum()))
    return clean, truth, recovered, omega, eC


def zoom_box(truth, recovered, pad=8, minimum=32):
    """Square pixel box around where the true or recovered convolved model exceeds 0.1% of the true peak."""
    level = 1e-3 * float(np.max(np.abs(truth)))
    yy, xx = np.nonzero((np.abs(truth) > level) | (np.abs(recovered) > level))
    cy, cx = (yy.min() + yy.max()) / 2, (xx.min() + xx.max()) / 2
    half = int(max(minimum / 2, (yy.max() - yy.min()) / 2 + pad, (xx.max() - xx.min()) / 2 + pad))
    n = truth.shape[0]
    y0 = int(min(max(cy - half, 0), n - 2 * half)) if 2 * half < n else 0
    x0 = int(min(max(cx - half, 0), n - 2 * half)) if 2 * half < n else 0
    return slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half)


def plot_case(path, clean, truth, recovered, omega, title):
    """CLEAN image over the full field (box = zoom), then the models and their difference zoomed to the source."""
    peak = float(np.max(np.abs(truth)))
    diff = np.where(omega, recovered - truth, np.nan)
    dmax = float(np.nanmax(np.abs(diff))) or 1.
    norm = SymLogNorm(linthresh=1e-2 * peak, vmin=-0.1 * peak, vmax=max(peak, float(np.max(recovered))), base=10)
    cpeak = float(np.nanmax(np.abs(clean))) or 1.
    ys, xs = zoom_box(truth, recovered)
    extent = (xs.start - .5, xs.stop - .5, ys.start - .5, ys.stop - .5)
    panels = ((clean, None, "CLEAN image (full field)", "inferno",
               SymLogNorm(linthresh=1e-2 * cpeak, vmin=-0.1 * cpeak, vmax=cpeak, base=10)),
              (truth[ys, xs], extent, "true model ⊛ beam (zoom)", "inferno", norm),
              (recovered[ys, xs], extent, "recovered model ⊛ beam (zoom)", "inferno", norm),
              (diff[ys, xs], extent, "recovered − true (zoom)", "RdBu_r",
               SymLogNorm(linthresh=1e-2 * peak, vmin=-dmax, vmax=dmax, base=10)))
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.45))
    for ax, (img, ext, label, cmap, nrm), full in zip(axes, panels, (clean, truth, recovered, diff)):
        im = ax.imshow(img, origin="lower", cmap=cmap, norm=nrm, interpolation="nearest", extent=ext)
        ax.set_title(f"{label}\nmax|·| {np.nanmax(np.abs(full)):.3g} Jy/beam", fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5)
    axes[0].add_patch(plt.Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2],
                                    fill=False, ec="cyan", lw=0.8))
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=85)
    plt.close(fig)


def plot_features(path, names, X_train, y_train, X_val, y_val, sel):
    cols = 4
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows))
    for j, ax in enumerate(axes.flat):
        if j >= len(names):
            ax.axis("off")
            continue
        xs = (X_train[:, j], X_val[:, j], X_val[sel, j])
        finite = np.concatenate([x[np.isfinite(x)] for x in xs])
        log_x = finite.size and finite.min() > 0 and finite.max() / finite.min() > 50
        nonzero = np.abs(finite[finite != 0])
        symlog_x = (not log_x and nonzero.size and finite.min() <= 0
                    and np.abs(finite).max() / np.median(nonzero) > 50)
        ax.scatter(X_train[:, j], y_train, s=3, c="#9a9a9a", alpha=0.35, lw=0, label=f"train ({len(y_train)})", rasterized=True)
        ax.scatter(X_val[:, j], y_val, s=5, c="#f2c200", alpha=0.8, lw=0, label=f"val ({len(y_val)})", rasterized=True)
        ax.scatter(X_val[sel, j], y_val[sel], s=11, c="#d7191c", lw=0.3, edgecolors="k",
                   label=f"selected ({int(sel.sum())})", rasterized=True)
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        elif symlog_x:
            ax.set_xscale("symlog", linthresh=float(np.percentile(nonzero, 10)))
        ax.set_title(names[j], fontsize=9)
        ax.set_ylabel("true eC", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ml/runs/matched-v2-full-all22-dirty")
    ap.add_argument("--model", default="xgb_mlp")
    ap.add_argument("--val", default="runs/dataset-v2-matched-val")
    ap.add_argument("--train", default="runs/dataset-v2-matched-train", help="training dataset folder (train.npz) for the scatter plots")
    ap.add_argument("--fraction", type=float, default=0.10, help="fraction of validation rows to select by squared eC error")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    run, val_root, train_root, output = (_resolve(p) for p in (args.run, args.val, args.train, args.output))
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} exists and is not empty; use a fresh output directory")
    assets = output / "assets"
    (assets / "cases").mkdir(parents=True, exist_ok=True)

    with np.load(val_root / "val.npz", allow_pickle=False) as z:
        val = {k: z[k] for k in z.files}
    with np.load(train_root / "train.npz", allow_pickle=False) as z:
        X_train, y_train, train_names = np.asarray(z["X"], float), np.asarray(z["eC"], float), z["feature_names"].astype(str)
    with np.load(run / f"predictions_val_{args.model}.npz", allow_pickle=False) as z:
        ids, pred = z["row_ids"].astype(str), np.asarray(z["decoded"], float)[:, EC_COLUMN]
    names = json.loads((run / "results.json").read_text())["feature_names"]
    row_ids = val["row_ids"].astype(str)
    if not np.array_equal(ids, row_ids):
        raise ValueError("prediction rows differ from the validation set")
    if list(val["feature_names"].astype(str)) != names or list(train_names) != names:
        raise ValueError("feature names of the run, val.npz and train.npz differ")
    rows = {r["id"]: r for r in json.loads((val_root / "rows.json").read_text())["rows"]}

    y, X, src = np.asarray(val["eC"], float), np.asarray(val["X"], float), val["source_ids"].astype(str)
    err = pred - y
    se = err ** 2
    total = se.sum()
    k = int(round(args.fraction * len(y)))
    order = np.argsort(-se)[:k]
    sel = np.zeros(len(y), bool)
    sel[order] = True

    share = {s: se[sel & (src == s)].sum() / total for s in np.unique(src[sel])}
    source_order = sorted(share, key=lambda s: -share[s])
    rmse = float(np.sqrt(se.mean()))
    rmse_rest = float(np.sqrt(se[~sel].mean()))

    table = []
    max_cols = 0
    for s in source_order:
        idx = [i for i in order if src[i] == s]   # already by squared error, descending
        max_cols = max(max_cols, len(idx))
        all_s = src == s
        cells = [f"<td class='src'><span class='big'>{html.escape(s)}</span><br>{len(idx)} of {int(all_s.sum())} rows selected"
                 f"<br>selected: {100 * share[s]:.1f}% of total SE<br>whole source: {100 * se[all_s].sum() / total:.1f}%"
                 f"<br>source RMSE {np.sqrt(se[all_s].mean()):.3f}</td>"]
        for i in idx:
            rid = row_ids[i]
            row = rows[rid]
            clean, truth, recovered, omega, eC = case_images(val_root, row)
            if not np.isclose(eC, y[i], rtol=1e-6, atol=1e-9):
                raise ValueError(f"{rid}: recomputed eC {eC} differs from stored {y[i]}")
            name = rid.replace("+", "p").replace(":", "_") + ".png"
            plot_case(assets / "cases" / name, clean, truth, recovered, omega,
                      f"{rid}  ({MASK_NAMES[int(val['mask'][i])]})")
            A, phi, a = val["params"][i]
            cls = "over" if err[i] > 0 else "under"
            feats = "".join(f"<tr><td>{html.escape(n)}</td><td>{fmt(v)}</td></tr>" for n, v in zip(names, X[i]))
            cells.append(
                f"<td><img src='assets/cases/{html.escape(name)}' width='714' alt='{html.escape(rid)}'>"
                f"<div class='big'>{html.escape(rid)} · {MASK_NAMES[int(val['mask'][i])]}</div>"
                f"<div class='big'>true eC {y[i]:.4f} · predicted {pred[i]:.4f} · "
                f"<span class='{cls}'>error {err[i]:+.4f}</span> · {100 * se[i] / total:.2f}% of total SE</div>"
                f"<div>gain A {A:.4g} · phase φ {phi:.4g}° · noise multiplier a {a:.4g} · recipe {html.escape(str(row['recipe']))}</div>"
                f"<table class='feat'>{feats}</table></td>")
        table.append(cells)
    body = "".join("<tr>" + "".join(c) + "<td></td>" * (max_cols + 1 - len(c)) + "</tr>" for c in table)
    header = "<tr><th>source</th>" + "".join(f"<th>case {j + 1}</th>" for j in range(max_cols)) + "</tr>"

    plot_features(assets / "features_vs_eC.png", names, X_train, y_train, X, y, sel)

    summary = dict(run=str(args.run), model=args.model, val=str(args.val), train=str(args.train), fraction=args.fraction,
                   selected_rows=int(k), val_rows=int(len(y)), rmse=rmse, rmse_without_selected=rmse_rest,
                   selected_share_of_se=float(se[sel].sum() / total),
                   sources=[dict(source=s, selected=int((sel & (src == s)).sum()), share=float(share[s])) for s in source_order],
                   rows=[dict(id=row_ids[i], source=src[i], eC=float(y[i]), pred=float(pred[i]), err=float(err[i]),
                              se_share=float(se[i] / total)) for i in order])
    (output / "summary.json").write_text(json.dumps(summary, indent=1))

    page = [f"<!doctype html><meta charset='utf-8'><title>Top eC error cases</title>{STYLE}",
            f"<h1>{html.escape(args.model)}: the validation cases that carry most of the eC error</h1>",
            f"<p class='note'>Run <code>{html.escape(str(args.run))}</code>, validation <code>{html.escape(str(args.val))}</code> "
            f"({len(y)} rows, {len(np.unique(src))} sources). Selected: the top {100 * args.fraction:g}% of rows by squared eC error "
            f"(<b>{k} rows</b>), carrying <b>{100 * se[sel].sum() / total:.1f}%</b> of the total squared error. "
            f"Overall RMSE {rmse:.4f}; RMSE of the other {len(y) - k} rows {rmse_rest:.4f}.</p>",
            "<h2>Selected cases by source</h2>",
            "<p class='note'>One row per source, ordered by the selected cases' share of the total squared error; cases "
            "left to right by squared error. Images: CLEAN image; true model and recovered CLEAN model, both convolved with "
            "the restoring beam (peak-one beam, the eC definition), on a shared colour scale; and their difference. "
            "eC = ‖recovered − true‖ / ‖true‖ over the support Ω, recomputed from these images (matches the stored value). "
            "Colour scales are symmetric-log (linear within 1% of the peak). Error = predicted − true "
            "(<span class='over'>red: overpredicted</span>, <span class='under'>blue: underpredicted</span>). "
            "Features are the model's inputs, unscaled.</p>",
            f"<div class='wrap'><table>{header}{body}</table></div>",
            "<h2>Input features vs true eC</h2>",
            f"<p class='note'>Grey: training rows (<code>{html.escape(str(args.train))}</code>, {len(y_train)}); yellow: validation "
            f"rows; red: the {k} selected rows. x axes are log when a feature is positive and spans more than a factor 50, and symmetric-log (linear below the 10th percentile of |x|) when it includes zero or negative values and its largest |x| exceeds 50 times its median |x|.</p>",
            "<img src='assets/features_vs_eC.png' alt='features vs eC' style='max-width:100%'>"]
    (output / "report.html").write_text("\n".join(page))
    print(f"wrote {output / 'report.html'}: {k} cases, {len(source_order)} sources, "
          f"{100 * se[sel].sum() / total:.1f}% of squared error")


if __name__ == "__main__":
    main()
