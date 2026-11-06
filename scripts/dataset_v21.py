"""Dataset v2.1: a manifest that joins verified matched-v2 runs, filtered, at load time.

Nothing is copied or modified. A manifest lists parts (verified runs with <split>.npz and
plan.json) and one row filter applied to every part:
  --drop-weak   drop a row if a corruption it contains came from a weak band
                (amp_band or phase_band "weak" in the run's plan, for an active type)
  --amp-max A   drop a row if it contains amplitude corruption with gain A > amp-max
Rows are dropped whole and never changed. Each row's labels come only from its own matched
cases (and its own run's control), so removing other rows or joining runs leaves them valid.
Row, case and recipe ids are prefixed with the part tag; source ids are kept, so source-level
splits and folds group a source's rows from every part together. File hashes are recorded and
re-checked on load.

Build (writes the manifest and a short HTML summary next to it):
  python -m scripts.dataset_v21 build --output runs/dataset-v2.1-train/manifest.json --split train \\
      --part v2=runs/dataset-v2-matched-train --part ext=runs/dataset-v2.1-matched-train-extension \\
      --drop-weak --amp-max 8.5468
ML scripts accept the manifest path wherever they take a dataset folder (e.g. --train).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ID_KEYS = ("row_ids", "case_ids", "recipe_ids")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_part(run, split):
    run = _resolve(run)
    with np.load(run / f"{split}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    plan = json.loads((run / "plan.json").read_text())
    if plan["split"] != split:
        raise ValueError(f"{run}: plan split is {plan['split']}, expected {split}")
    return data, plan


def row_filter(data, plan, drop_weak, amp_max):
    """Boolean keep-mask and the count of rows dropped for each reason."""
    recipes = {r["id"]: r for sp in plan["source_plans"] for r in sp["recipes"]}
    b = data["b"].astype(bool)
    keep = np.ones(len(b), bool)
    reasons = Counter()
    for i, rid in enumerate(map(str, data["recipe_ids"])):
        if not rid:
            continue   # control row: nothing injected
        r = recipes[rid]
        why = []
        if drop_weak and b[i, 0] and r["amp_band"] == "weak": why.append("weak amp")
        if drop_weak and b[i, 1] and r["phase_band"] == "weak": why.append("weak phase")
        if amp_max is not None and b[i, 0] and r["A"] > amp_max: why.append(f"A > {amp_max:g}")
        if why:
            keep[i] = False
            reasons[" + ".join(why)] += 1
    return keep, dict(reasons)


def build(output, split, parts, drop_weak, amp_max):
    output = _resolve(output)
    tags = [tag for tag, _ in parts]
    if len(set(tags)) != len(tags) or any(not t or "/" in t for t in tags):
        raise ValueError("Part tags must be unique, non-empty and without '/'")
    entries, names, sources, seeds = [], None, [], []
    for tag, run in parts:
        data, plan = read_part(run, split)
        if names is None:
            names = data["feature_names"].tolist()
        elif data["feature_names"].tolist() != names:
            raise ValueError(f"{run}: feature schema differs from the first part")
        seeds.append(plan["configuration"]["seed"])
        keep, reasons = row_filter(data, plan, drop_weak, amp_max)
        sources.append(set(map(str, data["source_ids"])))
        run_path = _resolve(run)
        entries.append(dict(tag=tag, run=str(run), npz_sha256=sha256(run_path / f"{split}.npz"),
                            plan_sha256=sha256(run_path / "plan.json"), seed=plan["configuration"]["seed"],
                            range_policy=plan["configuration"].get("range_policy"),
                            rows=int(len(keep)), kept=int(keep.sum()), dropped=reasons))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Parts share a sampling seed; their recipes and noise would repeat")
    manifest = dict(schema="dataset_v2.1_manifest", split=split, feature_names=names,
                    filter=dict(drop_weak=bool(drop_weak), amp_max=amp_max), parts=entries,
                    rows=int(sum(e["kept"] for e in entries)), sources=len(set().union(*sources)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    summary(output, manifest)
    print(f"Manifest: {output} ({manifest['rows']} rows from {len(entries)} parts, {manifest['sources']} sources)")
    return manifest


def load_manifest(path, split=None):
    """Joined, filtered arrays of every part, in the same layout as a <split>.npz."""
    path = _resolve(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "dataset_v2.1_manifest":
        raise ValueError(f"{path} is not a dataset v2.1 manifest")
    if split is not None and manifest["split"] != split:
        raise ValueError(f"{path} is a {manifest['split']} manifest, expected {split}")
    pieces = []
    for e in manifest["parts"]:
        run = _resolve(e["run"])
        if sha256(run / f"{manifest['split']}.npz") != e["npz_sha256"] or sha256(run / "plan.json") != e["plan_sha256"]:
            raise ValueError(f"Part {e['tag']} changed since the manifest was built; rebuild it")
        data, plan = read_part(run, manifest["split"])
        keep, _ = row_filter(data, plan, manifest["filter"]["drop_weak"], manifest["filter"]["amp_max"])
        if int(keep.sum()) != e["kept"]:
            raise ValueError(f"Part {e['tag']}: filtered row count differs from the manifest")
        n = len(keep)
        part = {k: (v[keep] if k != "feature_names" and v.shape[:1] == (n,) else v) for k, v in data.items()}
        for k in ID_KEYS:
            part[k] = np.asarray([f"{e['tag']}/{x}" if x else "" for x in map(str, part[k])])
        snr = {p["id"]: p["nominal_source_snr"] for p in plan["prepared_sources"]}
        part["snr0"] = np.asarray([snr[str(s)] for s in part["source_ids"]], float)
        part["noise_ratio"] = part["params"][:, 2] / part["snr0"]
        part["part"] = np.full(int(keep.sum()), e["tag"])
        pieces.append(part)
    joined = {k: (pieces[0][k] if k == "feature_names" else np.concatenate([p[k] for p in pieces])) for k in pieces[0]}
    if len(set(joined["row_ids"])) != len(joined["row_ids"]):
        raise ValueError("Duplicate row ids across parts")
    return joined


def summary(path, manifest):
    """Small HTML page: counts per part and the injected-parameter distributions of the kept rows."""
    import html
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    data = load_manifest(path)
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    b, A, phi, r = data["b"].astype(bool), data["params"][:, 0], data["params"][:, 1], data["noise_ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    panels = ((axes[0], A, b[:, 0], "gain A (log)", np.geomspace(max(A[b[:, 0]].min(), 1e-3), A[b[:, 0]].max(), 40), True),
              (axes[1], np.abs(phi), b[:, 1], "|φ| (deg)", np.linspace(0, 180, 37), False),
              (axes[2], r, b[:, 2], "noise r = a/SNR₀ (log)", np.geomspace(r[b[:, 2]].min(), r[b[:, 2]].max(), 40), True))
    for ax, x, active, label, bins, log in panels:
        for i, e in enumerate(manifest["parts"]):
            sel = active & (data["part"] == e["tag"])
            ax.hist(x[sel], bins=bins, histtype="step", lw=2, color=colors[i % 4], label=f"{e['tag']} ({sel.sum()} rows)")
        ax.set(xlabel=label, ylabel="rows with this type active")
        if log:
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    image = path.with_name(path.stem + "_parameters.png")
    fig.savefig(image, dpi=120); plt.close(fig)
    f = manifest["filter"]
    rows = "".join(f"<tr><td>{html.escape(e['tag'])}</td><td>{html.escape(e['run'])}</td><td>{e['seed']}</td>"
                   f"<td>{html.escape(str(e['range_policy']))}</td><td>{e['rows']}</td><td>{e['kept']}</td>"
                   f"<td>{html.escape(', '.join(f'{k}: {v}' for k, v in e['dropped'].items()) or '—')}</td></tr>" for e in manifest["parts"])
    counts = Counter(zip(data["part"], data["mask"].tolist()))
    names = ["control", "amp", "phase", "amp+phase", "noise", "amp+noise", "phase+noise", "amp+phase+noise"]
    mask_rows = "".join(f"<tr><td>{names[m]}</td>" + "".join(f"<td>{counts[(e['tag'], m)]}</td>" for e in manifest["parts"]) + "</tr>"
                        for m in range(8))
    path.with_suffix(".html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Dataset v2.1 manifest</title><style>body{font:15px system-ui;max-width:1200px;"
        "margin:auto;padding:1.5rem;color:#1b2932}table{border-collapse:collapse;margin:.5rem 0 1.2rem}td,th{border:1px solid #ccd;"
        "padding:.3rem .6rem;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}</style>"
        f"<h1>Dataset v2.1 ({html.escape(manifest['split'])}): {manifest['rows']} rows, {manifest['sources']} sources</h1>"
        f"<p>Joined at load time from the parts below; no files are copied or changed. Filter: drop weak-band rows = "
        f"{f['drop_weak']}, amplitude gain cap = {f['amp_max']}. A row is dropped whole when a corruption it contains violates "
        "the filter; its other rows (e.g. the phase-only row of the same recipe) stay.</p>"
        "<table><tr><th>Part</th><th>Run</th><th>Seed</th><th>Range policy</th><th>Rows</th><th>Kept</th><th>Dropped</th></tr>"
        f"{rows}</table><h2>Kept rows per injection mask</h2><table><tr><th>Mask</th>"
        + "".join(f"<th>{html.escape(e['tag'])}</th>" for e in manifest["parts"]) + f"</tr>{mask_rows}</table>"
        f"<h2>Injected parameters of kept rows</h2><img src='{image.name}' alt='Distributions of A, phase and noise per part'>")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output", type=Path, required=True, help="Manifest path (.json)")
    b.add_argument("--split", choices=("train", "val", "test"), required=True)
    b.add_argument("--part", action="append", required=True, metavar="TAG=RUN")
    b.add_argument("--drop-weak", action="store_true")
    b.add_argument("--amp-max", type=float, default=None)
    a = p.parse_args()
    parts = [tuple(x.split("=", 1)) for x in a.part]
    if any(len(x) != 2 for x in parts):
        p.error("--part must be TAG=RUN")
    build(a.output, a.split, parts, a.drop_weak, a.amp_max)


if __name__ == "__main__":
    main()
