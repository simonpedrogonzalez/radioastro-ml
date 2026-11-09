"""Controlled examples of negative recovery damage, picked from a matched Dataset v2 run.

Vault note: Notes/Negative Recovery Damage.md. Damage d_k = eC(all) - eC(all without k) is
negative when removing corruption k makes the CLEAN model further from the truth.

Mechanisms (each searched in the dataset rows, then re-simulated with a small sweep around the
strongest row; amplitude and phase always act on the same antenna):
  rescue       amplification lifts a source buried in thermal noise above CLEAN's threshold (d_amp < 0)
  hide         attenuation hides a phase error on the same antenna (d_amp < 0)
  noise_helps  thermal noise stops CLEAN before it fits a strong amplitude error's artifacts (d_noise < 0)

Stages (CHPC: run each ML stage with bin/run ml python -m ...):
  ML:   python -m scripts.negative_damage_examples --output OUT --select --dataset runs/dataset-v2-matched-train
  ML:   python -m scripts.create_dataset_v2_matched --output OUT --config OUT/prepare_config.json --prepare --parallel 3
  ML:   python -m scripts.negative_damage_examples --output OUT --plan
  ML:   python -m scripts.create_dataset_v2_matched --output OUT --execute --parallel 3
  ML:   python -m scripts.negative_damage_examples --output OUT --report
The dataset run is only read. Cases reuse its configuration, reference hashes and noise seed, so a
re-simulated dataset case has the same case id and its eC is checked against the dataset value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.create_dataset_v2 import file_sha
from scripts.dataset_v2 import canonical_case, stream_seed
from scripts.injection_bounds_report import digest
from scripts.preprocessing.schema import atomic_write_json

RESCUE_R = (.02, .05, .1, .2, .3)        # thermal-noise scan; the figure shows the lowest r with a clear effect
CLEAR = -.1                          # "clear" damage for choosing that r
HIDE_A = (.5, .2, .05)               # attenuation sweep, plus the dataset value
MECHANISMS = {
    "rescue": dict(mask=5, k=0, need=lambda R: R["A"] > 1, removed="amp",
                   title="Amplification rescues a source buried in noise",
                   claim="Noise only: CLEAN stops early (threshold 3σ) and recovers too little source flux, or misses the source "
                         "when it is near the threshold. Amplifying one antenna adds source flux on its baselines but not "
                         "noise (gains act before the noise is added), which partly cancels the missing flux."),
    "hide": dict(mask=3, k=0, need=lambda R: R["A"] < 1, removed="amp",
                 title="Attenuation hides a phase error on the same antenna",
                 claim="The attenuated antenna barely contributes, so its phase error barely reaches the image. "
                       "Removing the attenuation (A=1) brings the phase-corrupted antenna back at full weight."),
    "noise_helps": dict(mask=5, k=2, need=lambda R: R["A"] > 1, removed="noise",
                        title="Thermal noise stops CLEAN before it fits a strong amplitude error",
                        claim="Amp only: CLEAN goes deep and fits fake structure from the bad antenna. "
                              "Noise raises CLEAN's stopping threshold (3σ), so it stops earlier and fits less of it."),
}


def antenna_share(norms, ant):
    """Fraction of the noiseless visibility power on baselines with this antenna (each baseline counted twice)."""
    power = {a: float(n["V_Ak_over_sigma_L2"]) ** 2 for a, n in norms.items()}
    return power[str(ant)] / (sum(power.values()) / 2)


# ---------------------------------------------------------------- stage 1: select

def select(output, dataset, top):
    output, dataset = output.resolve(), dataset.resolve()
    plan = json.loads((dataset / "plan.json").read_text())
    rows = json.loads((dataset / "rows.json").read_text())["rows"]
    recipes = {r["id"]: r for sp in plan["source_plans"] for r in sp["recipes"]}
    prepared = {p["id"]: p for p in plan["prepared_sources"]}
    chosen, tables = {}, {}
    for name, m in MECHANISMS.items():
        found = []
        for row in rows:
            if row["mask"] != m["mask"] or row["recipe"] is None:
                continue
            R = recipes[row["recipe"]]
            if not m["need"](R) or row["d"][m["k"]] >= 0:
                continue
            p = prepared[row["source"]]
            found.append(dict(row=row["id"], case=row["case"], removal_case=row["removal"][m["k"]],
                              source=row["source"], antenna=R["antenna"], A=R["A"] if m["mask"] & 1 else 1.,
                              phi_deg=R["phi_deg"] if m["mask"] & 2 else 0., r=R["r"] if m["mask"] & 4 else None,
                              a=R["a"] if m["mask"] & 4 else 1., snr0=p["nominal_source_snr"], antennas=len(p["norms"]),
                              antenna_share=antenna_share(p["norms"], R["antenna"]),
                              eC=row["eC"], d=row["d"][m["k"]]))
        if not found:
            raise ValueError(f"No negative-damage rows for {name} in {dataset}")
        found.sort(key=lambda x: x["d"])
        tables[name], chosen[name] = found[:top], found[0]
        print(name, f"{len(found)} rows with d<0, strongest", {k: found[0][k] for k in
              ("source", "antenna", "A", "phi_deg", "r", "a", "d", "eC")})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "selection.json", dict(dataset=str(dataset), dataset_plan_sha256=file_sha(dataset / "plan.json"),
                                                      chosen=chosen, candidates=tables))
    # Same configuration as the dataset (same seeds and case ids), restricted to the chosen sources.
    config = dict(plan["configuration"], source_ids=sorted({c["source"] for c in chosen.values()}))
    atomic_write_json(output / "prepare_config.json", config)
    print("WROTE", output / "selection.json", output / "prepare_config.json")


# ---------------------------------------------------------------- stage 2: plan

def merged(grid, exact):
    """Sweep values plus the exact dataset value (kept bit-identical so the dataset case is reproduced)."""
    return sorted([v for v in grid if abs(v - exact) > .02 * exact] + [exact])


def items_for(name, c):
    """Cases of one mechanism: (label, A, phi, r, base label or None, group)."""
    A, phi, r = c["A"], c["phi_deg"], c["r"]
    out = [("control", 1., 0., None, None, "control")]
    if name == "rescue":
        half = max(1.5, round(A / 2, 2))
        out.append((f"A={A:.3g}", A, 0., None, None, "amp"))
        for rr in merged(RESCUE_R, r):
            n = f"r={rr:.3g}"
            out += [(n, 1., 0., rr, None, rr), (f"{n} + A={half:.3g}", half, 0., rr, n, rr),
                    (f"{n} + A={A:.3g}", A, 0., rr, n, rr)]
    elif name == "hide":
        base = f"φ={phi:.0f}°"
        out.append((base, 1., phi, None, None, "phase"))
        for aa in reversed(merged(HIDE_A, A)):
            out += [(f"A={aa:.3g}", aa, 0., None, None, aa), (f"{base} + A={aa:.3g}", aa, phi, None, base, aa)]
    else:
        base = f"A={A:.3g}"
        out.append((base, A, 0., None, None, "amp"))
        for rr in (r / 2, r):
            out += [(f"r={rr:.3g}", 1., 0., rr, None, rr), (f"{base} + r={rr:.3g}", A, 0., rr, base, rr)]
    return out


def plan(output):
    output = output.resolve()
    selection = json.loads((output / "selection.json").read_text())
    index = json.loads((output / "source_index.json").read_text())
    config = index["configuration"]
    prepared = {sid: json.loads((output / "prepared" / f"{sid}.json").read_text()) for sid in index["ids"]}
    seeds = {sid: stream_seed(config["seed"], sid, "noise:0") for sid in prepared}   # dataset slot-0 seed
    cases, owner, mechanisms = {}, {}, []
    for name, c in selection["chosen"].items():
        sid, p = c["source"], prepared[c["source"]]
        items = []
        for label, A, phi, r, base, group in items_for(name, c):
            case = canonical_case(sid, p["reference_sha256"], 0, seeds[sid], A, phi,
                                  1. if r is None else (c["a"] if r == c["r"] else max(1., r * p["nominal_source_snr"])), c["antenna"])
            cases[case["id"]] = case
            owner.setdefault(case["id"], f"{sid}:{name}")
            items.append(dict(label=label, A=A, phi_deg=phi, r=r, base=base, group=group, case=case["id"]))
        mechanisms.append(dict(id=name, source=sid, antenna=c["antenna"], chosen=c, items=items,
                               **{k: MECHANISMS[name][k] for k in ("title", "claim", "removed")}))
    source_plans = []
    for sid in prepared:
        units = [dict(id=u, source=sid, slot=0, recipe=None, cases=[i for i, o in owner.items() if o == u])
                 for u in sorted(set(owner.values())) if u.startswith(sid + ":")]
        mine = [x for x in cases.values() if x["source"] == sid]
        source_plans.append(dict(source=sid, split=config["split"], recipes=[], units=units, rows=[], cases=mine,
                                 noise_seeds=[seeds[sid]], counts=dict(units=len(units), cases=len(mine))))
    payload = json.loads(json.dumps(dict(
        schema=3, split=config["split"], smoke=False, configuration=config, configuration_hash=digest(config),
        source_index=index["audit"], prepared_sources=list(prepared.values()), source_plans=source_plans,
        experiment=dict(selection_sha256=file_sha(output / "selection.json"), mechanisms=mechanisms),
        counts=dict(sources=len(prepared), units=len(set(owner.values())), cases=len(cases)))))
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Plan changed; use a fresh output directory")
    atomic_write_json(path, payload)
    units = [u["id"] for sp in source_plans for u in sp["units"]]
    (output / "shards").mkdir(exist_ok=True)
    atomic_write_json(output / "shards" / "N1.json", dict(num_shards=1, plan_sha256=file_sha(path),
                                                          owner={u: 0 for u in units}, estimated_serial_seconds=[0]))
    print("PLAN", len(cases), "cases,", len(units), "units, sources", sorted(prepared))


# ---------------------------------------------------------------- stage 3: report

def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    from scripts.clean_recovery_noise_sweep import convolve_model

    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    prepared = {x["id"]: x for x in p["prepared_sources"]}
    masks = {}
    for sid, x in prepared.items():
        with np.load(output / x["reference_masks"]) as m:
            masks[sid] = (np.asarray(m["truth"], float), np.asarray(m["omega"], bool))

    def load(case_id, sid):
        rec = json.loads((output / "cases" / case_id / "case.json").read_text())
        planes = {}
        for key in ("dirty", "model", "clean"):
            with fits.open(output / rec["products"][key]) as h:
                planes[key] = np.squeeze(h[0].data).astype(float)
                if key == "clean":
                    header = h[0].header.copy()
        truth, omega = masks[sid]
        C = convolve_model(planes["model"], header)
        eC = float(np.sqrt(np.square((C - truth)[omega]).sum() / np.square(truth[omega]).sum()))
        if abs(eC - rec["eC"]) > 1e-6:
            raise ValueError(f"Recomputed eC differs from committed record: {case_id}")
        con = rec["convergence"]
        return dict(eC=rec["eC"], dirty=planes["dirty"], C=C, err=C - truth, iters=con.get("iterdone", 0),
                    threshold=con.get("final_cycle_threshold_jy_per_beam"),
                    flux_ratio=rec["model_flux"]["sum_all_pixels"] / float(truth.max()),
                    nonzero=int(np.count_nonzero(planes["model"])))

    html = ["<!doctype html><meta charset=utf-8><title>Negative recovery damage examples</title>",
            "<style>body{font-family:sans-serif;max-width:1250px;margin:2em auto;padding:0 1em}"
            "table{border-collapse:collapse;margin:.5em 0}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}"
            "th{background:#f3f3f3}td:first-child{text-align:left}.neg{color:#b2182b;font-weight:bold}"
            "img{max-width:100%}</style>",
            "<h1>Negative recovery damage: strongest dataset cases, re-simulated</h1>",
            f"<p>Cases picked from {selection['dataset']} (most negative d per mechanism) and swept around. "
            "eC = ‖B⊛M − B⊛S‖₂ / ‖B⊛S‖₂ (0 perfect, 1 empty model). d = eC(with) − eC(without), negative = removing "
            "the corruption makes recovery worse. r = a/SNR₀ (source peak S/N in the image ≈ 1/r), "
            "a = thermal-noise multiplier on the nominal noise.</p>"]
    results = {}
    for m in p["experiment"]["mechanisms"]:
        sid, chosen = m["source"], m["chosen"]
        data = {it["case"]: load(it["case"], sid) for it in m["items"]}
        by_label = {it["label"]: dict(it, **data[it["case"]]) for it in m["items"]}
        pairs = [dict(with_=it["label"], without=it["base"], group=it["group"], eC_with=by_label[it["label"]]["eC"],
                      eC_without=by_label[it["base"]]["eC"], d=by_label[it["label"]]["eC"] - by_label[it["base"]]["eC"])
                 for it in m["items"] if it["base"]]
        # Reproduction check against the dataset row (same case ids when the configuration matches).
        repro = {k: data[chosen[k]]["eC"] for k in ("case", "removal_case") if chosen[k] in data}
        if m["id"] == "rescue":
            clear = [q for q in pairs if q["d"] <= CLEAR]
            r_fig = min((q["group"] for q in clear), default=min(pairs, key=lambda q: q["d"])["group"])
            shown = [it for it in m["items"] if it["group"] in ("amp", r_fig) and it["label"] != "control"]
        else:
            shown = [it for it in m["items"] if it["label"] != "control"]
        control = data[next(it["case"] for it in m["items"] if it["label"] == "control")]
        truth = masks[sid][0]
        cy, cx = np.unravel_index(np.argmax(truth), truth.shape)
        half, flux = 30, float(truth.max())
        crop = lambda a: a[cy - half:cy + half + 1, cx - half:cx + half + 1] / flux
        panels = [("dirty", "Dirty image", "inferno", lambda d: d["dirty"]),
                  ("ddirty", "Dirty − uncorrupted dirty", "RdBu_r", lambda d: d["dirty"] - control["dirty"]),
                  ("C", "CLEAN model ⊛ beam", "inferno", lambda d: d["C"]),
                  ("err", "Model − truth", "RdBu_r", lambda d: d["err"])]
        fig, axes = plt.subplots(len(panels), len(shown), figsize=(2.3 * len(shown) + .8, 9.4),
                                 layout="constrained", squeeze=False)
        for row, (key, name, cmap, fn) in enumerate(panels):
            arrays = [crop(fn(data[it["case"]])) for it in shown]
            hi = max(np.abs(a).max() for a in arrays) or 1.
            lo = -hi if cmap == "RdBu_r" else min(0., min(a.min() for a in arrays))
            for col, (it, a) in enumerate(zip(shown, arrays)):
                ax = axes[row, col]
                im = ax.imshow(a, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    d = data[it["case"]]
                    ax.set_title(f"{it['label']}\neC = {d['eC']:.2f}, {d['iters']} iters",
                                 fontsize=9, color="#b2182b" if it["base"] else "#444")
                if col == 0:
                    ax.set_ylabel(name, fontsize=9)
            fig.colorbar(im, ax=axes[row, :], shrink=.85, label="÷ true flux")
        fig.suptitle(f"{m['title']}  (source {sid}, antenna {m['antenna']}, "
                     f"{chosen['antennas']} antennas, central 61×61 px)", fontsize=11)
        fig.savefig(output / f"{m['id']}.png", dpi=130)
        plt.close(fig)

        results[m["id"]] = dict(source=sid, antenna=m["antenna"], chosen=chosen, reproduced_dataset_eC=repro,
                                damage=pairs, cases=[{k: v for k, v in by_label[it["label"]].items()
                                                      if k not in ("dirty", "C", "err")} for it in m["items"]])
        snr0 = chosen["snr0"]
        html.append(f"<h2>{m['title']}</h2><p>{m['claim']}</p>")
        html.append(f"<p>Source {sid} (SNR₀ = {snr0:.0f}, {chosen['antennas']} antennas), antenna {m['antenna']} "
                    f"carries {100 * chosen['antenna_share']:.1f}% of the visibility power. Dataset row {chosen['row']}: "
                    f"d = {chosen['d']:+.3f}" + (f", reproduced eC {repro.get('case', float('nan')):.3f} vs dataset "
                    f"{chosen['eC']:.3f}" if "case" in repro else "") + ".</p>")
        html.append(f"<table><tr><th>with → without (removed: {m['removed']})</th><th>a</th>"
                    "<th>eC with</th><th>eC without</th><th>d</th></tr>")
        for q in pairs:
            r = by_label[q["with_"]]["r"]
            a = "1" if r is None else f"{max(1., r * snr0):.0f}"
            cls = " class=neg" if q["d"] < 0 else ""
            html.append(f"<tr><td>{q['with_']} → {q['without']}</td><td>{a}</td><td>{q['eC_with']:.3f}</td>"
                        f"<td>{q['eC_without']:.3f}</td><td{cls}>{q['d']:+.3f}</td></tr>")
        html.append("</table><table><tr><th>case</th><th>eC</th><th>CLEAN iters</th><th>stop threshold (Jy/beam)</th>"
                    "<th>model flux / true</th><th>nonzero model px</th></tr>")
        for it in m["items"]:
            x = by_label[it["label"]]
            th = "–" if x["threshold"] is None else f"{x['threshold']:.3g}"
            html.append(f"<tr><td>{it['label']}</td><td>{x['eC']:.3f}</td><td>{x['iters']}</td><td>{th}</td>"
                        f"<td>{x['flux_ratio']:.2f}</td><td>{x['nonzero']}</td></tr>")
        html.append(f"</table><img src='{m['id']}.png'>")
        cand = selection["candidates"][m["id"]]
        html.append("<details><summary>Strongest dataset rows for this mechanism</summary><table><tr><th>row</th>"
                    "<th>A</th><th>φ</th><th>r</th><th>a</th><th>SNR₀</th><th>ant share</th><th>eC</th><th>d</th></tr>")
        for c in cand:
            html.append(f"<tr><td>{c['row']}</td><td>{c['A']:.3g}</td><td>{c['phi_deg']:.0f}</td><td>{'–' if c['r'] is None else format(c['r'], '.3g')}</td>"
                        f"<td>{c['a']:.0f}</td><td>{c['snr0']:.0f}</td><td>{100 * c['antenna_share']:.1f}%</td>"
                        f"<td>{c['eC']:.3f}</td><td class=neg>{c['d']:+.3f}</td></tr>")
        html.append("</table></details>")
    atomic_write_json(output / "results.json", results)
    (output / "report.html").write_text("\n".join(html))
    print("REPORT", output / "report.html")
    for name, r in results.items():
        print(name, r["source"], "ant", r["antenna"], [f"{q['with_']}: {q['d']:+.3f}" for q in r["damage"]],
              "reproduced", r["reproduced_dataset_eC"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, help="Matched v2 run with plan.json and rows.json (--select)")
    ap.add_argument("--top", type=int, default=10, help="Candidate rows kept per mechanism (--select)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--plan", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.select:
        if a.dataset is None:
            ap.error("--select needs --dataset")
        select(a.output, a.dataset, a.top)
    elif a.plan:
        plan(a.output)
    else:
        report(a.output)


if __name__ == "__main__":
    main()
