"""Small controlled experiment: when does removing a corruption make CLEAN recovery worse?

Vault note: Notes/Negative Recovery Damage.md. Damage d_k = eC(all) - eC(all without k)
is negative when removing corruption k makes the CLEAN model further from the truth.
Three hand-picked scenarios on one source (the matched-v2 smoke source 0449+113),
one antenna, one noise seed, imaged with the unchanged matched-v2 pipeline:

  1. noise_helps_amp      strong amplification A=14.5, add thermal noise r in {0.03, 0.1, 0.2}
  2. amp_rescues_faint    very strong noise r in {0.5, 1.0}, add amplification A in {3, 6.4}
  3. attenuation_hides_phase  phase 129 deg, add attenuation A in {0.5, 0.17, 0.05}

Stages (same layout as scripts/create_dataset_v2_matched.py, which does the CASA work):
  ML:   python -m scripts.create_dataset_v2_matched --output OUT --smoke --prepare
  ML:   python -m scripts.negative_damage_examples --output OUT --plan
  ML:   python -m scripts.create_dataset_v2_matched --output OUT --execute --parallel 3
  ML:   python -m scripts.negative_damage_examples --output OUT --report
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

ANTENNA = 3
PHI = 129.
SCENARIOS = [
    dict(id="noise_helps_amp", removed="noise",
         title="1. Thermal noise makes a strong amplitude error look better",
         claim="Amp only: the image is very bright, CLEAN goes deep and fits fake structure from the bad antenna. "
               "Adding noise raises CLEAN's stopping threshold (3σ), so it stops earlier and fits less fake structure.",
         # (label, A, phi, r) ; r=None means ordinary nominal noise (a=1)
         without=[("A=14.5", 14.5, 0., None)],
         levels=[("r=0.03", .03), ("r=0.1", .1), ("r=0.2", .2)],
         combine=lambda r: (14.5, 0., r), alone=lambda r: (1., 0., r)),
    dict(id="amp_rescues_faint", removed="amp",
         title="2. Amplification rescues a source buried in noise",
         claim="Noise only: the source sits near or below CLEAN's threshold, so little or nothing is fitted (eC near 1). "
               "Amplifying one antenna boosts the source on its baselines but not the noise, so CLEAN finds the source.",
         without=[("r=0.5", 1., 0., .5), ("r=1.0", 1., 0., 1.)],
         levels=[("A=3", 3.), ("A=6.4", 6.4)],
         combine=None, alone=lambda A: (A, 0., None)),
    dict(id="attenuation_hides_phase", removed="amp",
         title="3. Attenuation hides a phase error",
         claim=f"With the antenna attenuated it barely contributes, so its {PHI:g}° phase error barely matters. "
               "Removing the attenuation (A=1) brings the phase-corrupted antenna back at full weight.",
         without=[(f"φ={PHI:g}°", 1., PHI, None)],
         levels=[("A=0.5", .5), ("A=0.17", .17), ("A=0.05", .05)],
         combine=lambda A: (A, PHI, None), alone=lambda A: (A, 0., None)),
]


def scenario_grid(s):
    """Ordered (label, A, phi, r) rows: [without k ...], then for each level 'k alone' and 'with k'."""
    rows = [("control", "control", 1., 0., None)]
    rows += [("without", lab, A, phi, r) for lab, A, phi, r in s["without"]]
    for lab, v in s["levels"]:
        rows.append(("alone", lab, *s["alone"](v)))
        if s["combine"] is not None:
            rows.append(("with", f"{s['without'][0][0]} + {lab}", *s["combine"](v)))
    if s["id"] == "amp_rescues_faint":   # every noise level crossed with every amplification
        for lab_n, _, _, r in s["without"]:
            for lab_a, A in s["levels"]:
                rows.append(("with", f"{lab_n} + {lab_a}", A, 0., r))
    return rows


def plan(output):
    output = output.resolve()
    index = json.loads((output / "source_index.json").read_text())
    config = index["configuration"]
    (sid,) = index["ids"]
    prepared = json.loads((output / "prepared" / f"{sid}.json").read_text())
    snr0, ref = prepared["nominal_source_snr"], prepared["reference_sha256"]
    seed = stream_seed(config["seed"], sid, "noise:0")      # same nominal-noise realization as the smoke run
    cases, scenarios, owner = {}, [], {}
    for s in SCENARIOS:
        items = []
        for role, label, A, phi, r in scenario_grid(s):
            c = canonical_case(sid, ref, 0, seed, A, phi, 1. if r is None else r * snr0, ANTENNA)
            cases[c["id"]] = c
            owner.setdefault(c["id"], s["id"])
            items.append(dict(role=role, label=label, A=A, phi_deg=phi, r=r, case=c["id"]))
        scenarios.append(dict({k: v for k, v in s.items() if k in ("id", "removed", "title", "claim")}, items=items))
    units = [dict(id=f"{sid}:{s['id']}", source=sid, slot=0, recipe=None,
                  cases=[cid for cid, o in owner.items() if o == s["id"]]) for s in SCENARIOS]
    source_plan = dict(source=sid, split=config["split"], recipes=[], units=units, rows=[], cases=list(cases.values()),
                       noise_seeds=[seed], counts=dict(units=len(units), cases=len(cases)))
    payload = json.loads(json.dumps(dict(
        schema=3, split=config["split"], smoke=True, configuration=config, configuration_hash=digest(config),
        source_index=index["audit"], prepared_sources=[prepared], source_plans=[source_plan],
        experiment=dict(antenna=ANTENNA, snr0=snr0, scenarios=scenarios),
        counts=dict(sources=1, units=len(units), cases=len(cases)))))
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Plan changed; use a fresh output directory")
    atomic_write_json(path, payload)
    shard = output / "shards" / "N1.json"
    shard.parent.mkdir(exist_ok=True)
    atomic_write_json(shard, dict(num_shards=1, plan_sha256=file_sha(path), owner={u["id"]: 0 for u in units},
                                  estimated_serial_seconds=[0]))
    print("PLAN", len(cases), "cases in", len(units), "units, antenna", ANTENNA, "SNR0", round(snr0))


# ---------------------------------------------------------------- report

def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    from scripts.clean_recovery_noise_sweep import convolve_model

    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    sid = p["prepared_sources"][0]["id"]
    with np.load(output / p["prepared_sources"][0]["reference_masks"]) as m:
        truth, omega = np.asarray(m["truth"], float), np.asarray(m["omega"], bool)
    flux = float(truth.max())
    cy, cx = np.unravel_index(np.argmax(truth), truth.shape)
    half = 30

    def load(case_id):
        cell = output / "cases" / case_id
        rec = json.loads((cell / "case.json").read_text())
        planes = {}
        for key in ("dirty", "model", "residual"):
            with fits.open(output / rec["products"][key]) as h:
                planes[key] = np.squeeze(h[0].data).astype(float)
        with fits.open(output / rec["products"]["clean"]) as h:
            header = h[0].header.copy()
        C = convolve_model(planes["model"], header)
        eC = float(np.sqrt(np.square((C - truth)[omega]).sum() / np.square(truth[omega]).sum()))
        if abs(eC - rec["eC"]) > 1e-6:
            raise ValueError(f"Recomputed eC differs from committed record: {case_id}")
        con = rec["convergence"]
        return dict(eC=rec["eC"], dirty=planes["dirty"], C=C, err=C - truth, iters=con.get("iterdone", 0),
                    threshold=con.get("final_cycle_threshold_jy_per_beam"), model_flux=rec["model_flux"]["sum_all_pixels"],
                    nonzero=int(np.count_nonzero(planes["model"])))

    crop = lambda a: a[cy - half:cy + half + 1, cx - half:cx + half + 1] / flux
    results, sections = {}, []
    for s in p["experiment"]["scenarios"]:
        data = {it["case"]: load(it["case"]) for it in s["items"]}
        rows = [dict(it, **{k: data[it["case"]][k] for k in ("eC", "iters", "threshold", "model_flux", "nonzero")})
                for it in s["items"]]
        # Damage pairs: 'with' case vs the same case without the removed corruption.
        pairs = []
        by_key = {(round(r["A"], 6), round(r["phi_deg"], 6), r["r"]): r for r in rows}
        for r in rows:
            if r["role"] != "with":
                continue
            if s["removed"] == "noise":
                base = by_key[(round(r["A"], 6), round(r["phi_deg"], 6), None)]
            else:
                base = by_key[(1., round(r["phi_deg"], 6), r["r"])]
            pairs.append(dict(with_=r["label"], without=base["label"], eC_with=r["eC"], eC_without=base["eC"],
                              d=r["eC"] - base["eC"]))
        results[s["id"]] = dict(rows=[{k: v for k, v in r.items() if k != "case"} | dict(case=r["case"]) for r in rows],
                                damage=pairs)
        # Figure: one column per case (control omitted), rows dirty / convolved CLEAN model / model - truth.
        shown = [it for it in s["items"] if it["role"] != "control"]
        fig, axes = plt.subplots(3, len(shown), figsize=(2.35 * len(shown), 7.4), layout="constrained", squeeze=False)
        for row, (key, name, cmap) in enumerate([("dirty", "Dirty image", "inferno"),
                                                 ("C", "CLEAN model ⊛ beam", "inferno"),
                                                 ("err", "Model − truth", "RdBu_r")]):
            arrays = [crop(data[it["case"]][key]) for it in shown]
            hi = max(np.abs(a).max() for a in arrays)
            lo = -hi if key == "err" else min(0., min(a.min() for a in arrays))
            for col, (it, a) in enumerate(zip(shown, arrays)):
                ax = axes[row, col]
                im = ax.imshow(a, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    d = data[it["case"]]
                    colour = {"without": "#555", "alone": "#555", "with": "#b2182b"}[it["role"]]
                    ax.set_title(f"{it['label']}\neC = {d['eC']:.2f}\n{d['iters']} CLEAN iters",
                                 fontsize=9, color=colour)
                if col == 0:
                    ax.set_ylabel(name, fontsize=10)
            fig.colorbar(im, ax=axes[row, :], shrink=.85, label="÷ true flux")
        fig.suptitle(s["title"] + f"   (source {sid}, antenna {p['experiment']['antenna']}, central 61×61 px)",
                     fontsize=11)
        name = f"{s['id']}.png"
        fig.savefig(output / name, dpi=130)
        plt.close(fig)
        sections.append((s, rows, pairs, name))

    atomic_write_json(output / "results.json", dict(source=sid, antenna=p["experiment"]["antenna"],
                                                     snr0=p["experiment"]["snr0"], scenarios=results))
    html = ["<!doctype html><meta charset=utf-8><title>Negative recovery damage examples</title>",
            "<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}"
            "table{border-collapse:collapse;margin:.5em 0}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}"
            "th{background:#f3f3f3}.neg{color:#b2182b;font-weight:bold}img{max-width:100%}</style>",
            f"<h1>Negative recovery damage: controlled examples</h1><p>Source {sid}, antenna "
            f"{p['experiment']['antenna']}, one nominal-noise realization, matched-v2 imaging. "
            "eC = ‖B⊛M − B⊛S‖₂ / ‖B⊛S‖₂ (0 = perfect, 1 = empty model). "
            "d = eC(with corruption) − eC(same case without it). Negative d: removing the corruption makes recovery worse. "
            "r = a/SNR₀ is the thermal-noise strength.</p>"]
    for s, rows, pairs, name in sections:
        html.append(f"<h2>{s['title']}</h2><p>{s['claim']}</p><table><tr><th>removed: {s['removed']}</th>"
                    "<th>eC with</th><th>eC without</th><th>d</th></tr>")
        for q in pairs:
            cls = ' class=neg' if q["d"] < 0 else ""
            html.append(f"<tr><td style=text-align:left>{q['with_']} → {q['without']}</td><td>{q['eC_with']:.3f}</td>"
                        f"<td>{q['eC_without']:.3f}</td><td{cls}>{q['d']:+.3f}</td></tr>")
        html.append("</table><table><tr><th>case</th><th>eC</th><th>CLEAN iters</th><th>stop threshold (Jy/beam)</th>"
                    "<th>model flux / true</th><th>nonzero model px</th></tr>")
        for r in rows:
            th = "–" if r["threshold"] is None else f"{r['threshold']:.3g}"
            html.append(f"<tr><td style=text-align:left>{r['label']}</td><td>{r['eC']:.3f}</td><td>{r['iters']}</td>"
                        f"<td>{th}</td><td>{r['model_flux'] / flux:.2f}</td><td>{r['nonzero']}</td></tr>")
        html.append(f"</table><img src='{name}'>")
    (output / "report.html").write_text("\n".join(html))
    print("REPORT", output / "report.html")
    for s, _, pairs, _ in sections:
        print(s["id"], [f"{q['with_']}: d={q['d']:+.3f}" for q in pairs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    plan(a.output) if a.plan else report(a.output)


if __name__ == "__main__":
    main()
