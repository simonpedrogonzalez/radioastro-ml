"""Negative recovery damage: dimming sweeps.

Vault note: Notes/Negative Recovery Damage.md. Damage d_k = eC(all) - eC(all without k) is negative when
removing corruption k makes the CLEAN model further from the truth.

Each experiment fixes one corruption (the "target", whose error gets dimmed) at a level chosen from the earlier
controlled examples, then sweeps a second corruption (the "dimmer") on the same antenna and measures eC for
every value. Change in error = eC(target + dimmer) - eC(target only); negative means the dimmer reduces it.
  rescue       target: thermal noise (source buried in noise)    dimmer: amplification of one antenna
  hide         target: phase error on one antenna                 dimmer: attenuation of that antenna
  noise_helps  target: strong amplification of one antenna        dimmer: thermal noise

Stages (prefix each with bin/run ml, locally and on CHPC):
  ML:   python -m scripts.negative_damage_examples --output OUT --setup --dataset runs/dataset-v2-matched-train
  ML:   python -m scripts.create_dataset_v2_matched --output OUT --config OUT/prepare_config.json --prepare --parallel 3
  ML:   python -m scripts.negative_damage_examples --output OUT --plan
  ML:   python -m scripts.create_dataset_v2_matched --output OUT --execute --parallel 3
  ML:   python -m scripts.negative_damage_examples --output OUT --report       -> OUT/report.html
The dataset run is only read, for its configuration (imaging setup and noise seeds).
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

SOURCE = "0449+113"   # SNR0 ~ 13400, 24 antennas; every earlier controlled example used it
# Fixed target levels picked from the earlier runs (runs/negative-damage-examples-20261109 and -v2-smoke-20261109):
#   rescue: r=0.5 gave eC 0.84 noise-only and 0.52 with A=6.4 (d=-0.32), the largest effect seen; at r=1 the
#           source is lost even with amplification, at r<=0.3 the effect was at most -0.12.
#   hide: the phase-only error grows toward 180 deg (129 deg: 0.18, 171 deg: 0.20) and d kept growing as A fell.
#   noise_helps: A=14.5 was the strongest case on this source (d=-0.04 at r=0.2); every other try was weaker.
EXPERIMENTS = {
    "rescue": dict(
        antenna=3, target=("noise", .5), dimmer="amp", sweep=(1.25, 1.5, 2., 3., 4., 5., 6.4, 8., 10., 12.5, 15., 20.),
        title="Amplification rescues a source buried in noise",
        claim="Noise only: CLEAN stops early (3σ threshold) and recovers too little source flux. Amplifying one "
              "antenna adds source flux on its baselines but not noise, which partly makes up the missing flux."),
    "hide": dict(
        antenna=6, target=("phase", 180.), dimmer="amp", sweep=(.9, .75, .6, .5, .4, .3, .2, .1, .05, .02, .01),
        title="Attenuation hides a phase error on the same antenna",
        claim="The attenuated antenna barely contributes, so its phase error barely reaches the image."),
    "noise_helps": dict(
        antenna=3, target=("amp", 14.5), dimmer="noise", sweep=(.005, .01, .02, .05, .1, .15, .2, .3, .4, .5, .7, 1.),
        title="Thermal noise against a strong amplitude error",
        claim="Amp only: CLEAN goes deep and fits fake structure from the bad antenna. Noise raises CLEAN's "
              "stopping threshold (3σ), so it should stop earlier and fit less of it."),
}
AXIS = {"amp": "amplitude gain A on the antenna", "noise": "thermal-noise strength r = a / SNR₀",
        "phase": "phase φ (deg)"}


def label(kind, value):
    return {"amp": f"A={value:.3g}", "noise": f"r={value:.3g}", "phase": f"φ={value:.0f}°"}[kind]


def physics(*settings):
    """(A, phi, r) for a list of (kind, value) corruptions; r None = nominal noise only."""
    out = dict(A=1., phi=0., r=None)
    for kind, value in settings:
        out[{"amp": "A", "phase": "phi", "noise": "r"}[kind]] = float(value)
    return out


def antenna_share(norms, ant):
    """Fraction of the noiseless visibility power on baselines with this antenna (each baseline counted twice)."""
    power = {a: float(n["V_Ak_over_sigma_L2"]) ** 2 for a, n in norms.items()}
    return power[str(ant)] / (sum(power.values()) / 2)


# ---------------------------------------------------------------- stage 1: setup

def setup(output, dataset):
    output, dataset = output.resolve(), dataset.resolve()
    plan = json.loads((dataset / "plan.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    # Same configuration (imaging setup, seeds) as the dataset, restricted to the experiment source.
    atomic_write_json(output / "prepare_config.json", dict(plan["configuration"], source_ids=[SOURCE]))
    atomic_write_json(output / "setup.json", dict(dataset=str(dataset), dataset_plan_sha256=file_sha(dataset / "plan.json")))
    print("WROTE", output / "prepare_config.json")


# ---------------------------------------------------------------- stage 2: plan

def plan(output):
    output = output.resolve()
    index = json.loads((output / "source_index.json").read_text())
    config = index["configuration"]
    if SOURCE not in index["ids"]:
        raise ValueError(f"{SOURCE} was not prepared in {output}")
    p = json.loads((output / "prepared" / f"{SOURCE}.json").read_text())
    snr0 = float(p["nominal_source_snr"])
    seed = stream_seed(config["seed"], SOURCE, "noise:0")   # dataset slot-0 seed
    cases, owner, experiments = {}, {}, []
    for name, e in EXPERIMENTS.items():
        items = [("control", None, physics()), ("target", None, physics(e["target"]))]
        items += [("sweep", v, physics(e["target"], (e["dimmer"], v))) for v in e["sweep"]]
        rows = []
        for role, value, x in items:
            a = 1. if x["r"] is None else max(1., x["r"] * snr0)
            case = canonical_case(SOURCE, p["reference_sha256"], 0, seed, x["A"], x["phi"], a, e["antenna"])
            cases[case["id"]] = case
            owner.setdefault(case["id"], f"{SOURCE}:{name}")
            rows.append(dict(role=role, value=value, a=a, case=case["id"], **x))
        experiments.append(dict(id=name, antenna=e["antenna"], target=list(e["target"]), dimmer=e["dimmer"],
                                title=e["title"], claim=e["claim"], items=rows))
    units = [dict(id=u, source=SOURCE, slot=0, recipe=None, cases=[i for i, o in owner.items() if o == u])
             for u in sorted(set(owner.values()))]
    source_plans = [dict(source=SOURCE, split=config["split"], recipes=[], units=units, rows=[], cases=list(cases.values()),
                         noise_seeds=[seed], counts=dict(units=len(units), cases=len(cases)))]
    payload = json.loads(json.dumps(dict(
        schema=3, split=config["split"], smoke=False, configuration=config, configuration_hash=digest(config),
        source_index=index["audit"], prepared_sources=[p], source_plans=source_plans,
        experiment=dict(kind="dimming_sweeps", snr0=snr0, antenna_share={str(e["antenna"]): antenna_share(p["norms"], e["antenna"])
                                                                         for e in EXPERIMENTS.values()},
                        experiments=experiments),
        counts=dict(sources=1, units=len(units), cases=len(cases)))))
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Plan changed; use a fresh output directory")
    atomic_write_json(path, payload)
    (output / "shards").mkdir(exist_ok=True)
    atomic_write_json(output / "shards" / "N1.json", dict(num_shards=1, plan_sha256=file_sha(path),
                                                          owner={u["id"]: 0 for u in units}, estimated_serial_seconds=[0]))
    print("PLAN", len(cases), "cases,", len(units), "units, source", SOURCE)


# ---------------------------------------------------------------- stage 3: report

def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    from scripts.clean_recovery_noise_sweep import convolve_model

    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    x = p["prepared_sources"][0]
    sid, snr0 = x["id"], p["experiment"]["snr0"]
    with np.load(output / x["reference_masks"]) as m:
        truth, omega = np.asarray(m["truth"], float), np.asarray(m["omega"], bool)
    assets = output / "report_assets"
    assets.mkdir(exist_ok=True)

    def load(case_id):
        rec = json.loads((output / "cases" / case_id / "case.json").read_text())
        planes = {}
        for key in ("dirty", "model", "clean"):
            with fits.open(output / rec["products"][key]) as h:
                planes[key] = np.squeeze(h[0].data).astype(float)
                if key == "clean":
                    header = h[0].header.copy()
        C = convolve_model(planes["model"], header)
        eC = float(np.sqrt(np.square((C - truth)[omega]).sum() / np.square(truth[omega]).sum()))
        if abs(eC - rec["eC"]) > 1e-6:
            raise ValueError(f"Recomputed eC differs from committed record: {case_id}")
        con = rec["convergence"]
        return dict(eC=rec["eC"], dirty=planes["dirty"], C=C, err=C - truth, iters=con.get("iterdone", 0),
                    flux_ratio=rec["model_flux"]["sum_all_pixels"] / float(truth.max()),
                    nonzero=int(np.count_nonzero(planes["model"])))

    cy, cx = np.unravel_index(np.argmax(truth), truth.shape)
    half, flux = 30, float(truth.max())
    crop = lambda a: a[cy - half:cy + half + 1, cx - half:cx + half + 1] / flux
    html = ["<!doctype html><meta charset=utf-8><title>Negative recovery damage: dimming sweeps</title>",
            "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;line-height:1.45}"
            "img{max-width:100%;display:block;margin:.6em 0}.neg{color:#b2182b;font-weight:bold}</style>",
            "<h1>Negative recovery damage: dimming sweeps</h1>",
            f"<p>Source {sid} (SNR₀ = {snr0:.0f}), one noise realization, matched-v2 imaging. Each section fixes one "
            "corruption (the target) and sweeps a second one on the same antenna. eC = ‖B⊛M − B⊛S‖₂ / ‖B⊛S‖₂ "
            "(0 perfect, 1 empty model). Change in error = eC(target + dimmer) − eC(target only); negative = the "
            "dimmer reduces the target's error. r = a/SNR₀, with a the thermal-noise multiplier.</p>"]
    results = {}
    for e in p["experiment"]["experiments"]:
        data = {it["case"]: load(it["case"]) for it in e["items"]}
        control = data[next(it["case"] for it in e["items"] if it["role"] == "control")]
        target = next(it for it in e["items"] if it["role"] == "target")
        base = data[target["case"]]["eC"]
        sweep = [dict(value=it["value"], case=it["case"], eC=data[it["case"]]["eC"], change=data[it["case"]]["eC"] - base,
                      iters=data[it["case"]]["iters"], flux_ratio=data[it["case"]]["flux_ratio"],
                      nonzero=data[it["case"]]["nonzero"]) for it in e["items"] if it["role"] == "sweep"]
        best = min(sweep, key=lambda q: q["change"])
        tkind, tval = e["target"]
        tname, bname = label(tkind, tval), f"{label(tkind, tval)} + {label(e['dimmer'], best['value'])}"

        # Images: target only vs. strongest dimming.
        shown = [(tname, data[target["case"]], "#444"), (bname, data[best["case"]], "#b2182b")]
        panels = [("Dirty image", "inferno", lambda d: d["dirty"]),
                  ("Dirty − uncorrupted dirty", "RdBu_r", lambda d: d["dirty"] - control["dirty"]),
                  ("CLEAN model ⊛ beam", "inferno", lambda d: d["C"]),
                  ("Model − truth", "RdBu_r", lambda d: d["err"])]
        fig, axes = plt.subplots(len(panels), 2, figsize=(5.6, 10.2), layout="constrained", squeeze=False)
        for row, (name, cmap, fn) in enumerate(panels):
            arrays = [crop(fn(d)) for _, d, _ in shown]
            hi = max(np.abs(a).max() for a in arrays) or 1.
            lo = -hi if cmap == "RdBu_r" else min(0., min(a.min() for a in arrays))
            for col, ((title, d, color), a) in enumerate(zip(shown, arrays)):
                ax = axes[row, col]
                im = ax.imshow(a, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{title}\neC = {d['eC']:.3f}, {d['iters']} iters", fontsize=9, color=color)
                if col == 0:
                    ax.set_ylabel(name, fontsize=9)
            fig.colorbar(im, ax=axes[row, :], shrink=.85, label="÷ true flux")
        fig.savefig(assets / f"{e['id']}_images.png", dpi=130)
        plt.close(fig)

        # Line plot: change in error vs. the dimmer, at the fixed target.
        xs = [q["value"] for q in sweep]
        ys = [q["change"] for q in sweep]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        ax.axhline(0, color="#888", lw=1, ls="--", label=f"{tname} only (eC = {base:.3f})")
        ax.plot(xs, ys, "o-", color="#2166ac", lw=1.6, ms=5, label=f"{tname} + {AXIS[e['dimmer']].split(' ')[0]}")
        ax.plot([best["value"]], [best["change"]], "o", ms=11, mfc="none", mec="#b2182b", mew=2,
                label=f"strongest: {label(e['dimmer'], best['value'])}, {best['change']:+.3f}")
        ax.set_xscale("log")
        if e["dimmer"] == "amp" and max(xs) < 1:
            ax.invert_xaxis()   # stronger attenuation to the right
        ax.set_xlabel(AXIS[e["dimmer"]])
        ax.set_ylabel("change in eC vs. target only")
        ax.set_title(f"{tname} fixed, antenna {e['antenna']}", fontsize=10)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, frameon=False)
        fig.savefig(assets / f"{e['id']}_sweep.png", dpi=130)
        plt.close(fig)

        share = p["experiment"]["antenna_share"][str(e["antenna"])]
        results[e["id"]] = dict(antenna=e["antenna"], antenna_share=share, target=e["target"], dimmer=e["dimmer"],
                                eC_target=base, eC_control=control["eC"], strongest=best, sweep=sweep)
        cls = " class=neg" if best["change"] < 0 else ""
        html += [f"<h2>{e['title']}</h2>",
                 f"<p>{e['claim']} Fixed: <b>{tname}</b> on antenna {e['antenna']} ({100 * share:.1f}% of the "
                 f"visibility power), eC = {base:.3f}. Swept: {AXIS[e['dimmer']]}. Strongest dimming at "
                 f"<b>{label(e['dimmer'], best['value'])}</b>: eC = {best['eC']:.3f}, change "
                 f"<span{cls}>{best['change']:+.3f}</span>.</p>",
                 f"<img src='report_assets/{e['id']}_images.png' alt='{e['id']} images'>",
                 f"<img src='report_assets/{e['id']}_sweep.png' alt='{e['id']} sweep'>"]
    atomic_write_json(output / "results.json", results)
    (output / "report.html").write_text("\n".join(html))
    print("REPORT", output / "report.html")
    for name, r in results.items():
        print(name, "target eC %.3f" % r["eC_target"], "strongest", r["strongest"]["value"],
              "change %+.3f" % r["strongest"]["change"], [f"{q['value']:.3g}:{q['change']:+.3f}" for q in r["sweep"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, help="Matched v2 run with plan.json (--setup)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--setup", action="store_true")
    g.add_argument("--plan", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.setup:
        if a.dataset is None:
            ap.error("--setup needs --dataset")
        setup(a.output, a.dataset)
    elif a.plan:
        plan(a.output)
    else:
        report(a.output)


if __name__ == "__main__":
    main()
