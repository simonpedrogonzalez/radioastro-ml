"""Why does hif_selfcal skip cases? Run experiment B on controls and corrupted cases, show its own images.

hif_selfcal builds its model with auto-multithresh. The mask threshold is the larger of
  sidelobe threshold = sidelobethreshold (2.0) x PSF max sidelobe x peak
  noise threshold    = noisethreshold (5.0) x rms
and regions smaller than minbeamfrac (0.3) of a beam are pruned. For a point source the region above a
level L (as a fraction of the peak) covers ln(1/L) beams, so it is pruned when L > exp(-0.3) = 0.74.
An empty mask means no model, so self-calibration never solves. This report shows, per case, the pipeline's
initial image with its mask, the PSF and these numbers, so controls and corruptions can be compared.

  python -m scripts.vla_selfcal_mask_report --output OUT --run --source-run RUN --cases representative --parallel 2
  python -m scripts.vla_selfcal_mask_report --output OUT            # report only (after --run)
--cases accepts the vla_pipeline_qa syntax, e.g. controls:10 (controls of 10 sources spread in S/N) or
representative:3 on a full matched-v2 run on CHPC. The recipe is B:phase_amp:flagging_on (documented flags).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

RECIPE = "B:phase_amp:flagging_on"
SIDELOBE_T, NOISE_T, MINBEAMFRAC = 2.0, 5.0, 0.3
KIND = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise", 7: "all three"}


def run(output, source_run, cases, parallel, imsize=None, masking="auto"):
    from scripts import vla_pipeline_qa as qa
    qa.plan(output, source_run, "B", [RECIPE], cases, "none", None, None, native_diagnostics=True,
            selfcal_imsize=imsize, selfcal_masking=masking)
    qa.execute(output, parallel)


def log_numbers(notes):
    """Peak, rms and solint S/N from the pipeline decision lines (last occurrence = hif_selfcal's image)."""
    out = {}
    for text in notes:
        for key, pattern in (("peak_jy", r"Peak intensity of source: ([0-9.eE+-]+) mJy"),
                             ("rms_jy", r"rms: ([0-9.eE+-]+) mJy"), ("peak_snr", r"Peak SNR: ([0-9.eE+-]+)")):
            m = re.search(pattern, text)
            if m:                                  # last value wins: hif_selfcal's own image, not regcal imaging
                out[key] = float(m.group(1)) / (1000 if key != "peak_snr" else 1)
        m = re.match(r"^(inf_EB|inf|int|[0-9.]+s): ([0-9.]+)$", text)
        if m:
            out.setdefault("solint_snr", {})[m.group(1)] = float(m.group(2))
    return out


def automask_sidelobe(lines):
    """PSF sidelobe level as logged by CASA auto-multithresh, if present."""
    for line in lines:
        m = re.search(r"[Ss]idelobe\s*[Ll]evel[^0-9.]*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)   # CASA: "SidelobeLevel = 0.385"
        if m:
            return float(m.group(1))
    return None


def psf_sidelobe(psf):
    """Largest PSF value outside the main lobe (main lobe = connected psf > 0 region around the centre)."""
    from scipy import ndimage
    labels, _ = ndimage.label(psf > 0)
    c = np.unravel_index(np.argmax(psf), psf.shape)
    outside = labels != labels[c]
    return float(psf[outside].max()) if outside.any() else math.nan


def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = output.resolve()
    p = json.loads((output / "plan.json").read_text())
    source_run = Path(p["source_run"])
    sp = json.loads((source_run / "plan.json").read_text())
    snr0 = {x["id"]: x["nominal_source_snr"] for x in sp["prepared_sources"]}
    rows, figs = [], []
    for path in sorted((output / "results").glob("*.json")):
        r = json.loads(path.read_text())
        rec, ev = r["recipe_record"], r["evaluation"]
        committed = json.loads((source_run / "cases" / r["case"] / "case.json").read_text())
        c = committed["case"]
        kind = KIND[(c["A"] != 1) + 2 * (c["phi_deg"] != 0) + 4 * (c["a"] != 1)]
        from scripts.vla_pipeline_qa import pipeline_notes, safe
        log = output / "logs" / f"unit_{r['case']}_{safe(r['recipe'])}.log"
        nums = log_numbers(pipeline_notes(log) if log.exists() else rec.get("pipeline_notes", []))
        ours_sidelobe = committed["convergence"].get("raw", {}).get("maxpsfsidelobe")
        native = rec.get("native_crops")
        arrays, meta = {}, {}
        if native and Path(native).exists():
            with np.load(native) as z:
                meta = json.loads(str(z["meta"]))
                arrays = {k: z[k] for k in z.files if k != "meta"}
        sidelobe = psf_sidelobe(arrays["initial_psf_tt0"]) if "initial_psf_tt0" in arrays else math.nan
        logged = automask_sidelobe(rec.get("automask_log", {}).get("lines", []))
        peak, rms = nums.get("peak_jy"), nums.get("rms_jy")
        # Sidelobe used for the estimate: CASA's logged value, else the pipeline PSF crop, else our imaging's.
        used, used_from = next(((v, k) for v, k in ((logged, "CASA log"), (sidelobe, "pipeline PSF"),
                                                     (ours_sidelobe, "matched-v2 PSF"))
                                if v is not None and math.isfinite(v)), (None, None))
        level = None
        if peak and rms and used is not None:
            threshold = max(SIDELOBE_T * used * peak, NOISE_T * rms)
            level = threshold / peak
        mask_pixels = meta.get("initial.mask", {}).get("full_sum")
        row = dict(case=r["case"], source=c["source"], kind=kind, A=c["A"], phi=c["phi_deg"], a=c["a"],
                   snr0=snr0.get(c["source"]), disposition=rec.get("disposition"),
                   sc_success=next((t.get("sc_success") for t in rec.get("selfcal_result", {}).get("targets", [])), None),
                   peak=peak, rms=rms, peak_snr=nums.get("peak_snr"), sidelobe=sidelobe, sidelobe_ours=ours_sidelobe,
                   sidelobe_logged=logged, sidelobe_used_from=used_from,
                   threshold_level=level, region_beams=None if level is None or level >= 1 else math.log(1 / level),
                   mask_pixels=mask_pixels, solint_snr=nums.get("solint_snr"),
                   eC_before=ev.get("before", {}).get("eC"), eC_after=ev.get("after", {}).get("eC"),
                   new_flags=sum(s.get("added") or 0 for s in rec.get("flag_stages", [])) or rec.get("new_flag_fraction"),
                   seconds=rec.get("seconds"), mask_usage=rec.get("mask_usage"), reused=ev.get("evaluation_reused"),
                   native_error=None if arrays else (rec.get("native_crops_error") or
                                                     (json.dumps(rec["native_crops_errors"]) if rec.get("native_crops_errors")
                                                      else f"no crops saved; images found: {rec.get('native_images_found')}")))
        rows.append(row)
        if not arrays:
            continue
        keys = [k for k in ("dirty_image_tt0", "initial_image_tt0", "initial_residual_tt0", "initial_psf_tt0", "final_image_tt0")
                if k in arrays]
        fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4), squeeze=False, layout="constrained")
        for ax, k in zip(axes[0], keys):
            a = arrays[k]
            lim = np.nanmax(np.abs(a)) or 1
            ax.imshow(a, origin="lower", cmap="RdBu_r" if "psf" not in k else "viridis",
                      vmin=-lim if "psf" not in k else min(0, float(np.nanmin(a))), vmax=lim)
            mask = arrays.get("initial_mask" if not k.startswith("final") else "final_mask")
            if mask is not None and np.any(mask > 0) and "psf" not in k:
                ax.contour(mask > 0, levels=[.5], colors="lime", linewidths=1)
            ax.set_title(k.replace("_tt0", "").replace("_", " "), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        empty = "EMPTY mask" if not mask_pixels else f"mask {int(mask_pixels)} px"
        fig.suptitle(f"{c['source']}  {kind}  A={c['A']:.3g} φ={c['phi_deg']:.0f}° a={c['a']:.0f}  ·  "
                     f"{row['disposition']} · {empty}", fontsize=10)
        name = f"native_{r['case']}.png"
        fig.savefig(output / name, dpi=110); plt.close(fig)
        figs.append((row, name))

    fmt = lambda v, f="{:.3g}": "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f.format(v)
    applied = sum(r["disposition"] == "applied" for r in rows)
    html = ["<!doctype html><meta charset=utf-8><title>hif_selfcal mask check</title><style>body{font-family:sans-serif;"
            "max-width:1300px;margin:2em auto;padding:0 1em}table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:3px 6px;text-align:right;font-size:13px}th{background:#f3f3f3}td:nth-child(-n+3){text-align:left}"
            ".bad{color:#b2182b;font-weight:bold}img{max-width:100%}</style>",
            f"<h1>Why hif_selfcal skips cases</h1><p>{RECIPE} ({p.get('variant') or 'imsize=auto, masking=auto'}), "
            f"{len(rows)} cases, {applied} self-calibrated. "
            f"Mask level = max({SIDELOBE_T} × PSF sidelobe × peak, {NOISE_T} × rms) / peak (estimate). A point source "
            f"keeps a region of ln(1/level) beams, pruned below {MINBEAMFRAC} beam (level &gt; 0.74). "
            "Peak and rms are the pipeline's own values for its initial image. Images: central 256 px of the "
            "pipeline's own self-cal images, mask in green.</p>",
            "<table><tr><th>source</th><th>case</th><th>kind</th><th>SNR₀</th><th>result</th><th>peak (Jy)</th>"
            "<th>rms (Jy)</th><th>peak S/N</th><th>PSF sidelobe (CASA log / pipe PSF / ours)</th><th>mask level (sidelobe from)</th>"
            "<th>region (beams)</th><th>mask px</th><th>eC before → after</th><th>min</th></tr>"]
    for r in sorted(rows, key=lambda r: (r["source"], r["kind"])):
        bad = " class=bad" if r["disposition"] != "applied" else ""
        html.append(f"<tr><td>{r['source']}</td><td>{r['case'][:8]}</td><td>{r['kind']}</td><td>{fmt(r['snr0'], '{:.0f}')}</td>"
                    f"<td{bad}>{r['disposition']}</td><td>{fmt(r['peak'])}</td><td>{fmt(r['rms'])}</td>"
                    f"<td>{fmt(r['peak_snr'], '{:.0f}')}</td><td>{fmt(r['sidelobe_logged'], '{:.2f}')} / {fmt(r['sidelobe'], '{:.2f}')} / {fmt(r['sidelobe_ours'], '{:.2f}')}</td>"
                    f"<td>{fmt(r['threshold_level'], '{:.2f}')} ({r['sidelobe_used_from'] or '–'})</td><td>{fmt(r['region_beams'], '{:.2f}')}</td>"
                    f"<td>{fmt(r['mask_pixels'], '{:.0f}')}</td><td>{fmt(r['eC_before'], '{:.3f}')} → {fmt(r['eC_after'], '{:.3f}')}</td>"
                    f"<td>{fmt(None if r['seconds'] is None else r['seconds'] / 60, '{:.0f}')}</td></tr>")
    html.append("</table>")
    missing = [r for r in rows if r["native_error"]]
    if missing:
        html.append(f"<p class=bad>No pipeline images for {len(missing)} case(s): "
                    f"{missing[0]['native_error'][-600:]}</p>")
    for r, name in sorted(figs, key=lambda x: (x[0]["source"], x[0]["kind"])):
        html.append(f"<h3>{r['source']} · {r['kind']}</h3><img src='{name}'>")
    (output / "mask_report.html").write_text("\n".join(html))
    (output / "mask_rows.json").write_text(json.dumps(rows, indent=2, default=str))
    print("REPORT", output / "mask_report.html")
    for r in rows:
        print(r["source"], r["kind"], r["disposition"], "level", fmt(r["threshold_level"], "{:.2f}"),
              "mask", r["mask_pixels"], "sidelobe", fmt(r["sidelobe"], "{:.2f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run", action="store_true", help="Plan and execute B before reporting")
    ap.add_argument("--source-run", type=Path)
    ap.add_argument("--cases", default="representative")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--selfcal-imsize", type=int, help="Internal self-cal image size N (default: pipeline's own)")
    ap.add_argument("--selfcal-masking", choices=("auto", "center"), default="auto")
    a = ap.parse_args()
    if a.run:
        if not a.source_run:
            ap.error("--run needs --source-run")
        run(a.output.resolve(), a.source_run, a.cases, a.parallel, a.selfcal_imsize, a.selfcal_masking)
    report(a.output)


if __name__ == "__main__":
    main()
