"""Small local experiments for the Tanmoy questions (docs/tanmoy_local_experiments.md).

  ML:   python -m scripts.tanmoy_local_experiments plan --output OUT
  ML:   python -m scripts.tanmoy_local_experiments run --output OUT --experiment order --smoke
  ML:   python -m scripts.tanmoy_local_experiments run --output OUT --experiment {order,stopping,sampling} --resume
  ML:   python -m scripts.tanmoy_local_experiments audit-selfcal --output OUT
  ML:   python -m scripts.tanmoy_local_report --input OUT

Experiment 1b (order validation on new archived rescue cases; separate output, runs on CHPC):
  ML:   python -m scripts.tanmoy_local_experiments select-order --output runs/tanmoy-order-validation-20261123
  ML:   python -m scripts.tanmoy_local_experiments plan --study order-validation --output runs/tanmoy-order-validation-20261123
  ML:   python -m scripts.tanmoy_local_experiments run --output runs/tanmoy-order-validation-20261123 --experiment order [--smoke|--resume]

Experiment 3b (separate study and output; the 27-case plan is untouched):
  ML:   python -m scripts.tanmoy_local_experiments plan --study uvgrid --output runs/tanmoy-uvgrid-20261123
  ML:   python -m scripts.tanmoy_local_experiments run --output runs/tanmoy-uvgrid-20261123 --experiment uvgrid [--smoke|--resume]

`run` launches one serial CASA worker (bin/run casa ... casa-run) per call. Inputs (collect/, earlier runs and the
synced reports mirror) are only read; every simulated MS is a scratch copy, deleted after its case is extracted.
The module stays importable without CASA: casatools is imported only inside CASA-side functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "tanmoy-local-v1"
EXPERIMENTS = ("order", "stopping", "sampling", "uvgrid")
UVGRID_PROTOCOL = "tanmoy-uvgrid-v1"
ORDERVAL_PROTOCOL = "tanmoy-order-validation-v1"
# Experiment 1b candidate search (read-only matched records). Training records first; validation only as a
# labelled exploratory fallback; test is never read.
ORDERVAL_TRAIN_RUNS = ("runs/dataset-v2-matched-train", "runs/dataset-v2.1-matched-train-extension")
ORDERVAL_VAL_RUNS = ("runs/dataset-v2-matched-val",)
ORDERVAL_RESCUE = -0.05     # illustrative effect-size cutoff on d_amp, not a significance threshold
ORDERVAL_AMP_MAX = 8.5468   # current production amplification range
ORDERVAL_EXCLUDE = ("0449+113",)   # the original experiment-1 source
ORDERVAL_SOURCES = 2
UVGRID_SIZES = (256, 512, 1024)
F32_EPS = float(np.finfo(np.float32).eps)

# Historical inputs (read-only).
HIST_ORDER_RUN = ROOT / "runs/negative-damage-examples-v2-smoke-20261109"   # 0449+113 rescue sweep, Week 11 note
HIST_ORDER_CASES = dict(current="1331c0677101030493ac", noise="548d1617a01f707bb2ad")   # A=3.003, a=2113 (r=0.158)
HIST_2241_REFERENCE = ROOT / "runs/clean-recovery-original-flux-20261104/sources/2241+098/recovery"
SELFCAL_RUN = "vla-pipeline-qa-v2-val-256-center-on-20261112-0054"
SELFCAL_SOURCES = (ROOT / "runs" / SELFCAL_RUN,
                   Path("/Users/u1528314/Documents/chpc-radioastroml-sync/reports/runs") / SELFCAL_RUN)

ADAPTIVE = dict(niter=1_000_000, nsigma=3.0, dirty_peak_fraction=1e-7, threshold=None)   # = DefaultImagingConfig.clean
CEILING = 10_000          # diagnostic minor-iteration ceiling (stopping, sampling)
STOP_R = (0.02, 0.08, 0.16)
CUT_FRACTION = 0.5        # angular fraction removed: atan2(v,u) mod 180 deg in [0, 90) and its conjugate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- pure physics/bookkeeping (tested)

def baseline_gain(ant1, ant2, antenna, A, phi_deg=0.):
    """G = g_i conj(g_j) per row, g = A exp(i phi) on one antenna and 1 elsewhere."""
    g = A * np.exp(1j * math.radians(phi_deg))
    return np.where(ant1 == antenna, g, 1.) * np.conj(np.where(ant2 == antenna, g, 1.))


def construct(V, n, G, order):
    """Stored DATA (complex64) for one case. V is the stored noiseless DATA, n the scaled noise, G per row.

    noise:       Y = V + n
    current:     Y = G V + n        (G V stored first, as the dataset helper does, then n added)
    alternative: Y = G (V + n)      (V + n stored first, then the gain applied)
    """
    V = np.asarray(V, np.complex128)
    G = np.asarray(G, np.complex128)[None, None, :]
    f32 = lambda x: np.asarray(x, np.complex64).astype(np.complex128)
    if order == "noise":
        return np.asarray(V + n, np.complex64)
    if order == "current":
        return np.asarray(f32(G * V) + n, np.complex64)
    if order == "alternative":
        return np.asarray(G * f32(V + n), np.complex64)
    raise ValueError(f"Unknown order {order!r}")


def exact(V, n, G, order):
    """The same equations in complex128 without intermediate storage."""
    G = np.asarray(G, np.complex128)[None, None, :]
    return {"noise": V + n, "current": G * V + n, "alternative": G * (V + n)}[order]


def storage_tolerance(*arrays):
    """A few float32 ulps of the largest magnitude involved: storage roundoff, not a physics tolerance."""
    return 8 * F32_EPS * max([float(np.abs(a).max()) for a in arrays if np.size(a)] + [1e-30])


def angular_cut(u, v, fraction):
    """Conjugate-paired angular cut (scripts/uv_angular_coverage_experiment.py): remove atan2(v,u) mod pi in
    [0, pi*fraction)."""
    return np.mod(np.arctan2(v, u), math.pi) < math.pi * fraction


def scaled_grid(reference, imsize, factor):
    """Same angular field and centre: imsize pixels of reference cell / factor."""
    cell = [c / factor for c in reference["image_grid"]["cell_arcsec"]]
    out = json.loads(json.dumps(reference))
    out["image_grid"] = dict(imsize=[imsize, imsize], cell_arcsec=cell, field_of_view_arcsec=[imsize * c for c in cell])
    return out


def uv_grid(reference, imsize):
    """Image pixel size unchanged, imsize pixels: computational field imsize*c, Fourier cell 1/(imsize c)."""
    out = json.loads(json.dumps(reference))
    cell = list(reference["image_grid"]["cell_arcsec"])
    out["image_grid"] = dict(imsize=[imsize, imsize], cell_arcsec=cell, field_of_view_arcsec=[imsize * c for c in cell])
    return out


def fourier_cell(imsize, cell_arcsec):
    """Nominal uv-grid spacing (wavelengths) of an imsize-pixel FFT image with cell_arcsec pixels: 1/(N c)."""
    return 1 / (imsize * math.radians(cell_arcsec / 3600))


def central_crop(n, m):
    """Slice of length m whose index m//2 is native pixel n//2 (CASA's reference/phase-centre pixel)."""
    if m > n:
        raise ValueError(f"Crop {m} larger than image {n}")
    return slice(n // 2 - m // 2, n // 2 - m // 2 + m)


def check_cached(record, key):
    """Resume only a completed record with the identical cache key; anything else is a hard error."""
    if record is None:
        return False
    if record.get("key") != key:
        raise ValueError(f"Case {record.get('id')} exists with different settings (cache key); use a fresh output")
    return record.get("status") in ("complete", "reused")


def robust_stats(x):
    x = np.asarray(x, float)
    med = float(np.median(x))
    return dict(std=float(np.std(x)), scaled_mad=float(1.4826 * np.median(np.abs(x - med))),
                abs_peak=float(np.abs(x).max()), pixels=int(x.size))


def measure(model, residual, header, valid, *, flux, ra_rad, dec_rad, region, mask_major_arcsec, crop=None):
    """Recovery metrics on one native grid. model Jy/pixel; residual Jy/beam; header from the restored image.

    eC = ||C - T||2 / ||T||2 over the valid image, C = raw model convolved with the peak-one restoring Gaussian,
    T = true point source convolved identically (scripts/clean_recovery_noise_sweep.geometry/convolve_model).
    crop=m: convolve the full native model first, then evaluate everything on the central m x m pixels.
    """
    from scripts.clean_recovery_noise_sweep import convolve_model, geometry
    from scripts.imaging.metrics import beam_region_mask
    from scripts.imaging.models import BeamRegion

    model, residual, valid = np.asarray(model, float), np.asarray(residual, float), np.asarray(valid, bool)
    rho, T = geometry(header, model.shape, flux, ra_rad, dec_rad)
    C = convolve_model(model, header)
    native = dict(shape=list(model.shape), model_flux_ratio=float(model[valid].sum() / flux))
    if crop:
        s = central_crop(model.shape[0], crop), central_crop(model.shape[1], crop)
        model, residual, valid, rho, T, C = (x[s] for x in (model, residual, valid, rho, T, C))
    eC = float(np.sqrt(np.square((C - T)[valid]).sum() / np.square(T[valid]).sum()))
    aperture = valid & (rho <= 1)
    cell = abs(float(header["CDELT2"])) * 3600
    annulus = beam_region_mask(model.shape, (cell, cell), mask_major_arcsec, BeamRegion(*region)) & valid
    bmaj, bmin = header["BMAJ"] * 3600, header["BMIN"] * 3600
    return dict(
        eC=eC, empty_model=not bool(np.any(model[valid])), nonzero_model_pixels=int(np.count_nonzero(model)),
        flux_ratio_total=float(model[valid].sum() / flux), flux_ratio_aperture=float(model[aperture].sum() / flux),
        signed_flux_error_aperture=float((model[aperture].sum() - flux) / flux),
        outside_aperture_abs_flux_ratio=float(np.abs(model[valid & ~aperture]).sum() / flux),
        residual_offsource=robust_stats(residual[annulus]), residual_aperture=robust_stats(residual[aperture]),
        residual_full=robust_stats(residual[valid]),
        grid=dict(shape=list(model.shape), cell_arcsec=cell, field_of_view_arcsec=cell * model.shape[0],
                  pixels_per_beam_minor=bmin / cell, pixels_per_beam_area=math.pi * bmaj * bmin / (4 * math.log(2)) / cell ** 2,
                  aperture_pixels=int(aperture.sum()), offsource_pixels=int(annulus.sum()), valid_pixels=int(valid.sum()),
                  aperture="rho<=1 (restoring-beam FWHM ellipse)", native=native, crop=crop,
                  offsource=f"{region[0]:g}-{region[1]:g} grid-beam majors from image centre"),
        units=dict(model="Jy/pixel", residual="Jy/beam", flux_jy=flux))


def clean_diagnostics(summary, requested):
    raw = summary.get("raw") or {}
    niter = raw.get("niter", requested.get("niter"))
    it = summary.get("iterdone")
    return dict(
        iterdone=it, nmajordone=summary.get("nmajordone"), stopcode=summary.get("stopcode"),
        stop_description=summary.get("stop_description"), niter=niter,
        iteration_limit_reached=None if it is None or niter is None else bool(it >= niter),
        requested=requested, tclean_threshold_jy=raw.get("threshold"), tclean_nsigma=raw.get("nsigma"),
        final_minor_cycle_threshold_jy=summary.get("final_cycle_threshold_jy_per_beam"),
        final_peak_residual_jy=summary.get("final_peak_residual_jy_per_beam"),
        tclean_model_flux_jy=summary.get("final_model_flux_jy"),
        note="final_minor_cycle_threshold is the last minor-cycle (cycleThresh) value, not necessarily the global "
             "stopping threshold; CASA does not return its internal n-sigma threshold.")


# ---------------------------------------------------------------- plan

def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return dict(head=head, tracked_changes=dirty)
    except OSError:
        return None


def source_spec(sid, config, reference_dir, antenna=None):
    from scripts import create_dataset_v1 as g
    from scripts.create_dataset_v2 import original_flux_spec, thermal_source

    source = thermal_source(config, sid)
    flux = original_flux_spec(source, config)
    sigma = g._noise_request(source)[2]
    sky = flux["sky"][0]
    ra, dec = [float(x[:-3]) for x in sky["dir"].split()[1:]]
    ref = json.loads((reference_dir / "reference.json").read_text())
    if abs(ref["flux_jy"] - flux["flux_jy"]) > 1e-12 * flux["flux_jy"] or abs(ref["ra_rad"] - ra) > 1e-12:
        raise ValueError(f"Reference {reference_dir} does not match the source flux/position")
    return dict(
        id=sid, configuration=config, thermal_metadata=str(source.simulation_metadata),
        thermal_metadata_sha256=file_sha(source.simulation_metadata),
        original_ms=g._simulation_payload(source)["input_ms"], simulation_flags=config.get("simulation_flags"),
        flux_spec=flux, flux_jy=flux["flux_jy"], ra_rad=ra, dec_rad=dec, sigma0_jy=sigma,
        snr0=flux["nominal_source_snr"], antenna=antenna,
        region=[source.metric_min_radius_beams, source.metric_max_radius_beams],
        reference=dict(dir=str(reference_dir), beam=ref["beam"], grid=ref["grid"],
                       sha256={n: file_sha(reference_dir / n) for n in ("reference.json", "reference_grid.json", "reference_masks.npz")}))


def build_plan():
    from scripts.dataset_v2 import stream_seed

    hist = json.loads((HIST_ORDER_RUN / "plan.json").read_text())
    config = hist["configuration"]
    archived = {k: json.loads((HIST_ORDER_RUN / "cases" / c / "case.json").read_text()) for k, c in HIST_ORDER_CASES.items()}
    cur = archived["current"]["case"]
    if archived["noise"]["case"]["a"] != cur["a"] or cur["antenna"] != 3 or cur["phi_deg"] != 0:
        raise ValueError("Archived rescue cases do not match the documented example (antenna 3, zero phase)")
    s0449 = source_spec("0449+113", config, HIST_ORDER_RUN / "sources/0449+113/recovery", antenna=cur["antenna"])
    s2241 = source_spec("2241+098", dict(config, source_ids=["2241+098"]), HIST_2241_REFERENCE)
    seeds = [cur["noise_seed"]] + [stream_seed(config["seed"], "0449+113", f"tanmoy-order:{i}") for i in (1, 2)]
    A, a = cur["A"], cur["a"]
    cases = []

    def add(**c):
        c["key"] = digest(dict(protocol=PROTOCOL, case={k: v for k, v in c.items() if k != "derived_threshold"},
                               source=digest(dict(s0449 if c["source"] == "0449+113" else s2241))))
        cases.append(c)

    ref0449 = s0449["reference"]
    for cid, order, gain, a_case, seed in [("order-ctrl-V", "noise", 1., 0., None),
                                           ("order-ctrl-GV", "current", A, 0., None),
                                           ("order-ctrl-GV0", "alternative", A, 0., None)]:
        add(id=cid, experiment="order", source="0449+113", order=order, A=gain, phi_deg=0., antenna=s0449["antenna"],
            a=a_case, r=0., seed=seed, coverage="full", grid=ref0449["grid"], clean=ADAPTIVE)
    for i, seed in enumerate(seeds):
        tag = "archived" if i == 0 else f"new{i}"
        for order, gain in (("noise", 1.), ("current", A), ("alternative", A)):
            add(id=f"order-{order}-{tag}", experiment="order", source="0449+113", order=order, A=gain, phi_deg=0.,
                antenna=s0449["antenna"], a=a, r=a / s0449["snr0"], seed=seed, seed_label=tag, coverage="full",
                grid=ref0449["grid"], clean=ADAPTIVE)
    seed2241 = stream_seed(config["seed"], "2241+098", "noise:0")
    ref2241 = s2241["reference"]
    for coverage in ("full", "cut50"):
        for policy in ("adaptive", "fixed"):
            for r in STOP_R:
                clean = (dict(ADAPTIVE, niter=CEILING) if policy == "adaptive" else
                         dict(niter=CEILING, nsigma=0.0, dirty_peak_fraction=None, threshold=f"derived:{coverage}"))
                add(id=f"stop-{coverage}-{policy}-r{r:g}", experiment="stopping", source="2241+098", order="noise",
                    A=1., phi_deg=0., antenna=None, a=r * s2241["snr0"], r=r, seed=seed2241, coverage=coverage,
                    policy=policy, grid=ref2241["grid"], clean=clean)
    flux = s2241["flux_jy"]
    for imsize, factor in ((256, 1), (512, 2), (1024, 4)):
        add(id=f"sample-{imsize}", experiment="sampling", source="2241+098", order="noise", A=1., phi_deg=0.,
            antenna=None, a=0., r=0., seed=None, coverage="full", grid=scaled_grid(ref2241["grid"], imsize, factor),
            factor=factor, clean=dict(niter=CEILING, nsigma=0.0, dirty_peak_fraction=None, threshold=f"{1e-7 * flux:.12g}Jy"))
    if len(cases) > 27:
        raise ValueError("Planned case count exceeds the 27-image budget")
    return dict(
        schema=1, protocol=PROTOCOL, sources={"0449+113": s0449, "2241+098": s2241}, cases=cases,
        smoke=["order-ctrl-V", "order-current-archived", "order-alternative-archived"],
        archived=dict(run=str(HIST_ORDER_RUN), cases=HIST_ORDER_CASES, eC={k: v["eC"] for k, v in archived.items()},
                      v_sha256=cur.get("v_sha256") or archived["current"]["v_sha256"],
                      nominal_noise_sha256=archived["current"]["nominal_noise_sha256"],
                      noiseless_record=str(HIST_ORDER_RUN / "sources/0449+113/recovery/noiseless/record.json")),
        rules=dict(
            noise="n = a * (regular - V), regular = V + nominal thermal noise (seeded CASA simulator); r = a / SNR0",
            fixed_threshold="3 x residual scaled MAD in the off-source annulus of the same coverage's r=0.02 adaptive run",
            cut="flag atan2(v,u) mod 180 deg in [0, 90) and conjugate; noise generated before flagging; weights unchanged",
            weights="WEIGHT/SIGMA from the nominal sigma0 for every case (not rescaled for the alternative order)",
            grid="fixed reference grid/beam/mask per source; sampling: same angular field, cell c/factor"),
        software=dict(python=platform.python_version(), numpy=np.__version__, casa="recorded per case", git=git_state()),
        selfcal_run=SELFCAL_RUN)


def build_uvgrid_plan():
    """Experiment 3b: noiseless 2241+098, fixed image pixel c, N = 256/512/1024 (Fourier cell 1/(N c)), natural
    weighting, fixed central CLEAN mask and evaluation on the central reference-size crop."""
    from scripts.imaging.config import DefaultImagingConfig as D

    config = json.loads((HIST_ORDER_RUN / "plan.json").read_text())["configuration"]
    s2241 = source_spec("2241+098", dict(config, source_ids=["2241+098"]), HIST_2241_REFERENCE)
    ref = s2241["reference"]["grid"]
    crop = ref["image_grid"]["imsize"][0]
    c0 = ref["image_grid"]["cell_arcsec"][0]
    imaging = dict(weighting="natural", gridder=D.gridder, deconvolver=D.deconvolver, nterms=D.nterms, gain=D.gain,
                   mask_nbeams=D.mask_nbeams, uvtaper=None, usemask="user (central circle, reference beam)")
    clean = dict(niter=CEILING, nsigma=0.0, dirty_peak_fraction=None, threshold=f"{1e-7 * s2241['flux_jy']:.12g}Jy")
    cases = []
    for n in UVGRID_SIZES:
        c = dict(id=f"uvgrid-{n}", experiment="uvgrid", source="2241+098", order="noise", A=1., phi_deg=0., antenna=None,
                 a=0., r=0., seed=None, coverage="full", grid=uv_grid(ref, n), crop=crop, imaging=imaging, clean=clean,
                 fourier_cell_wavelengths=fourier_cell(n, c0))
        c["key"] = digest(dict(protocol=UVGRID_PROTOCOL, case=c, source=digest(s2241)))
        cases.append(c)
    return dict(
        schema=1, protocol=UVGRID_PROTOCOL, study="uvgrid", sources={"2241+098": s2241}, cases=cases,
        smoke=["uvgrid-256", "uvgrid-512"],
        rules=dict(
            grid="image cell c fixed (reference grid); N grows, so the computational field N c grows and the nominal "
                 "Fourier cell 1/(N c) shrinks; CASA's own padding (tclean default, not set here) is not exposed",
            weighting="natural for all three (Briggs density weighting would also change with the field)",
            mask="CASA central circle of 6 reference-beam majors at pixel N//2 (same angular support); no automask",
            metric=f"full native model convolved first, then C, T, residual cropped to the central {crop}x{crop} pixels",
            clean="nsigma 0, threshold 1e-7 x true flux, no dirty-peak fraction, 10,000 minor iterations"),
        software=dict(python=platform.python_version(), numpy=np.__version__, casa="recorded per case", git=git_state()),
        parent_plan="runs/tanmoy-local-20261123/plan.json (experiment 3; read only)")


# ---------------------------------------------------------------- experiment 1b: order validation

def rescue_candidates(run_dir, split_label):
    """Every amplitude+noise row (mask 5) of one matched run with its noise-only counterpart (mask 4, same recipe,
    slot and noise realization). d_amp = eC(amp+noise) - eC(noise only) from the two unrounded stored eC values."""
    from scripts.dataset_v2_matched_extract import FEATURES
    run_dir = ROOT / run_dir
    plan = json.loads((run_dir / "plan.json").read_text())
    rows = json.loads((run_dir / "rows.json").read_text())["rows"]
    cases = {c["id"]: c for sp in plan["source_plans"] for c in sp["cases"]}
    recipes = {r["id"]: r for sp in plan["source_plans"] for r in sp["recipes"]}
    prepared = {p["id"]: p for p in plan["prepared_sources"]}
    by_case = {r["case"]: r for r in rows}
    inside = FEATURES.index("model.flux_inside_beam_jy")
    out = []
    for r in rows:
        if r["mask"] != 5:
            continue
        noise = by_case.get(r["removal"][0])
        c, recipe, p = cases[r["case"]], recipes[r["recipe"]], prepared[r["source"]]
        flux = p["flux_spec"]["flux_jy"]
        ratio = lambda row: None if row is None or row["X"][inside] is None else row["X"][inside] / flux
        out.append(dict(
            run=str(run_dir.relative_to(ROOT)), split=split_label, source=r["source"], row=r["id"], case=r["case"],
            noise_case=r["removal"][0], recipe=r["recipe"], amp_branch=recipe["amp_branch"], antenna=c["antenna"],
            A=c["A"], phi_deg=c["phi_deg"], a=c["a"], r=c["a"] / p["nominal_source_snr"], noise_seed=c["noise_seed"],
            slot=c["slot"], eC_amp_noise=r["eC"], eC_noise=None if noise is None else noise["eC"],
            d_amp=None if noise is None else r["eC"] - noise["eC"], d_amp_label=r["d"][0],
            aperture_ratio_amp_noise=ratio(r), aperture_ratio_noise=ratio(noise), flux_jy=flux,
            parent_original_ms=p["parent_original_ms"], noise_mask=None if noise is None else noise["mask"]))
    return out


def classify_candidate(c, seen):
    """Exclusion reason, or '' when eligible (the -0.05 cutoff is illustrative, not a significance test)."""
    if c["noise_mask"] != 4 or c["d_amp"] is None:
        return "no matched noise-only counterpart"
    if c["phi_deg"] != 0:
        return "nonzero injected phase"
    if not c["A"] > 1:
        return "attenuation (A <= 1)"
    if c["source"] in ORDERVAL_EXCLUDE:
        return "original experiment-1 source"
    if c["A"] > ORDERVAL_AMP_MAX:
        return f"A above production range ({ORDERVAL_AMP_MAX})"
    if not c["d_amp"] < 0:
        return "no rescue (d_amp >= 0)"
    if c["d_amp"] > ORDERVAL_RESCUE:
        return f"weak rescue ({ORDERVAL_RESCUE} < d_amp < 0)"
    an, aa = c["aperture_ratio_noise"], c["aperture_ratio_amp_noise"]
    if an is None or aa is None:
        return "aperture flux unavailable"
    if not an < 1:
        return "noise-only source not under-recovered in the aperture"
    if not abs(aa - 1) < abs(an - 1):
        return "amplification does not bring aperture flux closer to truth"
    if not (ROOT / c["parent_original_ms"]).exists() and not Path(c["parent_original_ms"]).exists():
        return "original MS unavailable"
    key = (c["source"], c["antenna"], c["A"], c["a"], c["noise_seed"])
    if key in seen:
        return "duplicate of an earlier source/antenna/recipe/seed"
    seen.add(key)
    return ""


def select_order_cases(output):
    """Candidate audit and frozen selection for experiment 1b, before any imaging. Training records first; the
    validation run is read only if fewer than ORDERVAL_SOURCES new sources qualify (labelled exploratory)."""
    output = Path(output).resolve()
    if (output / "selection.json").exists():
        raise ValueError(f"Selection already frozen: {output / 'selection.json'}")
    candidates, seen = [], set()
    pools = [("train", ORDERVAL_TRAIN_RUNS), ("val (exploratory)", ORDERVAL_VAL_RUNS)]
    for label, runs in pools:
        for run in runs:
            if (ROOT / run / "rows.json").exists():
                for c in rescue_candidates(run, label):
                    c["exclusion"] = classify_candidate(c, seen)
                    candidates.append(c)
        if len({c["source"] for c in candidates if not c["exclusion"]}) >= ORDERVAL_SOURCES:
            break
    eligible = sorted((c for c in candidates if not c["exclusion"]), key=lambda c: (c["d_amp"], c["source"], c["case"]))
    chosen, used = [], set()
    for c in eligible:
        if c["source"] not in used:
            chosen.append(c); used.add(c["source"])
        if len(chosen) == ORDERVAL_SOURCES:
            break
    for c in candidates:
        c["selected"] = any(c["case"] == s["case"] and c["run"] == s["run"] for s in chosen)
        c["eligible_rank"] = next((i + 1 for i, e in enumerate(eligible) if e is c), None)
    candidates.sort(key=lambda c: (c["exclusion"] != "", c["d_amp"] if c["d_amp"] is not None else 9, c["source"], c["case"]))
    output.mkdir(parents=True, exist_ok=True)
    import csv
    with (output / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0]))
        w.writeheader(); w.writerows(candidates)
    reasons = {}
    for c in candidates:
        reasons[c["exclusion"] or "eligible"] = reasons.get(c["exclusion"] or "eligible", 0) + 1
    weaker = sorted((c for c in candidates if c["exclusion"].startswith("weak rescue")), key=lambda c: c["d_amp"])[:10]
    payload = dict(protocol=ORDERVAL_PROTOCOL, frozen=time.strftime("%FT%T"), rules=dict(
        rescue_cutoff=ORDERVAL_RESCUE, amp_max=ORDERVAL_AMP_MAX, excluded_sources=list(ORDERVAL_EXCLUDE),
        mixture="amplitude amplification + noise (mask 5), zero injected phase, single antenna",
        counterpart="noise-only row (mask 4) of the same recipe, slot and noise realization",
        aperture="model.flux_inside_beam_jy (raw model in the restoring-beam FWHM aperture) / true flux; noise-only "
                 "must be < 1 and the amplified case closer to 1",
        order="most negative d_amp, then source and case id; strongest eligible case on each of the first distinct sources",
        pools=[label for label, _ in pools]), counts=reasons, n_sources_eligible=len({c["source"] for c in eligible}),
        selected=chosen, weaker_candidates=weaker,
        limitation=None if len(chosen) >= ORDERVAL_SOURCES else
        f"only {len(chosen)} new source(s) qualify; no replacement or weakened cutoff applied automatically")
    write_json(output / "selection.json", payload)
    print("SELECT", reasons, "selected", [(c["source"], c["split"], round(c["d_amp"], 4), round(c["A"], 3)) for c in chosen])
    return payload


def build_order_validation_plan(output):
    """Experiment 1b: 11 images per frozen case (noiseless V, zero-noise G V, G(V+0) reused if bit-identical, and
    noise-only / current / alternative for the archived seed and two new deterministic seeds)."""
    from scripts.dataset_v2 import stream_seed
    sel = json.loads((Path(output).resolve() / "selection.json").read_text())
    if not sel["selected"]:
        raise ValueError("No selected cases; see selection.json")
    sources, cases, archived = {}, [], {}
    for s in sel["selected"]:
        run = ROOT / s["run"]
        config = json.loads((run / "plan.json").read_text())["configuration"]
        arch = {k: json.loads((run / "cases" / s[f] / "case.json").read_text()) for k, f in (("current", "case"), ("noise", "noise_case"))}
        cur = arch["current"]["case"]
        if (cur["A"], cur["a"], cur["antenna"], cur["phi_deg"], cur["noise_seed"]) != (s["A"], s["a"], s["antenna"], 0, s["noise_seed"]):
            raise ValueError(f"Archived case {s['case']} does not match the frozen selection")
        if arch["noise"]["case"]["a"] != cur["a"] or arch["noise"]["case"]["noise_seed"] != cur["noise_seed"]:
            raise ValueError(f"Archived counterpart {s['noise_case']} is not the matched noise-only case")
        sid = s["source"]
        ref = run / "sources" / sid / "recovery"
        src = sources[sid] = source_spec(sid, config, ref, antenna=cur["antenna"])
        masks = str(ref / "reference_masks.npz")
        archived[sid] = dict(run=s["run"], split=s["split"], cases=dict(current=s["case"], noise=s["noise_case"]),
                             eC={k: v["eC"] for k, v in arch.items()}, v_sha256=arch["current"]["v_sha256"],
                             nominal_noise_sha256=arch["current"]["nominal_noise_sha256"],
                             noise_seed=cur["noise_seed"], d_amp=s["d_amp"], row=s["row"])
        seeds = [cur["noise_seed"]] + [stream_seed(config["seed"], sid, f"tanmoy-order:{i}") for i in (1, 2)]

        def add(**c):
            c.update(experiment="order", source=sid, phi_deg=0., antenna=cur["antenna"], coverage="full",
                     grid=src["reference"]["grid"], clean=ADAPTIVE, matched_masks=masks)
            c["key"] = digest(dict(protocol=ORDERVAL_PROTOCOL, case=c, source=digest(src)))
            cases.append(c)

        for cid, order, gain in (("ctrl-V", "noise", 1.), ("ctrl-GV", "current", cur["A"]), ("ctrl-GV0", "alternative", cur["A"])):
            add(id=f"{sid}-{cid}", order=order, A=gain, a=0., r=0., seed=None,
                **(dict(reuse_from=f"{sid}-ctrl-GV") if cid == "ctrl-GV0" else {}))
        for i, seed in enumerate(seeds):
            tag = "archived" if i == 0 else f"new{i}"
            for order, gain in (("noise", 1.), ("current", cur["A"]), ("alternative", cur["A"])):
                add(id=f"{sid}-{order}-{tag}", order=order, A=gain, a=cur["a"], r=cur["a"] / src["snr0"], seed=seed,
                    seed_label=tag)
    first = sel["selected"][0]["source"]
    return dict(
        schema=1, protocol=ORDERVAL_PROTOCOL, study="order-validation", sources=sources, cases=cases, archived=archived,
        selection=dict(path="selection.json", sha256=file_sha(Path(output).resolve() / "selection.json"),
                       limitation=sel.get("limitation")),
        smoke=[f"{first}-ctrl-V", f"{first}-noise-archived", f"{first}-current-archived"],
        rules=dict(
            noise="n = a * (regular - V), regular = V + nominal thermal noise (seeded CASA simulator), exactly as the "
                  "dataset generator; r = a / SNR0",
            weights="WEIGHT/SIGMA from the nominal sigma0 for every case (not rescaled for the alternative order)",
            grid="each source's matched-run reference grid, restoring beam and CLEAN mask; adaptive CLEAN = DefaultImagingConfig",
            eC="runner eC over the valid image (as experiment 1) and eC_matched over the generator's fixed support Omega "
               "(reference_masks.npz), which must reproduce the archived eC",
            budget="11 images per case: G(V+0) reuses the G V image only if its stored DATA is bit-identical"),
        software=dict(python=platform.python_version(), numpy=np.__version__, casa="recorded per case", git=git_state()),
        parent="experiment 1 (0449+113) results live in runs/tanmoy-local-20261123 on the local machine; read-only")


def plan_stage(output, study="local"):
    output = output.resolve()
    builders = {"uvgrid": build_uvgrid_plan, "order-validation": lambda: build_order_validation_plan(output), "local": build_plan}
    payload = json.loads(json.dumps(builders[study](), default=str))
    path = output / "plan.json"
    if path.exists():
        old = json.loads(path.read_text())
        strip = lambda p: {k: v for k, v in p.items() if k != "software"}
        if strip(old) != strip(payload):
            raise ValueError("Frozen plan differs from the current inputs/settings; use a fresh output directory")
        print("PLAN unchanged", path)
        return old
    write_json(path, payload)
    counts = {e: sum(c["experiment"] == e for c in payload["cases"]) for e in EXPERIMENTS if any(c["experiment"] == e for c in payload["cases"])}
    print("PLAN", path, counts, "total", len(payload["cases"]))
    return payload


def load_plan(output):
    path = Path(output).resolve() / "plan.json"
    if not path.is_file():
        raise ValueError(f"Run `plan` first: {path} missing")
    return json.loads(path.read_text())


def case_record(output, cid):
    path = Path(output) / "cases" / cid / "record.json"
    return json.loads(path.read_text()) if path.is_file() else None


# ---------------------------------------------------------------- CASA side

def tree_bytes(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.exists() else 0


def column_hash(ms, minus=None):
    """SHA-256 of DATA (or DATA - minus.DATA) in complex128, row-chunk order (as create_dataset_v2_matched)."""
    from casatools import table
    a, b = table(), table()
    a.open(str(ms))
    if minus is not None:
        b.open(str(minus))
    h = hashlib.sha256()
    try:
        for start in range(0, a.nrows(), 4096):
            kw = dict(startrow=start, nrow=min(4096, a.nrows() - start))
            data = np.asarray(a.getcol("DATA", **kw), np.complex128)
            if minus is not None:
                data = data - np.asarray(b.getcol("DATA", **kw), np.complex128)
            h.update(np.ascontiguousarray(data).tobytes())
    finally:
        if minus is not None:
            b.close()
        a.close()
    return h.hexdigest()


def invariant_hashes(ms):
    from casatools import table
    tb = table()
    tb.open(str(ms))
    try:
        out = {}
        for key in ("WEIGHT", "SIGMA", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "FLAG", "FLAG_ROW", "UVW"):
            if key in tb.colnames():
                h = hashlib.sha256()
                for start in range(0, tb.nrows(), 4096):
                    h.update(np.ascontiguousarray(tb.getcol(key, startrow=start, nrow=min(4096, tb.nrows() - start))).tobytes())
                out[key] = h.hexdigest()
        return out
    finally:
        tb.close()


def write_case_data(v, regular, ms, c):
    """Write DATA by the case equation, verify the stored values and summarize the effective noise."""
    from casatools import table
    a, b, o = table(), table(), table()
    a.open(str(v)); o.open(str(ms), nomodify=False)
    if regular is not None:
        b.open(str(regular))
    worst, h_all, h_unaff, h_aff = 0., hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    acc = {k: [0., 0, 0., 0.] for k in ("affected", "unaffected")}   # sum|e|^2, count, sum|n|^2 base, sum|Y-GV-n|^2
    try:
        for start in range(0, a.nrows(), 4096):
            kw = dict(startrow=start, nrow=min(4096, a.nrows() - start))
            V = np.asarray(a.getcol("DATA", **kw), np.complex128)
            n = (c["a"] * (np.asarray(b.getcol("DATA", **kw), np.complex128) - V)) if regular is not None else np.zeros_like(V)
            ant1, ant2 = a.getcol("ANTENNA1", **kw), a.getcol("ANTENNA2", **kw)
            G = baseline_gain(ant1, ant2, c["antenna"], c["A"], c["phi_deg"]) if c["antenna"] is not None else np.ones(V.shape[-1])
            Y = construct(V, n, G, c["order"])
            o.putcol("DATA", Y, **kw)
            stored = np.asarray(o.getcol("DATA", **kw), np.complex128)
            want = exact(V, n, G, c["order"])
            tol = storage_tolerance(want, n, V)
            err = float(np.abs(stored - want).max())
            if err > tol:
                raise ValueError(f"Stored DATA deviates from the {c['order']} equation: {err} > {tol}")
            worst = max(worst, err / tol)
            affected = G != 1
            h_all.update(np.ascontiguousarray(stored).tobytes())
            h_unaff.update(np.ascontiguousarray(stored[..., ~affected]).tobytes())
            h_aff.update(np.ascontiguousarray(stored[..., affected]).tobytes())
            valid = ~a.getcol("FLAG", **kw) & ~a.getcol("FLAG_ROW", **kw)[None, None, :]
            e = stored - G[None, None, :] * V            # effective additive noise relative to the gained sky
            for name, rows in (("affected", affected), ("unaffected", ~affected)):
                m = valid & rows[None, None, :]
                acc[name][0] += float(np.square(np.abs(e[m])).sum())
                acc[name][1] += int(m.sum())
                acc[name][2] += float(np.square(np.abs(n[m])).sum())
    finally:
        o.close(); a.close()
        if regular is not None:
            b.close()
    rms = {k: dict(samples=x[1], effective_noise_rms_component_jy=math.sqrt(x[0] / x[1] / 2) if x[1] else None,
                   base_noise_rms_component_jy=math.sqrt(x[2] / x[1] / 2) if x[1] else None) for k, x in acc.items()}
    return dict(max_error_over_storage_tolerance=worst, data_sha256=h_all.hexdigest(),
                unaffected_sha256=h_unaff.hexdigest(), affected_sha256=h_aff.hexdigest(), noise=rms)


def order_check(v, regular, c):
    """Before imaging: both orders from identical V and n. Unaffected rows identical; affected differ by (G-1)n."""
    from casatools import table
    a, b = table(), table()
    a.open(str(v)); b.open(str(regular))
    worst, identical, rows_aff = 0., True, 0
    try:
        for start in range(0, a.nrows(), 4096):
            kw = dict(startrow=start, nrow=min(4096, a.nrows() - start))
            V = np.asarray(a.getcol("DATA", **kw), np.complex128)
            n = c["a"] * (np.asarray(b.getcol("DATA", **kw), np.complex128) - V)
            G = baseline_gain(a.getcol("ANTENNA1", **kw), a.getcol("ANTENNA2", **kw), c["antenna"], c["A"], c["phi_deg"])
            cur = construct(V, n, G, "current").astype(np.complex128)
            alt = construct(V, n, G, "alternative").astype(np.complex128)
            aff = G != 1
            rows_aff += int(aff.sum())
            identical &= bool(np.array_equal(cur[..., ~aff], alt[..., ~aff]))
            want = (G[None, None, :] - 1) * n
            tol = storage_tolerance(G[None, None, :] * V, G[None, None, :] * n)
            worst = max(worst, float(np.abs((alt - cur) - want)[..., aff].max(initial=0.)) / tol)
    finally:
        a.close(); b.close()
    if not identical or worst > 1:
        raise ValueError(f"Order check failed: unaffected identical={identical}, (G-1)n error/tol={worst}")
    return dict(unaffected_rows_identical=identical, affected_rows=rows_aff,
                max_difference_error_over_storage_tolerance=worst, a=c["a"], A=c["A"], seed=c["seed"])


def apply_cut(ms, fraction):
    from casatools import table
    tb = table()
    tb.open(str(ms), nomodify=False)
    try:
        uvw = tb.getcol("UVW")
        flag, row = tb.getcol("FLAG"), tb.getcol("FLAG_ROW")
        before = int((~flag & ~row[None, None, :]).sum())
        cut = angular_cut(uvw[0], uvw[1], fraction)
        flag |= cut[None, None, :]
        row |= cut
        tb.putcol("FLAG", flag); tb.putcol("FLAG_ROW", row)
        after = int((~flag & ~row[None, None, :]).sum())
    finally:
        tb.close()
    return dict(angular_fraction_removed=fraction, rows_cut=int(cut.sum()), rows_total=int(cut.size),
                usable_samples_before=before, usable_samples_after=after,
                usable_fraction_removed=1 - after / before)


def image_case(ms, directory, grid, beam, clean, imaging=None):
    """One dirty + CLEAN call with explicit per-case overrides of DefaultImagingConfig (no default changes).
    imaging: optional {"weighting": ...} override; the other recorded fields must equal the defaults."""
    from dataclasses import replace
    from casatools import image as casa_image
    from scripts import create_dataset_v1 as g
    from scripts.imaging import Beam, BeamRegion, DefaultImagingConfig, image_ms
    from scripts.imaging.config import CleanIterationsConfig, read_beam
    from scripts.imaging.fits import export_casa_fits
    from scripts.imaging.models import ImageGrid, ResolvedGrid
    from scripts.imaging.qa import json_safe
    from scripts.uv_angular_coverage_experiment import FixedGrid

    ig = grid["image_grid"]
    resolved_grid = ResolvedGrid(Beam(**grid["beam"]), ImageGrid(tuple(ig["imsize"]), tuple(ig["cell_arcsec"]),
                                                                  tuple(ig["field_of_view_arcsec"])), Path(directory) / "firstpass")
    controls = CleanIterationsConfig(niter=clean["niter"], threshold=clean["threshold"],
                                     dirty_peak_fraction=clean["dirty_peak_fraction"], nsigma=clean["nsigma"])
    config = replace(DefaultImagingConfig, grid=FixedGrid(resolved_grid), restoring_beam=Beam(**beam), clean=controls)
    if imaging:
        for k in ("gridder", "deconvolver", "nterms", "gain", "mask_nbeams"):
            if imaging[k] != getattr(config, k):
                raise ValueError(f"Planned {k}={imaging[k]} differs from the imaging configuration {getattr(config, k)}")
        config = replace(config, weighting=imaging["weighting"])
    imaging = image_ms(ms, config, directory, imsize=tuple(ig["imsize"]), metric_region=BeamRegion(),
                       keep_intermediate_products=True, fits_invalid_policy="fill", fits_fill_value=0.,
                       pblimit=g.DATASET_PBLIMIT)
    final_beam = read_beam(imaging.clean_image)
    model_fits = export_casa_fits(imaging.model_image, Path(directory) / "model.fits.gz", fallback_beam=Beam(**beam),
                                  fallback_unit="Jy/pixel", invalid_policy="fill")

    def chunk(path, mask=False):
        ia = casa_image(); ia.open(str(path))
        try:
            return np.squeeze(np.asarray(ia.getchunk(getmask=mask))).T
        finally:
            ia.close()

    valid = np.logical_and.reduce([chunk(p, True).astype(bool) for p in (imaging.dirty_image, imaging.clean_image, imaging.residual_image)])
    search = chunk(imaging.mask_image) != 0
    np.savez_compressed(Path(directory) / "support.npz", valid=valid, clean_search=search)
    products = dict(dirty=imaging.dirty_fits, clean=imaging.clean_fits, residual=imaging.residual_fits,
                    psf=imaging.psf_fits, model=model_fits, support=Path(directory) / "support.npz")
    for p in Path(directory).iterdir():   # keep FITS, support and QA only
        if p.name.endswith(".fits.gz") or p.name in ("support.npz", "qa.json", "qa.txt"):
            continue
        remove_scratch(p)
    return dict(products=products, summary=json_safe(imaging.qa.tclean_summary),
                effective=json_safe(imaging.resolved_config.effective_imaging_parameters),
                warnings=list(imaging.qa.warnings), final_beam=json_safe(final_beam),
                mask_pixels=int(search.sum()))


def remove_scratch(p):
    """Delete a CASA by-product. On NFS a file CASA still holds open is silly-renamed to .nfs*; unlinking it raises
    EBUSY. Such placeholders are left for NFS to remove once the handle closes."""
    import errno
    if p.name.startswith(".nfs"):
        return
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except OSError as e:
        if e.errno != errno.EBUSY:
            raise
        if p.is_dir() and any(not x.name.startswith(".nfs") for x in p.rglob("*") if x.is_file()):
            raise


def load_planes(products):
    from astropy.io import fits
    out = {}
    for key in ("dirty", "clean", "residual", "model"):
        with fits.open(products[key], memmap=False) as h:
            out[key] = np.asarray(np.squeeze(h[0].data), np.float64)
            if key == "clean":
                out["header"] = h[0].header.copy()
    with np.load(products["support"]) as s:
        out["valid"] = np.asarray(s["valid"], bool)
    return out


def extract(output, products, src, case):
    planes = load_planes({k: Path(output) / v for k, v in products.items()})
    h = planes["header"]
    cell = abs(float(h["CDELT2"])) * 3600
    want = case["grid"]["image_grid"]
    if list(planes["model"].shape) != list(want["imsize"])[::-1] or abs(cell - want["cell_arcsec"][1]) > 1e-6 * cell:
        raise ValueError(f"Grid was not honoured: {planes['model'].shape} cell {cell} vs {want}")
    out = measure(planes["model"], planes["residual"], h, planes["valid"], flux=src["flux_jy"], ra_rad=src["ra_rad"],
                  dec_rad=src["dec_rad"], region=src["region"], mask_major_arcsec=case["grid"]["beam"]["major_arcsec"],
                  crop=case.get("crop"))
    if case.get("matched_masks"):   # experiment 1b: the dataset generator's eC on its fixed support (Omega)
        out.update(matched_eC(planes["model"], h, case["matched_masks"]))
    return out


def matched_eC(model, header, masks_path):
    """eC exactly as scripts/dataset_v2_matched_extract.extract: sqrt(sum((C - truth)^2) / sum(truth^2)) over omega,
    with C = raw model convolved with the header's restoring beam and truth/omega from reference_masks.npz."""
    from scripts.clean_recovery_noise_sweep import convolve_model
    with np.load(masks_path, allow_pickle=False) as m:
        truth, omega = np.asarray(m["truth"], np.float64), np.asarray(m["omega"], bool)
    if truth.shape != model.shape or not omega.any():
        raise ValueError(f"Matched masks {masks_path} do not fit the image {model.shape}")
    C = convolve_model(np.asarray(model, np.float64), header)
    return dict(eC_matched=float(np.sqrt(np.square((C - truth)[omega]).sum() / np.square(truth[omega]).sum())),
                matched_support_pixels=int(omega.sum()))


def derived_threshold(output, coverage):
    rec = case_record(output, f"stop-{coverage}-adaptive-r{STOP_R[0]:g}")
    if rec is None or rec.get("status") != "complete":
        raise ValueError(f"Fixed threshold for {coverage} needs the completed r={STOP_R[0]} adaptive case")
    mad = rec["metrics"]["residual_offsource"]["scaled_mad"]
    if not (math.isfinite(mad) and mad > 0):
        raise ValueError(f"Invalid off-source scaled MAD for {coverage}: {mad}; not inventing a threshold")
    return dict(value_jy=3 * mad, rule="3 x off-source residual scaled MAD", from_case=rec["id"], scaled_mad_jy=mad)


def casa_run(output, experiment, smoke):
    """CASA worker: prepare the source once, then build, image and extract each pending case serially."""
    import casatools
    from scripts import create_dataset_v1 as g
    from scripts.create_dataset_v2 import thermal_source
    from scripts.injection_bounds_pilot import prepare_v
    from scripts.simulation import add_thermal_noise_inplace, initialize_weights

    output = Path(output).resolve()
    plan = load_plan(output)
    selected = [c for c in plan["cases"] if c["experiment"] == experiment and (not smoke or c["id"] in plan["smoke"])]
    pending = [c for c in selected if not check_cached(case_record(output, c["id"]), c["key"])]
    for c in selected:                     # reused cases: retained products must still be the recorded ones
        rec = case_record(output, c["id"])
        if c not in pending and rec.get("status") == "complete":
            bad = [k for k, h in (rec.get("product_sha256") or {}).items() if file_sha(output / rec["products"][k]) != h]
            if bad:
                raise ValueError(f"{c['id']}: retained products changed since imaging: {bad}")
    print(f"CASA {experiment}: {len(selected)} cases, {len(pending)} pending", flush=True)
    if not pending:
        return
    started = time.time()
    scratch = output / "scratch" / experiment
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    run_log = dict(experiment=experiment, smoke=smoke, started=time.strftime("%FT%T"), casa=casatools.version_string(),
                   host=os.uname().nodename, cases=[])
    cwd = Path.cwd()
    os.chdir(scratch)   # CASA temporary lattices stay in scratch
    peak = 0
    try:
        preps = {}
        for sid in dict.fromkeys(c["source"] for c in pending):   # one source preparation at a time
            src = plan["sources"][sid]
            src_pending = [c for c in pending if c["source"] == sid]
            for old in scratch.iterdir():
                shutil.rmtree(old) if old.is_dir() else old.unlink()
            config = src["configuration"]
            source = thermal_source(config, src["id"])
            if file_sha(source.simulation_metadata) != src["thermal_metadata_sha256"]:
                raise ValueError("Parent thermal simulation metadata changed since the plan")
            noise_model, noise_params, sigma = g._noise_request(source)
            if sigma != src["sigma0_jy"]:
                raise ValueError("Nominal noise changed since the plan")
            t0 = time.time()
            v = prepare_v(source, scratch, src["flux_spec"]["sky"], v2=True, simulation_flags=src["simulation_flags"])
            initialize_weights(v, sigma)
            prep = dict(v_sha256=column_hash(v), v_invariants=invariant_hashes(v), seconds=None, regular={})
            prep["seconds"] = round(time.time() - t0, 1)
            regulars = {}

            def regular_for(seed):
                if seed is None:
                    return None
                if seed not in regulars:
                    for old in list(regulars):           # keep only one noise realization at a time
                        shutil.rmtree(regulars.pop(old))
                    t1 = time.time()
                    path = scratch / f"regular_{seed}.ms"
                    shutil.copytree(v, path)
                    add_thermal_noise_inplace(path, noise_model=noise_model, noise_parameters=noise_params, seed=seed)
                    initialize_weights(path, sigma)
                    prep["regular"][str(seed)] = dict(noise_sha256=column_hash(path, minus=v), seconds=round(time.time() - t1, 1))
                    regulars[seed] = path
                return regulars[seed]

            checks = {}
            for c in sorted(src_pending, key=lambda c: (str(c["seed"]), c["id"])):
                cell = output / "cases" / c["id"]
                t1 = time.time()
                record = dict(id=c["id"], key=c["key"], case=c, experiment=experiment, casa=casatools.version_string(),
                              started=time.strftime("%FT%T"))
                try:
                    clean = dict(c["clean"])
                    if str(clean.get("threshold") or "").startswith("derived:"):
                        record["derived_threshold"] = derived_threshold(output, clean["threshold"].split(":", 1)[1])
                        clean["threshold"] = f"{record['derived_threshold']['value_jy']:.12g}Jy"
                    record["clean_resolved"] = clean
                    regular = regular_for(c["seed"])
                    if experiment == "order" and regular is not None and c["seed"] not in checks:
                        gained = next(x for x in plan["cases"] if x["experiment"] == "order" and x["seed"] == c["seed"]
                                      and x["order"] == "alternative" and x["source"] == c["source"])
                        checks[c["seed"]] = order_check(v, regular, gained)
                        write_json(output / "checks" / f"order_{c['seed']}.json", checks[c["seed"]])
                    record["order_check"] = checks.get(c["seed"])
                    ms = scratch / "case.ms"
                    if ms.exists():
                        shutil.rmtree(ms)
                    shutil.copytree(v, ms)
                    record["data"] = write_case_data(v, regular, ms, c)
                    initialize_weights(ms, sigma)
                    inv = record["invariants"] = invariant_hashes(ms)
                    if inv != prep["v_invariants"]:
                        raise ValueError("Weights/flags/UVW differ from V before the coverage cut")
                    if c["coverage"] == "cut50":
                        record["cut"] = apply_cut(ms, CUT_FRACTION)
                    peak = max(peak, tree_bytes(scratch))
                    reuse = c.get("reuse_from") or ("order-ctrl-GV" if c["id"] == "order-ctrl-GV0" else None)
                    if reuse:
                        other = case_record(output, reuse)
                        if other and other.get("status") == "complete" and other["data"]["data_sha256"] == record["data"]["data_sha256"]:
                            record.update(status="reused", reused_from=reuse, products=other["products"],
                                          metrics=other["metrics"], clean=other["clean"], seconds=round(time.time() - t1, 1),
                                          note="G(V+0) DATA bit-identical to G V; image reused")
                            shutil.rmtree(ms)
                            write_json(cell / "record.json", record)
                            run_log["cases"].append(dict(id=c["id"], status="reused", seconds=record["seconds"]))
                            print("REUSED", c["id"], flush=True)
                            continue
                    directory = cell / f"img_{c['key'][:12]}"   # fresh CASA prefix for every distinct setting
                    if directory.exists():
                        directory.rename(cell / f"incomplete_{directory.name}_{time.time_ns()}")
                    img = image_case(ms, directory, c["grid"], src["reference"]["beam"], clean, c.get("imaging"))
                    peak = max(peak, tree_bytes(scratch))
                    products = {k: str(Path(p).relative_to(output)) for k, p in img["products"].items()}
                    record.update(products=products, imaging=dict(effective=img["effective"], warnings=img["warnings"],
                                                                  final_beam=img["final_beam"], mask_pixels=img["mask_pixels"]),
                                  clean=clean_diagnostics(img["summary"], clean),
                                  metrics=extract(output, products, src, c),
                                  product_sha256={k: file_sha(output / p) for k, p in products.items()})
                    record.update(status="complete", seconds=round(time.time() - t1, 1))
                    write_json(cell / "record.json", record)
                    (cell / "failure.json").unlink(missing_ok=True)
                    shutil.rmtree(ms)
                    run_log["cases"].append(dict(id=c["id"], status="complete", seconds=record["seconds"]))
                    print(f"COMPLETE {c['id']} eC={record['metrics']['eC']:.4f} iters={record['clean']['iterdone']} "
                          f"{record['seconds']:.0f}s", flush=True)
                except Exception:
                    write_json(cell / "failure.json", dict(record, status="failed", traceback=traceback.format_exc()[-12000:]))
                    run_log["cases"].append(dict(id=c["id"], status="failed", seconds=round(time.time() - t1, 1)))
                    print(traceback.format_exc(), flush=True)
            preps[sid] = prep
        run_log.update(source_preparation=prep if len(preps) == 1 else None, source_preparations=preps)
    finally:
        os.chdir(cwd)
        run_log.update(seconds=round(time.time() - started, 1), peak_scratch_bytes=int(peak), finished=time.strftime("%FT%T"))
        write_json(output / "runs" / f"{experiment}{'_smoke' if smoke else ''}_{time.strftime('%Y%m%dT%H%M%S')}.json", run_log)
        if scratch.exists():
            shutil.rmtree(scratch)
    if any(x["status"] == "failed" for x in run_log["cases"]):
        raise SystemExit(f"{experiment}: failed cases; see {output / 'cases'}/*/failure.json")


# ---------------------------------------------------------------- ML side

def run_stage(output, experiment, smoke, resume):
    output = output.resolve()
    plan = load_plan(output)
    done = [c["id"] for c in plan["cases"] if c["experiment"] == experiment and case_record(output, c["id"])]
    if done and not resume:
        raise ValueError(f"{len(done)} {experiment} cases already exist; pass --resume to continue matching cases")
    log = output / "logs" / f"{experiment}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / "bin/run"), "casa", "scripts/tanmoy_local_experiments.py", "casa-run",
               "--output", str(output), "--experiment", experiment] + (["--smoke"] if smoke else [])
    t0 = time.time()
    with log.open("a") as handle:
        handle.write(f"\n=== {time.strftime('%FT%T')} {' '.join(command)}\n"); handle.flush()
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    print(f"CASA {experiment} exit {code} after {time.time() - t0:.0f}s; log {log}", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.tanmoy_local_report", "--input", str(output)], cwd=ROOT, check=True)
    if code:
        raise SystemExit(code)


# ---------------------------------------------------------------- experiment 4: read-only self-calibration audit

CATEGORIES = ("low_initial_snr_block", "rejected_correction", "retained_improved", "retained_similar_or_worse",
              "mask_or_model_failure", "other")


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def selfcal_case(d):
    rr = d.get("recipe_record") or {}
    v5 = d.get("v5") or {}
    targets = _get(rr, "selfcal_result", "targets") or []
    lib = (targets[0].get("sc_lib") if targets else None) or {}
    field = lib.get("0") or {}
    vis = next((k for k in field if isinstance(field.get(k), dict) and k.endswith(".ms")), None)
    per_vis = field.get(vis) or {}
    solints = lib.get("solints") or (targets[0].get("sc_solints") if targets else None) or []
    attempts = [dict(solint=s, passed=per_vis[s].get("Pass"), fail_reason=per_vis[s].get("Fail_Reason"),
                     snr_pre=per_vis[s].get("SNR_pre"), snr_post=per_vis[s].get("SNR_post"),
                     rms_pre=per_vis[s].get("RMS_pre"), rms_post=per_vis[s].get("RMS_post"))
                for s in solints if isinstance(per_vis.get(s), dict)]
    notes = rr.get("pipeline_notes") or []
    solint_snr = {}
    if "Estimated SNR per solint:" in notes:     # pipeline log lines "inf_EB: 3.53" after this header
        for line in notes[notes.index("Estimated SNR per solint:") + 1:]:
            name, sep, value = line.partition(": ")
            try:
                if not sep or name.strip() not in solints:
                    break
                solint_snr[name.strip()] = float(value)
            except ValueError:
                break
    flag_after = next((st.get("after") for st in rr.get("flag_stages") or [] if st.get("after")), None)
    blocked, app, recovery = v5.get("blocked_reason"), v5.get("application"), v5.get("recovery")
    text = " ".join(str(x) for x in [rr.get("reason"), *(v5.get("stop_reasons") or [])]).lower()
    if blocked == "low_initial_snr":
        cat = "low_initial_snr_block"
    elif blocked == "solution_lowers_snr":
        cat = "rejected_correction"
    elif app == "applied" and recovery in ("improved", "solved"):
        cat = "retained_improved"
    elif app == "applied" and recovery in ("similar", "degraded"):
        cat = "retained_similar_or_worse"
    elif "mask" in text or "model" in text:
        cat = "mask_or_model_failure"
    else:
        cat = "other"
    return dict(
        case=d.get("case"), label=d.get("label"), category=cat, application=app, disposition=rr.get("disposition"),
        blocked_reason=blocked, stop_reasons=v5.get("stop_reasons"), stop_reason_lib=lib.get("Stop_Reason"),
        sc_success=lib.get("SC_success"), final_solint=lib.get("final_solint"), solints_configured=solints,
        attempted=attempts, retained_interval=lib.get("final_solint") if app == "applied" else None,
        estimated_snr_per_solint=solint_snr or None,
        estimated_snr=dict(per_EB=field.get("per_EB_SNR") or lib.get("per_EB_SNR"),
                           per_scan=field.get("per_scan_SNR") or lib.get("per_scan_SNR"),
                           image_orig=lib.get("SNR_orig"), image_final=lib.get("SNR_final")),
        eC_before=v5.get("eC_before"), eC_after=v5.get("eC_after"), eC_reference=v5.get("reference"),
        recovery=recovery, group=v5.get("group"), subgroup=v5.get("subgroup"), D=v5.get("D"),
        image_snr=dict(before=v5.get("snr_before"), after=v5.get("snr_after")),
        image_rms=dict(before=v5.get("rms_before"), after=v5.get("rms_after")),
        flag_fraction_after=(flag_after["flagged"] / flag_after["total"]) if flag_after else None,
        reason=rr.get("reason"), description=d.get("description"), pipeline=rr.get("pipeline"),
        selfcal_arguments=rr.get("selfcal_arguments"), image=None)


def audit_selfcal(output, source=None):
    output = Path(output).resolve()
    src = Path(source) if source else next((p for p in SELFCAL_SOURCES if (p / "descriptions").is_dir()), None)
    if src is None:
        write_json(output / "selfcal_audit.json", dict(status="missing_input", searched=[str(p) for p in SELFCAL_SOURCES]))
        print("SELFCAL input missing")
        return
    files = sorted((src / "descriptions").glob("*.json"))
    cases = []
    for f in files:
        try:
            cases.append(selfcal_case(json.loads(f.read_text())))
        except (ValueError, KeyError, TypeError) as exc:
            cases.append(dict(case=f.stem, category="unreadable", error=str(exc)))
    cases.sort(key=lambda c: str(c["case"]))
    images = src / "images"
    for c in cases:
        found = sorted(images.glob(f"*{c['case']}*")) if images.is_dir() else []
        c["image"] = [str(p) for p in found]
    selected = {}
    for cat in CATEGORIES:
        first = next((c for c in cases if c["category"] == cat), None)
        if first:
            selected[cat] = first["case"]
    counts = {}
    for key in ("category", "application", "disposition", "blocked_reason", "recovery", "final_solint", "stop_reason_lib"):
        counts[key] = {}
        for c in cases:
            k = str(c.get(key))
            counts[key][k] = counts[key].get(k, 0) + 1
    missing = ["configured minsnr_to_proceed / gaincal minsnr (recipe arguments record only the non-default ones)",
               "gain-solution flagged fractions per interval (gain tables were removed with the CHPC work dirs)",
               "native weblog / casalogs (synced mirror holds HTML, PNG and description JSON only)"]
    first = next((c for c in cases if c.get("pipeline")), {})
    write_json(output / "selfcal_audit.json", dict(
        status="ok", source=str(src), run=SELFCAL_RUN, files=len(files), pipeline=first.get("pipeline"),
        recorded_arguments=first.get("selfcal_arguments"), counts=counts, selected=selected,
        missing_native_fields=missing, cases=cases))
    print("SELFCAL", len(cases), "cases; selected", selected)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "casa-run", "audit-selfcal", "select-order"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name in ("run", "casa-run"):
            p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
            p.add_argument("--smoke", action="store_true")
        if name == "run":
            p.add_argument("--resume", action="store_true")
        if name == "plan":
            p.add_argument("--study", choices=("local", "uvgrid", "order-validation"), default="local")
        if name == "audit-selfcal":
            p.add_argument("--source", type=Path)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan_stage(a.output, a.study)
    elif a.cmd == "run":
        run_stage(a.output, a.experiment, a.smoke, a.resume)
    elif a.cmd == "casa-run":
        casa_run(a.output, a.experiment, a.smoke)
    elif a.cmd == "select-order":
        select_order_cases(a.output)
    else:
        audit_selfcal(a.output, a.source)


if __name__ == "__main__":
    main()
