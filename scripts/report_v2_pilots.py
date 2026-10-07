#!/usr/bin/env python3
"""Fresh training-only section-1 pilots and explanatory HTML reports.

CASA: casa --nogui --nologger -c scripts/report_v2_pilots.py --generate-only
Report: ml/.venv/bin/python scripts/report_v2_pilots.py --report-only
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/radioastro-v2-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from astropy.wcs import WCS
from scipy import sparse

from scripts.preprocessing.fits import load_fits_plane
from scripts.preprocessing.partitions import TRAIN_IDS, partition_for_sample
from scripts.report_fourier_toy import OUTPUT, figure_uri, phase_centered_fft, write_html

DATASET = ROOT / "collect/experiments/dataset_v1_20260921_all_psf"
SOURCES = ("0005+383", "0012-399", "0201-115")
SEEDS = (202610051, 202610052, 202610053)
VARIANTS = ("no_gain", "amplitude", "phase", "mixture", "double_noise")
LABELS = dict(no_gain="No gain error", amplitude="Amplitude only", phase="Phase only",
              mixture="Amplitude + phase", double_noise="Twice the thermal σ",
              single_baseline="Single-baseline amplitude error")
CHUNK = 512
DISPLAY_PER_BASELINE = 96
C = 299792458.0


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_provenance(value, base):
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    if not path.exists() and "/radioastro-ml/" in str(path):
        path = ROOT / str(path).split("/radioastro-ml/", 1)[1]
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def source_info(sid):
    if sid not in TRAIN_IDS:
        raise ValueError(f"Exploration is training-only: {sid}")
    index = json.loads((DATASET / "dataset.json").read_text())
    indexed = f"samples/{sid}_not_corrupted/sample.json"
    if indexed not in index["samples"]:
        raise ValueError(f"No indexed baseline for {sid}")
    # Only open the chosen training manifest; no held-out sample products.
    sample_path = DATASET / indexed
    sample = json.loads(sample_path.read_text())
    report = json.loads((DATASET / "report.json").read_text())
    thermal_root = resolve_provenance(report["source_run"], DATASET)
    thermal = json.loads((thermal_root / "report.json").read_text())
    entry = next(e for e in thermal["samples"] if e["id"] == sid)
    meta_path = resolve_provenance(entry["simulation_metadata"], thermal_root)
    meta = json.loads(meta_path.read_text())
    residual = resolve_provenance(sample["products"]["residual"], sample_path.parent)
    plane = load_fits_plane(residual)
    qa = json.loads(resolve_provenance(sample["metadata"]["imaging_qa"], sample_path.parent).read_text())
    cfg = qa["effective_imaging_parameters"]
    expected = dict(stokes="I", specmode="mfs", deconvolver="mtmfs", nterms=1,
                    weighting="briggs", robust=0.5, imsize=[256, 256])
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unexpected pilot imaging {sid}: {key}={cfg.get(key)}")
    for key in ("spw", "field", "scan", "antenna", "uvrange"):
        if cfg.get(key) not in (None, ""):
            raise ValueError(f"This pilot requires all-row imaging selection: {key}")
    row = next(e for e in report["sources"] if e["id"] == sid)
    return dict(id=sid, partition=partition_for_sample(sid), manifest=str(sample_path),
        parent_ms=str(resolve_provenance(meta["input_ms"], meta_path.parent)),
        simulation_metadata=str(meta_path), components=meta["components"],
        sigma0=float(meta["noise"]["simplenoise_jy"]),
        s0=float(row["predicted_image_rms_jy_per_beam"]),
        source_snr=float(entry["source_snr"]), header=dict(plane.celestial_header),
        imaging_reference=cfg, shape=list(plane.shape))


def bilinear_cells(u, v, header, shape):
    """Deposit in the interior FFT domain, then fold each corner to one pair member.

    The representative is fy>0 or (fy==0 and fx>0). DC and Nyquist boundaries
    are discarded. Out-of-domain samples are excluded, never wrapped.
    Returns cell IDs, deposited weights, and inside-domain sample mask.
    """
    ny, nx = shape
    j = np.deg2rad(WCS(header, fix=False).celestial.pixel_scale_matrix)
    f = j.T @ np.array([u, v])
    x, y = f[0] * nx, f[1] * ny
    inside = (x >= -nx/2 + 1) & (x <= nx/2 - 1) & (y >= -ny/2 + 1) & (y <= ny/2 - 1)
    # Clip only invalid samples to make integer conversion safe; their weights remain zero.
    x = np.where(inside, x, 0.)
    y = np.where(inside, y, 0.)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    cells, weights = [], []
    for ox, oy, weight in [(0,0,(1-dx)*(1-dy)), (1,0,dx*(1-dy)),
                           (0,1,(1-dx)*dy), (1,1,dx*dy)]:
        kx, ky = x0 + ox, y0 + oy
        valid = inside & (abs(kx) < nx//2) & (abs(ky) < ny//2) & ((kx != 0) | (ky != 0))
        flip = (ky < 0) | ((ky == 0) & (kx < 0))
        kx, ky = np.where(flip, -kx, kx), np.where(flip, -ky, ky)
        cells.append(np.where(valid, (ky + ny//2)*nx + kx + nx//2, 0))
        weights.append(np.where(valid, weight, 0.))
    return np.array(cells), np.array(weights), inside


def geometry(ms, info, output):
    from casatools import table
    from scripts.corruption.metrics import _read_correlation_indices, _valid_mask
    started = time.perf_counter()
    tb = table()
    tb.open(str(ms / "DATA_DESCRIPTION"))
    spw = np.asarray(tb.getcol("SPECTRAL_WINDOW_ID"), int)
    pol = np.asarray(tb.getcol("POLARIZATION_ID"), int)
    tb.close()
    tb.open(str(ms / "SPECTRAL_WINDOW"))
    freqs = {i: np.asarray(tb.getcell("CHAN_FREQ", i), float) for i in range(tb.nrows())}
    tb.close()
    tb.open(str(ms / "POLARIZATION"))
    corr_types = {i: np.asarray(tb.getcell("CORR_TYPE", i), int) for i in range(tb.nrows())}
    tb.close()
    corr = _read_correlation_indices(ms)
    tb.open(str(ms / "ANTENNA"));names = list(tb.getcol("NAME"));tb.close()
    tb.open(str(ms / "OBSERVATION"))
    info["observation"] = {key: tb.getcol(key).tolist() for key in ("PROJECT", "TIME_RANGE", "TELESCOPE_NAME")}
    tb.close()
    tb.open(str(ms))
    a1 = np.asarray(tb.getcol("ANTENNA1"), int); a2 = np.asarray(tb.getcol("ANTENNA2"), int)
    times = np.asarray(tb.getcol("TIME"), float); intervals = np.asarray(tb.getcol("INTERVAL"), float)
    ddids = np.asarray(tb.getcol("DATA_DESC_ID"), int); uvw = np.asarray(tb.getcol("UVW"), float)
    pairs = np.unique(np.sort(np.c_[a1[a1 != a2], a2[a1 != a2]], axis=1), axis=0)
    pair_map = {tuple(pair): i for i, pair in enumerate(pairs)}
    pair_id = np.array([pair_map.get(tuple(sorted((a,b))), -1) for a,b in zip(a1,a2)])
    h = sparse.csr_matrix((len(pairs), 256**2), dtype=float)
    counts = np.zeros(len(pairs)); outside = 0.; total = 0.
    # Display sample selection spans rows/time; channels and hands vary across selected rows.
    selected = {}
    for b in range(len(pairs)):
        rows = np.flatnonzero(pair_id == b)
        rows = rows[np.argsort(times[rows], kind="stable")]
        for row in rows[np.linspace(0, len(rows)-1, min(DISPLAY_PER_BASELINE,len(rows))).astype(int)]:
            selected[int(row)] = b
    evidence = {k: [] for k in ("row", "channel", "correlation", "ddid", "baseline", "time", "u", "v")}
    flag_digest = hashlib.sha256()
    weight_digest = hashlib.sha256()
    for start in range(0, tb.nrows(), CHUNK):
        n = min(CHUNK, tb.nrows()-start); sl = slice(start,start+n)
        data = tb.getcol("DATA", startrow=start, nrow=n)
        flags = tb.getcol("FLAG", startrow=start, nrow=n)
        flag_row = tb.getcol("FLAG_ROW", startrow=start, nrow=n)
        flag_digest.update(flags.tobytes());flag_digest.update(flag_row.tobytes())
        weight_digest.update(tb.getcol("WEIGHT", startrow=start, nrow=n).tobytes())
        valid = _valid_mask(data, flags, a1[sl], a2[sl], flag_row, ddids[sl], corr)
        for dd in np.unique(ddids[sl]):
            channel, row = np.where(valid.sum(axis=0) * (ddids[sl] == dd)[None,:] > 0)
            if not len(row):continue
            weight = valid[:,channel,row].sum(axis=0)
            freq = freqs[int(spw[dd])][channel]
            u = uvw[0,start+row] * freq/C;v = uvw[1,start+row] * freq/C
            cells, deposit, inside = bilinear_cells(u,v,info["header"],info["shape"])
            ids = pair_id[start+row]
            counts += np.bincount(ids, weights=weight, minlength=len(pairs))
            total += weight.sum();outside += weight[~inside].sum()
            for k in range(4):
                good = deposit[k] > 0
                h += sparse.coo_matrix((deposit[k,good]*weight[good], (ids[good],cells[k,good])),shape=h.shape).tocsr()
        for row in range(start,start+n):
            if row not in selected:continue
            cc, ch = np.where(valid[:,:,row-start])
            if not len(ch):continue
            pick = (row * 104729) % len(ch); channel=int(ch[pick]); correlation=int(cc[pick])
            frequency=freqs[int(spw[ddids[row]])][channel]
            values=(row,channel,correlation,int(ddids[row]),int(pair_id[row]),float(times[row]),
                    float(uvw[0,row]*frequency/C),float(uvw[1,row]*frequency/C))
            for key,value in zip(evidence,values):evidence[key].append(value)
    tb.close()
    if outside/total > 0.01:
        raise ValueError(f"Important uv coverage outside FFT grid: {outside/total:.2%}")
    for key,values in evidence.items():
        evidence[key] = np.asarray(values, dtype=float if key in ("time","u","v") else int)
    sparse.save_npz(output / "coverage.npz", h)
    np.savez_compressed(output / "geometry.npz", pairs=pairs, counts=counts, **evidence)
    info.update(antenna_names=names, pairs=pairs.tolist(), valid_samples=int(total),
        outside_fraction=float(outside/total), deposited_fraction=float(h.sum()/total),
        nrows=len(a1), time_range=[float(times.min()),float(times.max())],
        interval_range=[float(intervals.min()),float(intervals.max())],
        frequency_range_hz=[float(min(f.min() for f in freqs.values())),float(max(f.max() for f in freqs.values()))],
        max_uv_wavelengths=float(max(np.hypot(uvw[0],uvw[1]))*max(f.max() for f in freqs.values())/C),
        sampling_selection="All fields, scans, DDIDs/channels; unflagged finite parallel hands; cross-correlations only",
        ddid_mapping={str(i):dict(spw_id=int(s),polarization_id=int(pol[i]),
            channels=len(freqs[int(s)]),corr_types=corr_types[int(pol[i])].tolist()) for i,s in enumerate(spw)},
        flags_sha256=flag_digest.hexdigest(), weights_sha256=weight_digest.hexdigest(),
        geometry_seconds=time.perf_counter()-started, displayed_samples=len(evidence["row"]))
    return h, evidence, pair_id


def read_evidence(ms, evidence):
    from casatools import table
    tb=table();tb.open(str(ms)); result=np.zeros(len(evidence["row"]),complex)
    for start in range(0,tb.nrows(),CHUNK):
        choose=np.flatnonzero((evidence["row"]>=start)&(evidence["row"]<start+CHUNK))
        if len(choose):
            data=tb.getcol("DATA",startrow=start,nrow=min(CHUNK,tb.nrows()-start))
            result[choose]=data[evidence["correlation"][choose],evidence["channel"][choose],evidence["row"][choose]-start]
    tb.close();return result


def write_gained_model(model_ms, observed_ms, antenna, gain, baseline=None):
    from casatools import table
    model=table(); out=table(); model.open(str(model_ms));out.open(str(observed_ms),nomodify=False)
    for start in range(0,model.nrows(),CHUNK):
        n=min(CHUNK,model.nrows()-start)
        data=model.getcol("DATA",startrow=start,nrow=n)
        a=model.getcol("ANTENNA1",startrow=start,nrow=n);b=model.getcol("ANTENNA2",startrow=start,nrow=n)
        if baseline is None:
            factor=np.where(a==antenna,gain,1)*np.where(b==antenna,np.conj(gain),1)
        else:
            affected=((a==baseline[0])&(b==baseline[1]))|((a==baseline[1])&(b==baseline[0]))
            factor=np.where(affected,gain,1)
        out.putcol("DATA",data*factor[None,None,:],startrow=start,nrow=n)
    out.close();model.close()


def add_shared_draw(model_ms, noise_ms, observed_ms, kappa, expected_flags, expected_weights):
    from casatools import table
    model=table();noisy=table();out=table()
    model.open(str(model_ms));noisy.open(str(noise_ms));out.open(str(observed_ms),nomodify=False)
    noise_digest=hashlib.sha256();flags_digest=hashlib.sha256();weights_digest=hashlib.sha256()
    max_roundoff=0.;noise_sq=0.;noise_n=0
    for start in range(0,model.nrows(),CHUNK):
        n=min(CHUNK,model.nrows()-start)
        truth=model.getcol("DATA",startrow=start,nrow=n)
        noise=noisy.getcol("DATA",startrow=start,nrow=n)-truth
        signal=out.getcol("DATA",startrow=start,nrow=n)
        combined=signal+kappa*noise
        out.putcol("DATA",combined,startrow=start,nrow=n)
        # DATA is the imaged column; inherited CORRECTED_DATA is not used.
        max_roundoff=max(max_roundoff,float(np.max(abs((combined-signal)-kappa*noise))))
        noise_digest.update(noise.tobytes());noise_sq+=float(np.sum(abs(noise.astype(complex))**2));noise_n+=noise.size
        flags_digest.update(out.getcol("FLAG",startrow=start,nrow=n).tobytes())
        flags_digest.update(out.getcol("FLAG_ROW",startrow=start,nrow=n).tobytes())
        weights_digest.update(out.getcol("WEIGHT",startrow=start,nrow=n).tobytes())
    out.close();noisy.close();model.close()
    assert flags_digest.hexdigest()==expected_flags, "Flags changed"
    assert weights_digest.hexdigest()==expected_weights, "Weights changed"
    return dict(shared_noise_sha256=noise_digest.hexdigest(),noise_component_rms=float(np.sqrt(noise_sq/(2*noise_n))),
        addition_roundoff_max_jy=max_roundoff,flags_unchanged=True,weights_unchanged=True)


def score_map(z,h,pairs):
    mass=np.asarray(h.sum(axis=1)).ravel()
    scores=np.divide(h @ (abs(z)**2).ravel(),mass,out=np.zeros(len(mass)),where=mass>0)
    antennas=np.unique(pairs[mass>0])
    ant_scores=np.array([np.median(scores[np.any(pairs==a,axis=1)&(mass>0)]) for a in antennas])
    return scores,antennas,ant_scores


def audit_parent(sid, output, info=None):
    """Read-only comparison of the sampling copy with its original observation."""
    from casatools import table
    out=output/sid
    info=info or json.loads((out/"pilot.json").read_text())
    parent=table();model=table();checked=[]
    columns={"":("ANTENNA1","ANTENNA2","UVW","TIME","INTERVAL","DATA_DESC_ID","FIELD_ID","SCAN_NUMBER","FLAG","FLAG_ROW"),
             "DATA_DESCRIPTION":("SPECTRAL_WINDOW_ID","POLARIZATION_ID"),
             "SPECTRAL_WINDOW":("CHAN_FREQ",), "POLARIZATION":("CORR_TYPE",),
             "FIELD":("PHASE_DIR",), "OBSERVATION":("PROJECT","TIME_RANGE",)}
    try:
        for sub,keys in columns.items():
            parent.open(str(Path(info["parent_ms"])/sub));model.open(str(out/"model.ms"/sub))
            assert parent.nrows()==model.nrows(), f"Row count changed: {sub}"
            for key in keys:
                for start in range(0,parent.nrows(),CHUNK):
                    n=min(CHUNK,parent.nrows()-start)
                    np.testing.assert_array_equal(parent.getcol(key,startrow=start,nrow=n),
                                                  model.getcol(key,startrow=start,nrow=n),err_msg=f"{sub}/{key}")
                checked.append(f"{sub or 'MAIN'}/{key}")
            parent.close();model.close()
    finally:
        parent.close();model.close()
    result=dict(parent_ms=info["parent_ms"],exactly_preserved_columns=checked,passed=True)
    dump(out/"parent_audit.json",result)
    return result


def generate(sid, output):
    from casatools import table
    from scripts.simulation import simulate_ms, add_thermal_noise_inplace, initialize_weights
    from scripts.corruption import ConstantGainSpec, measure_constant_gain_norms, solve_constant_gain, measure_corruption_metrics
    from scripts.imaging import DefaultImagingConfig, image_ms
    from scripts.corruption.metrics import _read_correlation_indices, _valid_mask
    out=output/sid;out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();info=source_info(sid)
    config=dict(source=sid,seeds=list(SEEDS),target=30.,variants=list(VARIANTS),parent=info["parent_ms"],version=1)
    config_path=out/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError("Output configuration differs; use a new output directory")
    dump(config_path,config)
    model_ms=out/"model.ms"
    if not model_ms.exists():
        simulate_ms(info["parent_ms"],info["components"],model_ms,noise_model=None)
    audit_parent(sid,output,info)
    initialize_weights(model_ms,info["sigma0"])
    h,evidence,pair_id=geometry(model_ms,info,out)
    pairs=np.array(info["pairs"]);counts=np.asarray(np.load(out/"geometry.npz")["counts"])
    active=np.unique(pairs[counts>0]); antenna=int(active[len(active)//2])
    amp_spec=ConstantGainSpec(model_ms,antenna,"amp",30.,info["sigma0"],sign=1)
    norms=measure_constant_gain_norms(amp_spec)
    amp=solve_constant_gain(amp_spec,norms)
    phase=solve_constant_gain(ConstantGainSpec(model_ms,antenna,"phase",30.,info["sigma0"],sign=1),norms)
    info.update(antenna_id=antenna,antenna_name=info["antenna_names"][antenna],
        amplitude_solution=asdict(amp),phase_solution=asdict(phase))
    # Dataclass solutions contain only numerical fields and nested numerical norms.
    model_vis=read_evidence(model_ms,evidence)
    np.savez_compressed(out/"model_evidence.npz",visibility=model_vis)
    # Match V1's rule, but freeze the numerical CLEAN threshold across pilot variants.
    threshold=info["imaging_reference"].get("threshold")
    clean=replace(DefaultImagingConfig.clean,threshold=threshold,dirty_peak_fraction=None)
    imaging=replace(DefaultImagingConfig,clean=clean)
    case_records=[]
    baseline_pair=tuple(pairs[np.flatnonzero((counts>0)&np.any(pairs==antenna,axis=1))[0]])
    if sid==SOURCES[0]:
        tb=table();tb.open(str(model_ms));norm_sq=0.;corr=_read_correlation_indices(model_ms)
        for start in range(0,tb.nrows(),CHUNK):
            n=min(CHUNK,tb.nrows()-start);d=tb.getcol("DATA",startrow=start,nrow=n)
            a=tb.getcol("ANTENNA1",startrow=start,nrow=n);b=tb.getcol("ANTENNA2",startrow=start,nrow=n)
            valid=_valid_mask(d,tb.getcol("FLAG",startrow=start,nrow=n),a,b,
                tb.getcol("FLAG_ROW",startrow=start,nrow=n),tb.getcol("DATA_DESC_ID",startrow=start,nrow=n),corr)
            valid &= (((a==baseline_pair[0])&(b==baseline_pair[1]))|((a==baseline_pair[1])&(b==baseline_pair[0])))[None,None,:]
            norm_sq+=float(np.sum(abs(d[valid].astype(complex)/info["sigma0"])**2))
        tb.close();baseline_gain=1+30/np.sqrt(norm_sq)
    for seed in SEEDS:
        noise_ms=out/f"noise_{seed}.ms"
        pending=list(VARIANTS)+(["single_baseline"] if seed==SEEDS[0] and sid==SOURCES[0] else [])
        if any(not (out/f"{seed}_{variant}"/"case.json").exists() for variant in pending):
            if not noise_ms.exists():
                shutil.copytree(model_ms,noise_ms)
                add_thermal_noise_inplace(noise_ms,noise_model="simplenoise",
                    noise_parameters={"simplenoise":f"{info['sigma0']:.17g}Jy"},seed=seed)
        for variant in pending:
            folder=out/f"{seed}_{variant}";case_path=folder/"case.json"
            if case_path.exists():
                case_records.append(json.loads(case_path.read_text()));continue
            print(f"PILOT {sid} seed={seed} {variant}",flush=True)
            if folder.exists():
                # Only an uncommitted folder in this configuration-matched pilot run.
                from scripts.create_dataset_v1 import _safe_remove_partial
                _safe_remove_partial(folder,out)
            folder.mkdir();ms=folder/"observed.ms"
            t0=time.perf_counter();shutil.copytree(model_ms,ms)
            gain=1+0j;kappa=2. if variant=="double_noise" else 1.;pair=None
            if variant in ("amplitude","mixture"):gain*=amp.g_amp
            if variant in ("phase","mixture"):gain*=np.exp(1j*phase.phi_rad)
            if variant=="single_baseline":gain=complex(baseline_gain);pair=baseline_pair
            write_gained_model(model_ms,ms,antenna,gain,pair)
            measured_snr=(0. if variant in ("no_gain","double_noise") else
                measure_corruption_metrics(model_ms,ms,info["sigma0"]).SNR_corr)
            if variant in ("amplitude","phase","single_baseline"):
                np.testing.assert_allclose(measured_snr,30.,rtol=0.01)
            if variant in ("no_gain","double_noise"):
                assert measured_snr==0
            shared=add_shared_draw(model_ms,noise_ms,ms,kappa,info["flags_sha256"],info["weights_sha256"])
            observed=read_evidence(ms,evidence)
            result=image_ms(ms,imaging,folder/"imaging",imsize=(256,256),pblimit=-0.1,
                            keep_intermediate_products=False,fits_invalid_policy="error")
            residual=load_fits_plane(result.residual_fits);dirty=load_fits_plane(result.dirty_fits)
            psf=load_fits_plane(result.psf_fits)
            if residual.shape!=(256,256):raise ValueError("Unexpected image shape")
            np.testing.assert_allclose(WCS(residual.celestial_header, fix=False).pixel_scale_matrix,
                WCS(info["header"], fix=False).pixel_scale_matrix,rtol=1e-8,atol=1e-15)
            np.testing.assert_allclose(WCS(residual.celestial_header, fix=False).wcs.crpix,
                WCS(info["header"], fix=False).wcs.crpix,atol=1e-8)
            if seed==SEEDS[0] and variant=="no_gain":
                np.savez_compressed(out/"reference_psf.npz",psf=psf.values)
            else:
                np.testing.assert_allclose(psf.values,np.load(out/"reference_psf.npz")["psf"],rtol=2e-5,atol=2e-6)
            z,u,v=phase_centered_fft(residual.values,residual.celestial_header,info["s0"])
            scores,antennas,ant_scores=score_map(z,h,pairs)
            rank=int(np.flatnonzero(antennas[np.argsort(-ant_scores,kind="stable")]==antenna)[0])+1
            np.savez_compressed(folder/"evidence.npz",visibility=observed,fft=z,residual=residual.values,
                dirty=dirty.values,baseline_scores=scores,antennas=antennas,antenna_scores=ant_scores)
            case=dict(id=folder.name,variant=variant,seed=seed,gain_real=float(gain.real),gain_imag=float(gain.imag),
                gain_amplitude=float(abs(gain)),gain_phase_deg=float(np.angle(gain,deg=True)),
                target_amplitude=30. if variant in ("amplitude","mixture") else 0.,
                target_phase=30. if variant in ("phase","mixture") else 0.,
                kappa=kappa,measured_SNR_corr=measured_snr,injected_baseline=None if pair is None else list(map(int,pair)),
                injected_antenna_rank=rank,seconds=time.perf_counter()-t0,checks=shared,
                effective_imaging=result.effective_imaging_parameters,
                residual_rms_over_s0=float(np.sqrt(np.mean((residual.values/info["s0"])**2))))
            dump(case_path,case);case_records.append(case)
            # Only delete this script's copied variant MS after evidence and FITS are committed.
            shutil.rmtree(ms)
            print(f"DONE {sid} {folder.name}: rank={rank}, {case['seconds']:.1f}s",flush=True)
        if noise_ms.exists():shutil.rmtree(noise_ms)
    # All seed-matched gain comparisons must have exactly the same saved draw checksum.
    for seed in SEEDS:
        hashes={c['checks']['shared_noise_sha256'] for c in case_records if c['seed']==seed}
        assert len(hashes)==1
    assert len({c['checks']['shared_noise_sha256'] for c in case_records if c['variant']=='no_gain'})==3
    info.update(cases=case_records,total_seconds=time.perf_counter()-started,
        visibility_note="At most 96 actual valid samples per baseline for display; scoring uses all valid samples. Copied variant MSs removed after FITS/evidence saved; original parent and noiseless model retained.")
    dump(out/"pilot.json",info)
    print(f"SOURCE COMPLETE {sid}",flush=True)


def visibility_display_limits(arrays):
    """Pool actual displayed measurements, independent of the selected case/pair."""
    bounds = {}
    for key, transform in (("amplitude", np.abs), ("phase", lambda v: np.angle(v, deg=True))):
        values = [transform(a["visibility"]) for a in arrays.values()]
        lo = min(float(v.min()) for v in values)
        hi = max(float(v.max()) for v in values)
        pad = max((hi-lo)*.04, 1e-9)
        bounds[key] = dict(min=lo, max=hi, axis_min=lo-pad, axis_max=hi+pad)
    return bounds


def render_report(sid, output):
    import base64
    import matplotlib.pyplot as plt
    from scripts.imaging.plot_utils import casa_image_to_png, shared_fits_display_limits
    out=output/sid;info=json.loads((out/"pilot.json").read_text())
    assert json.loads((out/"parent_audit.json").read_text())["passed"]
    geom=np.load(out/"geometry.npz");pairs=geom["pairs"];names=info["antenna_names"]
    h=sparse.load_npz(out/"coverage.npz");mass=np.asarray(h.sum(axis=1)).ravel()
    antennas=np.unique(pairs[mass>0]);ant=int(info["antenna_id"])
    cases=info["cases"];arrays={c["id"]:dict(np.load(out/c["id"]/"evidence.npz")) for c in cases}
    _,u,v=phase_centered_fft(arrays[cases[0]["id"]]["residual"],info["header"],info["s0"])
    color_limit=max(float(np.log1p(abs(a["fft"])**2).max()) for a in arrays.values())
    assert all(np.log1p(a["baseline_scores"]).max() <= color_limit+1e-10 for a in arrays.values())
    residual_paths=[out/c["id"]/"imaging/residual.fits.gz" for c in cases]
    residual_limits=shared_fits_display_limits(residual_paths)
    visibility_limits=visibility_display_limits(arrays)
    uvlim=1.06*max(abs(geom["u"]).max(),abs(geom["v"]).max())/1000
    n=len(antennas);ant_index={int(a):i for i,a in enumerate(antennas)}
    bid=geom["baseline"]
    uv_ids=[]
    for b in range(len(pairs)):
        ids=np.flatnonzero(bid==b)
        if len(ids):uv_ids.extend(ids[np.linspace(0,len(ids)-1,min(16,len(ids))).astype(int)].tolist())
    data_cases=[];recipes={}
    for case,residual_path in zip(cases,residual_paths):
        a=arrays[case["id"]]
        residual_png=out/"report_assets"/f"{case['id']}_residual.png"
        recipes[case["id"]]=casa_image_to_png(residual_path,residual_png,title="CLEAN residual",
            draw_beam=True,display_limits_mjy_per_beam=residual_limits)
        residual_uri="data:image/png;base64,"+base64.b64encode(residual_png.read_bytes()).decode()
        fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
        m=ax.pcolormesh(u/1000,v/1000,np.log1p(abs(a["fft"])**2),cmap="Reds",
                       vmin=0,vmax=color_limit,shading="nearest",rasterized=True)
        ax.set(title="Fourier power P = |Z|²",xlabel="u (kλ)",ylabel="v (kλ)",
               xlim=(-uvlim,uvlim),ylim=(-uvlim,uvlim),aspect="equal")
        fig.colorbar(m,ax=ax,shrink=.85,label="ln(1 + Fourier power P)")
        power_uri=figure_uri(fig,out/"report_assets"/f"{case['id']}_power.png")
        matrix=np.full((n,n),np.nan)
        for b,(left,right) in enumerate(pairs):
            if mass[b]<=0:continue
            i,j=ant_index[int(left)],ant_index[int(right)]
            matrix[i,j]=matrix[j,i]=np.log1p(a["baseline_scores"][b])
        fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
        m=ax.imshow(matrix,origin="lower",cmap="Reds",vmin=0,vmax=color_limit)
        ax.set_xticks(range(n),[str(x) for x in antennas],rotation=90)
        ax.set_yticks(range(n),[str(x) for x in antennas])
        ax.set(title="Baseline score B: mean Fourier power along each pair",xlabel="Antenna ID",ylabel="Antenna ID")
        fig.colorbar(m,ax=ax,shrink=.85,label="ln(1 + baseline score B)")
        matrix_uri=figure_uri(fig,out/"report_assets"/f"{case['id']}_baseline.png")
        vis=a["visibility"]
        data_cases.append(dict(id=case["id"],variant=case["variant"],
            label=f"{LABELS[case['variant']]} · example {SEEDS.index(case['seed'])+1}",
            residual=residual_uri,power=power_uri,matrix=matrix_uri,
            real=vis.real.tolist(),imag=vis.imag.tolist(),scores=a["baseline_scores"].tolist(),
            antenna_scores=a["antenna_scores"].tolist(),injected_baseline=case["injected_baseline"]))
    coverage_ant=np.array([np.median(geom["counts"][np.any(pairs==x,axis=1)&(mass>0)]) for x in antennas])
    coverage_rank=int(np.flatnonzero(antennas[np.argsort(-coverage_ant,kind="stable")]==ant)[0])+1
    rank_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for k,variant in enumerate(VARIANTS):
        rows=[c for c in cases if c["variant"]==variant]
        ranks=[c["injected_antenna_rank"] for c in rows]
        values=[float(np.log1p(arrays[c["id"]]["antenna_scores"][ant_index[ant]])) for c in rows]
        rank_rows.append((variant,ranks,values))
        xx=k+np.linspace(-.12,.12,len(rows))
        axes[0].scatter(xx,ranks,s=70,color="#a50f15")
        axes[1].scatter(xx,values,s=70,color="#a50f15")
    for ax in axes:
        ax.set_xticks(range(5),["No gain error","Amplitude","Phase","Both","2× noise"],rotation=15)
    axes[0].set(ylabel=f"Antenna {ant} rank (1 = highest)",title="Does the known faulty antenna rank highly?",ylim=(n+.5,.5))
    axes[1].set(ylabel="ln(1 + antenna score A)",title="The same antenna score across cases",ylim=(0,color_limit))
    summary_uri=figure_uri(fig,out/"report_assets"/"score_comparison.png")
    gain_rows=[c for c in cases if c["variant"] in ("amplitude","phase","mixture")]
    top1=sum(c["injected_antenna_rank"]==1 for c in gain_rows)
    top3=sum(c["injected_antenna_rank"]<=3 for c in gain_rows)
    payload=dict(cases=data_cases,pairs=pairs.tolist(),antennas=antennas.tolist(),names=names,
        antenna=ant,valid_baselines=np.flatnonzero(mass>0).tolist(),baseline=bid.tolist(),
        row=geom["row"].tolist(),channel=geom["channel"].tolist(),correlation=geom["correlation"].tolist(),
        ddid=geom["ddid"].tolist(),time=((geom["time"]-info["time_range"][0])/60).tolist(),
        u=(geom["u"]/1000).tolist(),v=(geom["v"]/1000).tolist(),uv_ids=uv_ids,
        color_limit=color_limit,uv_limit=uvlim,visibility_limits=visibility_limits,
        colors=[plt.colormaps["Reds"](i/255)[:3] for i in range(256)])
    body=f"""<style>
main{{max-width:1800px;padding:24px}}.report-image{{display:block;margin:12px auto}}
.image-row,.rank-row,.measurement-row{{display:grid;gap:10px;align-items:start}}
.image-row{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.rank-row{{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:18px}}
.measurement-row{{grid-template-columns:minmax(0,1fr) minmax(0,2fr)}}
.image-row figure,.rank-row figure{{margin:0;min-width:0}}
.image-row img,.rank-row svg,.measurement-row canvas{{display:block;width:100%;height:auto;margin:0}}
.rank-row h3{{font-size:16px;text-align:center;margin:4px 0}}figcaption{{font-size:13px;line-height:1.4}}
.zoomable{{cursor:zoom-in}}select{{max-width:100%}}
dialog{{border:1px solid #cbd5e1;border-radius:10px;width:min(1500px,94vw);max-height:94vh;padding:12px}}
dialog::backdrop{{background:#142c46bb}}#zoom-content img,#zoom-content svg{{display:block;width:100%;height:auto;max-height:82vh;object-fit:contain}}
#zoom-close{{display:block;margin-left:auto}}
@media(max-width:650px){{main{{padding:8px}}section{{padding:12px}}.image-row,.rank-row,.measurement-row{{grid-template-columns:1fr}}}}
</style>
<section><h2>1. From residual pixels to a candidate antenna</h2>
<pre>R / s₀ → complex Fourier response Z → Fourier power P = |Z|²
Baseline score B = coverage-weighted mean of P along one antenna pair
Antenna score A = median B over that antenna's partner baselines
Rank = position after sorting antenna scores from largest to smallest</pre>
<p><b>R</b> is the residual image; <b>s₀</b> is its fixed reference noise level.
<b>Z</b> contains the cosine and sine responses of R/s₀. <b>P</b> is their combined power.
Large B means strong residual power where a pair sampled the uv plane; large A means
several partners of an antenna have high B. The score is a candidate ranking, not a probability.</p>
<p>The Fourier map, baseline matrix and measurement dots use the same <b>ln(1 + value)</b> colour scale, from pale to dark red.
Fourier pixels show P; the matrix, tracks and visibility points show B.
B averages P, so their colours need not be identical. The logarithm only changes display;
scores are calculated before taking it.</p></section>
<section><h2>2. Residual → Fourier power → baseline score</h2>
<label>Case <select id='case-select' aria-label='Case'></select></label>
<p id='case-description'></p>
<div class='image-row'>
<figure><img class='zoomable' id='residual-panel' alt='CLEAN residual with RA and Dec axes' src='{data_cases[0]['residual']}'>
<figcaption>Residual brightness in mJy/beam; green ellipse = synthesized beam. Fixed colour limits.</figcaption></figure>
<figure><img class='zoomable' id='power-panel' alt='Fourier power with no sample overlay' src='{data_cases[0]['power']}'>
<figcaption>Fourier power P. Darker red means a stronger stripe response.</figcaption></figure>
<figure><img class='zoomable' id='baseline-panel' alt='Baseline score matrix' src='{data_cases[0]['matrix']}'>
<figcaption>Cell (a,b) = baseline score B for pair a–b. The matrix is symmetric; unavailable pairs are blank.</figcaption></figure>
</div>
<div class='rank-row'>
<figure><h3>Antenna scores · raw</h3><svg class='zoomable' id='antenna-raw' viewBox='0 0 600 440' role='img' aria-label='Ranked raw antenna scores'></svg></figure>
<figure><h3>Antenna scores · ln(1 + A)</h3><svg class='zoomable' id='antenna-log' viewBox='0 0 600 440' role='img' aria-label='Ranked logarithmic antenna scores'></svg></figure>
<figure><h3>Baseline scores · raw</h3><svg class='zoomable' id='baseline-raw' viewBox='0 0 600 440' role='img' aria-label='Ranked raw baseline scores'></svg></figure>
<figure><h3>Baseline scores · ln(1 + B)</h3><svg class='zoomable' id='baseline-log' viewBox='0 0 600 440' role='img' aria-label='Ranked logarithmic baseline scores'></svg></figure>
</div>
<p>One bar per antenna or pair, sorted from highest to lowest score; rank 1 is at the left.
The raw and logarithmic views have the same ordering. Their y axes start at zero and fit the selected case.
Teal bars and triangles mark the known injection. <span id='rank-truth'></span>
Hover over a bar for its ID and both score values. Click any plot to enlarge.</p></section>
<section><h2>3. Follow a candidate into the visibility measurements</h2>
<label>Candidate <select id='candidate-select' aria-label='Antenna or baseline candidate'></select></label>
<p id='candidate-description'></p><p id='ranking'></p>
<div class='measurement-row'>
<canvas class='zoomable' id='tracks' width='1100' height='850' aria-label='Selected candidate uv tracks'></canvas>
<canvas class='zoomable' id='timeseries' width='1100' height='460' aria-label='Visibility amplitude and phase versus time'></canvas>
</div>
<p id='hover-detail' class='muted'>Hover over a visibility point for its antenna pair, time and score.</p>
<p>The dropdown is the only candidate selection. Only its tracks are shown, with uniformly sized dots;
there are no truth rings or size codes. All dots use <b>ln(1 + baseline score B)</b>, exactly as the matrix.
Up to 16 samples per pair are drawn in uv and {DISPLAY_PER_BASELINE} in time to limit crowding;
all valid samples enter the score. Exact overlaps remain at their true coordinates.</p>
<p>Amplitude and phase are measured visibilities on the selected pairs. Their y ranges are fixed
across all cases and candidates. The two dashed lines mark the global minimum and maximum displayed
values for that measurement. Actual times are retained, so gaps are empty. Phase wraps at ±180°.
A pair's image score is constant along its time series; it does not locate the time of a fault.</p></section>
<section><h2>4. Is the score useful here?</h2>
<img class='report-image' alt='Antenna rank and antenna score by case type' src='{summary_uri}'>
<p>In these strong-error examples, the known faulty antenna {ant} ranks first in <b>{top1}/{len(gain_rows)}</b> gain-error cases.
Each dot represents a run. Both panels follow that same antenna; for no-gain cases it is only a reference,
since there is no faulty antenna. The right panel shows its antenna score A directly, with the same
logarithm as the candidate list; no score ratio is used.</p>
<p>Noise can also raise these scores. A high rank identifies where to inspect, but cannot establish
that a gain error exists. Shared uv cells can give several pairs high scores.</p>"""
    if any(c['variant']=='single_baseline' for c in cases):
        single=next(c for c in cases if c['variant']=='single_baseline')
        a=arrays[single['id']];b=int(np.flatnonzero(np.all(pairs==single['injected_baseline'],axis=1))[0])
        brank=int(np.flatnonzero(np.argsort(-a['baseline_scores'],kind='stable')==b)[0])+1
        body+=f"<p>In the single-baseline example, the injected pair {single['injected_baseline']} ranks {brank} of {int(np.sum(mass>0))} pairs.</p>"
    body+="</section><dialog id='plot-zoom'><button id='zoom-close' aria-label='Close enlarged plot'>Close</button><div id='zoom-content'></div></dialog>"
    script="const D="+json.dumps(payload,separators=(",",":"),allow_nan=False)+";\n"+PILOT_JS
    write_html(output/f"pilot_{sid}.html",f"Residual power and antenna candidates · {sid}",body,script)
    dump(out/"report_display.json",dict(residual_recipes=recipes,color_limit=color_limit,palette="Reds",
         visibility_limits=visibility_limits,uv_samples_per_baseline=16))
    dump(out/"inspection_summary.json",dict(source=sid,top1=top1,top3=top3,gain_cases=len(gain_rows),
        coverage_only_rank=coverage_rank,all_seed_top3_gate=top3==len(gain_rows),rank_rows=rank_rows,
        rank_rows_value="ln(1 + antenna score A)"))
    print(output/f"pilot_{sid}.html",flush=True)


def render_index(output):
    """Short reading guide and cross-observation comparison, once all reports exist."""
    if not all((output/f"pilot_{sid}.html").exists() for sid in SOURCES):return
    records=[json.loads((output/sid/"pilot.json").read_text()) for sid in SOURCES]
    summaries=[json.loads((output/sid/"inspection_summary.json").read_text()) for sid in SOURCES]
    parents=[r["parent_ms"] for r in records]
    projects=[tuple(r["observation"]["PROJECT"]) for r in records]
    assert len(set(parents))==3 and len(set(projects))==3, "Review shared pilot parent observations"
    top1=sum(s["top1"] for s in summaries)
    body=f"""<p>Vault V2 Setup · section 1 only · 3 training observations · 3 noise seeds · 46 fresh simulations.</p>
<div class='note'><b>The result:</b> the fixed injected antenna ranks first in {top1}/27 strong gain-error
cases. The toy checks agree with direct dot products. This supports continuing the investigation;
it is not a trained detector or an estimate of performance on unseen observations.</div>
<section><h2>Read these reports in order</h2><ol>
<li><a href='toy_fourier.html'>One wave → one Fourier response</a>: cosine, sine, their sum, amplitude scaling and between-bin leakage.</li>"""
    for r in records:
        body+=f"<li><a href='pilot_{r['id']}.html'>{r['id']}: residual → Fourier power → candidate pairs</a> "
        body+=f"({r['frequency_range_hz'][0]/1e9:.2f}–{r['frequency_range_hz'][1]/1e9:.2f} GHz; maximum {r['max_uv_wavelengths']/1000:.1f} kλ).</li>"
    body+="""</ol><p>In each pilot, select <b>Amplitude only</b>, then <b>Phase only</b>, and compare the same seed.
Next select a candidate antenna or pair to inspect actual contributing visibility amplitudes and phases.
Switch to <b>No gain error</b> and <b>Twice the thermal σ</b> before reading the three-seed summary.
Compare the Fourier power with the baseline scores, then inspect the selected visibility samples.</p></section>
<section><h2>Different sampling, same controlled question</h2><div class='scroll'><table>
<tr><th>Observation</th><th>Valid samples</th><th>Pixel (arcsec)</th><th>Gain cases: top 1</th><th>No-gain reference ranks</th><th>Coverage-only rank</th></tr>"""
    for r,s in zip(records,summaries):
        pixel=abs(WCS(r['header'],fix=False).pixel_scale_matrix[0,0])*3600
        controls=next(x[1] for x in s['rank_rows'] if x[0]=='no_gain')
        body+=f"<tr><td>{r['id']}</td><td>{r['valid_samples']:,}</td><td>{pixel:.4f}</td><td>{s['top1']}/9</td>"
        body+=f"<td>{' / '.join(map(str,controls))}</td><td>{s['coverage_only_rank']}</td></tr>"
    body+="""</table></div><p>All pilots have distinct parent MS paths and distinct project identifiers in their
original OBSERVATION tables. Each source uses a different observing epoch; seeds repeat that source's
sampling and do not create additional independent observations. No held-out products were explored.</p>
<p>Noise controls still produce candidate rankings. Their scores rise when noise doubles; the unchanged
rank ordering is expected when the same draw is scaled. Ranking alone cannot distinguish a gain fault
from noise. Shared uv cells and CLEAN can also spread a baseline's evidence.</p></section>
<section><h2>What was checked and retained</h2><p>Analytical FFT sign, phase centre, rotated WCS, direct
response, amplitude and power scaling; bilinear coverage and conjugate folding; training selection;
exact parent sampling and flags; fixed imaging weights, WCS and PSFs; pure injection strengths near 30;
matching noise within seeds and independent draws across seeds. All 46 cases retain FITS, complex
Fourier maps, coverage maps, sampled visibility evidence and numerical metadata.</p>
<p>Each pilot HTML embeds its figures, data and controls and works offline. The index links to those
files, so keep the five HTML files together when sharing. Variant MS copies were removed after saving
evidence; the original archival data and V1 products are unchanged.</p></section>
<div class='caution'><b>Boundary:</b> no dataset expansion, feature-bank construction, regression or physical
gain fitting was performed. Figure 3 is deferred until fitting is authorized. Strong controlled
localization is encouraging; weak errors, unfamiliar sky structure and detection thresholds remain untested.</div>"""
    write_html(output/"index.html","Making directional features visible",body)
    print(output/"index.html",flush=True)


PILOT_JS = r"""
const byId=id=>document.getElementById(id), cs=byId('case-select'), ca=byId('candidate-select');
let points=[];
function color(s){let i=Math.round(255*Math.min(1,Math.max(0,Math.log1p(s)/D.color_limit)));return 'rgb('+D.colors[i].map(x=>Math.round(x*255)).join(',')+')'}
D.cases.forEach((c,i)=>cs.add(new Option(c.label,i)));
cs.value=String(Math.max(0,D.cases.findIndex(c=>c.variant==='amplitude')));
function candidateMask(b){if(ca.value==='all')return true;let v=ca.value.split(':');return v[0]==='a'?D.pairs[b].includes(+v[1]):b===+v[1]}
function rankedScores(c,kind){
 const gain=['amplitude','phase','mixture'].includes(c.variant);
 let rows=kind==='antenna'?D.antennas.map((id,i)=>({id,label:'Antenna '+id,raw:c.antenna_scores[i],affected:gain&&id===D.antenna})):
 D.valid_baselines.map(id=>({id,label:'Pair '+D.pairs[id].join('–'),raw:c.scores[id],affected:
   c.injected_baseline?D.pairs[id].every((a,i)=>a===c.injected_baseline[i]):gain&&D.pairs[id].includes(D.antenna)}));
 return rows.sort((a,b)=>b.raw-a.raw||a.id-b.id);
}
function drawRankedScores(c){
 const fmt=x=>x>=100?x.toFixed(0):x>=10?x.toFixed(1):x.toFixed(2);
 for(let kind of ['antenna','baseline']){
  const rows=rankedScores(c,kind),n=rows.length;
  for(let mode of ['raw','log']){
   const ymax=Math.max(1e-9,(mode==='log'?Math.log1p(rows[0].raw):rows[0].raw)*1.07);
   const x0=76,y0=24,w=504,h=326,step=w/n;
   let s='<rect width="600" height="440" fill="white"/><g font-family="system-ui" font-size="18" fill="#233044">';
   for(let k=0;k<=4;k++){
    let value=ymax*k/4,y=y0+h-h*k/4;
    s+='<text x="68" y="'+(y+6)+'" text-anchor="end">'+fmt(value)+'</text>';
   }
   rows.forEach((r,k)=>{
    const value=mode==='log'?Math.log1p(r.raw):r.raw,height=h*value/ymax,x=x0+k*step,y=y0+h-height;
    const title=r.label+' · rank '+(k+1)+' · raw score '+r.raw.toPrecision(7)+' · ln(1 + score) '+Math.log1p(r.raw).toFixed(6)+(r.affected?' · known injection':'');
    s+='<rect data-entity="'+r.id+'" data-affected="'+r.affected+'" x="'+x+'" y="'+y+'" width="'+(step*.92)+'" height="'+height+'" fill="'+(r.affected?'#007f78':'#ce5551')+'"><title>'+title+'</title></rect>';
    if(r.affected){let cx=x+step*.46;s+='<path d="M '+cx+' '+(y-3)+' l -4 -7 h 8 z" fill="#007f78"><title>'+title+'</title></path>';}
   });
   s+='<path d="M '+x0+' '+y0+' V '+(y0+h)+' H '+(x0+w)+'" stroke="#64748b" fill="none"/>';
   for(let k of [...new Set([0,Math.round((n-1)/4),Math.round((n-1)/2),Math.round(3*(n-1)/4),n-1])])
    s+='<text x="'+(x0+(k+.5)*step)+'" y="380" text-anchor="middle">'+(k+1)+'</text>';
   const symbol=kind==='antenna'?'A':'B',label=mode==='log'?'ln(1 + '+symbol+')':symbol+' (raw)';
   s+='<text x="328" y="418" text-anchor="middle">Rank (highest score first)</text><text transform="translate(20 187) rotate(-90)" text-anchor="middle">'+label+'</text></g>';
   byId(kind+'-'+mode).innerHTML=s;
  }
 }
 const na=rankedScores(c,'antenna').filter(r=>r.affected).length,nb=rankedScores(c,'baseline').filter(r=>r.affected).length;
 byId('rank-truth').textContent=na?'Antenna '+D.antenna+' and its '+nb+' affected baselines are marked.':nb?'Only pair '+c.injected_baseline.join('–')+' was corrupted; no antenna gain was injected.':'This case has no injected gain error, so no bars are marked.';
}
function axes(ctx,x,y,w,h,xmin,xmax,ymin,ymax,title,xlabel,ylabel){
 ctx.fillStyle='#233044';ctx.font='16px system-ui';ctx.fillText(title,x,y-20);ctx.strokeStyle='#94a3b8';ctx.strokeRect(x,y,w,h);
 ctx.font='12px system-ui';for(let i=0;i<=4;i++){let t=i/4;ctx.fillText((xmin+t*(xmax-xmin)).toFixed(1),x+t*w-12,y+h+20);ctx.fillText((ymax-t*(ymax-ymin)).toFixed(2),x-53,y+t*h+4)}
 ctx.fillText(xlabel,x+w/2-60,y+h+43);ctx.save();ctx.translate(x-65,y+h/2);ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,-50,0);ctx.restore();
 return (a,b)=>[x+(a-xmin)/(xmax-xmin)*w,y+h-(b-ymin)/(ymax-ymin)*h];
}
function populate(){let c=D.cases[+cs.value],old=ca.value;ca.innerHTML='';
 D.antennas.map((a,i)=>[a,c.antenna_scores[i]]).sort((a,b)=>b[1]-a[1]||a[0]-b[0]).forEach(([a,s],rank)=>ca.add(new Option('Antenna '+a+' ('+D.names[a]+') · rank '+(rank+1)+' · ln(1 + A) = '+Math.log1p(s).toFixed(3),'a:'+a)));
 D.valid_baselines.map(i=>[D.pairs[i],i,c.scores[i]]).sort((a,b)=>b[2]-a[2]||a[1]-b[1]).forEach(([p,i,s],rank)=>ca.add(new Option('Pair '+p.join('–')+' · rank '+(rank+1)+' · ln(1 + B) = '+Math.log1p(s).toFixed(3),'b:'+i)));
 ca.add(new Option('All antenna pairs','all'));
 if([...ca.options].some(o=>o.value===old))ca.value=old;
 byId('residual-panel').src=c.residual;byId('power-panel').src=c.power;byId('baseline-panel').src=c.matrix;
 let truth=c.variant==='no_gain'||c.variant==='double_noise'?'No gain error was injected.':c.injected_baseline?'Known injected fault: pair '+c.injected_baseline.join('–')+'.':'Known injected fault: antenna '+D.antenna+' ('+D.names[D.antenna]+').';
 byId('case-description').textContent=c.label+'. '+truth;
 drawRankedScores(c);
 draw();
}
function draw(){let c=D.cases[+cs.value];
 let ids=D.baseline.map((b,i)=>candidateMask(b)?i:-1).filter(i=>i>=0);
 let uvIds=D.uv_ids.filter(i=>candidateMask(D.baseline[i]));
 byId('candidate-description').textContent=ca.options[ca.selectedIndex].text+'. Showing only this selection: '+uvIds.length+' uv samples (and their conjugates), '+ids.length+' visibility samples.';
 let ranked=D.antennas.map((a,i)=>[a,c.antenna_scores[i]]).sort((a,b)=>b[1]-a[1]||a[0]-b[0]).slice(0,5);
 byId('ranking').textContent='Top antenna candidates, ln(1 + A): '+ranked.map(([a,s])=>a+' ('+D.names[a]+') = '+Math.log1p(s).toFixed(3)).join(' · ');
 let cv=byId('tracks'),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
 let transform=axes(ctx,85,55,720,720,-D.uv_limit,D.uv_limit,-D.uv_limit,D.uv_limit,'Selected uv tracks','u (kλ)','v (kλ)');
 uvIds.forEach(i=>{let b=D.baseline[i];for(let sign of [1,-1]){let [x,y]=transform(sign*D.u[i],sign*D.v[i]);ctx.fillStyle=color(c.scores[b]);ctx.beginPath();ctx.arc(x,y,3.3,0,2*Math.PI);ctx.fill()}});
 let barX=865,barY=170,barH=430;
 for(let i=0;i<256;i++){ctx.fillStyle='rgb('+D.colors[i].map(x=>Math.round(x*255)).join(',')+')';ctx.fillRect(barX,barY+barH-i*barH/255,25,barH/255+1)}
 ctx.fillStyle='#233044';ctx.font='14px system-ui';ctx.fillText('ln(1 + baseline score B)',barX-10,barY-35);
 for(let i=0;i<=4;i++)ctx.fillText((D.color_limit*i/4).toFixed(2),barX+38,barY+barH-i*barH/4+5);
 ctx.fillText('Same scale as the matrix',barX-10,barY+barH+35);
 cv=byId('timeseries');ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);points=[];
 let tmax=Math.max(...D.time),amp=D.visibility_limits.amplitude,phase=D.visibility_limits.phase;
 let ta=axes(ctx,85,55,425,310,0,tmax,amp.axis_min,amp.axis_max,'Visibility amplitude','Minutes from first integration','Amplitude (Jy)');
 let tp=axes(ctx,650,55,390,310,0,tmax,phase.axis_min,phase.axis_max,'Visibility phase','Minutes from first integration','Phase (degrees)');
 for(let [tr,range] of [[ta,amp],[tp,phase]]){
  ctx.strokeStyle='#64748b';ctx.lineWidth=1;ctx.setLineDash([5,4]);
  for(let [label,value] of [['global min',range.min],['global max',range.max]]){let [x0,y]=tr(0,value),[x1]=tr(tmax,value);ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();ctx.fillStyle='#475569';ctx.font='11px system-ui';ctx.fillText(label+' '+value.toFixed(4),x0+8,y+(label==='global max'?-3:13))}
  ctx.setLineDash([]);
 }
 ids.forEach(i=>{let b=D.baseline[i],re=c.real[i],im=c.imag[i],av=Math.hypot(re,im),pv=Math.atan2(im,re)*180/Math.PI;
  for(let [tr,val] of [[ta,av],[tp,pv]]){let [x,y]=tr(D.time[i],val);ctx.fillStyle=color(c.scores[b]);ctx.beginPath();ctx.arc(x,y,2.8,0,2*Math.PI);ctx.fill();points.push({x,y,i,amp:av,phase:pv})}});
}
cs.addEventListener('change',populate);ca.addEventListener('change',draw);
function enlargePlot(node){
 const tag=node.tagName.toLowerCase();
 byId('zoom-content').innerHTML=tag==='svg'?'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 440">'+node.innerHTML+'</svg>':
  '<img alt="Enlarged plot" src="'+(tag==='canvas'?node.toDataURL('image/png'):node.src)+'">';
 byId('plot-zoom').showModal();
}
for(let id of ['residual-panel','power-panel','baseline-panel','antenna-raw','antenna-log','baseline-raw','baseline-log','tracks','timeseries']){
 const node=byId(id);node.tabIndex=0;node.addEventListener('click',()=>enlargePlot(node));
 node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();enlargePlot(node)}});
}
byId('zoom-close').addEventListener('click',()=>byId('plot-zoom').close());
byId('timeseries').addEventListener('mousemove',event=>{let box=event.target.getBoundingClientRect(),x=(event.clientX-box.left)*event.target.width/box.width,y=(event.clientY-box.top)*event.target.height/box.height;
 let nearest=null,best=100;for(let p of points){let d=(p.x-x)**2+(p.y-y)**2;if(d<best){best=d;nearest=p}}if(nearest){let i=nearest.i,c=D.cases[+cs.value];byId('hover-detail').textContent='Pair '+D.pairs[D.baseline[i]].join('–')+' · '+D.time[i].toFixed(3)+' min · amplitude '+nearest.amp.toFixed(6)+' Jy · phase '+nearest.phase.toFixed(2)+'° · ln(1 + baseline score B) = '+Math.log1p(c.scores[D.baseline[i]]).toFixed(4)}});
populate();
"""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUTPUT)
    p.add_argument("--source",choices=SOURCES,action="append")
    p.add_argument("--generate-only",action="store_true")
    p.add_argument("--report-only",action="store_true")
    p.add_argument("--audit-only",action="store_true",help="CASA: verify retained model sampling against its original parent")
    args=p.parse_args()
    if sum((args.generate_only,args.report_only,args.audit_only))>1:p.error("Choose one mode")
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    for sid in args.source or SOURCES:
        if args.audit_only:
            audit_parent(sid,args.output);continue
        if not args.report_only:generate(sid,args.output)
        if not args.generate_only:render_report(sid,args.output)
    if not args.generate_only and not args.audit_only:render_index(args.output)


if __name__=="__main__":
    main()
