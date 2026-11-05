"""Point-source CLEAN recovery metrics and thermal-noise report."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

LEVELS = (0., 1., 2.5, 5., 10., 15., 20., 30., 40., 50., 100., 200., 400., 700., 1000., 2000., 3000., 4000.)
METRICS = ('e_TI_1','e_TI_2','e_TM_1','e_TM_2','e_II_1','e_II_2','e_MM_1','e_MM_2','e_Fin','e_Fout','e_pos','robust_rms')


def geometry(header, shape, flux, ra_rad, dec_rad):
    """Analytic restoring Gaussian centred on stored true coordinates."""
    wcs = WCS(header).celestial
    yy, xx = np.indices(shape)
    centre = SkyCoord(ra_rad*u.rad, dec_rad*u.rad, frame='fk5', equinox='J2000')
    east, north = centre.spherical_offsets_to(wcs.pixel_to_world(xx, yy))
    east, north = east.to_value(u.arcsec), north.to_value(u.arcsec)
    pa = np.deg2rad(header['BPA'])
    major = east*np.sin(pa) + north*np.cos(pa)
    minor = east*np.cos(pa) - north*np.sin(pa)
    rho = np.hypot(major/(header['BMAJ']*3600), minor/(header['BMIN']*3600))
    return rho, flux*np.exp(-4*np.log(2)*rho**2)


def convolve_model(model, header):
    """Zero-padded convolution: Jy/pixel components, peak-one restoring beam.

    Pixel offsets use the FITS tangent-plane matrix, including rotation and
    handedness. An odd kernel spanning all image lags preserves pixel centres.
    """
    from scipy.signal import fftconvolve
    model = np.asarray(model, dtype=np.float64)
    if model.ndim != 2 or not np.isfinite(model).all():
        raise ValueError('Expected finite 2D model')
    if not np.any(model): return np.zeros_like(model)
    ny, nx = model.shape
    dy, dx = np.mgrid[-ny+1:ny, -nx+1:nx]
    matrix = WCS(header).celestial.pixel_scale_matrix * 3600
    east = matrix[0,0]*dx + matrix[0,1]*dy
    north = matrix[1,0]*dx + matrix[1,1]*dy
    pa = np.deg2rad(header['BPA'])
    major = (east*np.sin(pa) + north*np.cos(pa))/(header['BMAJ']*3600)
    minor = (east*np.cos(pa) - north*np.sin(pa))/(header['BMIN']*3600)
    if header['BMAJ'] <= 0 or header['BMIN'] <= 0: raise ValueError('Invalid restoring beam')
    kernel = np.exp(-4*np.log(2)*(major**2 + minor**2))
    return fftconvolve(model, kernel, mode='same')


def measure(I, R, M, T, rho, omega, annulus, flux, regular_I, regular_R,
            *, smoothed_model=None, regular_smoothed_model=None, compute_position=True):
    """Float64 full-support recovery and component-flux metrics."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (I,R,M,T,rho,regular_I,regular_R)]
    I,R,M,T,rho,regular_I,regular_R = arrays
    omega, annulus = np.asarray(omega,bool), np.asarray(annulus,bool)
    if not omega.any() or any(x.shape != omega.shape for x in arrays):
        raise ValueError('Empty support or mismatched array shapes')
    if not all(np.isfinite(x[omega]).all() for x in arrays):
        raise ValueError('Nonfinite pixels inside fixed support')
    if not annulus.any() or not np.isfinite(R[annulus]).all():
        raise ValueError('Invalid residual annulus')
    if flux <= 0: raise ValueError('Flux must be positive')
    source = omega & (rho <= 1)
    if not source.any(): raise ValueError('Empty source region')
    C = I-R if smoothed_model is None else np.asarray(smoothed_model, dtype=np.float64)
    C1 = regular_I-regular_R if regular_smoothed_model is None else np.asarray(regular_smoothed_model, dtype=np.float64)
    if any(x.shape != omega.shape or not np.isfinite(x[omega]).all() for x in (C,C1)):
        raise ValueError('Invalid smoothed model')
    denom1, denom2 = np.abs(T[omega]).sum(), np.square(T[omega]).sum()
    if denom1 <= 0 or denom2 <= 0: raise ValueError('Invalid true-sky denominators')
    metrics = {}
    for name, a, b in (('TI',I,T), ('TM',C,T), ('II',I,regular_I), ('MM',C,C1)):
        delta = (a-b)[omega]
        metrics[f'e_{name}_1'] = float(np.abs(delta).sum()/denom1)
        metrics[f'e_{name}_2'] = float(np.sqrt(np.square(delta).sum()/denom2))
    signed = float((M[source].sum()-flux)/flux)
    peak = float(C[omega].max())
    pos = None
    if compute_position and peak > 0:
        candidates = np.argwhere(omega & (C == peak))
        # argwhere is row/column sorted; stable argmin resolves remaining ties.
        chosen = candidates[np.argmin(rho[tuple(candidates.T)])]
        pos = float(rho[tuple(chosen)])
    residual = R[annulus]
    metrics.update(e_Fin=abs(signed), signed_flux_error=signed,
        e_Fout=float(np.abs(M[omega & ~source]).sum()/flux), e_pos=pos,
        missing_detection=pos is None,
        empty_raw_model=not bool(np.any(M[omega]!=0)),
        position_roundoff_warning=not bool(np.any(M[omega]!=0)) and pos is not None,
        robust_rms=float(1.4826*np.median(np.abs(residual-np.median(residual)))))
    return metrics


def load_products(root, record):
    from scripts.preprocessing.fits import load_fits_plane, _wcs_signature
    products={key:load_fits_plane(root/record['products'][key]) for key in ('dirty','clean','residual','model','psf')}
    ref=products['clean']
    for key, p in products.items():
        if p.shape != ref.shape or _wcs_signature(p.celestial_header) != _wcs_signature(ref.celestial_header):
            raise ValueError(f'Mismatched geometry: {key}')
        if not np.allclose(p.beam,ref.beam,rtol=1e-6,atol=1e-10): raise ValueError('Beam changed')
    for key in ('dirty','clean','residual'):
        if products[key].unit != 'Jy/beam': raise ValueError('Expected Jy/beam')
    if products['model'].unit != 'Jy/pixel': raise ValueError('Expected Jy/pixel raw model')
    return {key:p.values.astype(np.float64) for key,p in products.items()}


def collect_metrics(root: Path, plan):
    """Use completed checkpoints only; missing cases remain explicit."""
    levels,seeds=plan['levels'],plan['seeds']; rows=[]; failures=[]; all_records={}
    for source in plan['sources']:
        folder=root/'sources'/source/'recovery'
        try:
            reference=json.loads((folder/'reference.json').read_text())
            with np.load(folder/'reference_masks.npz') as masks:
                omega,annulus,rho,T=[masks[k] for k in ('omega','annulus','rho','truth')]
        except (OSError,ValueError,KeyError) as exc:
            failures.append(dict(source=source,error=f'Invalid reference: {exc}'));continue
        for seed in seeds:
            regular_path=folder/f'seed_{seed:02d}/std_1/record.json'
            if not regular_path.exists():
                failures.append(dict(source=source,seed=seed,error='Missing paired regular image')); continue
            try:
                regular_record=json.loads(regular_path.read_text()); regular=load_products(root,regular_record)
            except (OSError,ValueError,KeyError) as exc:
                failures.append(dict(source=source,seed=seed,error=f'Invalid paired regular image: {exc}'));continue
            for level in levels:
                path=folder/('noiseless' if level==0 else f'seed_{seed:02d}/std_{level:g}')/'record.json'
                try:
                    rec=json.loads(path.read_text()); products=load_products(root,rec)
                    support=np.load(root/rec['products']['support'])
                    if not support['valid'][omega].all(): raise ValueError('New invalid support inside Omega')
                    values=measure(products['clean'],products['residual'],products['model'],T,rho,omega,annulus,
                        reference['flux_jy'],regular['clean'],regular['residual'])
                    row=dict(source=source,seed=seed,noise_multiplier=level,**values)
                    rows.append(row); all_records[source,seed,level]=rec
                except (OSError, ValueError, KeyError) as exc:
                    failures.append(dict(source=source,seed=seed,level=level,error=str(exc)))
    return rows,failures,all_records


def progress(root, plan):
    counts={}
    expected=1+len(plan['seeds'])*sum(a>0 for a in plan['levels'])
    for source in plan['sources']:
        folder=root/'sources'/source/'recovery'
        paths=[folder/'noiseless/record.json',*[folder/f'seed_{s:02d}/std_{a:g}/record.json'
            for s in plan['seeds'] for a in plan['levels'] if a>0]]
        counts[source]=sum(p.is_file() for p in paths)
    return dict(completed_images=sum(counts.values()),expected_images=expected*len(counts),
        complete_sources=sum(n==expected for n in counts.values()),expected_sources=len(counts),
        images_per_source=expected,source_counts=counts,
        updated_utc=datetime.now(timezone.utc).isoformat(),
        generation_failures=[p.name for p in root.glob('failure_*.json')])

PAIRS = ('TI', 'TM', 'II', 'MM')
DIAGNOSTICS = (('e_Fin', r'$e_{\mathrm{Fin}}$'),
               ('e_Fout', r'$e_{\mathrm{Fout}}$'),
               ('e_pos', r'$e_{\mathrm{pos}}$ (beam FWHM)'),
               ('robust_rms', 'Robust residual RMS (Jy/beam)'))


def nominal_snr(root, source):
    settings = json.loads((root/'sources'/source/'recovery'/'settings.json').read_text())
    snr = settings.get('nominal_source_snr')
    if snr is None or not np.isfinite(snr) or snr <= 0:
        raise ValueError('Recovery settings lack valid nominal_source_snr')
    return float(snr)


def metric_grid(root, source, rows, levels, seeds, snr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = np.asarray(levels, float)
    r = a/snr
    fig, axes = plt.subplots(7, 2, figsize=(12, 18), sharex=True)
    fig.subplots_adjust(left=.12, right=.87, top=.995, bottom=.075, hspace=0, wspace=0)
    panels = [(f'e_{pair}_{norm}', rf'$e_{{{pair},{norm}}}$', f'$L_{norm}$')
              for pair in PAIRS for norm in (2, 1)]
    panels += [(key, label, None) for key, label in DIAGNOSTICS[:3]]
    panels += [(None, None, None), (*DIAGNOSTICS[3], None), (None, None, None)]

    def plotted_value(item, key):
        # An empty raw model can leave a spurious I-R peak at FITS roundoff.
        if key == 'e_pos' and item.get('position_roundoff_warning'):
            return np.nan
        value = item.get(key)
        return np.nan if value is None else float(value)

    for row_index in range(7):
        for col_index in range(2):
            key, label, corner = panels[2*row_index+col_index]
            ax = axes[row_index, col_index]
            if key is None:
                ax.set_visible(False)
                continue
            for seed in seeds:
                by_level = {item['noise_multiplier']: plotted_value(item, key) for item in rows if item['seed']==seed}
                ax.plot(r, [by_level.get(level, np.nan) for level in levels],
                        color='#527cb6', alpha=.23, lw=.8)
            grouped = [[value for item in rows if item['noise_multiplier']==level
                        if np.isfinite(value := plotted_value(item, key))] for level in levels]
            mean = np.array([np.mean(values) if values else np.nan for values in grouped])
            sigma = np.array([np.std(values, ddof=1) if len(values)>1 else 0. for values in grouped])
            ax.plot(r, mean, color='#101820', lw=1.8)
            ax.fill_between(r, mean-sigma, mean+sigma, color='#527cb6', alpha=.2)
            ax.set_xscale('symlog', linthresh=1/snr, linscale=.7)
            ax.set_xlim(-.25/snr, r[-1]*1.2)
            ax.grid(True, which='both', alpha=.25, lw=.5)
            if corner: ax.text(.98,.95,corner,transform=ax.transAxes,ha='right',va='top',fontsize=10)
            ax.set_ylabel(label)
            if col_index==1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            if row_index<6: ax.tick_params(labelbottom=False)
    indices = sorted({i for i in (0,1,4,10,14,len(a)-1) if i<len(a)})
    ticks = r[indices]
    labels = [f'{v:.2g}\n({a[i]:g})' for i,v in zip(indices,ticks)]
    for ax in (axes[-1,0], axes[4,1]):
        ax.set_xticks(ticks,labels,fontsize=8)
        ax.set_xlabel(r'$r=a/\mathrm{SNR}_0$  (parentheses: $a$)')
        ax.tick_params(labelbottom=True)
    name=f'{source}_recovery_metrics_seven_rows.png'
    fig.savefig(root/name,dpi=150)
    plt.close(fig)
    return name


def aggregate_training_metric(rows, levels, sources, metric):
    curves={};pending=[]
    for source in sources:
        selected=[row for row in rows if row['source']==source]
        by_level={row['noise_multiplier']:row for row in selected}
        if len(by_level)!=len(selected): raise ValueError('Training aggregation requires one seed/row per source and level')
        if set(by_level)!=set(levels): pending.append(source);continue
        curves[source]=[by_level[a][metric] for a in levels]
    median=[];counts=[]
    for index in range(len(levels)):
        values=[v[index] for v in curves.values() if v[index] is not None and np.isfinite(v[index])]
        counts.append(len(values));median.append(float(np.median(values)) if values else None)
    return dict(levels=list(levels),curves=curves,complete_sources=list(curves),pending_sources=pending,
        source_counts=counts,median_values=median)


def report(root: Path):
    root=Path(root).resolve()
    plan=json.loads((root/'recovery_plan.json').read_text())
    rows,failures,_=collect_metrics(root,plan)
    status=progress(root,plan)
    for row in rows:
        snr=nominal_snr(root,row['source'])
        row['a']=row['noise_multiplier']
        row['r']=row['noise_multiplier']/snr
        row['nominal_snr0']=snr
    if rows:
        with (root/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else: (root/'metrics.csv').write_text('source,seed,noise_multiplier,a,r,nominal_snr0\n')
    figures=[]
    for source in plan['sources']:
        source_rows=[row for row in rows if row['source']==source]
        if not source_rows: continue
        snr=nominal_snr(root,source)
        name=metric_grid(root,source,source_rows,plan['levels'],plan['seeds'],snr)
        settings=json.loads((root/'sources'/source/'recovery'/'settings.json').read_text())
        ref=json.loads((root/'sources'/source/'recovery'/'reference.json').read_text())
        provenance=settings.get('flux_provenance',{})
        unreliable_positions=sum(bool(row['position_roundoff_warning']) for row in source_rows)
        figures.append(f'<h2>{html.escape(source)}</h2><p>Point-source flux {ref["flux_jy"]:.9g} Jy from original CLEAN model; original image peak {provenance.get("clean_peak_jy_per_beam",float("nan")):.9g} Jy/beam. Predicted nominal image RMS {settings["predicted_nominal_image_rms_jy_per_beam"]:.9g} Jy/beam; SNR₀ {snr:.5g}. Same ideal thermal-noise request and nominal weights at every level.</p><img src="{name}" alt="Seven-row recovery metric grid with robust residual RMS in the bottom-left panel"><p>The bottom-left panel shows robust residual RMS: scaled median absolute deviation of residual pixels in the fixed annulus, in Jy/beam. Flux errors use raw CLEAN components. Position is measured in beam FWHM from the true source; {unreliable_positions} position values flagged as unreliable for empty raw models are omitted from the plot and its summary, but retained in the CSV.</p>')
    (root/'progress.json').write_text(json.dumps(status,indent=2))
    (root/'report_status.json').write_text(json.dumps(dict(status='complete' if status['completed_images']==status['expected_images'] and not failures else 'partial',metric_rows=len(rows),failures=failures,**status),indent=2))
    report_html='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="300"><title>CLEAN recovery metric comparison</title><style>body{font:16px sans-serif;margin:24px;max-width:1250px}img{display:block;width:100%}p{line-height:1.4}</style><h1>CLEAN recovery metric comparison</h1>'
    report_html+=f'<p>{status["completed_images"]}/{status["expected_images"]} images; {len(rows)} metric rows. <a href="metrics.csv">All metrics CSV</a> · <a href="recovery_plan.json">Plan</a> · <a href="progress.json">Progress</a>. Each thin curve is one thermal realization; dark line and band are mean ± 1 SD. Zero is on the same axis.</p>'
    report_html+=''.join(figures)+f'<p>Missing or invalid cases: {html.escape(str(failures))}</p>'
    (root/'report.html').write_text(report_html)
    return dict(rows=len(rows),failures=failures,progress=status)


def training_report(root: Path):
    return report(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    report(parser.parse_args().output)
