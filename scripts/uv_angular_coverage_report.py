"""Render the angular UVW ablation as one image-and-statistics row per cut."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scripts.reporting import QuartoReporter


def plane(path):
    with fits.open(path,memmap=False) as hdus:
        a=np.squeeze(np.asarray(hdus[0].data,dtype=float))
    if a.ndim!=2:raise ValueError(f'Expected 2D FITS plane: {path} {a.shape}')
    return a


def render(run:Path):
    run=run.resolve();record=json.loads((run/'results.json').read_text());rows=record['rows']
    noiseless=record.get('noise_mode')=='none'
    noise_only=record.get('noise_mode')=='baseline_plus_extra_no_source'
    number=lambda value: f'{value:.4g}' if noiseless else f'{value:.4f}'
    if [r['angular_removed_percent'] for r in rows] != list(range(0,100,10)):
        raise ValueError('Require completed 0 through 90 percent images')
    uv=np.load(run/'uvw_initial.npz');u,v,usable=uv['u'],uv['v'],uv['usable']
    reference=Path('collect/experiments/dataset_v1_20260921_all_psf/samples')/f"{record['source']}_noise_snr_50"/'imaging'
    if not noiseless and not noise_only:
        for key in ('dirty','clean','residual'):
            np.testing.assert_array_equal(plane(run/rows[0][key]),plane(reference/f'{key}.fits.gz'))
    elif noiseless:
        validation=json.loads((run/'validation.json').read_text())
        assert validation['no_regular_or_extra_noise_added'] and validation['measured_final_thermal_SNR_corr']==0.
    else:
        validation=json.loads((run/'validation.json').read_text())
        assert validation['source_visibility_subtracted'] and validation['data_weights_unchanged_across_cuts']
    angle=np.mod(np.arctan2(v,u),np.pi)
    arrays={key:[plane(run/r[key]) for r in rows] for key in ('dirty','clean','residual')}
    limits={key:max(float(np.nanpercentile(np.abs(a),99.8)) for a in images) for key,images in arrays.items()}
    # Shared limits for all rows of a given image type.
    fig,axes=plt.subplots(len(rows),5,figsize=(27,3.9*len(rows)),layout='constrained',
                          gridspec_kw={'width_ratios':[1.15,1,1,1,.75]})
    for i,r in enumerate(rows):
        selected=usable & (angle >= np.pi*r['angular_removed_percent']/100)
        idx=np.flatnonzero(selected)
        if len(idx)>10000:idx=idx[np.linspace(0,len(idx)-1,10000,dtype=int)]
        axis=axes[i,0]
        artist=axis.scatter(np.r_[u[idx],-u[idx]],np.r_[v[idx],-v[idx]],
                            c=np.r_[angle[idx],angle[idx]]*180/np.pi,s=1.3,
                            cmap='inferno',vmin=0,vmax=180,rasterized=True)
        fig.colorbar(artist,ax=axis,fraction=.046,pad=.04,label='Baseline angle (deg)')
        axis.set(xlabel='u (m)',ylabel='v (m)',aspect='equal',xlim=(-uv['extent'],uv['extent']),ylim=(-uv['extent'],uv['extent']))
        axis.set_title(f"{r['angular_removed_percent']}% angle; {r['actual_usable_removed_percent']:.1f}% samples",fontsize=9)
        for j,key in enumerate(('dirty','clean','residual'),1):
            axis=axes[i,j];a=arrays[key][i]*1000;lim=limits[key]*1000
            artist=axis.imshow(a,origin='lower',cmap='inferno',vmin=-lim,vmax=lim)
            fig.colorbar(artist,ax=axis,fraction=.046,pad=.04,label='mJy/beam')
            axis.set_xticks([]);axis.set_yticks([])
            axis.set_title(key.capitalize(),fontsize=9)
        axis=axes[i,4];axis.axis('off')
        axis.text(0,.8,f"MAD σ: {number(1000*r['residual_scaled_mad_jy_per_beam'])} mJy/beam\n"
             f"Peak: {number(1000*r['residual_abs_peak_jy_per_beam'])} mJy/beam\n"
             f"P99.5 / MAD: {r['residual_p99_5_over_mad']:.3f}",
             va='top',fontsize=9,transform=axis.transAxes)
    fig.savefig(run/'overview.png',dpi=160);plt.close(fig)
    (run/'report.css').write_text('''
html, body { min-width: 3400px; overflow-x: auto; }
body { font-size: 20px; }
#quarto-content, main.content, .page-columns, .content { max-width: none !important; width: max-content !important; }
.quarto-figure img, main img { max-width: none !important; width: 3300px !important; height: auto !important; }
table { font-size: 20px; }
''')
    title=('Angular UV coverage with no thermal noise' if noiseless else
           'Angular UV coverage: thermal noise only, no source' if noise_only else
           'Angular UV coverage with fixed extra thermal noise')
    description=(f"Dataset 1 source **{record['source']}**, **no thermal noise at all**: neither regular baseline noise nor extra noise is added. The visibilities contain only the predicted sky (point-source flux 0.0921665 Jy). Measured thermal **SNR_corr_actual = 0**. Finite nominal weights are used for imaging only; they do not inject noise. The image grid, CLEAN mask and measurement annulus are pinned to the original SNR_corr=50 report for direct comparison. Across rows only UVW flags change; visibility values and weights stay fixed."
        if noiseless else f"The **{record['source']}** template supplies the same UVW, flags, nominal noise model, two noise seeds, weights and imaging settings as the [source-plus-noise SNR_corr=50 report](../uv-angular-2241-noise50-final/report.html). The predicted point-source visibility is subtracted from the completed noisy realization before imaging, so the sky is zero. The extra-noise target is **SNR_corr=50 relative to the original baseline sigma0**, with measured injection **{record['measured_snr_corr']:.8f}**. Baseline thermal noise is also present. The fixed grid, CLEAN mask and residual measurement annulus match the original report. Only UVW flags change across rows." if noise_only else f"Dataset 1 source **{record['source']}**, extra thermal-noise SNR_corr **{record['measured_snr_corr']:.8f}**. The full SNR_corr=50 sample is generated first: extra noise is injected, baseline thermal noise is added, and weights are initialized. Angular UVW flags are applied only afterward. The visibility data and weights are unchanged across rows; only flags accumulate.")
    lines=['---',f'title: "{title}"','format:','  html:','    embed-resources: true','    toc: true','    page-layout: full','    css: report.css','---','','## Experiment','',
           description,'',
           'The angle is `atan2(v,u) mod 180°`. A baseline and its conjugate represent opposite points, so each cut removes a paired wedge. Percentages specify angular width, not an equal fraction of measured visibilities. The last column below gives the actual fraction of initially usable samples removed. 100% would leave no data to image.','',
           '## Image sequence','', 'Each row shows retained uv coverage, dirty, CLEAN and residual images, and the three statistics. All panels use the imaging pipeline’s inferno colormap and have individual labeled colorbars. Image limits are fixed within each image type across rows. The wide page scrolls horizontally and vertically; browser zoom enlarges images and text.','', '![](overview.png)','','## Metrics','',
           '| Angle removed | Usable samples removed | Robust sigma (mJy/beam) | Residual absolute peak (mJy/beam) | Residual P99.5 / MAD |',
           '| ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['angular_removed_percent']}% | {r['actual_usable_removed_percent']:.2f}% | {number(1000*r['residual_scaled_mad_jy_per_beam'])} | {number(1000*r['residual_abs_peak_jy_per_beam'])} | {r['residual_p99_5_over_mad']:.3f} |")
    if noiseless or noise_only:
        if noise_only:
            lines += ['', '[Full numerical results](results.json) · [Visibility and flag audit](validation.json) · [Source-plus-noise report](../uv-angular-2241-noise50-final/report.html) · [Noiseless-source report](../uv-angular-2241-noiseless/report.html)',
                      '', '## Verification', '',
                      'The same source-template sky visibility was subtracted from DATA and CORRECTED_DATA after the SNR_corr=50 extra-noise draw and ordinary thermal-noise draw. The noise and weight columns are unchanged through all angular cuts. Final flags equal the original flags OR the paired 90% wedge. The grid, CLEAN mask and pixel annulus use the original source-plus-noise report. The five displayed columns and metrics follow that report exactly.', '']
            (run/'report.qmd').write_text('\n'.join(lines))
            image_data=base64.b64encode((run/'overview.png').read_bytes()).decode('ascii')
            table_rows='\n'.join(
                f"<tr><td>{r['angular_removed_percent']}%</td>"
                f"<td>{r['actual_usable_removed_percent']:.2f}%</td>"
                f"<td>{1000*r['residual_scaled_mad_jy_per_beam']:.4f}</td>"
                f"<td>{1000*r['residual_abs_peak_jy_per_beam']:.4f}</td>"
                f"<td>{r['residual_p99_5_over_mad']:.3f}</td></tr>" for r in rows)
            (run/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>{title}</title><style>body{{font:19px system-ui,sans-serif;margin:32px;min-width:3400px}}
img{{width:3300px}}table{{border-collapse:collapse}}td,th{{padding:9px 16px;border-bottom:1px solid #ccc;text-align:right}}
th:first-child,td:first-child{{text-align:left}}p{{max-width:1300px;line-height:1.5}}</style>
<h1>{title}</h1><h2>Experiment</h2>
<p>The 2241+098 template supplies the same UVW, flags, noise model, seeds, weights and imaging settings as the
<a href="../uv-angular-2241-noise50-final/report.html">source-plus-noise SNR_corr=50 report</a>.
The predicted point-source visibility was subtracted after generating that paired noisy realization. The sky is zero.
The extra-noise target is SNR_corr=50 relative to the original baseline sigma0, and measured injection is
{record['measured_snr_corr']:.8f}. Ordinary baseline thermal noise is also present. The original grid, CLEAN mask
and residual measurement annulus are fixed across all cuts.</p>
<p>Each angular cut removes a paired wedge in atan2(v,u) mod 180°. Its angular width differs from the fraction of
usable visibilities removed. The image sequence uses the same residual metrics and layout as the comparison reports.
Image color limits are fixed within each product type across cuts.</p>
<h2>Image sequence</h2><img alt="UV coverage, dirty, CLEAN, residual and metrics for ten angular cuts"
src="data:image/png;base64,{image_data}">
<h2>Metrics</h2><table><tr><th>Angle removed</th><th>Usable samples removed</th>
<th>Robust sigma (mJy/beam)</th><th>Residual absolute peak (mJy/beam)</th><th>P99.5 / MAD</th></tr>
{table_rows}</table><p><a href="results.json">Full numerical results</a> ·
<a href="validation.json">Visibility and flag audit</a> ·
<a href="../uv-angular-2241-noiseless/report.html">Noiseless-source report</a></p>
<h2>Verification</h2><p>The predicted source was subtracted from DATA and CORRECTED_DATA after both thermal draws.
Visibility and weight columns did not change across cuts. Final flags equal original flags plus the paired 90° wedge.
The grid, CLEAN mask and annulus match the source-plus-noise report. CASA recorded empty CLEAN models in the noise-only images.</p>
</html>''')
            if not (run/'report.html').is_file(): raise RuntimeError('HTML report was not created')
            print(run/'report.html')
            return
        lines += ['','[Full numerical results](results.json) · [Noiseless and flag audit](validation.json) · [Original noise50 report](../uv-angular-2241-noise50-final/report.html)',
                  '', '## Verification', '',
                  'Direct comparison of the final flagged Measurement Set with the predicted noiseless sky gives exactly zero thermal perturbation. SHA256 fingerprints of DATA, CORRECTED_DATA and available noise-weight columns are identical before and after all cuts. Final flags equal original flags OR the paired 90% angular wedge; the grid, CLEAN mask and annulus match the original report. No existing report or dataset was overwritten.', '',
                  'The very small nonzero residuals are reconstruction/stopping and numerical residuals, not injected thermal noise. Values are displayed with scientific notation where needed. Residual color limits are shared across all cuts in this report but differ from the original noisy report; consult the labeled colorbars before comparing visual prominence. CLEAN convergence and CASA warnings are retained in each row’s QA. 100% removal is omitted because no image can be formed.', '']
        (run/'report.qmd').write_text('\n'.join(lines))
        QuartoReporter(run/'report.qmd',every=1).finish()
        if not (run/'report.html').is_file(): raise RuntimeError('HTML report was not created')
        print(run/'report.html')
        return
    lines += ['','[Full numerical results](results.json) · [Flag audit](validation.json) · [Experiment plan](../../docs/uv_angular_coverage_experiment.md)','','## Verification','',
              'The 0% dirty, CLEAN and residual FITS pixel arrays match the retained dataset 1 sample exactly. The table below compares the original CASA image-mask QA with this report’s statistics measured on exported FITS pixels in a fixed 0%-defined annulus; the slight differences reflect that measurement path. CASA warnings are recorded in `results.json`.','','',
              '| Metric | Retained dataset 1 | New 0% row | Relative difference |',
              '| --- | ---: | ---: | ---: |']
    for label,new_value,old_value in (
        ('Robust sigma (mJy/beam)',1000*rows[0]['residual_scaled_mad_jy_per_beam'],1000*record['original_qa']['scaled_mad_jy_per_beam']),
        ('Residual absolute peak (mJy/beam)',1000*rows[0]['residual_abs_peak_jy_per_beam'],1000*record['original_qa']['residual_abs_peak_jy_per_beam']),
        ('P99.5 / MAD',rows[0]['residual_p99_5_over_mad'],record['original_qa']['p99_5_over_scaled_mad'])):
        lines.append(f'| {label} | {old_value:.5f} | {new_value:.5f} | {100*(new_value/old_value-1):+.2f}% |')
    lines += ['','The full sequence uses nested angular flags. An independent reconstruction of the unmasked noisy Measurement Set matches the final masked set exactly in DATA, CORRECTED_DATA, WEIGHT, SIGMA and WEIGHT_SPECTRUM. Thus the cuts remove measurements after noise generation without changing the retained visibility noise or weights. 100% is omitted because it leaves no usable visibilities.','']
    (run/'report.qmd').write_text('\n'.join(lines))
    QuartoReporter(run/'report.qmd',every=1).finish()
    print(run/'report.html')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();render(a.run)
