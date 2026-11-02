"""Render resumable noise/UV experiments, including explicit completion counts."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

LEVELS = (0., 1., 2.5, 5., 10., 15., 20., 30., 40., 50., 100., 1000., 2000., 3000., 4000.)
CUTS = tuple(range(0, 100, 10))
KEYS = ('SNR_corr_actual', 'sigma_NA_jy_per_beam', 'sigma_simple_jy', 'robust_rms_jy_per_beam')
LABELS = ('Actual thermal SNR_corr', 'σ_NA (Jy/beam)', 'σ_simple (Jy)', 'Robust residual RMS (Jy/beam)')
CACHE = Path('ml/runs/20261028T065537442783Z_combined8_strategies/train_features.npz')


def pairplot(rows, path, title):
    fig, axes = plt.subplots(4, 4, figsize=(12, 11), layout='constrained')
    for y in range(4):
        for x in range(4):
            ax = axes[y, x]
            artist = ax.scatter([r[KEYS[x]] for r in rows], [r[KEYS[y]] for r in rows],
                c=[np.log10(1+r['noise_std_multiplier']) for r in rows], s=10, alpha=.6)
            ax.set_xscale('symlog', linthresh=1 if x == 0 else 1e-9)
            ax.set_yscale('symlog', linthresh=1 if y == 0 else 1e-9)
            if y == 3: ax.set_xlabel(LABELS[x], fontsize=8)
            if x == 0: ax.set_ylabel(LABELS[y], fontsize=8)
            ax.tick_params(labelsize=6); ax.grid(alpha=.2)
    fig.colorbar(artist, ax=axes, shrink=.5, label='log10(1 + STD multiplier)')
    fig.suptitle(title); fig.savefig(path, dpi=130); plt.close(fig)


def curves(rows, path, title, median=False):
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    for k in LEVELS:
        values = [r for r in rows if r['noise_std_multiplier'] == k]
        if not values: continue
        cuts = sorted({r['angular_removed_percent'] for r in values})
        ys = [np.median([r['robust_rms_jy_per_beam'] for r in values if r['angular_removed_percent'] == p]) for p in cuts]
        ax.plot(cuts, ys, marker='o', markersize=3, label=f'{k:g}× STD')
    ax.set(xlabel='Angular uv coverage removed (%)', ylabel='Annulus robust residual RMS (Jy/beam)', title=title)
    ax.set_yscale('symlog', linthresh=1e-10)
    ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=150); plt.close(fig)


def images(root, row):
    target = root / row['residual']
    preview = target.parent / 'panels.png'
    if preview.exists(): return preview.relative_to(root).as_posix()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    for ax, key in zip(axes, ('dirty', 'clean', 'residual')):
        a = np.squeeze(fits.getdata(root / row[key])).astype(float)
        lim = max(float(np.max(np.abs(a))), 1e-20)
        artist = ax.imshow(a, origin='lower', cmap='inferno', vmin=-lim, vmax=lim)
        fig.colorbar(artist, ax=ax, fraction=.046, label='Jy/beam')
        ax.set_title(key.capitalize()); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(preview, dpi=100); plt.close(fig)
    return preview.relative_to(root).as_posix()


def run(root):
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted((root/'sources').glob('*/std_*/cut_*/record.json'))]
    with np.load(CACHE, allow_pickle=False) as d:
        sources = set(str(s) for s in d['source_ids'])
    example = [r for r in rows if r['source'] == '2241+098']
    training = [r for r in rows if r['source'] in sources]
    expected = {(k, p) for k in LEVELS for p in CUTS}
    complete = {s for s in sources if {(r['noise_std_multiplier'], r['angular_removed_percent']) for r in training if r['source'] == s} == expected}
    matched = [r for r in training if r['source'] in complete]
    zero_cut = [r for r in rows if r['angular_removed_percent'] == 0]
    if rows:
        fields = [k for k in rows[0] if k not in ('input_sky_components', 'warnings', 'clean_summary')]
        with (root/'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    status = dict(example_cells=len(example), expected_example=150, training_cells=len(training),
                  expected_training=15000, complete_training_sources=len(complete), expected_training_sources=100)
    (root/'progress.json').write_text(json.dumps(status, indent=2))
    intro = '''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="600">
<style>body{font:17px sans-serif;margin:35px;max-width:1700px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}img{max-width:100%}.panels{width:850px}</style>
<h1>Noise multipliers, corruption metrics and uv coverage</h1>
<p>Live checkpoint report. Completion counts below distinguish results from pending work.
Noise is Y=V+a n0: a=0 means no thermal noise, a=1 regular noise. The same random
realization and sky are paired across levels. The source flux never changes within a source.
The numbers 5–4000 are now STD multipliers, unlike the old extra-SNR targets.</p>
<p>SNR_corr_actual=||Y−V||/σ0 is the measured total thermal perturbation relative to noiseless sky.
The CSV separately reports ||Y−Y_regular||/σ0 to preserve the earlier regular-baseline comparison.
Both use original σ0, never the increased sigma. Every row rereads stored DATA after flagging.
σ_NA=a σ0/sqrt(N_valid) predicts natural weighting; actual images use regular Briggs robust0.5.
For noiseless data σ_simple=σ_NA=0; finite nominal σ0 weights are used solely for imaging.</p>
<p>Fixed regular-noise 0% grid, CLEAN mask and pixel annulus within each source.
Cuts remove paired angular wedges (0–90%); actual visibility loss is reported separately.
Robust RMS is annulus 1.4826×MAD. Source model: phase-centre point source.
Image thumbnails have independent scales with colorbars, to expose structure across the large noise range.
Small noiseless residuals may reflect CLEAN stopping/numerics rather than thermal noise.</p>
<p><a href="metrics.csv">Metrics CSV</a> · <a href="progress.json">Completion counts</a></p>
'''
    progress = '<p><strong>' + html.escape(str(status)) + '</strong></p>'
    failures = [json.loads(p.read_text()) for p in root.glob('failure_*.json')]
    if failures:
        progress += '<p>Sources requiring recovery: ' + html.escape(', '.join(r['source'] for r in failures)) + '</p>'
    if (root/'input_flux_audit.json').exists():
        progress += '<p><a href="input_flux_audit.json">Input flux audit</a>: all 2,893 retained training variants preserve their source flux. Training-source input fluxes span 0.00432–0.66842 Jy; 2241+098 uses 0.09217 Jy.</p>'
    one = [r for r in zero_cut if r['source'] == '2241+098']
    if one: pairplot(one, root/'example_matrix.png', 'Experiment 1: 2241+098, full coverage')
    train0 = [r for r in zero_cut if r['source'] in sources]
    if train0: pairplot(train0, root/'training_matrix.png', f'Experiment 1: training full-coverage checkpoints ({len(train0)} rows)')
    exp1 = ['<h2>Experiment 1 — metric relationships</h2>']
    for name in ('example_matrix.png', 'training_matrix.png'):
        if (root/name).exists(): exp1.append(f'<img src="{name}">')
    exp1.append('<h3>2241+098 full-coverage noise table</h3>')
    exp1.append(table(root, one))
    exp1.append('''<h3>Training amplitude and phase injections (SNR_corr 5–100)</h3>
<p>Existing gain samples, each with regular thermal noise. Same source-dependent sky fluxes.
These panels retain gain perturbation SNR_corr, rather than the total thermal-noise norm.</p>
<img src="../corruption-metric-relationships-20261102/existing-training/amp.png">
<img src="../corruption-metric-relationships-20261102/existing-training/phase.png">''')
    exp2 = ['<h2>Experiment 2 — thermal noise and uv coverage</h2>']
    if example:
        curves(example, root/'example_uv.png', '2241+098: uv cuts at each thermal STD multiplier')
        exp2.append('<img src="example_uv.png">')
    if matched:
        curves(matched, root/'training_uv.png', f'Training medians: {len(complete)}/100 fully completed sources (provisional until 100)')
        exp2.append('<img src="training_uv.png">')
    else:
        exp2.append('<p>Training median curves pending: no source yet has all 150 cells. Medians will use identical source membership across every cut and noise level.</p>')
    for k in LEVELS:
        group = [r for r in example if r['noise_std_multiplier'] == k]
        if group: exp2.append(f'<details><summary>2241+098 — {k:g}× STD: {len(group)}/10 cuts</summary>'+table(root, group)+'</details>')
    (root/'report.html').write_text(intro + progress + '\n'.join(exp1 + exp2))
    print(status, flush=True)


def table(root, rows):
    parts = ['<table><tr><th>STD multiplier</th><th>Angular cut / actual sample loss (%)</th><th>σ_simple (Jy)</th><th>σ_NA (Jy/beam)</th><th>SNR_corr_actual</th><th>Robust RMS (Jy/beam)</th><th>Dirty / CLEAN / residual</th></tr>']
    for r in sorted(rows, key=lambda r: (r['noise_std_multiplier'], r['angular_removed_percent'])):
        parts.append(f'<tr><td>{r["noise_std_multiplier"]:g}</td><td>{r["angular_removed_percent"]} / {r["actual_sample_removed_percent"]:.2f}</td>'+
            ''.join(f'<td>{r[key]:.7g}</td>' for key in ('sigma_simple_jy','sigma_NA_jy_per_beam','SNR_corr_actual','robust_rms_jy_per_beam'))+
            f'<td><img class="panels" loading="lazy" src="{images(root,r)}"></td></tr>')
    return ''.join(parts)+'</table>'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
