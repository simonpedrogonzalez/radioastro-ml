"""Training-only R/E/O examples, using the canonical imaging/reporting helpers.

Run from the repository root:
    uv run --project ml python -m ml.parity_report --dataset PATH/dataset.json \
        --export PATH/examples.html
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from scipy.ndimage import binary_propagation

from ml.nn_common import parity_channels
from scripts.imaging.plot_utils import casa_image_to_png, shared_fits_display_limits
from scripts.preprocessing import FitsSimulationDataset, source_dataset_id
from scripts.preprocessing.schema import load_sample_manifest
from scripts.reporting import QuartoReporter, export_detached_report


INTRO = r"""---
title: "Residual symmetry: R, E and O"
format:
  html:
    page-layout: full
    embed-resources: true
---

<style>
.parity-row {display:grid;grid-template-columns:repeat(3,minmax(0,1fr)) minmax(180px,.8fr);align-items:center;gap:8px;margin-bottom:24px}
.parity-row img {width:100%;height:auto}
.parity-values {font-size:.9em;overflow-wrap:anywhere}
@media(max-width:850px) {.parity-row {grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
"""


def support_mask(image):
    if image.shape != (4, 256, 256) or not torch.isfinite(image).all():
        raise ValueError("Expected finite (4,256,256) FITS planes")
    filled = (image[:3] == 0).all(dim=0).numpy()
    if filled.any():
        boundary = filled.copy(); boundary[1:-1, 1:-1] = False
        if not np.array_equal(binary_propagation(boundary, mask=filled), filled):
            raise ValueError("Unexpected interior shared-zero holes")
        if filled[128, 128]:
            raise ValueError("Invalid support: phase centre is zero-filled")
    return filled


def residual_scale(residual):
    y, x = np.indices((256, 256)); radius = np.hypot(x - 128, y - 128)
    values = residual.numpy()[(radius >= 32) & (radius < 72)]
    scale = float(1.4826 * np.median(np.abs(values - np.median(values))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Residual annulus has nonpositive/nonfinite MAD scale")
    return scale


def select_examples(dataset):
    groups = {}
    for index, manifest in enumerate(dataset.samples):
        groups.setdefault(source_dataset_id(manifest.sample_id), []).append(index)
    expected = {(0, 0)} | {(kind, level) for kind in (1, 2) for level in (10, 30, 50, 100)}
    selected = {}
    for source, indices in sorted(groups.items()):
        if len(indices) != 9:
            continue
        samples = {}
        for index in indices:
            manifest = dataset.samples[index]
            load_sample_manifest(manifest.path, require_files=True, verify_integrity=True)
            sample = dataset[index]
            if support_mask(sample['image']).any():
                break
            header = fits.getheader(sample['paths']['products']['residual'])
            if (header.get('CRPIX1'), header.get('CRPIX2')) != (129, 129):
                raise ValueError(f'Unexpected phase centre: {sample["sample_id"]}')
            samples[(sample['label'], sample['label_metadata']['corruption_snr_target'])] = sample
        if set(samples) == expected:
            selected[source] = samples
        if len(selected) == 3:
            return selected
    raise ValueError('Fewer than three complete square-support training sources')


def build_report(dataset_path, output):
    dataset = FitsSimulationDataset(dataset_path.parent, index=dataset_path.name,
                                    partition='train', validate=False,
                                    label_criterion='constant_antenna_type')
    selected = select_examples(dataset)
    output.mkdir(parents=True, exist_ok=False)
    records, sections = [], [INTRO]
    for source, samples in selected.items():
        print(f'Plotting {source}: nine variants × R/E/O', flush=True)
        limits = shared_fits_display_limits(
            [sample['paths']['products']['residual'] for sample in samples.values()])
        for (kind, level), sample in sorted(samples.items()):
            residual_path = sample['paths']['products']['residual']
            residual = sample['image'][2]
            channels = torch.cat((residual[None], parity_channels(residual)))
            scale = residual_scale(residual)
            energies = (channels[1:, 1:, 1:].double() / scale).square().mean((-2, -1)).numpy()
            np.testing.assert_allclose(channels[1:, 1:, 1:].sum(0), residual[1:, 1:],
                                       rtol=1e-6, atol=float(residual.abs().max()) * 1e-7)
            plots, recipes = [], []
            # Temporary derived FITS retain the original WCS/beam and axis layout.
            with TemporaryDirectory(prefix='parity-', dir=output) as temporary:
                with fits.open(residual_path) as hdul:
                    shape, header = hdul[0].data.shape, hdul[0].header.copy()
                for channel, name in zip(channels, ('R', 'E', 'O')):
                    path = residual_path
                    if name != 'R':
                        path = Path(temporary) / f'{name}.fits'
                        fits.writeto(path, channel.numpy().reshape(shape), header)
                    png = Path('images') / f'{sample["sample_id"]}_{name}.png'
                    recipes.append(casa_image_to_png(path, output / png, title=name,
                                    draw_beam=True, display_limits_mjy_per_beam=limits))
                    plots.append(f'<img src="{png.as_posix()}" alt="{escape(sample["sample_id"])} {name}">')
            values = (f'<div class="parity-values"><b>{escape(sample["sample_id"])}</b>'
                      f'<p>s = {scale * 1000:.6g} mJy/beam</p>'
                      f'<p>q<sub>E</sub> = {energies[0]:.6g}<br>q<sub>O</sub> = {energies[1]:.6g}</p></div>')
            sections.append('<div class="parity-row">' + ''.join(plots) + values + '</div>')
            records.append({'source': source, 'sample_id': sample['sample_id'], 'partition': 'train',
                            'label': kind, 'target_level': level, 'scale_jy_per_beam': scale,
                            'energies': energies.tolist(),
                            'residual_fits': str(residual_path), 'plots': recipes})
    qmd = output / 'report.qmd'
    qmd.write_text('\n\n'.join(sections), encoding='utf-8')
    (output / 'examples.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    QuartoReporter(qmd).finish()
    if not qmd.with_suffix('.html').is_file():
        raise RuntimeError('Report rendering failed')
    return qmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('ml/runs'))
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_parity_examples')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        qmd = build_report(args.dataset.resolve(), output)
    exported = export_detached_report(qmd, args.export, overwrite=args.overwrite)
    print(f'Report: {qmd.with_suffix(".html")}\nExport: {exported.html_path}', flush=True)


if __name__ == '__main__':
    main()
