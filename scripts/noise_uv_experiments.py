"""Paired STD-multiplier/UV experiments using the existing CASA pipeline.

Run under CASA. Each source/cell checkpoints independently; source shards may
run concurrently. Original datasets are read only. Report with noise_uv_report.py.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from casatools import table

from scripts import create_dataset_v1 as g
from scripts.imaging import (Beam, BeamRegion, DefaultImagingConfig, beam_region_mask,
                             image_ms, summarize_residual_pixels)
from scripts.imaging.models import ImageGrid, ResolvedGrid
from scripts.noise_snr_audit import measure_pair
from scripts.preprocessing.schema import atomic_write_json
from scripts.simulation import add_thermal_noise_inplace, initialize_weights
from scripts.uv_angular_coverage_experiment import FixedGrid

LEVELS = (0., 1., 2.5, *g.NOISE_SNR_CORR_TARGETS)
CUTS = tuple(range(0, 100, 10))
CACHE = Path('ml/runs/20261028T065537442783Z_combined8_strategies/train_features.npz')


def training_sources():
    with np.load(CACHE, allow_pickle=False) as saved:
        return sorted(set(str(s) for s in saved['source_ids']))


def read_grid(path):
    d = json.loads(path.read_text())
    return ResolvedGrid(Beam(**d['beam']), ImageGrid(**d['image_grid']), Path(d['first_pass_imagename']))


def set_cut(ms, reference, pct):
    """Write exact original OR wedge flags, restoring them before each level."""
    a, b = table(), table()
    a.open(str(reference)); b.open(str(ms), nomodify=False)
    try:
        for start in range(0, a.nrows(), 4096):
            n = min(4096, a.nrows()-start)
            uv = a.getcol('UVW', startrow=start, nrow=n)
            selected = np.mod(np.arctan2(uv[1], uv[0]), math.pi) < math.pi*pct/100
            flag = a.getcol('FLAG', startrow=start, nrow=n) | selected[None, None, :]
            flagrow = a.getcol('FLAG_ROW', startrow=start, nrow=n) | selected
            b.putcol('FLAG', flag, startrow=start, nrow=n)
            b.putcol('FLAG_ROW', flagrow, startrow=start, nrow=n)
    finally:
        b.close(); a.close()


def source_run(output, source_id, levels, cuts):
    folder = output / 'sources' / source_id
    folder.mkdir(parents=True, exist_ok=True)
    expected = [folder / f'std_{k:g}' / f'cut_{p:02d}' / 'record.json' for k in levels for p in cuts]
    if all(p.is_file() for p in expected):
        print('Already complete', source_id, flush=True)
        return
    manifest = json.loads((g.THERMAL_DIR / 'report.json').read_text())
    source = g._thermal_source(g.THERMAL_DIR, manifest, source_id)
    g._check_disk_space([source], folder)
    model, params, sigma = g._noise_request(source)
    sky = g._simulation_payload(source)['components']
    seed = g._stable_seed(source_id, 'thermal_noise')
    v = g._prepare_V_ms(source, folder)
    regular = folder / 'regular.ms'
    if not regular.exists():
        shutil.copytree(v, regular)
        add_thermal_noise_inplace(regular, noise_model=model, noise_parameters=params, seed=seed)
    grid_path = folder / 'reference_grid.json'
    grid = read_grid(grid_path) if grid_path.exists() else None
    region = BeamRegion(source.metric_min_radius_beams, source.metric_max_radius_beams)
    order = [1., *[k for k in levels if k != 1.]]
    for multiplier in order:
        level = folder / f'std_{multiplier:g}'
        level.mkdir(exist_ok=True)
        ms = level / 'observed.ms'
        requested_cuts = cuts if multiplier in levels else (0,)
        pending = [p for p in requested_cuts if not (level/f'cut_{p:02d}'/'record.json').exists()]
        if not pending and grid is not None:
            continue
        if ms.exists():
            # This exact scratch MS is owned by this driver and regenerated.
            shutil.rmtree(ms)
        shutil.copytree(v if multiplier == 0 else regular, ms)
        if multiplier not in (0., 1.):
            g._rescale_visibility_delta(v, ms, multiplier)
        # Zero-noise data still require finite imaging weights: use nominal
        # sigma0 weights, preserving the relative baseline weighting.
        initialize_weights(ms, sigma * multiplier if multiplier else sigma)
        injection = measure_pair(v, ms, sigma)
        base_norm = measure_pair(v, regular, sigma)
        np.testing.assert_allclose(injection['SNR_corr_actual'],
            multiplier * base_norm['SNR_corr_actual'], rtol=1e-6, atol=1e-7)
        for pct in sorted(set(pending + ([0] if grid is None else []))):
            target = level / f'cut_{pct:02d}'
            target.mkdir(exist_ok=True)
            record_path = target / 'record.json'
            if record_path.exists():
                continue
            set_cut(ms, v, pct)
            # Fresh reread of stored DATA for each surviving UV selection.
            thermal = measure_pair(v, ms, sigma)
            difference = measure_pair(regular, ms, sigma)
            if multiplier:
                expected_difference = abs(multiplier-1)/multiplier * thermal['SNR_corr_actual']
                np.testing.assert_allclose(difference['SNR_corr_actual'], expected_difference,
                                           rtol=1e-6, atol=1e-7)
            else:
                assert thermal['SNR_corr_actual'] == 0., 'Noiseless DATA differ from sky'
            image_dir = target / 'imaging'
            if image_dir.exists():
                image_dir.rename(target / f'incomplete_imaging_{time.time_ns()}')
            config = DefaultImagingConfig if grid is None else replace(DefaultImagingConfig, grid=FixedGrid(grid))
            started = time.monotonic()
            image = image_ms(ms, config, image_dir, imsize=source.imsize,
                metric_region=BeamRegion(), keep_intermediate_products=False,
                fits_invalid_policy='fill', fits_fill_value=0., pblimit=g.DATASET_PBLIMIT)
            if grid is None:
                grid = image.resolved_config.grid
                atomic_write_json(grid_path, asdict(grid))
            with fits.open(image.residual_fits, memmap=False) as hdus:
                plane = np.squeeze(np.asarray(hdus[0].data, float))
                header = hdus[0].header
            mask = beam_region_mask(plane.shape, grid.image_grid.cell_arcsec, grid.beam.major_arcsec, region)
            values = plane[mask & np.isfinite(plane)]
            metrics = summarize_residual_pixels(values,
                pixel_area_arcsec2=math.prod(grid.image_grid.cell_arcsec),
                beam=Beam(header['BMAJ']*3600, header['BMIN']*3600, header['BPA']))
            n = thermal['valid_complex_samples']
            row = dict(source=source_id, noise_std_multiplier=multiplier,
                angular_removed_percent=pct, valid_complex_samples=n,
                actual_sample_removed_percent=100*(1-n/base_norm['valid_complex_samples']),
                sigma_simple_reference_jy=sigma, sigma_simple_jy=multiplier*sigma,
                sigma_NA_jy_per_beam=multiplier*sigma/math.sqrt(n),
                sigma_NA_reference_jy_per_beam=sigma/math.sqrt(n),
                SNR_corr_actual=thermal['SNR_corr_actual'],
                SNR_difference_from_regular_actual=difference['SNR_corr_actual'],
                SNR_corr_actual_precut=injection['SNR_corr_actual'],
                robust_rms_jy_per_beam=metrics.scaled_mad_jy_per_beam,
                annulus_rms_jy_per_beam=metrics.rms_jy_per_beam,
                annulus_abs_peak_jy_per_beam=metrics.residual_abs_peak_jy_per_beam,
                annulus_pixels=int(values.size), input_sky_components=sky,
                baseline_seed=seed, seconds=time.monotonic()-started,
                restored_peak_jy_per_beam=image.qa.metrics.clean_peak_jy_per_beam,
                warnings=list(image.qa.warnings),
                clean_summary=asdict(image.qa.tclean_summary) if image.qa.tclean_summary else None,
                nominal_weights_for_noiseless=multiplier == 0,
                **{key: str(getattr(image, key+'_fits').relative_to(output)) for key in ('dirty','clean','residual','psf')})
            atomic_write_json(record_path, row)
            print('COMPLETE', source_id, 'STD', multiplier, 'cut', pct,
                  'RMS', row['robust_rms_jy_per_beam'], 'seconds', row['seconds'], flush=True)
        shutil.rmtree(ms)
    shutil.rmtree(regular)
    g._cleanup_V_work(source, folder)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', action='append')
    p.add_argument('--training', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--levels', type=float, nargs='+', default=LEVELS)
    p.add_argument('--cuts', type=int, nargs='+', default=CUTS)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards: p.error('Invalid shard')
    if any(k < 0 or not math.isfinite(k) for k in a.levels): p.error('Invalid multiplier')
    if any(k not in CUTS for k in a.cuts): p.error('Invalid cut')
    sources = training_sources() if a.training else a.source or ['2241+098']
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(a.output / f'plan_{"training" if a.training else "example"}_{a.shard}.json',
        dict(sources=sources[a.shard::a.shards], levels=a.levels, cuts=a.cuts,
             control='total thermal STD multiplier; Y=V+a*n0; 0 is truly noiseless',
             reference='Original sigma0 fixed; SNR_corr_actual is total thermal norm vs V',
             library_configuration='DefaultImagingConfig with regular0% grid and annulus fixed'))
    failures = []
    for source_id in sources[a.shard::a.shards]:
        try:
            source_run(a.output, source_id, a.levels, a.cuts)
        except Exception:
            detail = traceback.format_exc()
            atomic_write_json(a.output / f'failure_{source_id}.json', dict(source=source_id, traceback=detail))
            print(detail, flush=True); failures.append(source_id)
        else:
            (a.output / f'failure_{source_id}.json').unlink(missing_ok=True)
    if failures:
        raise RuntimeError(f'Sources requiring recovery: {failures}')
