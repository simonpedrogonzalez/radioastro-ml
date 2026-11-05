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


def source_run(output, source_id, levels, cuts, *, recovery=False, seeds=None):
    if recovery:
        if tuple(cuts) != (0,): raise ValueError('Recovery sweeps require fixed original UV coverage')
        return recovery_source_run(output, source_id, levels, tuple(range(10)) if seeds is None else seeds)
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


def recovery_source_run(output, source_id, levels, seeds):
    """Opt-in seed-paired recovery sweep with durable raw models and support."""
    from scripts.imaging.qa import json_safe
    import hashlib

    output=Path(output).resolve()
    folder=output/'sources'/source_id/'recovery';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((g.THERMAL_DIR/'report.json').read_text())
    source=g._thermal_source(g.THERMAL_DIR,manifest,source_id)
    model,params,sigma=g._noise_request(source)
    payload=g._simulation_payload(source)
    sky=payload['components']
    mode_path=output/'recovery_flux_mode.json'
    if not mode_path.exists() or json.loads(mode_path.read_text())!={'mode':'original_imaging_model_flux'}:
        raise ValueError('Recovery requires original-imaging-flux mode in a fresh run directory')
    qa_path=g.THERMAL_DIR/source_id/'original'/'default_imaging'/'qa.json'
    qa=json.loads(qa_path.read_text())
    flux=float(qa['tclean_summary']['final_model_flux_jy'])
    peak=float(qa['metrics']['clean_peak_jy_per_beam'])
    if not math.isfinite(flux) or flux<=0 or not math.isfinite(peak) or peak<=0:
        raise ValueError('Original imaging has no positive finite source flux')
    sky=[dict(component,flux=[flux]+list(component['flux'][1:])) for component in sky]
    flux_provenance=dict(qa_path=str(qa_path),final_model_flux_jy=flux,
        clean_peak_jy_per_beam=peak,method='original CLEAN model sum as point-source flux')
    if len(sky)!=1 or sky[0]['shape']!='point': raise ValueError('Recovery requires one point source')
    flux=float(sky[0]['flux'][0]); direction=sky[0]['dir'].split()
    if direction[0]!='J2000' or not all(s.endswith('rad') for s in direction[1:]):
        raise ValueError('Expected stored J2000 radian coordinates')
    ra,dec=[float(v[:-3]) for v in direction[1:]]
    configuration=json_safe(asdict(DefaultImagingConfig))
    specification=dict(schema=4,source=source_id,levels=list(levels),seeds=list(seeds),components=sky,
        sigma0_jy=sigma,nominal_weights_sigma_jy=sigma,configuration=configuration,pblimit=g.DATASET_PBLIMIT,
        seed_map={str(s):g._stable_seed(source_id,f'clean_recovery:noise_seed:{s}') for s in seeds},
        protocol='noiseless-first; fixed grid/beam/search mask/valid support; same noise realization per seed')
    specification.update(flux_provenance=flux_provenance,
        predicted_nominal_image_rms_jy_per_beam=source.predicted_image_rms_jy_per_beam,
        nominal_source_snr=flux/source.predicted_image_rms_jy_per_beam)
    if len(set(specification['seed_map'].values()))!=len(seeds): raise ValueError('CASA seed collision')
    settings_hash=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    settings_path=folder/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=specification:
        raise ValueError('Recovery settings changed: use a fresh output directory')
    if not settings_path.exists(): atomic_write_json(settings_path,specification)
    cells=[folder/'noiseless',*[folder/f'seed_{s:02d}'/f'std_{a:g}' for s in seeds for a in levels if a>0]]
    from scripts.recovery_products import recovery_products
    complete,_=recovery_products(output,folder,source,sky,sigma,settings_hash)
    if all(complete(cell) for cell in cells):
        for name in ('reference.json','reference_grid.json','reference_masks.npz'):
            if not (folder/name).is_file(): raise ValueError(f'Missing recovery reference: {name}')
        print('Recovery already complete',source_id,flush=True);return
    g._check_disk_space([source],folder)
    from scripts.simulation import simulate_ms
    work=folder/'work'/source_id
    v=work/f'{source_id}_V.ms'
    metadata=work/f'{source_id}_V.simulation.json'
    metadata_text=work/f'{source_id}_V.simulation.txt'
    if not (v.is_dir() and metadata.is_file() and metadata_text.is_file()):
        if work.exists():
            if work.is_symlink() or work.resolve().parent!=(folder/'work').resolve():
                raise RuntimeError('Unsafe recovery scratch directory')
            shutil.rmtree(work)
        work.mkdir(parents=True,exist_ok=True)
        v=simulate_ms(payload['input_ms'],sky,v,noise_model=None).ms_path
    # A resumed source may reconstruct V with unit weights. Its reference
    # weights must match every newly generated nominal-sigma0 realization.
    initialize_weights(v,sigma)
    complete,image_cell=recovery_products(output,folder,source,sky,sigma,settings_hash,v)

    zero=folder/'noiseless'
    if not complete(zero):
        initialize_weights(v,sigma)
        image_cell(zero,v,0.,None,None)
    for seed_label in seeds:
        remaining=[a for a in levels if a>0 and not complete(folder/f'seed_{seed_label:02d}/std_{a:g}')]
        if not remaining: continue
        regular=folder/f'regular_seed_{seed_label:02d}.ms'
        if regular.exists(): shutil.rmtree(regular)
        shutil.copytree(v,regular)
        casa_seed=specification['seed_map'][str(seed_label)]
        add_thermal_noise_inplace(regular,noise_model=model,noise_parameters=params,seed=casa_seed)
        base=measure_pair(v,regular,sigma)['SNR_corr_actual']
        for multiplier in remaining:
            cell=folder/f'seed_{seed_label:02d}/std_{multiplier:g}';cell.mkdir(parents=True,exist_ok=True)
            ms=cell/'observed.ms'
            if ms.exists(): shutil.rmtree(ms)
            shutil.copytree(regular,ms)
            if multiplier!=1.: g._rescale_visibility_delta(v,ms,multiplier)
            initialize_weights(ms,sigma)
            actual=measure_pair(v,ms,sigma)['SNR_corr_actual']
            np.testing.assert_allclose(actual,base*multiplier,rtol=1e-6,atol=1e-7)
            try:
                image_cell(cell,ms,multiplier,seed_label,casa_seed)
            except Exception:
                atomic_write_json(cell/'failure.json',dict(traceback=traceback.format_exc()))
                raise
            shutil.rmtree(ms)
        shutil.rmtree(regular)
    g._cleanup_V_work(source,folder)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', action='append')
    p.add_argument('--training', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--levels', type=float, nargs='+')
    p.add_argument('--cuts', type=int, nargs='+', default=CUTS)
    p.add_argument('--recovery',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+')
    a = p.parse_args()
    if a.recovery:
        from scripts.clean_recovery_noise_sweep import LEVELS as RECOVERY_LEVELS
        if a.training and a.source: p.error('Choose training sources or explicit sources')
        a.levels=RECOVERY_LEVELS if a.levels is None else a.levels
        a.cuts=[0]
        a.seeds=([0] if a.training else list(range(10))) if a.seeds is None else a.seeds
        if a.training and len(a.seeds)!=1: p.error('Training recovery experiment requires exactly one seed per source')
        if not a.seeds or len(set(a.seeds))!=len(a.seeds) or any(s<0 for s in a.seeds): p.error('Seeds must be distinct nonnegative labels')
        if 0. not in a.levels or 1. not in a.levels: p.error('Recovery requires levels0 and1')
        if list(a.levels)!=sorted(set(a.levels)): p.error('Recovery levels must be increasing and distinct')
    elif a.levels is None: a.levels=LEVELS
    if not 0 <= a.shard < a.shards: p.error('Invalid shard')
    if any(k < 0 or not math.isfinite(k) for k in a.levels): p.error('Invalid multiplier')
    if any(k not in CUTS for k in a.cuts): p.error('Invalid cut')
    sources = training_sources() if a.training else a.source or ['2241+098']
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    if a.recovery:
        plan=dict(sources=sources,levels=list(a.levels),seeds=a.seeds,
                  mode='clean_recovery_training' if a.training else 'clean_recovery')
        if a.training:
            from scripts.preprocessing.partitions import partition_for_sample
            if not all(partition_for_sample(s)=='train' for s in sources): raise ValueError('Non-training source in cache')
        path=a.output/'recovery_plan.json'
        if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('Recovery plan changed; use fresh output')
        if not path.exists(): atomic_write_json(path,plan)
    else:
        atomic_write_json(a.output / f'plan_{"training" if a.training else "example"}_{a.shard}.json',
            dict(sources=sources[a.shard::a.shards], levels=a.levels, cuts=a.cuts,
                 control='total thermal STD multiplier; Y=V+a*n0; 0 is truly noiseless',
                 reference='Original sigma0 fixed; SNR_corr_actual is total thermal norm vs V',
                 library_configuration='DefaultImagingConfig with regular0% grid and annulus fixed'))
    failures = []
    for source_id in sources[a.shard::a.shards]:
        try:
            source_run(a.output, source_id, a.levels, a.cuts,recovery=a.recovery,seeds=a.seeds)
        except Exception:
            detail = traceback.format_exc()
            atomic_write_json(a.output / f'failure_{source_id}.json', dict(source=source_id, traceback=detail))
            print(detail, flush=True); failures.append(source_id)
        else:
            (a.output / f'failure_{source_id}.json').unlink(missing_ok=True)
    if failures:
        raise RuntimeError(f'Sources requiring recovery: {failures}')
