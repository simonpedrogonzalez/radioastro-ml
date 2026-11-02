"""Reproduce one V1 noise sample and image nested angular UVW cuts (CASA)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from casatools import table

from scripts.create_dataset_v1 import (DATASET_DIR, THERMAL_DIR, DATASET_PBLIMIT,
    NoiseControl, _apply_noise_control, _noise_request, _prepare_V_ms, _stable_seed, _thermal_source)
from scripts.imaging import Beam, BeamRegion, DefaultImagingConfig, image_ms, beam_region_mask, summarize_residual_pixels
from astropy.io import fits
from scripts.simulation import add_thermal_noise_inplace, initialize_weights


class FixedGrid:
    """Reuse the uncut image grid and CLEAN-mask reference beam."""
    def __init__(self, reference):
        self.reference = reference

    def resolve(self, ms, *, workspace, imsize, _context):
        return replace(self.reference, first_pass_imagename=Path(workspace) / 'firstpass')


def run(output: Path, source_id: str = '2241+098', *, no_noise: bool = False):
    output=output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest=json.loads((THERMAL_DIR/'report.json').read_text())
    source=_thermal_source(THERMAL_DIR,manifest,source_id)
    model,params,sigma=_noise_request(source)
    v_ms=_prepare_V_ms(source,output)
    ms=output/f'{source_id}.ms'
    shutil.copytree(v_ms,ms)
    seed=_stable_seed(source_id,'thermal_noise')
    if no_noise:
        # Finite nominal weights for imaging; no noise is added to DATA.
        initialize_weights(ms,sigma)
        injected={'extra_noise_seed':None,'SNR_corr_measured':0.}
    else:
        control=NoiseControl(50.,'noise_snr_50')
        injected=_apply_noise_control(source,control,v_ms,ms,sigma,model,params,seed)
        add_thermal_noise_inplace(ms,noise_model=model,noise_parameters=params,seed=seed)
        initialize_weights(ms,injected['final_total_sigma_jy'])
    original=json.loads((DATASET_DIR/'samples'/f'{source_id}_noise_snr_50'/'imaging'/'qa.json').read_text())
    tb=table();tb.open(str(ms),nomodify=True)
    try:
        uvw=np.asarray(tb.getcol('UVW'),dtype=float)
        nrows=int(tb.nrows())
        # A baseline orientation represents its conjugate too: use modulo pi.
        angle=np.mod(np.arctan2(uvw[1],uvw[0]),math.pi)
        assert angle.size==nrows and np.isfinite(angle).all()
        baseline_flags=np.asarray(tb.getcol('FLAG'),dtype=bool)
        row_flags=np.asarray(tb.getcol('FLAG_ROW'),dtype=bool) if 'FLAG_ROW' in tb.colnames() else np.zeros(nrows,bool)
    finally:tb.close()
    usable=(~baseline_flags) & (~row_flags.reshape(1,1,-1))
    initial=int(usable.sum())
    assert initial>0
    rows=[]
    region=BeamRegion(min_radius_beams=source.metric_min_radius_beams,max_radius_beams=source.metric_max_radius_beams)
    fixed_config = DefaultImagingConfig
    reference_grid = None
    if no_noise:
        from scripts.imaging.models import ImageGrid, ResolvedGrid
        d=json.loads(Path('runs/uv-angular-2241-noise50-final/remove_00/imaging/qa.json').read_text())['resolved_config']['grid']
        reference_grid=ResolvedGrid(Beam(**d['beam']),ImageGrid(**d['image_grid']),Path(d['first_pass_imagename']))
        fixed_config=replace(DefaultImagingConfig,grid=FixedGrid(reference_grid))
    def fingerprints():
        tb.open(str(ms),nomodify=True)
        try:
            result={}
            for name in ('DATA','CORRECTED_DATA','WEIGHT','SIGMA','WEIGHT_SPECTRUM','SIGMA_SPECTRUM'):
                if name not in tb.colnames():continue
                digest=hashlib.sha256()
                for start in range(0,nrows,4096):
                    digest.update(np.ascontiguousarray(tb.getcol(name,startrow=start,nrow=min(4096,nrows-start))).tobytes())
                result[name]=digest.hexdigest()
            return result
        finally:tb.close()
    before=fingerprints() if no_noise else None
    uv_usable=np.any(~baseline_flags,axis=(0,1)) & ~row_flags
    extent=float(np.percentile(np.hypot(uvw[0][uv_usable],uvw[1][uv_usable]),99.5))*1.1
    np.savez_compressed(output/'uvw_initial.npz',u=uvw[0],v=uvw[1],usable=uv_usable,extent=extent)
    for pct in range(0,100,10):
        selected=angle < math.pi*pct/100
        tb.open(str(ms),nomodify=False)
        try:
            for start in range(0,nrows,4096):
                end=min(start+4096,nrows)
                flags=baseline_flags[:,:,start:end] | selected[start:end].reshape(1,1,-1)
                tb.putcol('FLAG',flags,startrow=start,nrow=end-start)
                tb.putcol('FLAG_ROW',row_flags[start:end] | selected[start:end],startrow=start,nrow=end-start)
        finally:tb.close()
        remaining=int((usable & ~selected.reshape(1,1,-1)).sum())
        row_dir=output/f'remove_{pct:02d}'
        image=image_ms(ms,fixed_config,row_dir/'imaging',imsize=source.imsize,
            metric_region=BeamRegion(),keep_intermediate_products=False,fits_invalid_policy='fill',
            fits_fill_value=0.0,pblimit=DATASET_PBLIMIT)
        if pct == 0 and reference_grid is None:
            reference_grid = image.resolved_config.grid
            fixed_config = replace(DefaultImagingConfig, grid=FixedGrid(reference_grid))
        with fits.open(image.residual_fits, memmap=False) as hdus:
            plane = np.squeeze(np.asarray(hdus[0].data, dtype=float))
            header = hdus[0].header
        mask = beam_region_mask(plane.shape, reference_grid.image_grid.cell_arcsec,
                                reference_grid.beam.major_arcsec, region)
        selected_pixels = plane[mask & np.isfinite(plane)]
        if not selected_pixels.size:
            raise RuntimeError('Fixed 0% metric annulus contains no pixels')
        m = summarize_residual_pixels(selected_pixels,
              pixel_area_arcsec2=math.prod(reference_grid.image_grid.cell_arcsec),
              beam=Beam(header['BMAJ']*3600, header['BMIN']*3600, header['BPA']))
        record={'angular_removed_percent':pct,'angular_width_degrees_each_half':180*pct/100,
            'usable_initial':initial,'usable_remaining':remaining,
            'actual_usable_removed_percent':100*(1-remaining/initial),
            'residual_scaled_mad_jy_per_beam':m.scaled_mad_jy_per_beam,
            'residual_abs_peak_jy_per_beam':m.residual_abs_peak_jy_per_beam,
            'residual_p99_5_over_mad':m.p99_5_over_scaled_mad,
            'dirty':str(image.dirty_fits.relative_to(output)),
            'clean':str(image.clean_fits.relative_to(output)),
            'residual':str(image.residual_fits.relative_to(output)),
            'qa':str(image.qa_json.relative_to(output)),
            'warnings':list(image.qa.warnings),
            'image_cell_arcsec':list(image.resolved_config.grid.image_grid.cell_arcsec)}
        rows.append(record)
        (output/'results.json').write_text(json.dumps({'source':source_id,'baseline_seed':None if no_noise else seed,
            'extra_seed':injected['extra_noise_seed'],'measured_snr_corr':injected['SNR_corr_measured'],
            'sigma_reference_jy':sigma,'original_qa':original['metrics']['residual'],
            'noise_mode':'none' if no_noise else 'baseline_plus_extra',
            'injected_noise_sigma_jy':0. if no_noise else injected['final_total_sigma_jy'],
            'grid_reference':'original SNR50 report 0% grid and mask' if no_noise else 'current 0% row',
            'mask':'atan2(v,u) mod 180 degrees; remove [0, 180*p/100) and conjugate',
            'rows':rows},indent=2)+'\n')
        print('Completed',pct,'actual removed',record['actual_usable_removed_percent'],flush=True)
    if no_noise:
        from scripts.noise_snr_audit import measure_pair
        measured=measure_pair(v_ms,ms,sigma)
        assert measured['SNR_corr_actual']==0.
        after=fingerprints();assert before==after
        tb.open(str(ms),nomodify=True)
        try:
            np.testing.assert_array_equal(tb.getcol('FLAG'),baseline_flags | selected[None,None,:])
            np.testing.assert_array_equal(tb.getcol('FLAG_ROW'),row_flags | selected)
        finally:tb.close()
        (output/'validation.json').write_text(json.dumps(dict(rows=len(rows),
            no_regular_or_extra_noise_added=True,measured_final_thermal_SNR_corr=0.,
            data_weights_unchanged_across_cuts=True,identical_column_hashes=after,
            final_flags_exact_original_or_90_percent_wedge=True,
            grid_and_clean_mask_pinned_to_original_report=True,
            fixed_metric_annulus=True),indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',default='2241+098')
    p.add_argument('--no-noise',action='store_true',help='Sky only: neither baseline nor extra thermal noise; pin original report grid')
    a=p.parse_args();run(a.output,a.source,no_noise=a.no_noise)
