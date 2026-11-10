"""Shared recovery imaging/checkpoints; CASA is imported only when invoked."""
from __future__ import annotations

def recovery_products(output, folder, source, sky, sigma, settings_hash, v=None, retain_casa_tables=False):
    import hashlib
    import json
    import math
    import shutil
    import time
    from dataclasses import asdict, replace
    import numpy as np
    from astropy.io import fits
    from casatools import table, image as casa_image
    from scripts import create_dataset_v1 as g
    from scripts.imaging import Beam, BeamRegion, DefaultImagingConfig, beam_region_mask, image_ms
    from scripts.imaging.config import read_beam
    from scripts.imaging.fits import export_casa_fits
    from scripts.imaging.qa import json_safe
    from scripts.noise_uv_experiments import read_grid
    from scripts.noise_snr_audit import measure_pair
    from scripts.preprocessing.fits import load_fits_plane
    from scripts.preprocessing.schema import atomic_write_json
    from scripts.clean_recovery_noise_sweep import geometry
    from scripts.uv_angular_coverage_experiment import FixedGrid

    source_id=source.source_id
    flux=float(sky[0]['flux'][0])
    direction=sky[0]['dir'].split()
    ra,dec=[float(x[:-3]) for x in direction[1:]]
    def complete(cell):
        path=cell/'record.json'
        if not path.exists(): return False
        r=json.loads(path.read_text())
        if r.get('settings_hash')!=settings_hash: raise ValueError('Checkpoint settings mismatch')
        for key in ('dirty','clean','residual','psf','model','support'):
            product=output/r['products'][key]
            if not product.is_file() or hashlib.sha256(product.read_bytes()).hexdigest()!=r['sha256'][key]:
                raise ValueError(f'Checkpoint product missing or modified: {product}')
        with np.load(output/r['products']['support']) as support:
            if not {'valid','clean_search'}.issubset(support.files): raise ValueError('Incomplete support checkpoint')
        for name,digest in r['reference_sha256'].items():
            if not (folder/name).exists() or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Reference changed: {name}')
        return True
    reference_path=folder/'reference.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
    grid=read_grid(folder/'reference_grid.json') if reference is not None else None
    beam=Beam(**reference['beam']) if reference is not None else None
    region=BeamRegion(source.metric_min_radius_beams,source.metric_max_radius_beams)

    def image_cell(cell,ms,multiplier,seed_label,casa_seed,metadata=None,require_same_samples=True):
        """require_same_samples=False (VLA pipeline QA): flags/weights may differ from V; recorded, not rejected."""
        nonlocal reference,grid,beam
        if complete(cell): return
        cell.mkdir(parents=True,exist_ok=True)
        # Archive only incomplete products owned by this run; never overwrite.
        directory=cell/'imaging'
        if directory.exists(): directory.rename(cell/f'incomplete_imaging_{time.time_ns()}')
        config=DefaultImagingConfig if grid is None else replace(DefaultImagingConfig,grid=FixedGrid(grid),restoring_beam=beam)
        imaging=image_ms(ms,config,directory,imsize=source.imsize,metric_region=BeamRegion(),
            keep_intermediate_products=True,fits_invalid_policy='fill',fits_fill_value=0.,pblimit=g.DATASET_PBLIMIT)
        final_beam=read_beam(imaging.clean_image)
        if grid is None:
            grid=imaging.resolved_config.grid;beam=final_beam
        else:
            np.testing.assert_allclose(list(asdict(final_beam).values()),list(asdict(beam).values()),rtol=1e-6,atol=1e-6)
        if imaging.model_image is None: raise ValueError('CLEAN model missing')
        model_fits=export_casa_fits(imaging.model_image,directory/'model.fits.gz',fallback_beam=beam,fallback_unit='Jy/pixel',invalid_policy='fill')
        def raw(path):
            ia=casa_image();ia.open(str(path))
            try:
                values=np.squeeze(np.asarray(ia.getchunk(),float)).T
                mask=np.squeeze(np.asarray(ia.getchunk(getmask=True),bool)).T
                return values,mask
            finally:ia.close()
        data={k:raw(path) for k,path in dict(dirty=imaging.dirty_image,clean=imaging.clean_image,
            residual=imaging.residual_image,model=imaging.model_image,psf=imaging.psf_image).items()}
        valid=np.logical_and.reduce([data[k][1] for k in ('dirty','clean','residual')])
        if not all(np.isfinite(values[valid]).all() for values,mask in data.values()):
            raise ValueError('Nonfinite product inside CASA valid support')
        if imaging.mask_image is None: raise ValueError('CLEAN search mask missing')
        search=raw(imaging.mask_image)[0]!=0
        support_path=directory/'support.npz'
        np.savez_compressed(support_path,valid=valid,clean_search=search,**{k+'_mask':p[1] for k,p in data.items()})
        header=fits.getheader(imaging.clean_fits)
        if reference is None:
            rho,truth=geometry(header,data['clean'][0].shape,flux,ra,dec)
            # The entire rho<=1 ellipse must lie within the valid image.
            if np.any((rho<=1)&~valid) or np.any(rho[[0,-1],:]<=1) or np.any(rho[:,[0,-1]]<=1):
                raise ValueError('True source ellipse does not fit valid support')
            annulus=beam_region_mask(rho.shape,grid.image_grid.cell_arcsec,grid.beam.major_arcsec,region)
            if not np.any(annulus&valid): raise ValueError('No valid annulus pixels')
            annulus &= valid
            np.savez_compressed(folder/'reference_masks.npz',omega=valid,annulus=annulus,rho=rho,truth=truth,source=valid&(rho<=1),clean_search=search)
            reference=dict(flux_jy=flux,ra_rad=ra,dec_rad=dec,beam=asdict(beam),grid=asdict(grid),
                support_rule='CASA dirty/image/residual validity masks and finite model/PSF; standard gridder pblimit<0 applies no PB cutoff',
                omega_pixels=int(valid.sum()),annulus_pixels=int(annulus.sum()),
                image_area_arcsec2=float(valid.sum()*math.prod(grid.image_grid.cell_arcsec)),
                area_beams=float(valid.sum()*math.prod(grid.image_grid.cell_arcsec)/(math.pi*beam.major_arcsec*beam.minor_arcsec/(4*math.log(2)))))
            atomic_write_json(folder/'reference_grid.json',asdict(grid))
            atomic_write_json(reference_path,reference)
        else:
            with np.load(folder/'reference_masks.npz') as masks:
                if not valid[masks['omega']].all(): raise ValueError('New invalid support inside fixed Omega')
                np.testing.assert_array_equal(search,masks['clean_search'])
        products={k:str(getattr(imaging,k+'_fits').relative_to(output)) for k in ('dirty','clean','residual','psf')}
        products.update(model=str(model_fits.relative_to(output)),support=str(support_path.relative_to(output)))
        try:actual=measure_pair(v,ms,sigma)
        except Exception:
            if require_same_samples:raise
            actual=dict(SNR_corr_actual=None,valid_complex_samples=None)
        def invariant_hashes(path):
            tb=table();tb.open(str(path))
            try:
                hashes={}
                for key in ('WEIGHT','SIGMA','WEIGHT_SPECTRUM','SIGMA_SPECTRUM','FLAG','FLAG_ROW','UVW'):
                    if key not in tb.colnames(): continue
                    digest=hashlib.sha256()
                    for start in range(0,tb.nrows(),4096):
                        digest.update(np.ascontiguousarray(tb.getcol(key,startrow=start,nrow=min(4096,tb.nrows()-start))).tobytes())
                    hashes[key]=digest.hexdigest()
                return hashes
            finally:tb.close()
        invariants=invariant_hashes(ms)
        reference_invariants=invariant_hashes(v)
        if require_same_samples and invariants!=reference_invariants: raise ValueError('Weights/flags/UVW changed within recovery sweep')
        record=dict(source=source_id,noise_multiplier=multiplier,seed=seed_label,casa_seed=casa_seed,
            settings_hash=settings_hash,products=products,components=sky,flux_jy=flux,sigma0_jy=sigma,
            SNR_corr_actual=actual['SNR_corr_actual'],valid_complex_samples=actual['valid_complex_samples'],
            resolved_configuration=json_safe(imaging.resolved_config),beam=asdict(final_beam),
            convergence=json_safe(imaging.qa.tclean_summary),warnings=list(imaging.qa.warnings),
            visibility_invariant_hashes=invariants,
            reference_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ('reference.json','reference_grid.json','reference_masks.npz')},
            sha256={k:hashlib.sha256((output/p).read_bytes()).hexdigest() for k,p in products.items()})
        for k,p in products.items():
            if k!='support': load_fits_plane(output/p)
        if not require_same_samples:
            record['samples_changed_from_V']=sorted(k for k in set(invariants)|set(reference_invariants)
                                                   if invariants.get(k)!=reference_invariants.get(k))
        if metadata is not None: record.update(metadata)
        atomic_write_json(cell/'record.json',record)
        # Retain compressed products and metadata, remove only generated CASA
        # temporary tables/PNGs listed by the completed imaging result, unless
        # the caller retains every native CASA image table as well.
        for path in ([] if retain_casa_tables else directory.iterdir()):
            if path.name.endswith('.fits.gz') or path.name in ('qa.json','qa.txt','support.npz'): continue
            if path.name.startswith(('clean.','dirty.','firstpass.','first_pass.')):
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
        print('RECOVERY COMPLETE',source_id,seed_label,multiplier,flush=True)

    return complete, image_cell
