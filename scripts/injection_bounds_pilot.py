"""Run the isolated, seed-paired injection-bounds pilot under CASA."""
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts import create_dataset_v1 as g
from scripts.injection_bounds_report import (case, cell, collect, decisions, digest,
                                            initial_cases)
from scripts.preprocessing.schema import atomic_write_json


def select_sources(*, v2=False):
    from scripts.noise_uv_experiments import training_sources, CACHE
    manifest=json.loads((g.THERMAL_DIR/'report.json').read_text()); ranked=[]
    for source_id in training_sources():
        source=g._thermal_source(g.THERMAL_DIR,manifest,source_id)
        sky=g._simulation_payload(source)['components']
        if len(sky)!=1 or sky[0]['shape']!='point':raise ValueError('Expected one stored point source')
        flux=float(sky[0]['flux'][0]); rms=source.predicted_image_rms_jy_per_beam
        provenance=None
        if v2:
            qa_path=g.THERMAL_DIR/source_id/'original'/'default_imaging'/'qa.json'
            qa=json.loads(qa_path.read_text())
            flux=float(qa['tclean_summary']['final_model_flux_jy'])
            peak=float(qa['metrics']['clean_peak_jy_per_beam'])
            if not math.isfinite(peak) or peak<=0:raise ValueError('Invalid original clean peak')
            provenance=dict(qa_path=str(qa_path),qa_sha256=hashlib.sha256(qa_path.read_bytes()).hexdigest(),
                clean_peak_jy_per_beam=peak,method='original CLEAN model sum')
        if not math.isfinite(flux/rms) or rms<=0:raise ValueError('Invalid nominal source SNR')
        ranked.append(dict(id=source_id,flux_jy=flux,nominal_image_rms=rms,nominal_source_snr=flux/rms,
            **({'flux_provenance':provenance} if provenance else {})))
    ranked.sort(key=lambda r:(r['nominal_source_snr'],r['id']))
    # Integer middle rank (upper middle for an even-sized training partition).
    selected=[ranked[i] for i in sorted({0,len(ranked)//2,len(ranked)-1})]
    return selected,dict(path=str(CACHE),sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        ranked_training_sources=ranked,median_rank_rule='sorted ascending (flux / nominal expected image RMS, source ID), index n//2')


def prepare_v(source, folder, sky, *, v2=False):
    if not v2:return g._prepare_V_ms(source,folder)
    from scripts.simulation import simulate_ms
    payload=g._simulation_payload(source)
    work=folder/'work'/source.source_id
    v=work/f'{source.source_id}_V.ms'
    metadata=work/f'{source.source_id}_V.simulation.json'
    metadata_text=work/f'{source.source_id}_V.simulation.txt'
    if v.is_dir() and metadata.is_file() and metadata_text.is_file():
        stored=json.loads(metadata.read_text())
        if stored.get('components')!=sky or stored.get('input_ms')!=payload['input_ms']:
            raise ValueError('Noiseless source simulation changed')
        return v
    if work.exists():
        if work.is_symlink() or work.resolve().parent!=(folder/'work').resolve():
            raise RuntimeError('Unsafe scratch work directory')
        shutil.rmtree(work)
    work.mkdir(parents=True,exist_ok=True)
    return simulate_ms(payload['input_ms'],sky,v,noise_model=None).ms_path


def source_metadata(root, selected, *, v2=False):
    from casatools import table
    from scripts.corruption import ConstantGainSpec, measure_constant_gain_norms
    from scripts.simulation import initialize_weights
    sources=[];manifest=json.loads((g.THERMAL_DIR/'report.json').read_text())
    for item in selected:
        source=g._thermal_source(g.THERMAL_DIR,manifest,item['id'])
        folder=root/'sources'/source.source_id/'recovery';folder.mkdir(parents=True,exist_ok=True)
        sky=g._simulation_payload(source)['components']
        if v2:sky=[dict(component,flux=[item['flux_jy']]+list(component['flux'][1:])) for component in sky]
        _,_,sigma=g._noise_request(source);v=prepare_v(source,folder,sky,v2=v2);initialize_weights(v,sigma)
        norms={}
        for ant,name in g._unflagged_antenna_choices(v):
            try:n=measure_constant_gain_norms(ConstantGainSpec(v,ant,'amp',1,sigma,sign=1))
            except ValueError as exc:
                if 'No finite positive visibility norm involves' in str(exc):continue
                raise
            norms[str(ant)]=dict(name=name,**asdict(n))
        if not norms:raise ValueError('No positive antenna visibility norms')
        # Both extrema break ties by the lowest antenna ID.
        low=min(norms,key=lambda a:(norms[a]['V_Ak_over_sigma_L2'],int(a)))
        high=min(norms,key=lambda a:(-norms[a]['V_Ak_over_sigma_L2'],int(a)))
        tb=table();tb.open(str(v/'ANTENNA'))
        try:antenna_count=tb.nrows()
        finally:tb.close()
        tb.open(str(v))
        try:
            rows=tb.nrows();uv=tb.getcol('UVW');uv_summary=dict(rows=rows,
                projected_baseline_min_m=float(np.hypot(uv[0],uv[1]).min()),
                projected_baseline_max_m=float(np.hypot(uv[0],uv[1]).max()))
        finally:tb.close()
        sources.append(dict(item,sigma0_jy=sigma,components=sky,
            selected_antennas=sorted(set((int(low),int(high)))),norms=norms,antenna_count=antenna_count,uv_summary=uv_summary,
            metadata_sha256=hashlib.sha256(source.simulation_metadata.read_bytes()).hexdigest()))
    return sources


def add_paired_noise(v, regular, observed, scale=1.):
    """Add scale*(regular−V) after the gain; audit storage rounding."""
    from casatools import table
    tables=[table() for _ in range(3)];a,b,c=tables
    a.open(str(v));b.open(str(regular));c.open(str(observed),nomodify=False)
    max_roundoff=0.;noise_power=0.;h=hashlib.sha256()
    try:
        for start in range(0,a.nrows(),4096):
            args=dict(startrow=start,nrow=min(4096,a.nrows()-start))
            base=np.asarray(a.getcol('DATA',**args),np.complex128)
            noise=scale*(np.asarray(b.getcol('DATA',**args),np.complex128)-base)
            gain=np.asarray(c.getcol('DATA',**args),np.complex128)
            h.update(np.ascontiguousarray(noise).tobytes());noise_power+=float(np.square(np.abs(noise)).sum())
            c.putcol('DATA',gain+noise,**args)
            actual=np.asarray(c.getcol('DATA',**args),np.complex128)-gain
            max_roundoff=max(max_roundoff,float(np.abs(actual-noise).max()))
            tolerance=8*np.finfo(np.float32).eps*max(float(np.abs(gain).max()),float(np.abs(noise).max()),1e-30)
            if not np.allclose(actual,noise,rtol=0,atol=tolerance):raise ValueError('Paired additive noise changed')
    finally:
        for tb in reversed(tables):tb.close()
    return dict(noise_delta_sha256=h.hexdigest(),noise_delta_L2_all_samples=math.sqrt(noise_power),
        maximum_storage_roundoff_jy=max_roundoff)


def render_report(root, *, final=False):
    """Use the ML environment for plotting; CASA handles simulation only."""
    plan=json.loads((root/'plan.json').read_text())
    module='scripts.simulation_parametrization_v2_report' if plan.get('schema')==2 else 'scripts.injection_bounds_report'
    command=[str(g.ROOT/'ml/.venv/bin/python'),'-m',module,'--output',str(root)]
    if final and plan.get('schema')==2:command.append('--final')
    result=subprocess.run(command,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    with (root/'report_render.log').open('a') as f:f.write(result.stdout)
    if result.returncode:raise RuntimeError('Report rendering failed: see report_render.log')


def run_batch(root, plan, manifest, status):
    from scripts.corruption import (AntennaGainCorruption, ConstantGainSpec, ConstantGainNorms,
        TimeGrid, solve_constant_gain, measure_corruption_metrics)
    from scripts.recovery_products import recovery_products
    from scripts.simulation import add_thermal_noise_inplace, initialize_weights
    from scripts.noise_snr_audit import measure_pair
    from scripts.imaging import DefaultImagingConfig
    last_report=time.monotonic()
    thermal=json.loads((g.THERMAL_DIR/'report.json').read_text())
    for item in plan['sources']:
        source=g._thermal_source(g.THERMAL_DIR,thermal,item['id']);sky=item['components']
        folder=root/'sources'/source.source_id/'recovery';sigma=item['sigma0_jy']
        model,params,actual_sigma=g._noise_request(source)
        if actual_sigma!=sigma or hashlib.sha256(source.simulation_metadata.read_bytes()).hexdigest()!=item['metadata_sha256']:
            raise ValueError('Parent simulation metadata changed')
        specification=dict(schema=1,source=item,configuration=plan['configuration'],seeds=plan['seeds'],
            software=plan['software'],protocol=plan['protocol'],seed_map={str(s):g._stable_seed(source.source_id,f'injection_bounds:noise_seed:{s}') for s in plan['seeds']})
        settings_hash=digest(specification);settings=folder/'settings.json'
        if settings.exists() and json.loads(settings.read_text())!=specification:raise ValueError('Immutable source specification changed')
        if not settings.exists():atomic_write_json(settings,specification)
        complete,_=recovery_products(root,folder,source,sky,sigma,settings_hash)
        cases=[c for c in manifest['cases'] if c['source']==source.source_id]
        pending=[]
        for c in cases:
            if complete(cell(root,c)):
                if json.loads((cell(root,c)/'record.json').read_text()).get('case')!=c:raise ValueError('Case checkpoint mismatch')
                continue
            pending.append(c)
        if not pending:continue
        v=prepare_v(source,folder,sky,v2=plan.get('schema')==2);initialize_weights(v,sigma)
        complete,image_cell=recovery_products(root,folder,source,sky,sigma,settings_hash,v)
        for label in (None,*plan['seeds']):
            selected=[c for c in pending if c['seed']==label]
            if not selected:continue
            regular=folder/f'regular_{label}.ms'
            casa_seed=None
            if label is not None:
                if regular.exists():shutil.rmtree(regular)
                shutil.copytree(v,regular);casa_seed=specification['seed_map'][str(label)]
                add_thermal_noise_inplace(regular,noise_model=model,noise_parameters=params,seed=casa_seed)
                initialize_weights(regular,sigma)
            base=measure_pair(v,regular,sigma)['SNR_corr_actual'] if label is not None else 0
            # Noiseless before ordinary, ordinary before gain metrics.
            for c in sorted(selected,key=lambda c:(c['kind']!='noise',c['value']!=1,c['value'],c['id'])):
                target=cell(root,c);target.mkdir(parents=True,exist_ok=True);ms=target/'observed.ms'
                if ms.exists():shutil.rmtree(ms)
                started=time.monotonic();extra=dict(case=c,case_hash=digest(dict(case=c,settings=settings_hash)),S_A=0.,S_P=0.)
                try:
                    if c['kind']=='noise':
                        shutil.copytree(v if c['value']==0 else regular,ms)
                        if c['value'] not in (0,1):g._rescale_visibility_delta(v,ms,c['value'])
                        actual=measure_pair(v,ms,sigma)['SNR_corr_actual']
                        np.testing.assert_allclose(actual,base*c['value'],rtol=1e-6,atol=1e-7)
                    else:
                        shutil.copytree(v,ms)
                        norms=ConstantGainNorms(**{k:item['norms'][str(c['antenna'])][k] for k in ConstantGainNorms.__dataclass_fields__})
                        Q=norms.V_Ak_over_sigma_L2
                        strength=Q*abs(c['value']-1) if c['kind']=='amp' else 2*Q*abs(math.sin(math.radians(c['value'])/2))
                        sign=1 if (c['value']>1 if c['kind']=='amp' else c['value']>0) else -1
                        solution=solve_constant_gain(ConstantGainSpec(v,c['antenna'],c['kind'],strength,sigma,sign=sign),norms)
                        gained=solution.g_amp if c['kind']=='amp' else solution.phi_deg
                        np.testing.assert_allclose(gained,c['value'],rtol=1e-7,atol=1e-7)
                        corr=AntennaGainCorruption.from_constant_gain_solution(TimeGrid(g.CORRUPTION_SOLINT),solution)
                        tab=target/'gain.G'
                        if tab.exists():shutil.rmtree(tab)
                        corr.build_corrtable(str(ms),str(tab),seed=1,diagnostic_plot=target/'gain.png')
                        from casatools import table
                        gain_table=table();gain_table.open(str(tab))
                        try:
                            gains=gain_table.getcol('CPARAM');ants=gain_table.getcol('ANTENNA1').reshape(-1)
                        finally:gain_table.close()
                        expected=complex(c['value']) if c['kind']=='amp' else np.exp(1j*math.radians(c['value']))
                        affected=gains[...,ants==c['antenna']]
                        if affected.size==0:raise ValueError('Selected antenna absent from gain table')
                        np.testing.assert_allclose(affected,expected,rtol=2e-6,atol=2e-7)
                        np.testing.assert_allclose(gains[...,ants!=c['antenna']],1,rtol=2e-6,atol=2e-7)
                        measured_gain=complex(np.mean(affected.astype(np.complex128)))
                        gained=abs(measured_gain) if c['kind']=='amp' else math.degrees(math.atan2(measured_gain.imag,measured_gain.real))
                        if c['kind']=='phase' and c['value']==180:gained=abs(gained)
                        corr.apply_corrtable(str(ms),str(tab),seed=1)
                        actual=measure_corruption_metrics(v,ms,sigma)
                        np.testing.assert_allclose(actual.SNR_corr,strength,rtol=g.SNR_CORR_REL_TOL,atol=1e-6)
                        extra.update(solution=solution.to_report_dict(),gain_actual=gained,
                            complex_gain_actual=[measured_gain.real,measured_gain.imag],
                            standalone_SNR_actual=actual.SNR_corr,**{'S_A' if c['kind']=='amp' else 'S_P':strength})
                        extra['paired_noise']=add_paired_noise(v,regular,ms)
                        shutil.rmtree(tab)
                    initialize_weights(ms,sigma)
                    a=c['value'] if c['kind']=='noise' else 1.
                    image_cell(target,ms,a,label,casa_seed,metadata=extra)
                    record=json.loads((target/'record.json').read_text());record['seconds']=time.monotonic()-started
                    atomic_write_json(target/'record.json',record)
                    (target/'failure.json').unlink(missing_ok=True)
                    status['new_images']+=1;status['elapsed_seconds']=time.time()-status['started_epoch']
                    atomic_write_json(root/'status.json',status)
                    if time.monotonic()-last_report>=120:
                        render_report(root);last_report=time.monotonic()
                except Exception:
                    atomic_write_json(target/'failure.json',dict(case=c,traceback=traceback.format_exc()))
                    status['failures']+=1;atomic_write_json(root/'status.json',status)
                    print(traceback.format_exc(),flush=True)
                finally:
                    if ms.exists():shutil.rmtree(ms)
            if regular.exists():shutil.rmtree(regular)
            render_report(root)
            last_report=time.monotonic()
        g._cleanup_V_work(source,folder)


def main():
    from scripts.imaging import DefaultImagingConfig
    from scripts.imaging.qa import json_safe
    import casatools
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--v2',action='store_true',help='Original imaging flux and r-based thermal grid')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    plan_path=root/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if plan['smoke']!=args.smoke or (plan.get('schema')==2)!=args.v2:raise ValueError('Smoke/full scope changed')
        configuration=json_safe(asdict(DefaultImagingConfig))
        if configuration!=plan['configuration']:raise ValueError('Imaging defaults changed')
        if plan['software']!=dict(python=platform.python_version(),numpy=np.__version__,casa=casatools.version_string()):
            raise ValueError('Simulation software versions changed')
        from scripts.noise_uv_experiments import CACHE
        if hashlib.sha256(CACHE.read_bytes()).hexdigest()!=plan['selection']['sha256']:raise ValueError('Training partition changed')
        if args.v2:
            for item in plan['selection']['ranked_training_sources']:
                provenance=item['flux_provenance']
                if hashlib.sha256(Path(provenance['qa_path']).read_bytes()).hexdigest()!=provenance['qa_sha256']:
                    raise ValueError('Original imaging flux provenance changed')
    else:
        selected,selection=select_sources(v2=args.v2)
        if args.smoke:selected=[selected[len(selected)//2]]
        sources=source_metadata(root,selected,v2=args.v2)
        plan=dict(schema=2 if args.v2 else 1,smoke=args.smoke,sources=sources,seeds=[0] if args.smoke else [0,1,2],selection=selection,
            configuration=json_safe(asdict(DefaultImagingConfig)),
            software=dict(python=platform.python_version(),numpy=np.__version__,casa=casatools.version_string()),
            protocol='injection-bounds-v2: original-imaging flux; r-grid thermal; fixed nominal weights/grid/beam/mask; gain(V)+paired n; direct peak-one convolution' if args.v2 else 'injection-bounds-v1: fixed nominal weights/flux/grid/beam/mask; gain(V)+paired n; direct peak-one convolution',
            created_utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(plan_path,plan)
    manifest_path=root/'cases.json'
    if manifest_path.exists():manifest=json.loads(manifest_path.read_text())
    else:
        cases=initial_cases(plan)
        if args.smoke:
            source=plan['sources'][0];ant=source['selected_antennas'][0]
            cases=[case(source['id'],'noise',0),case(source['id'],'noise',1,0),case(source['id'],'noise',5,0),
                   case(source['id'],'amp',1.5,0,ant),case(source['id'],'phase',30,0,ant)]
        manifest=dict(cases=cases,refinement_rounds=0,extension_batches=0,status='running',initial_images=len(cases),history=[])
        atomic_write_json(manifest_path,manifest)
    manifest['status']='running'
    atomic_write_json(manifest_path,manifest)
    prior=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    status=dict(started_epoch=prior.get('started_epoch',time.time()),pid=os.getpid(),new_images=prior.get('new_images',0),
        failures=prior.get('failures',0),reused_images=sum((cell(root,c)/'record.json').exists() for c in manifest['cases']))
    atomic_write_json(root/'status.json',status)
    while True:
        run_batch(root,plan,manifest,status)
        rows=collect(root,plan,manifest)
        if any(r['validity']!='valid' for r in rows):
            manifest['status']='incomplete: missing/failed/invalid runs';break
        if args.smoke:manifest['status']='complete';break
        # Two rounds, each acting on medians after the complete preceding batch.
        refining=manifest['refinement_rounds']<2
        _,proposed=decisions(plan,rows,refine=refining,extend=not refining)
        existing={c['id'] for c in manifest['cases']};new=[c for c in proposed if c['id'] not in existing]
        if refining:manifest['refinement_rounds']+=1
        else:manifest['extension_batches']+=bool(new)
        if new:
            manifest['cases'].extend(new)
            manifest['history'].append(dict(kind='refinement' if refining else 'extension',images=len(new),ids=[c['id'] for c in new]))
        atomic_write_json(manifest_path,manifest)
        render_report(root)
        if not new and not refining:manifest['status']='complete';break
    atomic_write_json(manifest_path,manifest)
    status.update(status=manifest['status'],elapsed_seconds=time.time()-status['started_epoch'],
        planned_images=len(manifest['cases']),refinement_images=sum(h['images'] for h in manifest['history'] if h['kind']=='refinement'),
        extension_images=sum(h['images'] for h in manifest['history'] if h['kind']=='extension'))
    atomic_write_json(root/'status.json',status);render_report(root,final=manifest['status']=='complete')
    print('INJECTION BOUNDS',manifest['status'],root/'report.html',flush=True)
    if manifest['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
