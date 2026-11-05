"""Pure scheduling, measurements and HTML reporting for the injection pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.preprocessing.schema import atomic_write_json

BRANCHES = ('phase_positive', 'phase_negative', 'amp_attenuation', 'amp_amplification', 'noise')
PHASE = (1, 3, 10, 30, 60, 90, 135, 180)
AMPLITUDE = (.1, .25, .5, .75, .9, 1.1, 1.25, 1.5, 2, 4, 8)
NOISE = (1.5, 2, 5, 10, 20, 50, 100, 200, 400, 700, 1000)
NOISE_R_V2 = (.002, .005, .01, .015, .02, .03, .05, .08, .12, .2, .3, .4, .6, .8, 1.2)
EXTENSIONS = {'phase_positive': (), 'phase_negative': (),
              'amp_attenuation': (.03, .01), 'amp_amplification': (16, 32),
              'noise': (2000, 4000)}
METRIC_VERSION = 'direct-peak-one-gaussian-v4'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def case(source, kind, value, seed=None, antenna=None):
    """Canonical physical case, independent of the evolving search grid."""
    value = float(f'{value:.14g}')
    if kind == 'phase' and abs(value) == 180: value = 180.
    if (kind == 'phase' and value == 0) or (kind == 'amp' and value == 1):
        kind, value, antenna = 'noise', 1., None
    if kind == 'noise': antenna = None
    c = dict(source=source, kind=kind, value=value, seed=seed, antenna=antenna)
    c['id'] = digest(c)[:20]
    return c


def cell(root, c):
    return Path(root)/'sources'/c['source']/'recovery'/'cases'/c['id']


def context_list(plan):
    contexts = []
    for source in plan['sources']:
        for branch in BRANCHES:
            ants = [None] if branch == 'noise' else source['selected_antennas']
            for ant in ants:
                contexts.append(dict(source=source['id'], antenna=ant, branch=branch))
    return contexts


def context_id(ctx):
    return f"{ctx['source']} / {ctx['branch']}" + ('' if ctx['antenna'] is None else f" / ant {ctx['antenna']}")


def physical(branch, q):
    if branch.startswith('phase'): return q if branch.endswith('positive') else -q
    if branch.startswith('amp'): return math.exp(-q if branch.endswith('attenuation') else q)
    return 1+q


def coordinate(branch, value):
    if branch.startswith('phase'): return abs(value)
    if branch.startswith('amp'): return abs(math.log(value))
    return value-1


def cases_at(ctx, q, seeds):
    kind = ctx['branch'].split('_')[0]
    return [case(ctx['source'], kind, physical(ctx['branch'], q), s, ctx['antenna']) for s in seeds]


def initial_cases(plan):
    result = {}
    for source in plan['sources']:
        c = case(source['id'], 'noise', 0); result[c['id']] = c
        for seed in plan['seeds']:
            c = case(source['id'], 'noise', 1, seed); result[c['id']] = c
    for ctx in context_list(plan):
        branch = ctx['branch']
        if branch=='noise' and plan.get('schema')==2:
            snr=next(s['nominal_source_snr'] for s in plan['sources'] if s['id']==ctx['source'])
            values=[r*snr for r in NOISE_R_V2]
        else:
            values = PHASE if branch.startswith('phase') else NOISE if branch == 'noise' else [a for a in AMPLITUDE if (a<1) == branch.endswith('attenuation')]
        for value in values:
            for c in cases_at(ctx, coordinate(branch, value), plan['seeds']): result[c['id']] = c
    return list(result.values())


def curve(ctx, rows, seeds):
    """Keep incomplete levels as gaps; they never establish a crossing."""
    groups = defaultdict(dict)
    for r in rows:
        if r['source'] != ctx['source'] or r['seed'] not in seeds: continue
        branch, kind, value = ctx['branch'], r['kind'], r['value']
        control = kind == 'noise' and value == 1
        matches = control or (branch == 'noise' and kind == 'noise' and value >= 1)
        if branch.startswith('amp') and kind == 'amp' and r['antenna'] == ctx['antenna']:
            matches = (value < 1) == branch.endswith('attenuation')
        if branch.startswith('phase') and kind == 'phase' and r['antenna'] == ctx['antenna']:
            matches = value == 180 or (value > 0) == branch.endswith('positive')
        if not matches: continue
        q = 0. if control else coordinate(branch, value)
        groups[float(f'{q:.12g}')][r['seed']] = r
    result = []
    for q, group in sorted(groups.items()):
        valid = len(group) == len(seeds) and all(r.get('validity') == 'valid' for r in group.values())
        ys = [r['y'] for r in group.values()] if valid else []
        result.append(dict(q=q, value=physical(ctx['branch'], q), median=float(np.median(ys)) if ys else None,
                           minimum=min(ys) if ys else None, maximum=max(ys) if ys else None, rows=list(group.values())))
    return result


def analyze_curve(points):
    control = next((p['median'] for p in points if p['q'] == 0), None)
    good = control is not None and control <= .05
    severe = None; brackets = {'0.05': [], '0.9': []}
    nonmonotonic = False
    for a, b in zip(points, points[1:]):
        if a['median'] is None or b['median'] is None: continue
        if b['median'] < a['median']-1e-10: nonmonotonic = True
        if a['median'] >= .9 and b['median'] >= .9 and severe is None: severe = a['q']
        for guide in (.05, .9):
            if (a['median']-guide)*(b['median']-guide) < 0:
                brackets[str(guide)].append([a['q'], b['q']])
    finite = [p for p in points if p['median'] is not None and p['q']>0]
    missing = any(p['median'] is None for p in points)
    last = finite[-1] if finite else None
    status = []
    if control is None: status.append('missing control')
    elif not good: status.append('baseline-degraded')
    if missing: status.append('missing runs')
    if nonmonotonic: status.append('nonmonotonic')
    if severe is None:
        status.append('boundary crossing' if last and last['median'] >= .9 else 'no severe crossing')
        if any(p['median']>=.9 for p in finite): status.append('isolated severe point')
    return dict(baseline_good=good, control_median=control, severe_q=severe,
                ceiling_q=last['q'] if last else None, ceiling_y=last['median'] if last else None,
                brackets=brackets, status=status, missing=missing,
                achieved_y=[min(p['median'] for p in finite), max(p['median'] for p in finite)] if finite else None)


def decisions(plan, rows, *, refine=False, extend=False):
    """Note rules, with shared weak candidates evaluated in every context."""
    contexts = context_list(plan); results = []; proposals = {}
    points_by_id = {}
    for ctx in contexts:
        points = curve(ctx, rows, plan['seeds']); analysis = analyze_curve(points)
        branch=ctx['branch'];extensions=() if plan.get('schema')==2 and branch=='noise' else EXTENSIONS[branch]
        cap_q=(NOISE_R_V2[-1]*next(s['nominal_source_snr'] for s in plan['sources'] if s['id']==ctx['source'])-1
               if plan.get('schema')==2 and branch=='noise' else coordinate(branch,extensions[-1]) if extensions else 180.)
        if analysis['missing'] or analysis['control_median'] is None:reason='required runs missing or invalid'
        elif analysis['severe_q'] is not None:reason='two consecutive settings have median y >= 0.9'
        elif analysis['ceiling_q'] is not None and math.isclose(analysis['ceiling_q'],cap_q,rel_tol=1e-10):
            reason='full distinct constant-phase range reached; 0.9 not reached under one-ant constant error' if branch.startswith('phase') and (analysis['achieved_y'] or [0,0])[1]<.9 else 'declared search cap reached'
        else:reason='search/refinement or extension pending'
        analysis['search_stop_reason']=reason
        positive=[p for p in points if p['q']>0]
        if analysis['baseline_good'] and positive and positive[0]['median'] is not None and positive[0]['median']>.05:
            analysis['status'].append('weak-end unresolved')
        results.append(dict(**ctx, **analysis)); points_by_id[context_id(ctx)] = points
    for branch in BRANCHES:
        group = [r for r in results if r['branch']==branch]
        good = [r for r in group if r['baseline_good']]
        weak_candidates = set()
        if refine:
            for ctx in good:
                positive = [p for p in points_by_id[context_id(ctx)] if p['q']>0]
                if not (plan.get('schema')==2 and branch=='noise') and positive and positive[0]['median'] is not None and positive[0]['median']>.05:
                    weak_candidates.add(positive[0]['q']/2)
            # Shared lower bound needs the same tested q in every good context.
            for ctx in good:
                for q in weak_candidates:
                    for c in cases_at(ctx,q,plan['seeds']): proposals[c['id']]=c
        for ctx in group:
            points=points_by_id[context_id(ctx)]
            if refine:
                for brackets in ctx['brackets'].values():
                    for lo,hi in brackets:
                        for c in cases_at(ctx,(lo+hi)/2,plan['seeds']): proposals[c['id']]=c
            if extend and ctx['severe_q'] is None and not ctx['missing']:
                tested=[p['q'] for p in points]
                for value in (() if plan.get('schema')==2 and branch=='noise' else EXTENSIONS[branch]):
                    q=coordinate(branch,value)
                    if not any(math.isclose(q,x,rel_tol=1e-11,abs_tol=1e-11) for x in tested):
                        for c in cases_at(ctx,q,plan['seeds']): proposals[c['id']]=c
                        break
    bounds=[]
    for branch in BRANCHES:
        group=[r for r in results if r['branch']==branch]; good=[r for r in group if r['baseline_good']]
        candidates=None
        for ctx in good:
            small={p['q'] for p in points_by_id[context_id(ctx)] if p['q']>0 and p['median'] is not None and p['median']<=.05}
            candidates=small if candidates is None else candidates & small
        lower=min(candidates) if candidates else None
        uppers=[r['severe_q'] if r['severe_q'] is not None else r['ceiling_q'] for r in good]
        upper=max(uppers) if uppers and all(q is not None for q in uppers) else None
        statuses=set(s for r in good for s in r['status'])
        if not good: statuses.add('no baseline-good contexts')
        elif lower is None: statuses.add('weak-end unresolved')
        if not statuses: statuses.add('resolved')
        reached=sum(r['severe_q'] is not None for r in good)
        bounds.append(dict(branch=branch,lower_q=lower,upper_q=upper,
            lower_physical=physical(branch,lower) if lower is not None else None,
            upper_physical=physical(branch,upper) if upper is not None else None,
            baseline_good_contexts=len(good),sustained_severe_contexts=reached,
            damage_spanning=bool(good) and reached==len(good) and lower is not None and not any(r['missing'] for r in good),
            status=sorted(statuses)))
    return dict(contexts=results, recommendations=bounds), list(proposals.values())


def collect(root, plan, manifest):
    """Compute direct-model metrics once per immutable record, including failures."""
    from astropy.io import fits
    from scripts.clean_recovery_noise_sweep import convolve_model, load_products, measure
    rows=[]; arrays={}; references={}
    def products(c):
        path=cell(root,c)/'record.json'; rec=json.loads(path.read_text())
        if rec.get('case') != c: raise ValueError('Case parameters changed')
        for key,p in rec['products'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=rec['sha256'][key]: raise ValueError(f'Product changed: {p}')
        folder=root/'sources'/c['source']/'recovery'
        for name,sha in rec['reference_sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=sha: raise ValueError('Reference changed')
        data=load_products(root,rec)
        with np.load(root/rec['products']['support']) as support:
            if not support['valid'][references[c['source']]['omega']].all(): raise ValueError('Invalid support')
        data['C']=convolve_model(data['model'], fits.getheader(root/rec['products']['clean']))
        return rec,data
    for c in sorted(manifest['cases'],key=lambda c:(c['source'],c['kind']!='noise',c['value']!=1,c['id'])):
        folder=root/'sources'/c['source']/'recovery'; path=cell(root,c)
        r=dict(c,validity='missing',record=str((path/'record.json').relative_to(root)))
        if (path/'failure.json').exists(): r['validity']='failed'
        if not (path/'record.json').exists(): rows.append(r);continue
        try:
            signature=digest(dict(record=hashlib.sha256((path/'record.json').read_bytes()).hexdigest(),metric=METRIC_VERSION))
            cache=path/'metrics.json'
            if cache.exists():
                saved=json.loads(cache.read_text())
                if saved.get('signature')==signature: rows.append(saved['row']);continue
            if c['source'] not in references:
                with np.load(folder/'reference_masks.npz') as masks: references[c['source']]={k:masks[k] for k in masks.files}
            masks=references[c['source']]
            control=case(c['source'],'noise',1,c['seed']) if c['seed'] is not None else c
            key=control['id']
            if key not in arrays: arrays[key]=products(control)
            control_rec,regular=arrays[key]
            rec,data=arrays[key] if key==c['id'] else products(c)
            values=measure(data['clean'],data['residual'],data['model'],masks['truth'],masks['rho'],masks['omega'],masks['annulus'],
                rec['flux_jy'],regular['clean'],regular['residual'],smoothed_model=data['C'],regular_smoothed_model=regular['C'],compute_position=False)
            for optional in ('e_pos','missing_detection','position_roundoff_warning'):values.pop(optional)
            denom=np.square(masks['truth'][masks['omega']]).sum()
            y0=float(np.sqrt(np.square((regular['C']-masks['truth'])[masks['omega']]).sum()/denom))
            convergence=rec.get('convergence') or {}
            branches=['noise'] if c['kind']=='noise' else ['amp_attenuation' if c['value']<1 else 'amp_amplification'] if c['kind']=='amp' else ['phase_positive','phase_negative'] if c['value']==180 else ['phase_positive' if c['value']>0 else 'phase_negative']
            r.update(values,validity='valid',y=values['e_TM_2'],y0=y0 if c['seed'] is not None else None,
                d=values['e_TM_2']-y0 if c['seed'] is not None else None,branches=branches,
                control_id=control['id'] if c['seed'] is not None else None,
                flux_jy=rec['flux_jy'],sigma0_jy=rec['sigma0_jy'],
                raw_model_nonzero_count=int(np.count_nonzero(data['model'])),
                signed_total_model_flux_jy=float(data['model'].sum()),iterations=convergence.get('iterdone'),
                stopcode=convergence.get('stopcode'),stop_reason=convergence.get('stop_description'),
                seconds=rec.get('seconds'),a=rec['noise_multiplier'],q=coordinate(
                    'noise' if c['kind']=='noise' else 'phase_positive' if c['kind']=='phase' else 'amp_amplification',c['value']),
                S_A=rec.get('S_A',0),S_P=rec.get('S_P',0),SNR_corr_actual=rec['SNR_corr_actual'],
                gain_actual=rec.get('gain_actual'),standalone_SNR_actual=rec.get('standalone_SNR_actual'),
                products=rec['products'])
            atomic_write_json(cache,dict(signature=signature,row=r))
        except (OSError,ValueError,KeyError) as exc:
            r.update(validity='invalid',error=str(exc))
        rows.append(r)
    return rows


def render(root):
    import platform
    from importlib.metadata import version
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve(); started=time.monotonic()
    plan=json.loads((root/'plan.json').read_text()); manifest=json.loads((root/'cases.json').read_text())
    rows=collect(root,plan,manifest)
    bounds,next_cases=decisions(plan,rows,refine=manifest['refinement_rounds']<2,extend=True)
    planned={c['id'] for c in manifest['cases']}
    next_cases=[c for c in next_cases if c['id'] not in planned]
    incomplete=any(r['validity']!='valid' for r in rows)
    finished=manifest.get('status')=='complete' and not incomplete
    bounds.update(status=manifest.get('status','running'),complete=finished,metric=METRIC_VERSION,
        refinement_rounds=manifest['refinement_rounds'],failures=[r for r in rows if r['validity']!='valid'],
        measurement_software=dict(python=platform.python_version(),**{name:version(name) for name in ('numpy','scipy','astropy','matplotlib')}))
    if not finished:
        for b in bounds['recommendations']:
            b['status']=[s for s in b['status'] if s!='resolved']+['provisional: search incomplete']
    if plan.get('smoke'):
        bounds.update(scope='smoke validation only; no pilot-bound recommendations',recommendations=[])
    atomic_write_json(root/'bounds.json',bounds);atomic_write_json(root/'next_cases.json',next_cases)
    keys=sorted(set(k for r in rows for k in r))
    with (root/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    assets=root/'plots';assets.mkdir(exist_ok=True)
    links=[]
    # Each SVG point links to a durable case page with configuration and products.
    for r in rows:
        if r['validity']!='valid':continue
        page=cell(root,r)/'index.html'
        products=' '.join(f'<a href="../../../../../{html.escape(p)}">{k}</a>' for k,p in r['products'].items())
        references=' '.join(f'<a href="../../{name}">{name}</a>' for name in ('settings.json','reference.json','reference_grid.json','reference_masks.npz'))
        control_link=f'<a href="../{r["control_id"]}/index.html">Matched ordinary control</a>' if r.get('control_id') else ''
        page.write_text('<!doctype html><meta charset="utf-8"><a href="../../../../../report.html">Report</a>'
            +f'<p>{products}</p><p>{references}</p>{control_link} · <a href="record.json">Configuration and checksums</a>'
            +f'<pre>{html.escape(json.dumps(r,indent=2))}</pre>')
    for branch in BRANCHES:
        contexts=[c for c in context_list(plan) if c['branch']==branch]
        fig,axes=plt.subplots(len(contexts)+1,2,figsize=(13,3*(len(contexts)+1)),squeeze=False)
        for j,ctx in enumerate(contexts):
            points=curve(ctx,rows,plan['seeds'])
            if not any(r['validity']=='valid' for p in points for r in p['rows']):continue
            color=f'C{j%10}'; x=[p['value'] for p in points]
            for col,key in enumerate(('y','d')):
                ax=axes[j+1,col]; overview=axes[0,col]
                for seed in plan['seeds']:
                    rs=[next((r for r in p['rows'] if r['seed']==seed and r['validity']=='valid'),None) for p in points]
                    ax.plot(x,[r[key] if r is not None else np.nan for r in rs],color=color,alpha=.28,lw=.8)
                    for xx,r in zip(x,rs):
                        if r is None:continue
                        artist=ax.scatter([xx],[r[key]],s=13,color=color,alpha=.6)
                        artist.set_urls([f'../{cell(root,r).relative_to(root)}/index.html'])
                med=[float(np.median([r[key] for r in p['rows']])) if p['median'] is not None else np.nan for p in points]
                lo=[min(r[key] for r in p['rows']) if p['median'] is not None else np.nan for p in points]
                hi=[max(r[key] for r in p['rows']) if p['median'] is not None else np.nan for p in points]
                ax.fill_between(x,lo,hi,color=color,alpha=.12);ax.plot(x,med,color=color,lw=2)
                if np.isfinite(med).any():overview.plot(x,med,color=color,label=context_id(ctx),lw=1.5)
                ax.set_title(context_id(ctx));ax.set_ylabel('e_TM,2' if key=='y' else 'd = y − y0')
                result=next(r for r in bounds['contexts'] if context_id(r)==context_id(ctx))
                if result['severe_q'] is not None: ax.axvline(physical(branch,result['severe_q']),color=color,ls='--')
        for j,pair in enumerate(axes):
            for col,ax in enumerate(pair):
                for guide in ((.05,.9,1) if col==0 else (0,)):ax.axhline(guide,color='.45',ls=':',lw=.7)
                ax.grid(alpha=.15);ax.set_xlabel('Phase (degrees)' if branch.startswith('phase') else 'Gain A' if branch.startswith('amp') else 'Total noise STD multiplier a')
                if not branch.startswith('phase'):
                    ax.set_xscale('log');ax.set_xlim((.008,1.2) if branch=='amp_attenuation' else (.9,40) if branch.startswith('amp') else (.9,4500))
                if col==0:ax.set_ylim(bottom=0)
                if j==0 and ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6,loc='best')
        proposal=next((b for b in bounds['recommendations'] if b['branch']==branch),None)
        if proposal:
            for qkey,color,label in (('lower_q','tab:green','proposed weak bound'),('upper_q','tab:red','proposed upper / tested ceiling')):
                if proposal[qkey] is not None:
                    for ax in axes[0]:ax.axvline(physical(branch,proposal[qkey]),color=color,ls='--',label=label)
            for ax in axes[0]:
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6,loc='best')
        fig.suptitle(branch+' — seed curves, median and min–max; dashed vertical = sustained severe start')
        fig.tight_layout();fig.savefig(assets/f'{branch}.svg');plt.close(fig)
        links.append(f'<h2>{branch}</h2><object data="plots/{branch}.svg" type="image/svg+xml" style="width:100%"></object>')
    # Representative seed closest to median: ordinary and all guide brackets.
    representative={r['id'] for r in rows if r['validity']=='valid' and r['kind']=='noise' and r['value']==0}
    for ctx in bounds['contexts']:
        points=curve(ctx,rows,plan['seeds']); chosen={0.}
        for brackets in ctx['brackets'].values():
            for lo,hi in brackets:chosen.update((lo,hi))
        positive=[p for p in points if p['median'] is not None]
        if positive:chosen.add(positive[-1]['q'])
        for p in points:
            if p['q'] in chosen and p['median'] is not None:
                r=min(p['rows'],key=lambda r:(abs(r['y']-p['median']),r['seed']));representative.add(r['id'])
    images=[]
    from astropy.io import fits
    from scripts.clean_recovery_noise_sweep import convolve_model,load_products
    for source in plan['sources']:
        selected=[r for r in rows if r['id'] in representative and r['source']==source['id']]
        if not selected:continue
        with np.load(root/'sources'/source['id']/'recovery/reference_masks.npz') as masks:truth=masks['truth']
        panels=[]
        for r in selected:
            rec=json.loads((root/r['record']).read_text());data=load_products(root,rec)
            C=convolve_model(data['model'],fits.getheader(root/rec['products']['clean']))
            panels.append((r,[truth,C,C-truth,data['clean'],data['residual']]))
        limits=[max(float(np.abs(ps[i]).max()) for _,ps in panels) or 1 for i in range(5)]
        for r,ps in panels:
            fig,axs=plt.subplots(1,5,figsize=(15,3))
            for i,(ax,a,label) in enumerate(zip(axs,ps,('Truth','Smoothed model','Model − truth','Restored','Residual'))):
                im=ax.imshow(a,origin='lower',cmap='RdBu_r',vmin=-limits[i],vmax=limits[i]);ax.set_title(label);ax.set_axis_off();fig.colorbar(im,ax=ax,shrink=.65)
            fig.suptitle(f"{r['source']} {r['kind']} {r['value']:g}, ant {r['antenna']}, seed {r['seed']}, y={r['y']:.4g}")
            fig.tight_layout();fig.savefig(assets/f"images_{r['id']}.jpg",dpi=85);plt.close(fig)
            images.append(f'<a href="{cell(root,r).relative_to(root)}/index.html"><img loading="lazy" width="100%" src="plots/images_{r["id"]}.jpg"></a>')
    def table(records, fields):
        heading='<tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in fields)+'</tr>' for r in records)
        return '<table>'+heading+body+'</table>'
    valid=[r for r in rows if r['validity']=='valid']; counts=Counter(r['validity'] for r in rows)
    behavior=[]
    for ctx in context_list(plan):
        for p in curve(ctx,rows,plan['seeds']):
            rs=[r for r in p['rows'] if r['validity']=='valid']
            if not rs:continue
            behavior.append(dict(context=context_id(ctx),physical=p['value'],median_y=p['median'],
                empty_fraction=sum(r['raw_model_nonzero_count']==0 for r in rs)/len(rs),
                stopcodes=dict(Counter(str(r['stopcode']) for r in rs)),iterations=[r['iterations'] for r in rs],
                signed_flux=[r['signed_total_model_flux_jy'] for r in rs],false_flux=[r['e_Fout'] for r in rs]))
    control_rows=[r for r in valid if r['kind']=='noise' and r['value'] in (0,1)]
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    refresh='' if finished else '<meta http-equiv="refresh" content="120">'
    document='<!doctype html><meta charset="utf-8">'+refresh+'''<title>Injection bounds pilot</title>
<style>body{font:15px system-ui;margin:30px;max-width:1600px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}pre{white-space:pre-wrap}object{min-height:400px}</style>
<h1>Injection bounds pilot</h1>'''+f'<p><b>Status: {html.escape(manifest.get("status","running"))}</b> — {len(valid)}/{len(rows)} valid unique images; {dict(counts)}. Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}.</p>'
    document+='''<p>Recovery-testing coverage; these sweeps provide no evidence of empirical calibration-error realism. Gains affect the noiseless sky, followed by the same ordinary additive noise for each seed. Thermal a is the total STD multiplier. Flux and nominal weights stay fixed. y is full-support model-to-truth L2 error using direct zero-padded convolution with the peak-one restoring Gaussian; d=y−y0 retains its sign. Guides 0.05, 0.9 and 1 are declared coverage guides. Empty models give y=1. Curves are not forced monotone; missing runs cannot establish bounds. Exact no-error gain controls are shared ordinary a=1 cases; phase ±180° share one physical case. Three seeds/context/level are required.</p>
<p>Amplitude axes use positive A on a log scale (attenuation strength increases toward the left); noise uses log a; phase uses signed linear degrees. Ordinary controls remain at A=1 or a=1, and never become log(q=0). Each SVG point opens its case record and image products. Image scales are fixed per product across the representative cases of each source; units Jy/beam.</p>'''
    document+='<p><a href="plan.json">Immutable configuration/selection</a> · <a href="cases.json">Case manifest</a> · <a href="metrics.csv">Per-run CSV</a> · <a href="bounds.json">Bounds and crossing brackets</a> · <a href="next_cases.json">Next cases</a> · <a href="status.json">Run costs/status</a></p>'
    if plan.get('smoke'):document+='<p><b>Smoke validation only:</b> one template and one seed; this report establishes no pilot bounds.</p>'
    control_contexts=[c for c in bounds['contexts'] if c['branch']=='noise']
    if control_contexts and all(c['control_median'] is not None for c in control_contexts) and not any(c['baseline_good'] for c in control_contexts):
        document+='<p><b>All templates are baseline-degraded:</b> review the base simulation/imaging setup before choosing injection bounds. Results and signed effects are retained; no small-to-severe range is established.</p>'
    document+='<h2>Proposed bounds</h2>'+table(bounds['recommendations'],('branch','lower_physical','upper_physical','lower_q','upper_q','baseline_good_contexts','sustained_severe_contexts','damage_spanning','status'))
    document+='<p>Unreached 0.9 under a one-ant constant error is a tested ceiling, not a confirmed severe bound. At a cap, a lone crossing is unconfirmed. Recommendations use baseline-good contexts only; incomplete searches remain provisional. Lower/upper refer to increasing q: |phase|, |ln A| or a−1. Thus the weak attenuation bound has A nearer 1, and its stronger upper-q bound has a smaller positive A. Exact no-error settings remain separate.</p>'
    document+='<h2>Selection</h2>'+table(plan['sources'],('id','flux_jy','nominal_image_rms','nominal_source_snr','antenna_count','selected_antennas','norms','uv_summary'))
    document+='<h2>Control checks and noiseless numerical checks</h2>'+table(control_rows,('source','seed','value','y','raw_model_nonzero_count','signed_total_model_flux_jy','stop_reason'))
    document+='<h2>Context decisions</h2>'+table(bounds['contexts'],('source','antenna','branch','baseline_good','control_median','severe_q','ceiling_q','ceiling_y','achieved_y','brackets','search_stop_reason','status'))
    document+=''.join(links)+'<h2>Representative images</h2>'+''.join(images)
    document+='<h2>Failure behavior and fitted flux</h2><p>Empty fractions count valid raw models with zero nonzero pixels. Signed flux is the total raw-model sum in Jy. False flux is e_Fout: absolute component sum outside the source ellipse (rho≤1), inside fixed valid support, divided by true source flux. Source-flux errors and nonzero counts are retained per run in the CSV. Missing/failed runs are excluded from these fractions and shown separately.</p>'+table(behavior,('context','physical','median_y','empty_fraction','stopcodes','iterations','signed_flux','false_flux'))
    document+='<h2>Missing, failed or invalid cases</h2>'+table([r for r in rows if r['validity']!='valid'],('source','kind','value','antenna','seed','validity','error','record'))
    document+='<h2>Cost and configuration</h2><pre>'+html.escape(json.dumps(dict(imaging_seconds=sum(r.get('seconds') or 0 for r in valid),report_seconds=time.monotonic()-started,driver=state,software=plan.get('software'),measurement_software=bounds['measurement_software'],configuration=plan.get('configuration')),indent=2))+'</pre>'
    temporary=root/'report.tmp.html';temporary.write_text(document);temporary.replace(root/'report.html')
    return rows,bounds,next_cases


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    render(parser.parse_args().output)
