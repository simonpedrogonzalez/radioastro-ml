"""Combined bounds tables, source curves and all-case sliders for the v2 pilot."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.injection_bounds_report import case, collect, curve, decisions
from scripts.preprocessing.schema import atomic_write_json

COLORS = ('#2676bf', '#e58b22', '#299b58')
KINDS = ('amp', 'phase', 'noise')
TABLE_BRANCHES = ('phase_negative','phase_positive','amp_attenuation','amp_amplification','noise')
BRANCH_NAMES = {'amp_attenuation':'Amp attenuation', 'amp_amplification':'Amp amplification',
                'phase_negative':'Phase negative', 'phase_positive':'Phase positive', 'noise':'Thermal noise'}


def number(value):
    return '—' if value is None or not math.isfinite(float(value)) else f'{value:.5g}'


def source_role(source, ant):
    ordered=sorted(source['selected_antennas'],key=lambda a:source['norms'][str(a)]['V_Ak_over_sigma_L2'])
    return 'small' if ant==ordered[0] else 'large'


def context_label(plan, source_id, antenna):
    rank=next(i for i,s in enumerate(plan['sources']) if s['id']==source_id)
    label=('low','middle','high')[rank] if len(plan['sources'])==3 else 'middle'
    return label+' SNR'+('' if antenna is None else ' / '+source_role(plan['sources'][rank],antenna)+' antenna')


def physical_value(branch, q, snr):
    if q is None:return None
    if branch=='noise':return (1+q)/snr
    if branch=='amp_attenuation':return math.exp(-q)
    if branch=='amp_amplification':return math.exp(q)
    if branch=='phase_negative':return -q
    return q


def range_tables(plan, rows):
    from scripts.injection_bounds_report import analyze_curve, context_list
    full=[]
    for ctx in context_list(plan):
        source=next(s for s in plan['sources'] if s['id']==ctx['source'])
        points=curve(ctx,rows,plan['seeds']);a=analyze_curve(points);branch=ctx['branch'];snr=source['nominal_source_snr']
        low_bracket=a['brackets']['0.05'][0] if a['brackets']['0.05'] else None
        weak=physical_value(branch,low_bracket[1],snr) if low_bracket else None
        strong_q=a['severe_q'] if a['severe_q'] is not None else a['ceiling_q']
        strong=physical_value(branch,strong_q,snr)
        max_y=a['achieved_y'][1] if a['achieved_y'] else None
        full.append(dict(branch=branch,source=ctx['source'],antenna=ctx['antenna'],
            context=context_label(plan,ctx['source'],ctx['antenna']),
            low_bracket=None if low_bracket is None else [physical_value(branch,q,snr) for q in low_bracket],
            weak=weak,strong=strong,max_y=max_y,
            severe=a['severe_q'] is not None,baseline_good=a['baseline_good'],complete=not a['missing']))
    full.sort(key=lambda f:(TABLE_BRANCHES.index(f['branch']),
        next(i for i,s in enumerate(plan['sources']) if s['id']==f['source']),
        0 if f['antenna'] is None or 'small' in f['context'] else 1))
    summary=[];conclusion=[]
    for branch in TABLE_BRANCHES:
        group=[f for f in full if f['branch']==branch and f['baseline_good'] and f['complete']]
        def stats(key):
            entries=[(f[key],f['context']) for f in group if f[key] is not None]
            if not entries:return None
            values=[v for v,_ in entries]
            return dict(min=min(values),max=max(values),mean=float(np.mean(values)),
                min_context=next(c for v,c in entries if v==min(values)),
                max_context=next(c for v,c in entries if v==max(values)))
        weak,strong,high_y=stats('weak'),stats('strong'),stats('max_y')
        summary.append(dict(branch=branch,weak=weak,strong=strong,high_y=high_y,
            complete_contexts=len(group),severe_contexts=sum(f['severe'] for f in group)))
        if weak and strong:
            lo,hi=((strong['min'],weak['max']) if branch in ('phase_negative','amp_attenuation')
                   else (weak['min'],strong['max']))
        else:lo=hi=None
        conclusion.append(dict(branch=branch,parameter='r=a/SNR₀' if branch=='noise' else 'φ (deg)' if branch.startswith('phase') else 'A',
            min=lo,max=hi,coverage=f"{sum(f['severe'] for f in group)}/{len(group)} contexts reach 0.9" if group else 'pending',
            sampling='linear' if branch.startswith('phase') or branch=='amp_attenuation' else 'log'))
    return full,summary,conclusion


def table(records, headings):
    head='<tr>'+''.join('<th>'+html.escape(title)+'</th>' for title,_ in headings)+'</tr>'
    body=''.join('<tr>'+''.join('<td>'+str(formatter(row))+'</td>' for _,formatter in headings)+'</tr>' for row in records)
    return '<table>'+head+body+'</table>'


def table_html(plan, full, summary, conclusion, adjusted, status):
    context=lambda r:html.escape(r['context'])
    bracket=lambda r:'—' if r['low_bracket'] is None else '['+', '.join(number(x) for x in r['low_bracket'])+']'
    stat=lambda v:'—' if v is None else f"{number(v['min'])}, {number(v['max'])}, {number(v['mean'])}<br><small>{html.escape(v['min_context'])} → {html.escape(v['max_context'])}</small>"
    flux_rows=[dict(kind=('low','middle','high')[i] if len(plan['sources'])==3 else 'middle',**s) for i,s in enumerate(plan['sources'])]
    blocks=['<style>table{border-collapse:collapse;margin:8px 0 20px;font-size:.82rem}td,th{border:1px solid #506276;padding:5px 7px;vertical-align:top}th{color:#eaf3ff}small{color:#afc5d9}</style>',
            f'<p>Status: {html.escape(status)} · <a href="metrics.csv">all measured rows</a> · <a href="bounds.json">range data</a> · <a href="plan.json">configuration</a></p>',
            '<h2>Selected source templates</h2>',
            table(flux_rows,[('Kind',lambda r:r['kind']+' SNR'),('Source',lambda r:html.escape(r['id'])),('Original model flux (Jy)',lambda r:number(r['flux_jy'])),('Predicted nominal RMS (Jy/beam)',lambda r:number(r['nominal_image_rms'])),('SNR₀',lambda r:number(r['nominal_source_snr']))]),
            '<h2>Full table</h2>',
            table(full,[('Corruption',lambda r:BRANCH_NAMES[r['branch']]),('Template / antenna',context),('Near e=0.05 bracket',bracket),('High parameter',lambda r:number(r['strong'])),('High e',lambda r:number(r['max_y'])),('Coverage',lambda r:'0.9 reached' if r['severe'] else '0.9 not sustained')]),
            '<h2>Summary table</h2>',
            '<p>Each numeric cell is min, max, mean across complete baseline-good contexts; the labels below identify the extremes. Thermal values use r.</p>',
            table(summary,[('Corruption',lambda r:BRANCH_NAMES[r['branch']]),('Low parameter at e=0.05',lambda r:stat(r['weak'])),('High parameter',lambda r:stat(r['strong'])),('High e',lambda r:stat(r['high_y']))]),
            '<h2>Conclusion ranges</h2>',
            table(conclusion,[('Corruption',lambda r:BRANCH_NAMES[r['branch']]),('Parameter',lambda r:html.escape(r['parameter'])),('Min',lambda r:number(r['min'])),('Max',lambda r:number(r['max'])),('Coverage',lambda r:html.escape(r['coverage'])),('Sampling',lambda r:r['sampling'])]),
            '<h2>Adjusted conclusion ranges (interpolated)</h2>',
            '<p>The lower edge is the first 0.05 crossing of any individual source/antenna/seed curve. The upper edge is the furthest 0.9 crossing across those curves; a curve with no 0.9 crossing contributes its tested endpoint. Interpolation is linear in phase and logarithmic in amplitude or thermal r. Endpoint values are measured, not interpolated.</p>',
            table(adjusted,[('Corruption',lambda r:BRANCH_NAMES[r['branch']]),('Parameter',lambda r:html.escape(r['parameter'])),('Min',lambda r:number(r['min'])),('Max',lambda r:number(r['max'])),('Coverage',lambda r:html.escape(r['coverage'])),('Sampling',lambda r:r['sampling'])])]
    return ''.join(blocks)


def curve_rows(plan, rows, source, kind, ant, seed):
    sid=source['id'];snr=source['nominal_source_snr'];points=[]
    for row in rows:
        if row.get('validity')!='valid' or row['source']!=sid or row['seed']!=seed:continue
        if kind=='noise':
            if row['kind']!='noise' or row['value']<1:continue
            xs=[row['value']/snr]
        elif row['kind']=='noise' and row['value']==1:xs=[0.] if kind=='phase' else [1.]
        elif row['kind']==kind and row['antenna']==ant:
            xs=[-180.,180.] if kind=='phase' and row['value']==180 else [row['value']]
        else:continue
        for x in xs:points.append((x,row['y'],row['d']))
    return sorted(points)


def boundary_markers(plan, rows):
    """Interpolate the first 0.05 and outermost 0.9/endpoint over individual curves."""
    global_markers={kind:[] for kind in KINDS}
    source_markers={source['id']:{kind:[] for kind in KINDS} for source in plan['sources']}
    adjusted=[]

    def strength(kind,value):
        if kind=='phase':return abs(value)
        if kind=='amp':return abs(math.log(value))
        return value

    def crossing(points,threshold,kind):
        for (x0,y0,_),(x1,y1,_) in zip(points,points[1:]):
            if y0>=threshold:return x0,[x0,x0]
            if y0<threshold<=y1:
                fraction=(threshold-y0)/(y1-y0)
                value=(x0+fraction*(x1-x0) if kind=='phase' else
                       math.exp(math.log(x0)+fraction*(math.log(x1)-math.log(x0))))
                return value,[x0,x1]
        if points[-1][1]>=threshold:return points[-1][0],[points[-1][0]]*2
        return None,None

    for branch in TABLE_BRANCHES:
        kind='phase' if branch.startswith('phase') else 'amp' if branch.startswith('amp') else 'noise'
        def on_branch(value):
            if branch=='phase_negative':return value<=0
            if branch=='phase_positive':return value>=0
            if branch=='amp_attenuation':return value<=1
            if branch=='amp_amplification':return value>=1
            return value>0
        low_candidates=[];high_candidates=[];curve_count=0;high_crossings=0
        for source in plan['sources']:
            for antenna in ([None] if kind=='noise' else source['selected_antennas']):
                for seed in plan['seeds']:
                    points=[point for point in curve_rows(plan,rows,source,kind,antenna,seed)
                            if on_branch(point[0]) and point[1] is not None]
                    points.sort(key=lambda point:strength(kind,point[0]))
                    if len(points)<2:continue
                    curve_count+=1
                    provenance=dict(source=source['id'],antenna=antenna,seed=seed)
                    low,low_bracket=crossing(points,.05,kind)
                    high,high_bracket=crossing(points,.9,kind)
                    if low is not None:low_candidates.append(dict(value=low,bracket=low_bracket,**provenance))
                    if high is not None:high_crossings+=1
                    high_candidates.append(dict(value=high if high is not None else points[-1][0],
                                                bracket=high_bracket,endpoint=high is None,**provenance))
        low=min(low_candidates,key=lambda item:strength(kind,item['value'])) if low_candidates else None
        high=max(high_candidates,key=lambda item:strength(kind,item['value'])) if high_candidates else None
        for role,item in (('low',low),('high',high)):
            if item is None:continue
            reached=role=='high' and not item.get('endpoint',False)
            purpose=('first interpolated 0.05 crossing' if role=='low' else
                     f'outermost interpolated 0.9 crossing ({high_crossings}/{curve_count})' if reached else
                     f'tested endpoint ({high_crossings}/{curve_count} curves cross 0.9)')
            marker=dict(branch=branch,role=role,value=item['value'],purpose=purpose,
                        reached=reached,bracket=item['bracket'],
                        source=item['source'],antenna=item['antenna'],seed=item['seed'])
            global_markers[kind].append(marker)
            for source in plan['sources']:
                sid=source['id'];local=dict(marker)
                if kind=='noise':local.update(value=marker['value']*source['nominal_source_snr'],r=marker['value'])
                source_markers[sid][kind].append(local)
        reversed_branch=branch in ('phase_negative','amp_attenuation')
        adjusted.append(dict(branch=branch,parameter='r=a/SNR₀' if branch=='noise' else 'φ (deg)' if kind=='phase' else 'A',
                             min=(high['value'] if high else None) if reversed_branch else (low['value'] if low else None),
                             max=(low['value'] if low else None) if reversed_branch else (high['value'] if high else None),
                             coverage=f'{high_crossings}/{curve_count} curves cross 0.9'+
                                      ('; endpoint used' if high and high.get('endpoint') else ''),
                             sampling='linear' if kind=='phase' else 'log'))
    for kind in KINDS:global_markers[kind].sort(key=lambda marker:marker['value'])
    for source in source_markers.values():
        for kind in KINDS:source[kind].sort(key=lambda marker:marker['value'])
    return global_markers,source_markers,adjusted


def plot_overview(root, plan, rows, kind, markers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.4))
    fig.subplots_adjust(left=.07,right=.985,top=.94,bottom=.16,wspace=.17)
    for col,key in enumerate(('y','d')):
        ax=axes[col]
        for rank,source in enumerate(plan['sources']):
            color=COLORS[rank];label=('low','middle','high')[rank] if len(plan['sources'])==3 else 'middle'
            if kind=='noise':
                groups=defaultdict(list)
                for seed in plan['seeds']:
                    line=curve_rows(plan,rows,source,kind,None,seed)
                    if not line:continue
                    ax.plot([p[0] for p in line],[p[col+1] for p in line],color=color,alpha=.5,lw=1.25,ls='-',label=label+' SNR' if seed==plan['seeds'][0] else None)
                    for p in line:
                        if p[col+1] is not None:groups[p[0]].append(p[col+1])
                if groups:
                    x=sorted(groups)
                    ax.fill_between(x,[min(groups[z]) for z in x],[max(groups[z]) for z in x],color=color,alpha=.12)
            else:
                by_role={}
                for ant in source['selected_antennas']:
                    role=source_role(source,ant);style='-' if role=='large' else '--';series=[]
                    for seed in plan['seeds']:
                        line=curve_rows(plan,rows,source,kind,ant,seed)
                        if not line:continue
                        series.append(line)
                        ax.plot([p[0] for p in line],[p[col+1] for p in line],color=color,alpha=.15,lw=.75,ls=style)
                    groups=defaultdict(list)
                    for line in series:
                        for p in line:
                            if p[col+1] is not None:groups[p[0]].append(p[col+1])
                    if groups:
                        x=sorted(groups);y=[float(np.mean(groups[z])) for z in x]
                        ax.plot(x,y,color=color,ls=style,lw=2,label=f'{label} SNR · {role} antenna')
                        by_role[role]=(np.asarray(x),np.asarray(y))
                if set(by_role)=={'small','large'}:
                    lo,hi=by_role['small'],by_role['large']
                    shared=np.asarray(sorted(set(lo[0])|set(hi[0])))
                    shared=shared[(shared>=max(lo[0][0],hi[0][0]))&(shared<=min(lo[0][-1],hi[0][-1]))]
                    if len(shared)>1:
                        xp=np.log(shared) if kind=='amp' else shared
                        ylo=np.interp(xp,np.log(lo[0]) if kind=='amp' else lo[0],lo[1])
                        yhi=np.interp(xp,np.log(hi[0]) if kind=='amp' else hi[0],hi[1])
                        ax.fill_between(shared,ylo,yhi,color=color,alpha=.13)
        if kind!='phase':ax.set_xscale('log')
        ax.set_xlabel('φ (degrees)' if kind=='phase' else 'A' if kind=='amp' else 'r = a / SNR₀')
        ax.set_ylabel('e_TM,2' if key=='y' else 'd = e_TM,2 − y₀')
        ax.grid(alpha=.2)
        for threshold in (.05,.9):ax.axhline(threshold,color='#777',ls='--',lw=.8)
        for marker in markers:
            value=marker['value'];low=marker['role']=='low'
            color='#298dbf' if low else '#c2414d'
            ax.axvline(value,color=color,ls='--' if low else '-.',lw=1.3)
            if kind=='phase':value_label=f'φ={number(value)}°'
            elif kind=='amp':value_label=f'A={number(value)}'
            else:
                noise_a=[number(value*source['nominal_source_snr']) for source in plan['sources']]
                value_label=f'r={number(value)}\n'+'\n'.join(f'a({role})={a}' for role,a in zip('LMH',noise_a))
            left=kind=='noise' and low or kind=='phase' and value<0 and not low or kind=='amp' and value<1 and not low or kind in ('phase','amp') and value>0 and low
            label_height=.97 if not low else .66 if marker['branch'] in ('phase_positive','amp_amplification') else .79
            short_purpose='first e=0.05' if low else 'outermost e=0.9' if marker['reached'] else 'tested endpoint'
            ax.text(value,label_height,f'{value_label}\n{short_purpose}',
                    transform=ax.get_xaxis_transform(),ha='left' if left else 'right',va='top',
                    fontsize=7.2,color=color,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
        if col==0 and ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7,ncol=2,loc='best')
    name=f'overview_{kind}_interpolated.svg';directory=root/'plots';directory.mkdir(exist_ok=True)
    fig.savefig(directory/name);plt.close(fig)
    return name


def slider_manifest(plan, cases):
    sliders={};roles={};selected=[]
    available={c['id'] for c in cases}
    for source in plan['sources']:
        sid=source['id'];snr=source['nominal_source_snr'];ants=source['selected_antennas']
        roles[sid]={str(a):source_role(source,a)+' contribution' for a in ants}
        selected.append(dict(id=sid,nominal_source_snr=snr,selected_antennas=ants))
        source_cases=[c for c in cases if c['source']==sid]
        values={kind:set() for kind in KINDS}
        for c in source_cases:
            if c['kind']=='phase':values['phase'].add(c['value'])
            elif c['kind']=='amp':values['amp'].add(c['value'])
            elif c['kind']=='noise':values['noise'].add(c['value'])
        values['phase'].add(0.);values['amp'].add(1.)
        if 180. in values['phase']:values['phase'].add(-180.)
        sliders[sid]={}
        for kind in KINDS:
            stops=[]
            for value in sorted(values[kind]):
                stop=dict(value=value,badges=['ordinary control' if (kind=='phase' and value==0) or (kind in ('amp','noise') and value==1) else 'noiseless' if kind=='noise' and value==0 else 'measured'],admissible=True,cases={})
                if kind=='noise':stop.update(a=value,r=value/snr,sources=['sweep'])
                for seed in plan['seeds']:
                    for ant in ([None] if kind=='noise' else ants):
                        c=(case(sid,'noise',0) if kind=='noise' and value==0 else
                           case(sid,'noise',1,seed) if kind!='noise' and ((kind=='phase' and value==0) or (kind=='amp' and value==1)) else
                           case(sid,kind,value,seed,ant))
                        if c['id'] in available:
                            stop['cases'][f'{kind if kind=="noise" else ant}:{seed}']=c['id']
                if stop['cases']:stops.append(stop)
            sliders[sid][kind]=stops
    return dict(schema=2,sources=selected,seeds=plan['seeds'],antenna_roles=roles,sliders=sliders,
        cases=[dict(case=c,origin='pilot') for c in cases])


def render(root:Path, *, final=False):
    from scripts.injection_bounds_report import cell
    root=root.resolve();plan=json.loads((root/'plan.json').read_text());manifest=json.loads((root/'cases.json').read_text())
    if plan.get('schema')!=2:raise ValueError('Expected parametrization v2 plan')
    rows=collect(root,plan,manifest)
    full,summary,conclusion=range_tables(plan,rows)
    valid=[r for r in rows if r['validity']=='valid']
    status=manifest.get('status','running')
    _,next_cases=decisions(plan,rows,refine=manifest['refinement_rounds']<2,extend=True)
    planned={c['id'] for c in manifest['cases']}
    atomic_write_json(root/'next_cases.json',[c for c in next_cases if c['id'] not in planned])
    keys=sorted(set(k for r in rows for k in r)-{'q'})
    with (root/'metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items() if k in keys} for r in rows)
    slider=slider_manifest(plan,manifest['cases'])
    overview_markers,source_markers,adjusted=boundary_markers(plan,rows)
    atomic_write_json(root/'bounds.json',dict(status=status,full=full,summary=summary,conclusion=conclusion,
        adjusted_conclusion=adjusted,missing_or_invalid=[r for r in rows if r['validity']!='valid']))
    for kind in KINDS:plot_overview(root,plan,rows,kind,overview_markers[kind])
    overview=table_html(plan,full,summary,conclusion,adjusted,status)
    overview+='<p>Blue vertical lines mark the earliest interpolated 0.05 crossing of any curve in each branch. Red lines mark the furthest 0.9 crossing or tested endpoint. They use the same positions in the paired y/d plots and source slider plots. The interpolated positions can fall between slider stops; move to the neighboring measured stops to inspect their images. Thermal labels list the corresponding <em>a</em> for low, middle and high SNR.</p>'
    if final:
        if status!='complete' or len(valid)!=len(rows):raise ValueError('Cannot render all-case sliders before complete run')
        from scripts.injection_range_visual_report import _render_assets, verify_display, write_html
        cached=root/'render_data.json';display_path=root/'display.json'
        if cached.is_file() and display_path.is_file():
            data=json.loads(cached.read_text());display=json.loads(display_path.read_text())
            if set(data)!={case['id'] for case in manifest['cases']}:
                raise ValueError('Cached slider images do not match the complete case manifest')
        else:data,display=_render_assets(root,root,slider,plan=plan,peak_anchor=True)
        verify_display(slider,data,display)
        for entry in slider['cases']:
            record=data[entry['case']['id']]
            entry.update(origin=record['origin'],record=record['record'],record_sha256=record['record_sha256'])
        atomic_write_json(root/'case_manifest.json',slider)
        write_html(root,slider,data,display,overview_html=overview,
                   conclusion_boundaries=source_markers,by_kind=True)
    else:
        document='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="120"><title>Simulation parametrization v2</title><style>body{font:15px system-ui;background:#10151d;color:#e7edf5;margin:20px}a{color:#9cd5ff}img{width:100%}</style><h1>Simulation parametrization v2</h1>'
        document+=overview+''.join(f'<h2>{kind.title()}</h2><img src="plots/overview_{kind}_interpolated.svg">' for kind in KINDS)
        (root/'report.html').write_text(document)
    print('Report',root/'report.html',len(valid),'/',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    render(args.output,final=args.final)
