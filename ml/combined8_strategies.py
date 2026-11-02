"""Compare combined8 multi-output XGBoost strategies with source-safe stacking."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, ParameterGrid
from xgboost import XGBRegressor

from ml.logreg import COMBINED8_FEATURES, FeatureSet
from ml.regression import GRID, TARGETS, RegressionData, _hash, cross_talk, evaluate_data, evaluation_definition, fit_model, write_json
from ml.regression_report import write_report

NAMES = ("1tree_all_data", "per_target_all_data", "per_target_only_non0_target", "2stages")


def predict(model, x):
    return np.column_stack([m.predict(x) for m in model]) if isinstance(model, list) else model.predict(x)


def fit_positive(x, y, params, jobs):
    models = []
    for j in range(3):
        mask = y[:, j] > 0
        if not mask.any():
            raise ValueError(f"No positive training rows for {TARGETS[j]}")
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=jobs, **params)
        models.append(model.fit(x[mask], y[mask, j]))
    return models


def tune_positive(x, y, folds, jobs):
    rows = []
    for params in ParameterGrid(GRID):
        scores = []
        for a, b in folds:
            scale = y[a].std(0)
            model = fit_positive(x[a], y[a], params, jobs)
            p = np.maximum(0, predict(model, x[b]))
            scores.append(float(np.mean(np.abs(p-y[b]) / scale)))
        rows.append({"params":params, "fold_scores":scores, "mean_scaled_mae":float(np.mean(scores))})
    best = min(rows,key=lambda r:r['mean_scaled_mae'])['params']
    return fit_positive(x,y,best,jobs),best,rows


def source_folds(groups):
    groups = np.asarray(groups)
    return list(GroupKFold(n_splits=3).split(np.zeros(len(groups)),groups=groups))


def base_oof(x,y,groups,params_all,params_positive,jobs):
    output = np.full((len(x),6),np.nan)
    for a,b in source_folds(groups):
        all_model = XGBRegressor(objective="reg:squarederror",tree_method="hist",multi_strategy="one_output_per_tree",random_state=42,n_jobs=jobs,**params_all).fit(x[a],y[a])
        positive_model = fit_positive(x[a],y[a],params_positive,jobs)
        output[b,:3] = np.maximum(0,predict(all_model,x[b]))
        output[b,3:] = np.maximum(0,predict(positive_model,x[b]))
    assert np.isfinite(output).all()
    return output


def stack_features(x,all_model,positive_model):
    return np.column_stack((x,np.maximum(0,predict(all_model,x)),np.maximum(0,predict(positive_model,x))))


def load_data(source, noise_cutoff=None):
    feature_config=json.loads((source/'config.json').read_text())
    reference=Path(feature_config['source_run'])
    config=json.loads((reference/'config.json').read_text())
    if config.get('target_definition_version') != 2 or config['cv_grid'] != GRID or config['seed'] != 42:
        raise ValueError('Use the combined8 scalar run sourced from a full regression run')
    with (reference/'predictions.csv').open() as f:
        rows=list(csv.DictReader(f))
    first=next(iter(config['features']))
    lookup={r['sample_id']:r for r in rows if r['model']==first}
    data={}
    for split in ('train','validation'):
        with np.load(source/f'{split}_features.npz') as s:
            ids=tuple(s['sample_ids'].tolist())
            assert tuple(s['feature_names'])==COMBINED8_FEATURES
            keep = np.ones(len(ids), dtype=bool) if noise_cutoff is None else ~((s['kinds'] == 'noise') & (s['strengths'] >= noise_cutoff))
            selected_ids = tuple(np.asarray(ids)[keep])
            features=FeatureSet(s['X'][keep],np.zeros(sum(keep)),selected_ids,tuple(s['source_ids'][keep].tolist()),tuple({} for _ in selected_ids),COMBINED8_FEATURES)
            products=tuple(json.loads(lookup[i]['products']) for i in selected_ids) if split=='validation' else tuple({} for _ in selected_ids)
            with np.load(reference/f'{split}_features.npz') as original:
                np.testing.assert_array_equal(s['sample_ids'], original['sample_ids'])
                np.testing.assert_array_equal(s['X'][:,:6], original['X'])
                targets=original['y'][keep]
            data[split]=RegressionData(features,targets,s['kinds'][keep],s['strengths'][keep],products)
    groups=np.asarray(data['train'].features.source_ids)
    folds=[];counts=np.zeros(len(groups),int)
    for f in config['cv_folds']:
        a=np.flatnonzero(np.isin(groups,f['train_sources']));b=np.flatnonzero(np.isin(groups,f['validation_sources']))
        assert len(a)+len(b)==len(groups) and not set(groups[a]) & set(groups[b])
        folds.append((a,b));counts[b]+=1
    np.testing.assert_array_equal(counts,np.ones(len(groups),int))
    return config,data,folds


def compare(source,output,jobs=4,noise_cutoff=None):
    source=source.resolve();old,data,folds=load_data(source,noise_cutoff)
    train,val=data['train'],data['validation'];x=train.features.X;y=train.targets;groups=np.asarray(train.features.source_ids)
    run=output.resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+('_combined8_below1000_strategies' if noise_cutoff == 1000 else '_combined8_strategies'))
    run.mkdir(parents=True);print('Run:',run,flush=True)
    for split,d in data.items():
        with np.load(source/f'{split}_features.npz') as original:
            values={key:original[key] for key in original.files}
        keep = np.ones(len(values['sample_ids']), dtype=bool) if noise_cutoff is None else ~((values['kinds'] == 'noise') & (values['strengths'] >= noise_cutoff))
        values = {key: value[keep] if getattr(value, 'shape', ())[:1] == (len(keep),) else value for key, value in values.items()}
        values['y']=d.targets
        np.savez_compressed(run/f'{split}_features.npz',**values)
    shutil.copy2(source/'config.json',run/'source_config.json')
    config=dict(old)
    for k in ('report_source_run','source_artifact_sha256','report_code_sha256','training_rows_per_output','training_description','model_files'):config.pop(k,None)
    config.update(experiment='combined8 output strategies and stacked stage',source_run=str(source),jobs=jobs,
                  features={name:list(COMBINED8_FEATURES)+([f'{base}_{target}' for base in ('per_target_all_data','per_target_only_non0_target') for target in TARGETS] if name=='2stages' else []) for name in NAMES},
                  model_strategies={'1tree_all_data':'multi_output_tree','per_target_all_data':'one_output_per_tree','per_target_only_non0_target':'three scalar trees','2stages':'multi_output_tree'},
                  model_files={},best_params={},evaluation_definition=evaluation_definition(),python=platform.python_version(),
                  versions={name:version(name) for name in old['versions']},
                  code_sha256={str(p.resolve()):_hash(p) for p in (Path(__file__),Path('ml/regression.py'),Path('ml/regression_report.py'))},
                  training_rows_per_output={name:{target:(int((y[:,j]>0).sum()) if name=='per_target_only_non0_target' else len(y)) for j,target in enumerate(TARGETS)} for name in NAMES},
                  training_description='1tree_all_data and per_target_all_data use all training rows. per_target_only_non0_target fits each scalar model only on rows with a positive value for its target. 2stages fits a shared tree on combined8 and six clipped first-stage predictions. Its training features are out-of-fold by source; stage CV regenerates these features within each training fold to avoid validation-source leakage.',
                  median_prediction_lines=True,stacking='nested source-fold out-of-fold first-stage predictions',
                  noise_strength_exclusion={'kind':'noise','at_or_above':noise_cutoff} if noise_cutoff is not None else None,
                  splits={split:{'rows':len(d.targets),'sources':len(set(d.features.source_ids))} for split,d in data.items()})
    predictions={};results={};matrices={};cv_rows=[];rows=[];models={}
    for name in NAMES[:3]:
        print('Fitting',name,flush=True)
        if name=='per_target_only_non0_target':model,best,candidates=tune_positive(x,y,folds,jobs)
        else:model,best,candidates=fit_model(x,y,folds,jobs=jobs,multi_strategy='multi_output_tree' if name=='1tree_all_data' else 'one_output_per_tree')
        models[name]=model;config['best_params'][name]=best;cv_rows.extend({'model':name,**r} for r in candidates)
    # Full-training out-of-fold features for the final stage fit.
    oof=base_oof(x,y,groups,config['best_params']['per_target_all_data'],config['best_params']['per_target_only_non0_target'],jobs)
    stage_x=np.column_stack((x,oof))
    # Nested source CV: each stage training fold gets inner OOF predictions.
    stage_folds=[]
    for a,b in folds:
        all_inner=base_oof(x[a],y[a],groups[a],config['best_params']['per_target_all_data'],config['best_params']['per_target_only_non0_target'],jobs)
        all_outer=XGBRegressor(objective='reg:squarederror',tree_method='hist',multi_strategy='one_output_per_tree',random_state=42,n_jobs=jobs,**config['best_params']['per_target_all_data']).fit(x[a],y[a])
        positive_outer=fit_positive(x[a],y[a],config['best_params']['per_target_only_non0_target'],jobs)
        stage_folds.append((np.column_stack((x[a],all_inner)),stack_features(x[b],all_outer,positive_outer),y[a],y[b]))
    print('Fitting 2stages',flush=True)
    candidates=[]
    for params in ParameterGrid(GRID):
        scores=[]
        for xa,xb,ya,yb in stage_folds:
            m=XGBRegressor(objective='reg:squarederror',tree_method='hist',multi_strategy='multi_output_tree',random_state=42,n_jobs=jobs,**params).fit(xa,ya)
            scores.append(float(np.mean(np.abs(np.maximum(0,m.predict(xb))-yb)/ya.std(0))))
        candidates.append({'params':params,'fold_scores':scores,'mean_scaled_mae':float(np.mean(scores))})
    best=min(candidates,key=lambda r:r['mean_scaled_mae'])['params']
    models['2stages']=XGBRegressor(objective='reg:squarederror',tree_method='hist',multi_strategy='multi_output_tree',random_state=42,n_jobs=jobs,**best).fit(stage_x,y)
    config['best_params']['2stages']=best;cv_rows.extend({'model':'2stages',**r} for r in candidates)
    np.savez_compressed(run/'stage_train_features.npz',X=stage_x,sample_ids=train.features.sample_ids)
    val_x=np.column_stack((val.features.X,np.maximum(0,predict(models['per_target_all_data'],val.features.X)),np.maximum(0,predict(models['per_target_only_non0_target'],val.features.X))))
    np.savez_compressed(run/'stage_validation_features.npz',X=val_x,sample_ids=val.features.sample_ids)
    for name,model in models.items():
        estimators=model if isinstance(model,list) else [model]
        filenames=[f'{name}_{t}.json' for t in TARGETS] if isinstance(model,list) else [f'{name}.json']
        config['model_files'][name]=filenames
        restored=[]
        for m,fn in zip(estimators,filenames,strict=True):
            m.save_model(run/fn);re=XGBRegressor();re.load_model(run/fn);restored.append(re)
        xx=val_x if name=='2stages' else val.features.X
        raw=predict(model,xx).astype(float)
        np.testing.assert_array_equal(raw,predict(restored if isinstance(model,list) else restored[0],xx))
        pred=np.maximum(0,raw);predictions[name]=pred;results[name]=evaluate_data(val,pred);matrices[name]=cross_talk(val.targets,pred,val.kinds)
        for i,sid in enumerate(val.features.sample_ids):
            row={'model':name,'split':'validation','sample_id':sid,'source_id':val.features.source_ids[i],'kind':val.kinds[i],'requested_snr':float(val.strengths[i]),'products':json.dumps(val.products[i])}
            for j,t in enumerate(TARGETS):row.update({f'true_{t}':float(val.targets[i,j]),f'pred_{t}':float(pred[i,j]),f'raw_{t}':float(raw[i,j])})
            rows.append(row)
        print(name,'Total MAE',results[name]['overall']['total']['mae'],flush=True)
    for name,target in (('train_mean',y.mean(0)),('no_error',np.zeros(3))):results[name]=evaluate_data(val,np.broadcast_to(target,val.targets.shape))
    with (run/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(run/'config.json',config);write_json(run/'cv_results.json',cv_rows);write_json(run/'results.json',{'evaluation':{'validation':results},'cross_talk':matrices})
    write_report(run,data,{'validation':predictions},{'validation':results},matrices,config)
    print('Report:',run/'report.html',flush=True)
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-run',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('ml/runs'));p.add_argument('--jobs',type=int,default=4);p.add_argument('--exclude-noise-at-or-above',type=float)
    a=p.parse_args();compare(a.from_run,a.output,a.jobs,a.exclude_noise_at_or_above)

if __name__=='__main__':main()
