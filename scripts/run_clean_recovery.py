"""Run or monitor a recovery experiment and refresh its checkpoint report.

Pilot: --attach-pid PID monitors an already-running CASA launcher.
Training: --training --workers 2 runs disjoint shards with one seed per source.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.clean_recovery_noise_sweep import LEVELS, progress
from scripts.preprocessing.schema import atomic_write_json


def run(output, *, training=False, workers=1, attach_pid=None, refresh_seconds=300):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'driver.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if training:
        if attach_pid is not None: raise ValueError('Training workers cannot attach to a pilot')
        cache=Path('ml/runs/20261028T065537442783Z_combined8_strategies/train_features.npz')
        with np.load(cache,allow_pickle=False) as d:sources=sorted(set(d['source_ids'].tolist()))
        from scripts.preprocessing.partitions import partition_for_sample
        if len(sources)!=100 or not all(partition_for_sample(s)=='train' for s in sources):
            raise ValueError('Expected the existing 100 training sources only')
        plan=dict(sources=sources,levels=list(LEVELS),seeds=[0],mode='clean_recovery_training')
        atomic_write_json(output/'selection.json',dict(cache=str(cache.resolve()),sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            source_partition='train',source_count=len(sources),pilot_excluded='2241+098' not in sources))
    else:
        plan=dict(sources=['2241+098'],levels=list(LEVELS),seeds=list(range(10)),mode='clean_recovery')
    flux_mode=output/'recovery_flux_mode.json'
    mode={'mode':'original_imaging_model_flux'}
    if flux_mode.exists() and json.loads(flux_mode.read_text())!=mode:
        raise ValueError('Existing run uses a different flux protocol')
    if not flux_mode.exists(): atomic_write_json(flux_mode,mode)
    plan_path=output/'recovery_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan: raise ValueError('Existing run uses different settings')
    if not plan_path.exists(): atomic_write_json(plan_path,plan)
    if attach_pid is not None:
        command=subprocess.check_output(['ps','-p',str(attach_pid),'-o','command='],text=True)
        if 'noise_uv_experiments.py' not in command or output.name not in command:
            raise ValueError('PID does not belong to this experiment')
    children={};logs={};completed=[];render_errors=[]
    if attach_pid is None:
        for shard in range(workers if training else 1):
            args=['bin/run','casa','scripts/noise_uv_experiments.py','--recovery','--output',str(output)]
            args += ['--training','--seeds','0','--shard',str(shard),'--shards',str(workers)] if training else ['--source','2241+098']
            log=(output/f'worker_{shard}.log').open('a')
            children[shard]=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);logs[shard]=log
            print('Started',shard,children[shard].pid,flush=True)
    last_render=-float('inf')
    while True:
        for shard,child in list(children.items()):
            if child.poll() is not None:
                completed.append(dict(worker=shard,exit_code=child.returncode));logs.pop(shard).close();del children[shard]
        attached_alive=False
        if attach_pid is not None:
            status=subprocess.run(['ps','-p',str(attach_pid),'-o','stat='],capture_output=True,text=True)
            attached_alive=status.returncode==0 and 'Z' not in status.stdout
        active=bool(children) or attached_alive
        state=progress(output,plan)
        complete=state['completed_images']==state['expected_images']
        status='running' if active else 'complete' if complete and not state['generation_failures'] else 'incomplete_or_failed'
        atomic_write_json(output/'progress.json',state)
        atomic_write_json(output/'driver_status.json',dict(status=status,worker_pids={str(k):v.pid for k,v in children.items()},
            attached_pid=attach_pid if attached_alive else None,completed_workers=completed,report_errors=render_errors,**state))
        if time.monotonic()-last_render>=refresh_seconds or not active:
            with (output/'report_render.log').open('a') as log:
                result=subprocess.run(['ml/.venv/bin/python','-m','scripts.clean_recovery_noise_sweep','--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:render_errors.append(dict(time=time.time(),exit_code=result.returncode))
            last_render=time.monotonic()
        if not active:
            if not complete or state['generation_failures'] or any(c['exit_code'] for c in completed) or result.returncode:
                raise RuntimeError('Experiment needs recovery; inspect driver_status.json and logs')
            print('All requested recovery images and report completed',flush=True);break
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--training',action='store_true')
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--attach-pid',type=int)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    run(args.output,training=args.training,workers=args.workers,attach_pid=args.attach_pid)
