"""Run two disjoint CASA workers and refresh the checkpoint HTML report."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


def run(output):
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output/'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queues = [
        [['--source', '2241+098'], ['--training', '--shard', '0', '--shards', '2']],
        [['--training', '--shard', '1', '--shards', '2']],
    ]
    active, logs, failures = {}, {}, []
    rendered = 0.
    while active or any(queues):
        for worker in range(2):
            if worker in active:
                result = active[worker].poll()
                if result is None: continue
                logs.pop(worker).close(); del active[worker]
                if result:
                    failures.append({'worker': worker, 'exit_code': result})
            if queues[worker]:
                args = queues[worker].pop(0)
                log = (output/f'worker_{worker}.log').open('a')
                proc = subprocess.Popen(['bin/run', 'casa', 'scripts/noise_uv_experiments.py',
                    '--output', str(output), *args], stdout=log, stderr=subprocess.STDOUT)
                active[worker] = proc; logs[worker] = log
                print('Started worker', worker, proc.pid, args, flush=True)
        (output/'driver_status.json').write_text(json.dumps(dict(
            worker_pids={str(k): v.pid for k,v in active.items()}, failures=failures,
            queued=queues, updated_unix=time.time()), indent=2))
        if time.monotonic()-rendered > 300 or not active:
            with (output/'report_render.log').open('a') as log:
                subprocess.run(['ml/.venv/bin/python', 'scripts/noise_uv_report.py',
                    '--output', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
            rendered = time.monotonic()
        if active: time.sleep(30)
    if failures: raise RuntimeError(f'CASA workers failed; inspect worker logs: {failures}')
    print('All imaging workers completed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
