"""Portable configuration, provenance and task execution for the release."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing as mp
import tempfile
import torch
from threadpoolctl import threadpool_limits
from extremilefed.reproducibility import load_frozen_config

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def configuration(group, setting):
    path = ROOT / 'configs' / group / (setting + '.json')
    return load_frozen_config(path), read(path)


def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    threadpool_limits(limits=1)


def sources():
    paths = list((ROOT / 'code').glob('*.py'))
    paths += list((ROOT / 'code/extremilefed').glob('*.py'))
    paths += list((ROOT / 'configs').rglob('*.json'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def cached(path, signature, task):
    if not path.exists():
        return False
    value = read(path)
    if value.get('release_signature') != signature or value.get('release_task_sha256') != digest(task):
        raise ValueError('Cached result belongs to another task or source version: ' + str(path))
    return True


def publish(path, result, signature, task):
    result.update(release_signature=signature, release_task_sha256=digest(task))
    write(path, result)


def execute(worker, tasks, output, kind, workers=1):
    output = Path(output).resolve()
    identity = {'kind': kind, 'tasks_sha256': digest(tasks), 'sources': sources()}
    signature = digest(identity)
    with output_lock(output):
        target = output / 'protocol.json'
        if target.exists() and read(target) != identity:
            raise ValueError('Output configuration changed; choose a fresh --output directory.')
        write(target, identity)
        arguments = [(task, str(output), signature) for task in tasks]
        if workers == 1:
            for argument in arguments:
                print(worker(argument), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                for result in pool.map(worker, arguments):
                    print(result, flush=True)


def check_positive(parser, **values):
    for name, value in values.items():
        if value < 1:
            parser.error(name + ' must be positive')
