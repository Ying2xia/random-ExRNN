"""Resume the original ERNN+CSL study without changing its selected implementation.

Default is a check-only dry run. Use --run to execute missing formal replications.
This launcher deliberately lives outside extremilefed/ so its provenance hash stays intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import warnings

import torch

from extremilefed.config import TAU_LEVELS
from extremilefed.development import implementation_hash
from extremilefed.experiments import ALL_METHODS, RESULT_SCHEMA, SimulationJob, run_job
from extremilefed.reproducibility import verify_simulation_selection
from extremilefed.validation import FORMAL_METHODS, validate_formal_coverage

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUTPUT = CODE / 'results/formal_v7'
RECORDS = ROOT / '工作记录与备份/2026-10-07_运行预检'
ERRORS = ('normal', 't3', 'lognormal', 'contaminated')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def atomic_json(path, value, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            temporary.replace(path)
        else:
            # Hard-link publication is atomic and fails if any writer created the target.
            os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_local_configs():
    """Relocate only the two evidence paths; retain and enforce every existing gate."""
    configs, migration = {}, []
    local_dir = CODE / 'configs/ready_local'
    local_dir.mkdir(parents=True, exist_ok=True)
    for scenario in (1, 2, 3):
        for error in ERRORS:
            key = f'scenario{scenario}_{error}'
            source = CODE / 'configs/ready' / (key + '.json')
            payload = json.loads(source.read_text())
            before = json.loads(source.read_text())
            for field in ('evidence_root', 'readiness_evidence'):
                original = Path(payload['selection'][field])
                parts = original.parts
                if parts.count('工作记录与备份') != 1:
                    raise ValueError(f'Cannot uniquely relocate {field}: {original}')
                relative = Path(*parts[parts.index('工作记录与备份'):])
                target = (ROOT / relative).resolve()
                if not target.is_relative_to((ROOT / '工作记录与备份').resolve()) or not target.exists():
                    raise ValueError(f'Evidence missing or outside this project: {target}')
                payload['selection'][field] = str(target)
            destination = local_dir / source.name
            # Verify the copied paths using the original package gate before adopting them.
            with tempfile.TemporaryDirectory(prefix='verify-', dir=local_dir) as folder:
                candidate = Path(folder) / source.name
                atomic_json(candidate, payload)
                config = verify_simulation_selection(candidate, scenario, error)
            if destination.exists():
                if json.loads(destination.read_text()) != payload:
                    raise ValueError(f'Local configuration changed: {destination}')
            else:
                atomic_json(destination, payload)
            configs[(scenario, error)] = config
            migration.append({'source': str(source), 'source_sha256': sha256(source),
                              'local_copy': str(destination), 'local_sha256': sha256(destination),
                              'config_fingerprint': config.fingerprint(),
                              'old_paths': {k: before['selection'][k] for k in ('evidence_root', 'readiness_evidence')},
                              'new_paths': {k: payload['selection'][k] for k in ('evidence_root', 'readiness_evidence')}})
    return configs, migration


def finite_tree(value, location='result'):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'Nonfinite number at {location}')
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, location + '.' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f'{location}[{index}]')


def verify_result(payload, job, config):
    finite_tree(payload)
    if payload['schema_version'] != RESULT_SCHEMA or payload['job'] != asdict(job):
        raise ValueError(f'Wrong schema or job metadata: {job.key}')
    if payload['config_fingerprint'] != config.fingerprint() or canonical(payload['config']) != canonical(asdict(config)):
        raise ValueError(f'Configuration mismatch: {job.key}')
    train_seed = job.seed + 100000 * job.scenario + 1000 * job.replication
    expected_seeds = {'training_data': train_seed, 'test_data': train_seed + 1,
                      'partition': train_seed + 2, 'model': config.ernn.seed}
    if payload['seeds'] != expected_seeds or payload['tau_levels'] != list(TAU_LEVELS):
        raise ValueError(f'Seeds or target levels mismatch: {job.key}')
    sizes = payload['partition_sizes']
    if len(sizes) != job.clients or sum(sizes) != job.n_train or min(sizes) <= 0 or max(sizes) - min(sizes) > 1:
        raise ValueError(f'Invalid client allocation: {job.key}')
    q = payload['quadrature']
    if q['training_nodes'] != config.ernn.quadrature.training_nodes or q['prediction_nodes'] != config.ernn.quadrature.prediction_nodes:
        raise ValueError(f'Quadrature mismatch: {job.key}')
    from extremilefed.quadrature import gauss_legendre_unit
    import numpy as np
    nodes, weights = gauss_legendre_unit(config.ernn.quadrature.prediction_nodes)
    if not (np.allclose(q['evaluation_levels'], nodes, rtol=0, atol=1e-14)
            and np.allclose(q['evaluation_weights'], weights, rtol=0, atol=1e-14)):
        raise ValueError(f'Quadrature grid mismatch: {job.key}')
    if set(payload['methods']) != FORMAL_METHODS:
        raise ValueError(f'Incomplete method set: {job.key}')
    required = {'extremile_mae', 'extremile_rmse', 'integrated_pinball', 'coverage_error', 'crossing_rate'}
    required |= {f'extremile_{metric}_tau_{tau:.2f}' for metric in ('mae', 'rmse') for tau in TAU_LEVELS}
    for name, result in payload['methods'].items():
        metrics = result['metrics']
        if not required.issubset(metrics) or any(not isinstance(metrics[k], (int, float)) or metrics[k] < 0 for k in required):
            raise ValueError(f'Invalid metrics for {name}: {job.key}')
        if any(not 0 <= metrics[k] <= 1 for k in ('coverage_error', 'crossing_rate')):
            raise ValueError(f'Invalid diagnostic proportion for {name}: {job.key}')
        if result['elapsed_seconds'] < 0 or result['communication_mb'] < 0:
            raise ValueError(f'Invalid resource values for {name}: {job.key}')
    rounds = payload['methods']['ExtremileFed']['gradient_diagnostics']
    if len(rounds) != config.rounds:
        raise ValueError(f'Missing CSL round diagnostics: {job.key}')
    for index, entry in enumerate(rounds, 1):
        if entry['round'] != index or not entry['finite_step_audit']['numerically_valid']:
            raise ValueError(f'CSL finite-step check failed: {job.key}, round {index}')


def jobs():
    return [SimulationJob(s, e, r, 100000, 5000, 20, 20260903)
            for s in (1, 2, 3) for e in ERRORS for r in range(100)]


def scan(configs):
    coverage = validate_formal_coverage(OUTPUT)
    if coverage['unexpected'] or coverage['invalid']:
        raise ValueError(f"Invalid/unexpected existing results: {coverage['invalid'] + coverage['unexpected']}")
    existing, missing = {}, []
    for job in jobs():
        path = OUTPUT / (job.key + '.json')
        if path.exists():
            verify_result(json.loads(path.read_text()), job, configs[(job.scenario, job.error)])
            existing[path.name] = sha256(path)
        else:
            missing.append(job)
    return existing, missing


def environment():
    from threadpoolctl import threadpool_info
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'libraries': {n: importlib.import_module(n).__version__ for n in ('numpy', 'pandas', 'scipy', 'sklearn', 'torch')},
            'torch_threads': torch.get_num_threads(), 'blas': threadpool_info(),
            'legacy_environment': 'Original result files did not record dependency versions; not claimed identical.',
            'requirements_text': (CODE / 'requirements.txt').read_text()}


@contextmanager
def exclusive_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.resume.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another continuation process is using this result directory.')
        handle.seek(0); handle.truncate(); handle.write(str(os.getpid()) + '\n'); handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='execute missing formal replications; otherwise check only')
    parser.add_argument('--max-new-jobs', type=int, help='stop after this many new complete replications')
    parser.add_argument('--scenario', type=int, choices=(1, 2, 3))
    parser.add_argument('--error', choices=ERRORS)
    args = parser.parse_args()
    if args.max_new_jobs is not None and args.max_new_jobs <= 0:
        parser.error('--max-new-jobs must be positive')
    torch.set_num_threads(1)
    with exclusive_lock(OUTPUT):
        configs, migration = load_local_configs()
        existing, missing = scan(configs)
        selected = [j for j in missing if (args.scenario is None or j.scenario == args.scenario)
                    and (args.error is None or j.error == args.error)]
        print(f'Validated {len(configs)} configurations; {len(existing)}/1200 completed; {len(missing)} missing.', flush=True)
        for setting in configs:
            count = sum(j.scenario == setting[0] and j.error == setting[1] for j in missing)
            print(f'  scenario{setting[0]}_{setting[1]}: {100-count}/100 complete', flush=True)
        selected = selected[:args.max_new_jobs] if args.max_new_jobs else selected
        if not args.run:
            print(f'CHECK ONLY: would run {len(selected)} missing replications; no model training started.', flush=True)
            return
        if not selected:
            print('Nothing to run for the selected scope.', flush=True)
            return
        launch = RECORDS / 'formal_launches' / (datetime.now().astimezone().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
        manifest = {'started_at': datetime.now().astimezone().isoformat(), 'output': str(OUTPUT),
                    'implementation_hash': implementation_hash(), 'launcher_sha256': sha256(__file__),
                    'environment': environment(), 'migration': migration, 'existing_sha256': existing,
                    'planned_jobs': [j.key for j in selected], 'completed_jobs': [], 'status': 'running'}
        atomic_json(launch / 'manifest.json', manifest)
        try:
            for index, job in enumerate(selected, 1):
                print(f'[{index}/{len(selected)}] START {job.key}', flush=True)
                path = OUTPUT / (job.key + '.json')
                if path.exists():
                    raise RuntimeError(f'Unexpected concurrent writer: {path}')
                start = time.monotonic()
                observed = []
                try:
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter('always', RuntimeWarning)
                        result = run_job(job, configs[(job.scenario, job.error)], ALL_METHODS)
                finally:
                    notices = [{'category': w.category.__name__, 'message': str(w.message),
                                'filename': w.filename, 'lineno': w.lineno} for w in observed]
                    atomic_json(launch / (job.key + '_warnings.json'), notices)
                verify_result(result, job, configs[(job.scenario, job.error)])
                if path.exists():
                    raise RuntimeError(f'Unexpected concurrent writer: {path}')
                atomic_json(path, result, overwrite=False)
                manifest['completed_jobs'].append(job.key)
                atomic_json(launch / (job.key + '_completed.json'),
                            {'sha256': sha256(path), 'seconds': time.monotonic()-start,
                             'runtime_warnings': len(notices), 'finished_at': datetime.now().astimezone().isoformat()})
                print(f'[{index}/{len(selected)}] SAVED {job.key} ({time.monotonic()-start:.1f}s, {len(notices)} recorded warnings)', flush=True)
            final_existing, final_missing = scan(configs)
            if any(sha256(OUTPUT / name) != digest for name, digest in existing.items()):
                raise RuntimeError('An existing result changed during this launch.')
            manifest.update(status='complete' if not final_missing else 'batch_complete',
                            final_existing_count=len(final_existing), final_missing_count=len(final_missing))
            print(f'DONE: {len(final_existing)}/1200 complete; {len(final_missing)} remaining.', flush=True)
        except BaseException as error:
            manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error_type=type(error).__name__, error=str(error))
            raise
        finally:
            # Preserve an audit even after Ctrl-C or a training failure.
            try:
                deviations = [name for name, digest in existing.items()
                              if not (OUTPUT / name).is_file() or sha256(OUTPUT / name) != digest]
                manifest['existing_results_unchanged'] = not deviations
                manifest['existing_result_deviations'] = deviations
                manifest['final_result_file_count'] = len(list(OUTPUT.glob('scenario*_rep*.json')))
                manifest['implementation_hash_after'] = implementation_hash()
                manifest['launcher_sha256_after'] = sha256(__file__)
                changed = (manifest['implementation_hash_after'] != manifest['implementation_hash']
                           or manifest['launcher_sha256_after'] != manifest['launcher_sha256'])
                manifest['provenance_changed'] = changed
                if deviations or changed:
                    manifest['status'] = 'audit_failed'
                    print('AUDIT FAILED: source or existing results changed during this launch.', file=sys.stderr, flush=True)
            except Exception as audit_error:
                manifest['final_audit_error'] = repr(audit_error)
                manifest['status'] = 'audit_failed'
            manifest['finished_at'] = datetime.now().astimezone().isoformat()
            atomic_json(launch / 'manifest.json', manifest)
        if manifest['status'] == 'audit_failed':
            raise RuntimeError(f'Final provenance audit failed; see {launch / "manifest.json"}')


if __name__ == '__main__':
    main()
