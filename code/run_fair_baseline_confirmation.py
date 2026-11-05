"""Locked, resumable large-sample comparison from a common trained Host.

Run without arguments for development, selection, confirmation, and export.
The validated pilot's numerical update functions are reused read-only.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
DEFAULT_OUTPUT = CODE / 'results/fair_baseline_confirmation_20261029'
HELPER = CODE / '_fair_updates.py'
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('fair_baseline_pilot_helpers', HELPER)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
np, torch = engine.np, engine.torch

SETTINGS = ('scenario1_normal', 'scenario2_contaminated', 'scenario3_t3')
BASELINES = ('fedavg', 'fedprox', 'scaffold')
METHODS = ('extremilefed', 'zero', *BASELINES)


def log(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + message, flush=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != json.loads(json.dumps(value)):
            raise RuntimeError(f'Frozen configuration differs: {path}. Use a new output directory.')
    else:
        atomic_json(path, value)


def protocol(smoke=False):
    grids = {
        'fedavg': [{'lr': lr, 'epochs': epochs, 'mu': 0.0}
                   for lr in (.0003, .001, .01, .03) for epochs in (5, 10)],
        'fedprox': [{'lr': lr, 'epochs': 10, 'mu': mu}
                    for lr in (.0003, .001, .01, .03) for mu in (.0001, .001, .01)],
        'scaffold': [{'lr': lr, 'epochs': epochs, 'mu': 0.0}
                     for lr in (.003, .01, .03, .1, .3) for epochs in (5, 10)],
    }
    plan = {
        'schema': 1, 'purpose': 'Independent confirmation of shared-initialization IID comparison',
        'settings': list(SETTINGS), 'training_n': 100000, 'validation_n': 5000,
        'test_n': 5000, 'clients': 20, 'replications': 30,
        'development_replications': 3, 'development_rounds': 3,
        'max_rounds': 20, 'checkpoints': [1, 2, 3, 5, 10, 20],
        'primary_round': 3, 'descriptive_round': 20, 'zero_correction_rounds': 3,
        'partition': 'IID balanced random allocation; client 0 is host',
        'initialization': 'Identical saved trained Host parameter for every method within a replication',
        'config_source': 'configs/main/*.json',
        'host_and_surrogate_budgets': 'Inherit frozen main-study configurations; no solver retuning',
        'csl_learning_rates': [0.001, 0.001, 0.0003],
        'csl_after_round3_lr': 0.0003,
        'baseline_grids': grids,
        'selection': 'Smallest mean primary-round validation IPS across independent development datasets',
        'selection_ties': 'Lexicographic canonical parameter JSON; no test-based selection',
        'numerical_candidate_failures': 'Record all failures; candidate must succeed in every development replication',
        'confirmation_failures': 'Retain and report; never silently omit from aggregate means',
        'seed_bases': {'development': 1360000000, 'confirmation': 1470000000},
        'seed_rule': 'base + setting_index * 100000 + replication * 100; roles train/valid/test/partition/network use +0/+1/+2/+3/+4',
        'training_nodes': 32, 'prediction_nodes': 128,
        'resources': 'Count gradient-observations including common Host; training messages exclude common preprocessing summaries',
        'timings': 'Single-machine simulations, not multi-machine distributed wall-clock times',
        'smoke': smoke,
    }
    if smoke:
        plan.update(training_n=200, validation_n=32, test_n=32, clients=4,
                    replications=2, development_replications=2, development_rounds=3,
                    max_rounds=3, checkpoints=[1, 2, 3], descriptive_round=3,
                    training_nodes=4, prediction_nodes=16)
        plan['baseline_grids'] = {
            method: [{'lr': .01, 'epochs': 1, 'mu': .001 if method == 'fedprox' else 0.0}]
            for method in BASELINES}
    return plan


def source_hashes():
    files = [Path(__file__).resolve(), HELPER, CODE / 'run_real_data_experiments.py']
    files.extend(sorted((CODE / 'extremilefed').glob('*.py')))
    files.extend(ROOT / 'configs/main' / f'{s}.json'
                 for s in SETTINGS)
    return {str(path.relative_to(ROOT)): file_hash(path) for path in files}


def prepare(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / 'protocol.json', plan)
    manifest = {'protocol_sha256': fingerprint(plan), 'sources': source_hashes(),
                'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}
    immutable_json(output / 'source_manifest.json', manifest)
    for relative in manifest['sources']:
        target = output / 'source_snapshot' / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    atomic_json(output / 'expected_coverage.json', {
        'confirmation_groups': len(SETTINGS) * plan['replications'],
        'confirmation_arms_excluding_host': len(SETTINGS) * plan['replications'] * len(METHODS),
        'development_groups': len(SETTINGS) * plan['development_replications'],
        'development_candidate_fits': len(SETTINGS) * plan['development_replications'] *
                                    sum(map(len, plan['baseline_grids'].values())),
        'primary_round': 3, 'smoke': plan['smoke']})
    return fingerprint(manifest)


def group_folder(output, stage, setting, rep):
    return output / stage / f'{setting}_rep{rep:03d}'


def finite(values):
    array = values if isinstance(values, np.ndarray) else np.fromiter(values, dtype=float)
    if not np.isfinite(array).all():
        raise FloatingPointError('Non-finite parameter or evaluation metric')


def scores(model, scaler, config, X, y, truth=None):
    fit = engine.ERNNResult(model, scaler, config)
    levels, weights = engine.gauss_legendre_unit(config.quadrature.prediction_nodes)
    q = engine.predict_quantiles(fit, X, levels)
    finite(q.ravel())
    residual = y[:, None] - q
    loss = np.where(residual >= 0, levels * residual, (levels - 1) * residual)
    # Elementwise reduction is algebraically the same weighted IPS as the main code.
    ips = float(np.mean(np.sum(loss * (weights / weights.sum())[None, :], axis=1)))
    result = {'ips': ips}
    if truth is not None:
        predictions = engine.predict_extremiles(fit, X, engine.TAU_LEVELS)
        finite(predictions.ravel())
        result.update(engine.extremile_errors(predictions, truth))
        result['ce'] = engine.coverage_error(y, q, levels)
        result['cr'] = engine.crossing_rate(engine.predict_quantiles(fit, X, np.linspace(.01, .99, 99)))
        for j, level in enumerate(engine.TAU_LEVELS):
            result[f'rmse_{level:.2f}'] = float(np.sqrt(np.mean((predictions[:, j] - truth[:, j]) ** 2)))
    finite(result.values())
    return result


def run_arm(folder, method, params, start, host, parts, scaler, config,
            validation, test, truth, stage, plan, signature):
    path = folder / f'{method}_{fingerprint(params)[:12]}.json'
    if path.exists():
        previous = read(path)
        assert previous['signature'] == signature and previous['params'] == params
        assert previous['initial_sha256'] == fingerprint(start.tolist())
        return previous
    model = engine.init(parts[0][0].shape[1], config.ernn, start)
    state = {'controls': [np.zeros_like(start) for _ in parts], 'server': np.zeros_like(start)}
    limit = (plan['development_rounds'] if stage == 'development' else
             plan['zero_correction_rounds'] if method == 'zero' else plan['max_rounds'])
    result = {'method': method, 'params': params, 'signature': signature,
              'initial_sha256': fingerprint(start.tolist()), 'points': [], 'rounds': [],
              'status': 'running', 'smoke': plan['smoke']}
    fit_seconds = 0.0
    try:
        with engine.count_work() as work:
            for t in range(limit):
                then = time.perf_counter()
                model, history = engine.round_update(method, model, parts, config.ernn,
                                                     params, state, t, config)
                fit_seconds += time.perf_counter() - then
                finite(engine.vector(model))
                result['rounds'].append(history)
                if t + 1 in plan['checkpoints'] or t + 1 == limit:
                    point = {'round': t + 1, 'validation': scores(model, scaler, config.ernn, *validation),
                             'gradient_observations': work['gradient_observations'] + host['work']['gradient_observations'],
                             'forward_observations': work['forward_observations'] + host['work']['forward_observations'],
                             'payload_bytes': (0 if method == 'zero' else
                                (4 if method == 'scaffold' else 2) * (t + 1) * (len(parts) - 1) * len(start) * 8),
                             'fit_seconds': fit_seconds + host['fit_seconds']}
                    if stage == 'confirmation':
                        point['test'] = scores(model, scaler, config.ernn, *test, truth)
                    result['points'].append(point)
        result['status'] = 'ok'
    except FloatingPointError as error:
        result.update(status='numerical_failure', error=repr(error), traceback=traceback.format_exc())
    atomic_json(path, result)
    log(f'{folder.name} {method} {params}: {result["status"]}, training {fit_seconds:.1f}s')
    return result


def run_group(task):
    output_text, stage, setting, rep, plan, signature = task
    output = Path(output_text)
    folder = group_folder(output, stage, setting, rep)
    complete_path = folder / 'complete.json'
    if complete_path.exists():
        assert read(complete_path)['signature'] == signature
        return {'group': folder.name, 'cached': True}
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    engine.threadpool_limits(limits=1)
    seed = plan['seed_bases'][stage] + SETTINGS.index(setting) * 100000 + rep * 100
    scenario, error = int(setting[8]), setting.split('_', 1)[1]
    X, y = engine.generate_data(plan['training_n'], scenario, error, seed)
    V, z = engine.generate_data(plan['validation_n'], scenario, error, seed + 1)
    ids = engine.partition_indices(len(y), plan['clients'], seed + 3)
    test, truth = None, None
    if stage == 'confirmation':
        T, w = engine.generate_data(plan['test_n'], scenario, error, seed + 2)
        test = (T, w)
        truth = engine.true_extremiles(T, scenario, error, np.asarray(engine.TAU_LEVELS))
    config, config_sha256 = engine.configuration(setting, seed + 4)
    if plan['smoke']:
        config = replace(config, ernn=replace(config.ernn, hidden_units=3, second_hidden_units=None,
                         min_epochs=1, max_epochs=2, patience=3, refinement_epochs=1,
                         quadrature=replace(config.ernn.quadrature, training_nodes=4, prediction_nodes=16)),
                         host_epochs=2, host_refinement_epochs=1, host_patience=3,
                         surrogate_epochs=2, surrogate_patience=3)
    parts = [(X[index], y[index]) for index in ids]
    scaler = engine.fit_federated_standardizer(parts)
    standardized = [(scaler.transform_X(a), scaler.transform_y(b)) for a, b in parts]
    provenance = {'signature': signature, 'stage': stage, 'setting': setting, 'rep': rep,
                  'seed': seed, 'config': asdict(config), 'config_sha256': config_sha256,
                  'partition_sizes': [len(a) for a in ids],
                  'partition_sha256': hashlib.sha256(np.concatenate(ids).tobytes()).hexdigest(),
                  'training_n': len(y), 'validation_n': len(z),
                  'test_n': 0 if test is None else len(test[1]), 'smoke': plan['smoke']}
    immutable_json(folder / 'provenance.json', provenance)
    if (folder / 'host.json').exists():
        host = read(folder / 'host.json')
        assert host['vector_file_sha256'] == file_hash(folder / 'host.npy')
        start = np.load(folder / 'host.npy', allow_pickle=False)
    else:
        then = time.perf_counter()
        with engine.count_work() as work:
            model, history = engine.training.fit_standardized_staged(*standardized[0], config.host_config())
        start = engine.vector(model)
        finite(start)
        buffer = io.BytesIO()
        np.save(buffer, start, allow_pickle=False)
        temporary = folder / 'host.npy.tmp'
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(folder / 'host.npy')
        host = {'work': dict(work), 'fit_seconds': time.perf_counter() - then,
                'vector_file_sha256': file_hash(folder / 'host.npy'), 'epochs': history['epochs'],
                'validation': scores(model, scaler, config.ernn, V, z)}
        if stage == 'confirmation':
            host['test'] = scores(model, scaler, config.ernn, *test, truth)
        atomic_json(folder / 'host.json', host)
        log(f'{folder.name} Host: {host["fit_seconds"]:.1f}s')
    failures = []
    if stage == 'development':
        arms = [(method, params) for method in BASELINES for params in plan['baseline_grids'][method]]
    else:
        locked = read(output / 'locked_selection.json')
        assert locked['signature'] == signature
        arms = [(method, locked['selected'][setting].get(method, {})) for method in METHODS]
    for method, params in arms:
        result = run_arm(folder, method, params, start, host, standardized, scaler, config,
                         (V, z), test, truth, stage, plan, signature)
        if result['status'] != 'ok':
            failures.append({'method': method, 'params': params, 'error': result['error']})
    atomic_json(complete_path, {'signature': signature, 'stage': stage, 'setting': setting,
                'rep': rep, 'smoke': plan['smoke'], 'arm_failures': failures})
    return {'group': folder.name, 'cached': False, 'arm_failures': len(failures)}


def run_stage(output, stage, plan, signature, workers):
    reps = plan['development_replications'] if stage == 'development' else plan['replications']
    tasks = [(str(output), stage, setting, rep, plan, signature)
             for rep in range(reps) for setting in SETTINGS]
    log(f'{stage}: {len(tasks)} groups, workers={workers}; completed arms will be reused.')
    then, errors = time.perf_counter(), []
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
    try:
        pending = {pool.submit(run_group, task): task for task in tasks}
        for index, future in enumerate(as_completed(pending), 1):
            try:
                result = future.result()
                log(f'{stage} [{index}/{len(tasks)}] {result}')
            except Exception as error:
                task = pending[future]
                item = {'stage': stage, 'setting': task[2], 'rep': task[3],
                        'error': repr(error), 'traceback': traceback.format_exc()}
                errors.append(item)
                log(f'FAILED {item}')
    except BaseException:
        for process in list(pool._processes.values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    assert read(output / 'source_manifest.json')['sources'] == source_hashes(), 'Source changed during execution.'
    atomic_json(output / f'{stage}_run_status.json', {'tasks': len(tasks), 'errors': errors,
                'elapsed_seconds': time.perf_counter() - then, 'signature': signature})
    if errors:
        raise RuntimeError(f'{stage}: {len(errors)} failed groups; inspect status and resume the same command.')


def select(output, plan, signature):
    selected, audit = {}, []
    for setting in SETTINGS:
        selected[setting] = {}
        for method in BASELINES:
            candidates = []
            for params in plan['baseline_grids'][method]:
                values, failures = [], []
                for rep in range(plan['development_replications']):
                    folder = group_folder(output, 'development', setting, rep)
                    assert read(folder / 'complete.json')['signature'] == signature
                    result = read(folder / f'{method}_{fingerprint(params)[:12]}.json')
                    assert result['signature'] == signature and result['params'] == params
                    assert all('test' not in p for p in result['points']), 'Development used test evaluation.'
                    if result['status'] == 'ok':
                        assert result['points'][-1]['round'] == plan['development_rounds']
                        values.append(result['points'][-1]['validation']['ips'])
                    else:
                        failures.append({'rep': rep, 'error': result['error']})
                score = statistics.mean(values) if len(values) == plan['development_replications'] else None
                audit.append({'setting': setting, 'method': method, 'params': params,
                              'validation_ips': values, 'failures': failures, 'mean_validation_ips': score})
                if score is not None:
                    candidates.append((score, json.dumps(params, sort_keys=True), params))
            if not candidates:
                raise RuntimeError(f'No fully successful development candidate for {setting}/{method}.')
            selected[setting][method] = min(candidates, key=lambda row: (row[0], row[1]))[2]
    immutable_json(output / 'locked_selection.json', {'signature': signature, 'selected': selected})
    atomic_json(output / 'baseline_selection_audit.json', audit)
    log('Baseline settings locked before confirmation. ' + json.dumps(selected, sort_keys=True))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(output, plan, signature):
    rows, failures, groups = [], [], 0
    for setting in SETTINGS:
        for rep in range(plan['replications']):
            folder = group_folder(output, 'confirmation', setting, rep)
            if not (folder / 'complete.json').exists():
                failures.append({'setting': setting, 'rep': rep, 'error': 'incomplete group'})
                continue
            complete = read(folder / 'complete.json')
            assert complete['signature'] == signature
            groups += 1
            host = read(folder / 'host.json')
            start = np.load(folder / 'host.npy', allow_pickle=False)
            initial = fingerprint(start.tolist())
            common = {'setting': setting, 'rep': rep, 'training_n': plan['training_n'], 'smoke': plan['smoke']}
            rows.append({**common, 'method': 'host', 'round': 0, **host['test'],
                         'gradient_observations': host['work']['gradient_observations'],
                         'payload_mib': 0.0, 'fit_seconds': host['fit_seconds']})
            locked = read(output / 'locked_selection.json')['selected'][setting]
            for method in METHODS:
                params = locked.get(method, {})
                result = read(folder / f'{method}_{fingerprint(params)[:12]}.json')
                assert result['signature'] == signature and result['initial_sha256'] == initial
                if result['status'] != 'ok':
                    failures.append({**common, 'method': method, 'error': result['error']})
                for point in result['points']:
                    finite(point['test'].values())
                    rows.append({**common, 'method': method, 'round': point['round'], **point['test'],
                                 'validation_ips': point['validation']['ips'],
                                 'gradient_observations': point['gradient_observations'],
                                 'forward_observations': point['forward_observations'],
                                 'payload_mib': point['payload_bytes'] / 2**20,
                                 'fit_seconds': point['fit_seconds']})
    write_csv(output / 'all_checkpoint_results.csv', rows)
    paired = []
    for endpoint in sorted(set([plan['primary_round'], plan['descriptive_round']])):
        summaries = []
        for setting in SETTINGS:
            for method in ('host', *METHODS):
                if method == 'zero' and endpoint > plan['zero_correction_rounds']:
                    continue
                filtered = [r for r in rows if r['setting'] == setting and r['method'] == method
                            and r['round'] == (0 if method == 'host' else endpoint)]
                values = [r['extremile_rmse'] for r in filtered]
                valid = len(values) == plan['replications']
                summaries.append({'setting': setting, 'method': method, 'round': endpoint,
                                  'count': len(values), 'expected_count': plan['replications'],
                                  'mean': statistics.mean(values) if valid else None,
                                  'mcse': statistics.stdev(values) / math.sqrt(len(values)) if valid else None})
            ours = {r['rep']: r['extremile_rmse'] for r in rows if r['setting'] == setting
                    and r['method'] == 'extremilefed' and r['round'] == endpoint}
            for method in BASELINES:
                baseline = {r['rep']: r['extremile_rmse'] for r in rows if r['setting'] == setting
                            and r['method'] == method and r['round'] == endpoint}
                reps = sorted(ours.keys() & baseline.keys())
                values = [baseline[r] - ours[r] for r in reps]
                valid = len(values) == plan['replications']
                paired.append({'setting': setting, 'round': endpoint, 'baseline': method,
                               'count': len(values), 'expected_count': plan['replications'],
                               'mean_difference_baseline_minus_extremilefed': statistics.mean(values) if valid else None,
                               'mcse_difference': statistics.stdev(values) / math.sqrt(len(values)) if valid else None,
                               'extremilefed_wins': sum(v > 0 for v in values),
                               'ties': sum(v == 0 for v in values)})
        write_csv(output / f'summary_round{endpoint}.csv', summaries)
    write_csv(output / 'paired_differences.csv', paired)
    atomic_json(output / 'completion_report.json', {'signature': signature,
                'completed_groups': groups, 'expected_groups': len(SETTINGS) * plan['replications'],
                'all_initial_parameters_identical_within_each_group': True,
                'failures': failures, 'success': not failures and groups == len(SETTINGS) * plan['replications'],
                'smoke': plan['smoke']})
    if failures:
        raise RuntimeError(f'{len(failures)} confirmation failures/incomplete groups retained in completion_report.json.')
    log(f'ALL DONE: {groups} groups; summaries and completion_report.json saved in {output}')


def pipeline(output, plan, workers, stage):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This output directory is already running in another terminal.')
        signature = prepare(output, plan)
        if stage in ('all', 'development'):
            run_stage(output, 'development', plan, signature, workers)
        if stage in ('all', 'select'):
            select(output, plan, signature)
        if stage in ('all', 'confirmation'):
            assert read(output / 'locked_selection.json')['signature'] == signature
            run_stage(output, 'confirmation', plan, signature, workers)
        if stage in ('all', 'summary'):
            summarize(output, plan, signature)


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=['all', 'prepare', 'development', 'select', 'confirmation', 'summary'], default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run', action='store_true', help='Print protocol without starting fits or writing results')
    parser.add_argument('--smoke', action='store_true', help='Run a temporary tiny end-to-end check, then check resume')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    plan = protocol(args.smoke)
    if args.dry_run:
        print(json.dumps({'output': str(args.output), 'protocol': plan}, indent=2, ensure_ascii=False))
        return
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix='extremile_fair_confirmation_smoke_') as temporary:
            output = Path(temporary)
            pipeline(output, plan, min(args.workers, 2), 'all')
            arm = next((output / 'confirmation').glob('*/extremilefed_*.json'))
            original_mtime = arm.stat().st_mtime_ns
            pipeline(output, plan, min(args.workers, 2), 'all')
            assert arm.stat().st_mtime_ns == original_mtime, 'Resume overwrote a completed method.'
            assert read(output / 'completion_report.json')['success']
            print('SMOKE AND RESUME CHECKS PASSED; no full-size experiments were run.', flush=True)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    with (args.output / f'run_{timestamp}.log').open('a', buffering=1) as logfile:
        terminal_out, terminal_error = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(terminal_out, logfile), Tee(terminal_error, logfile)
        try:
            log(f'Output: {args.output}; stage={args.stage}; N=100000; repetitions=30; three settings.')
            pipeline(args.output, plan, args.workers, args.stage)
        finally:
            sys.stdout, sys.stderr = terminal_out, terminal_error


if __name__ == '__main__':
    main()
