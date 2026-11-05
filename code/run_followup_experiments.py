"""Frozen, paired supplementary experiments; --run is required to train.

The original statistical package and its 1,200 main results remain immutable.
Depth covers all 12 settings; other components cover five declared settings.
"""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing import resource_tracker
from pathlib import Path
import signal
import sys
import time
import traceback
import warnings

import numpy as np
import torch

import continue_main_simulations as original
import continue_main_simulations_parallel as parallel
from extremilefed import experiments, training
from extremilefed.config import QuadratureConfig, TAU_LEVELS
from extremilefed.development import implementation_hash
from extremilefed.experiments import SimulationJob
from extremilefed.metrics import extremile_errors
from extremilefed.model import predict_extremiles
from followup_depth import depth_architecture, parameter_count

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
REPRESENTATIVE = ((1, 'normal'), (1, 'lognormal'), (2, 'contaminated'), (2, 'lognormal'), (3, 't3'))
ALL_SETTINGS = tuple((s, e) for s in (1, 2, 3) for e in original.ERRORS)
METHOD_NAMES = dict(zip(experiments.ALL_METHODS, (
    'Centralized ERNN', 'Pooled linear Extremile-SL', 'Host ERNN', 'One-shot ERNN',
    'FedAvg-ERNN', 'FedProx-ERNN', 'SCAFFOLD-ERNN', 'ExtremileFed')))
CHE = ('centralized', 'host', 'extremilefed')
VERSION = 1


def digest(value):
    return hashlib.sha256(original.canonical(value).encode()).hexdigest()


def source_hashes():
    return {'core': implementation_hash(), **{
        p.name: original.sha256(p) for p in (
            Path(__file__), CODE / 'followup_depth.py',
            CODE / 'continue_main_simulations.py', CODE / 'continue_main_simulations_parallel.py')}}


@dataclass(frozen=True)
class Task:
    variant: str
    family: str
    job: SimulationJob
    config: object
    depth: int = 1
    methods: tuple = CHE
    smoke: bool = False

    @property
    def key(self):
        return self.variant + '/' + self.job.key

    def specification(self):
        return {'variant': self.variant, 'family': self.family, 'job': asdict(self.job),
                'config': asdict(self.config), 'depth': self.depth,
                'methods': list(self.methods), 'smoke': self.smoke}


def smoke_config(config):
    neural = replace(config.ernn, hidden_units=3, second_hidden_units=None,
                     max_epochs=2, min_epochs=1, patience=3, refinement_epochs=1,
                     batch_size=64, quadrature=QuadratureConfig(4, 8))
    return replace(config, ernn=neural, host_epochs=2, host_patience=3,
                   host_refinement_epochs=1, surrogate_epochs=2, surrogate_patience=3,
                   fedavg_local_epochs=1, fedprox_local_epochs=1, scaffold_local_epochs=1)


def build_tasks(configs, replications=30, smoke=False):
    if not 1 <= replications <= 100:
        raise ValueError('replications must be between 1 and 100 for paired main seeds')
    tasks = []
    for scenario, error in ALL_SETTINGS:
        base = smoke_config(configs[(scenario, error)]) if smoke else configs[(scenario, error)]
        variants = [('reference', 'reference', base, 1, CHE, 20)]
        for depth in (2, 3, 4):
            conf = replace(base, ernn=replace(base.ernn, second_hidden_units=base.ernn.hidden_units))
            variants.append((f'depth_{depth}', 'depth', conf, depth, CHE, 20))
        if (scenario, error) in REPRESENTATIVE:
            for clients in (10, 50):
                variants.append((f'clients_{clients}', 'clients', base, 1, ('host', 'extremilefed'), clients))
            for rounds in (1, 2, 3):
                if rounds != base.rounds:
                    conf = replace(base, rounds=rounds,
                                   surrogate_learning_rates=(.001, .001, .0003)[:rounds])
                    variants.append((f'rounds_{rounds}', 'rounds', conf, 1, ('extremilefed',), 20))
            for nodes in ((3, 6) if smoke else (16, 64)):
                neural = replace(base.ernn, quadrature=replace(base.ernn.quadrature, training_nodes=nodes))
                variants.append((f'training_nodes_{nodes}', 'training_nodes', replace(base, ernn=neural), 1, CHE, 20))
            for bandwidth in (.01, .05, .10):
                variants.append((f'h_{bandwidth:.2f}', 'smoothing', replace(base, ernn=replace(base.ernn, h=bandwidth)), 1, CHE, 20))
            if base.ernn.eta != 0:
                variants.append(('eta_zero', 'penalty', replace(base, ernn=replace(base.ernn, eta=0.)), 1, CHE, 20))
            variants.append(('prediction_nodes', 'quadrature', base, 1, CHE, 20))
            variants.append(('gradient_budget', 'budget', base, 1,
                             ('extremilefed', 'fedavg', 'fedprox', 'scaffold'), 20))
        for variant, family, config, depth, methods, clients in variants:
            config.validate()
            for replication in range(replications):
                job = SimulationJob(scenario, error, replication, 200 if smoke else 100000,
                                    24 if smoke else 5000, clients, 20260903)
                tasks.append(Task(variant, family, job, config, depth, methods, smoke))
    return tasks


def task_selected(task, suite):
    return (suite == 'all' or task.family == 'reference' or
            (suite == 'depth' and task.family == 'depth') or
            (suite == 'sensitivity' and task.family in ('clients', 'rounds', 'training_nodes', 'smoothing', 'penalty')) or
            task.family == suite)


@contextmanager
def count_work():
    """Count observations passed through loss with/without autograd enabled.

    Includes the complete-gradient diagnostics; this is not a FLOP/time measure.
    """
    previous = training.data_loss
    counts = {'gradient_observations': 0, 'forward_only_observations': 0,
              'gradient_calls': 0, 'forward_only_calls': 0}
    def counted(model, X, *args, **kwargs):
        kind = 'gradient' if torch.is_grad_enabled() else 'forward_only'
        counts[kind + '_observations'] += len(X)
        counts[kind + '_calls'] += 1
        return previous(model, X, *args, **kwargs)
    training.data_loss = counted
    try:
        yield counts
    finally:
        training.data_loss = previous


def quadrature_run(task):
    original_metrics = experiments._neural_metrics
    evaluations = []
    grid = (4, 8, 16) if task.smoke else (64, 128, 256)
    def collect(fit, X, y, tau, levels, truth):
        ordinary = original_metrics(fit, X, y, tau, levels, truth)
        predictions = {g: predict_extremiles(fit, X, tau, quadrature=g) for g in grid}
        if any(not np.isfinite(p).all() for p in predictions.values()):
            raise FloatingPointError('Nonfinite quadrature prediction')
        reference = predictions[task.config.ernn.quadrature.prediction_nodes]
        summary = {}
        for g, values in predictions.items():
            summary[str(g)] = {
                **extremile_errors(values, truth),
                'prediction_rmse_gap_to_G128' if not task.smoke else 'prediction_rmse_gap_to_G8':
                    float(np.sqrt(np.mean((values-reference)**2))),
                **{f'extremile_rmse_tau_{t:.2f}': float(np.sqrt(np.mean((values[:, j]-truth[:, j])**2)))
                   for j, t in enumerate(tau)}}
        evaluations.append(summary)
        return ordinary
    experiments._neural_metrics = collect
    try:
        result = experiments.run_job(task.job, task.config, task.methods)
    finally:
        experiments._neural_metrics = original_metrics
    if len(evaluations) != len(CHE):
        raise ValueError('Missing same-fit quadrature evaluation')
    result['same_fit_quadrature'] = {METHOD_NAMES[m]: e for m, e in zip(CHE, evaluations)}
    return result


def budget_run(task):
    with count_work() as ef_work:
        result = experiments.run_job(task.job, task.config, ('extremilefed',))
    # Each client fit also evaluates one complete terminal data gradient.
    # Count that pass before rounding the number of optimization epochs.
    local_epochs = max(1, math.ceil(ef_work['gradient_observations'] /
                                  (task.job.n_train * task.config.rounds)) - 1)
    config = replace(task.config, fedavg_local_epochs=local_epochs,
                     fedprox_local_epochs=local_epochs, scaffold_local_epochs=local_epochs)
    measured = {'ExtremileFed': dict(ef_work)}
    for method in ('fedavg', 'fedprox', 'scaffold'):
        with count_work() as work:
            baseline = experiments.run_job(task.job, config, (method,))
        result['methods'].update(baseline['methods'])
        measured[METHOD_NAMES[method]] = dict(work)
    result['budget_matching'] = {
        'criterion': 'observations passed through data_loss with autograd enabled, including gradient diagnostics',
        'target_gradient_observations': ef_work['gradient_observations'],
        'baseline_local_epochs': local_epochs, 'actual_work': measured,
        'gradient_observation_difference': {
            name: work['gradient_observations'] - ef_work['gradient_observations']
            for name, work in measured.items()},
        'baseline_config': asdict(config), 'baseline_config_fingerprint': config.fingerprint(),
        'not_matched': ['FLOPs', 'forward-only objective evaluation', 'wall-clock time'],
        'learning_rates': 'original selected baseline rates; no test-based reselection',
        'rounding': 'max(1, ceil(target/(N*T))-1); one terminal gradient pass per client fit; report measured mismatch'}
    return result


def run_task(task):
    linked = None
    if task.family == 'reference' and not task.smoke:
        path = original.OUTPUT / (task.job.key + '.json')
        result = json.loads(path.read_text())
        original.verify_result(result, task.job, task.config)
        linked = {'path': str(path), 'sha256': original.sha256(path)}
        result['methods'] = {METHOD_NAMES[m]: result['methods'][METHOD_NAMES[m]] for m in CHE}
    else:
        with depth_architecture(task.depth):
            if task.family == 'quadrature':
                result = quadrature_run(task)
            elif task.family == 'budget':
                result = budget_run(task)
            else:
                result = experiments.run_job(task.job, task.config, task.methods)
    # The ordinary core config encodes one/two layers only. This complete spec,
    # its digest and depth extension source hash identify every deeper model.
    # Read feature count from the generator rather than assuming its dimension.
    X, _ = experiments.generate_data(2, task.job.scenario, task.job.error, 17)
    p = X.shape[1]
    result['followup'] = {
        'version': VERSION, 'specification': task.specification(),
        'task_fingerprint': digest(task.specification()), 'sources': source_hashes(),
        'hidden_layers': task.depth, 'width_per_layer': task.config.ernn.hidden_units,
        'parameter_count': parameter_count(p, task.config.ernn.hidden_units, task.depth),
        'reference_main_result': linked,
        'study_note': 'fixed width and prescribed optimizer budgets; capacity and computation may change',
        'statistical_status': 'supplementary paired sensitivity, not a newly selected main configuration'}
    verify_task_result(result, task)
    return result


def verify_task_result(result, task):
    original.finite_tree(result)
    if result.get('schema_version') != experiments.RESULT_SCHEMA or result.get('job') != asdict(task.job):
        raise ValueError('Schema or job mismatch: ' + task.key)
    if result.get('config_fingerprint') != task.config.fingerprint() or original.canonical(result['config']) != original.canonical(asdict(task.config)):
        raise ValueError('Configuration mismatch: ' + task.key)
    provenance = result.get('followup', {})
    if (provenance.get('task_fingerprint') != digest(task.specification()) or
        original.canonical(provenance.get('specification')) != original.canonical(task.specification()) or
        provenance.get('sources') != source_hashes()):
        raise ValueError('Supplementary protocol/source mismatch: ' + task.key)
    from extremilefed.simulation import SCENARIO_DIMENSIONS
    dimension = parameter_count(SCENARIO_DIMENSIONS[task.job.scenario],
                                task.config.ernn.hidden_units, task.depth)
    if (provenance.get('version') != VERSION or
            provenance.get('hidden_layers') != task.depth or
            provenance.get('width_per_layer') != task.config.ernn.hidden_units or
            provenance.get('parameter_count') != dimension):
        raise ValueError('Architecture metadata mismatch: ' + task.key)
    expected_seed = task.job.seed + 100000 * task.job.scenario + 1000 * task.job.replication
    if result['seeds'] != dict(training_data=expected_seed, test_data=expected_seed+1,
                               partition=expected_seed+2, model=task.config.ernn.seed):
        raise ValueError('Unpaired data/model seeds: ' + task.key)
    if result['tau_levels'] != list(TAU_LEVELS) or set(result['methods']) != {METHOD_NAMES[m] for m in task.methods}:
        raise ValueError('Incomplete methods/levels: ' + task.key)
    from extremilefed.quadrature import gauss_legendre_unit
    quadrature = result.get('quadrature', {})
    nodes, weights = gauss_legendre_unit(task.config.ernn.quadrature.prediction_nodes)
    if (quadrature.get('training_nodes') != task.config.ernn.quadrature.training_nodes or
            quadrature.get('prediction_nodes') != len(nodes) or
            len(quadrature.get('evaluation_levels', [])) != len(nodes) or
            len(quadrature.get('evaluation_weights', [])) != len(weights) or
            not np.allclose(quadrature['evaluation_levels'], nodes, rtol=0, atol=1e-14) or
            not np.allclose(quadrature['evaluation_weights'], weights, rtol=0, atol=1e-14)):
        raise ValueError('Quadrature metadata mismatch: ' + task.key)
    sizes = result['partition_sizes']
    if len(sizes) != task.job.clients or sum(sizes) != task.job.n_train or min(sizes) <= 0 or max(sizes)-min(sizes) > 1:
        raise ValueError('Invalid partitions: ' + task.key)
    expected_metrics = {'extremile_mae', 'extremile_rmse', 'integrated_pinball', 'coverage_error', 'crossing_rate'}
    expected_metrics |= {f'extremile_{m}_tau_{t:.2f}' for m in ('mae', 'rmse') for t in TAU_LEVELS}
    for name, method in result['methods'].items():
        metrics = method['metrics']
        if not expected_metrics <= set(metrics) or any(not isinstance(metrics[k], (int, float)) or metrics[k] < 0 for k in expected_metrics):
            raise ValueError('Missing/invalid metrics: ' + task.key)
        if not 0 <= metrics['crossing_rate'] <= 1 or not 0 <= metrics['coverage_error'] <= 1:
            raise ValueError('Invalid probability diagnostic')
        level_rmse = [metrics[f'extremile_rmse_tau_{t:.2f}'] for t in TAU_LEVELS]
        level_mae = [metrics[f'extremile_mae_tau_{t:.2f}'] for t in TAU_LEVELS]
        if not math.isclose(metrics['extremile_rmse'], math.sqrt(np.mean(np.square(level_rmse))), rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(metrics['extremile_mae'], np.mean(level_mae), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Aggregate metric identity failed')
        if method['elapsed_seconds'] < 0 or method['communication_mb'] < 0:
            raise ValueError('Negative resource count')
        rounds_expected = (0 if name in ('Centralized ERNN', 'Host ERNN', 'Pooled linear Extremile-SL')
                           else 1 if name == 'One-shot ERNN' else task.config.rounds)
        messages = (0 if rounds_expected == 0 else task.job.clients - 1 if name == 'One-shot ERNN'
                    else (4 if name == 'SCAFFOLD-ERNN' else 2) * rounds_expected * (task.job.clients - 1))
        bytes_per = 8 if task.config.ernn.dtype == 'float64' else 4
        payload_expected = messages * dimension * bytes_per / 1024.0**2
        if method.get('rounds') != rounds_expected or not math.isclose(
                method['communication_mb'], payload_expected, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Rounds/communication metadata mismatch: ' + task.key)
        if name == 'ExtremileFed':
            rounds = method['gradient_diagnostics']
            if len(rounds) != task.config.rounds or any(r['round'] != i or not r['finite_step_audit']['numerically_valid'] for i, r in enumerate(rounds, 1)):
                raise ValueError('Invalid CSL diagnostics: ' + task.key)
    linked = provenance.get('reference_main_result')
    if task.family == 'reference' and not task.smoke:
        source = original.OUTPUT / (task.job.key + '.json')
        if linked != {'path': str(source), 'sha256': original.sha256(source)}:
            raise ValueError('Main reference changed')
    elif linked is not None:
        raise ValueError('Only formal reference tasks may link main results')
    if task.family == 'quadrature':
        q = result.get('same_fit_quadrature', {})
        grid = {'4', '8', '16'} if task.smoke else {'64', '128', '256'}
        if set(q) != {METHOD_NAMES[m] for m in CHE} or any(set(values) != grid for values in q.values()):
            raise ValueError('Incomplete same-fit quadrature records')
        reference_node = str(task.config.ernn.quadrature.prediction_nodes)
        gap_name = 'prediction_rmse_gap_to_G8' if task.smoke else 'prediction_rmse_gap_to_G128'
        required = {'extremile_mae', 'extremile_rmse', gap_name} | {
            f'extremile_rmse_tau_{tau:.2f}' for tau in TAU_LEVELS}
        for name, evaluations in q.items():
            for values in evaluations.values():
                if not required <= set(values) or any(
                        not isinstance(values[key], (int, float)) or values[key] < 0 for key in required):
                    raise ValueError('Invalid same-fit quadrature metric')
                aggregate = math.sqrt(np.mean([values[f'extremile_rmse_tau_{tau:.2f}']**2 for tau in TAU_LEVELS]))
                if not math.isclose(values['extremile_rmse'], aggregate, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Same-fit quadrature aggregate mismatch')
            if evaluations[reference_node][gap_name] != 0 or any(not math.isclose(
                    evaluations[reference_node][metric], result['methods'][name]['metrics'][metric],
                    rel_tol=1e-10, abs_tol=1e-12) for metric in ('extremile_mae', 'extremile_rmse')):
                raise ValueError('Same-fit reference quadrature mismatch')
    if task.family == 'budget':
        b = result.get('budget_matching', {})
        if set(b.get('actual_work', {})) != {METHOD_NAMES[m] for m in task.methods} or b.get('target_gradient_observations', 0) <= 0:
            raise ValueError('Missing measured gradient budget')
        target = b['target_gradient_observations']
        expected_epochs = max(1, math.ceil(target / (task.job.n_train * task.config.rounds)) - 1)
        expected_config = replace(task.config, fedavg_local_epochs=expected_epochs,
                                  fedprox_local_epochs=expected_epochs, scaffold_local_epochs=expected_epochs)
        if (b.get('baseline_local_epochs') != expected_epochs or
                b.get('baseline_config_fingerprint') != expected_config.fingerprint() or
                original.canonical(b.get('baseline_config')) != original.canonical(asdict(expected_config))):
            raise ValueError('Measured gradient budget/config mismatch')
        for name, work in b['actual_work'].items():
            keys = {'gradient_observations', 'forward_only_observations', 'gradient_calls', 'forward_only_calls'}
            if set(work) != keys or any(type(work[key]) is not int or work[key] < 0 for key in keys):
                raise ValueError('Invalid measured work counts')
        if (b['actual_work']['ExtremileFed']['gradient_observations'] != target or
                b.get('gradient_observation_difference') != {
                    name: work['gradient_observations'] - target for name, work in b['actual_work'].items()}):
            raise ValueError('Measured gradient difference mismatch')


def worker_entry(task, sender):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    envelope = {'ok': False, 'task': task.key}
    seen = []
    started = time.monotonic()
    try:
        limiter = parallel.configure_threads()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always', RuntimeWarning)
            envelope['result'] = run_task(task)
        envelope['ok'] = True
    except BaseException as error:
        envelope.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        envelope['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
                                 'filename': w.filename, 'lineno': w.lineno} for w in seen]
        envelope['seconds'] = time.monotonic()-started
        try:
            sender.send(envelope)
        finally:
            sender.close()


def start_worker(task):
    ctx = mp.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=worker_entry, args=(task, sender), name=task.key)
    resource_tracker.ensure_running()
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        process.start()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        sender.close()
    return parallel.Running(task, process, receiver, time.monotonic())


def execute(tasks, output, workers, launch, manifest):
    stop = parallel.StopController()
    previous = signal.signal(signal.SIGINT, stop)
    active, failures = [], []
    index = 0
    try:
        while active or (index < len(tasks) and not stop.requested and not failures):
            while len(active) < workers and index < len(tasks) and not stop.requested and not failures:
                task = tasks[index]
                if (output / (task.key + '.json')).exists():
                    raise RuntimeError('Unexpected concurrent writer: ' + task.key)
                active.append(start_worker(task))
                index += 1
                print(f'[{index}/{len(tasks)}] START {task.key}', flush=True)
            for running in list(active):
                if running.connection.poll() or not running.process.is_alive():
                    task = running.job
                    try:
                        envelope = parallel.receive_worker(running)
                        original.atomic_json(launch / (task.key + '_execution.json'),
                                             {k: v for k, v in envelope.items() if k != 'result'})
                        if not envelope['ok'] or envelope['task'] != task.key:
                            raise RuntimeError(envelope.get('error', 'Worker identity mismatch'))
                        result = envelope['result']
                        verify_task_result(result, task)
                        result['followup_execution'] = {
                            'workers': workers, 'threads_per_worker': 1,
                            'start_method': 'spawn', 'launch': str(launch),
                            'timing_note': 'concurrent load; main reference timings keep their original regime'}
                        target = output / (task.key + '.json')
                        original.atomic_json(target, result, overwrite=False)
                        manifest['completed'][task.key] = original.sha256(target)
                        print(f'SAVED {task.key} ({envelope["seconds"]:.1f}s)', flush=True)
                    except Exception as error:
                        failures.append({'task': task.key, 'error': repr(error)})
                        manifest['failures'] = failures
                        print(f'FAILED {task.key}: {error}; draining active tasks.', file=sys.stderr, flush=True)
                    finally:
                        parallel.close_worker(running)
                        active.remove(running)
                        original.atomic_json(launch / 'manifest.json', manifest)
            if active:
                time.sleep(.1)
        if failures:
            raise RuntimeError('Some tasks failed; see ' + str(launch))
        return stop.requested
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            parallel.stop_workers(active)
        finally:
            signal.signal(signal.SIGINT, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--suite', choices=('all', 'depth', 'sensitivity', 'budget', 'quadrature'), default='all')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--replications', type=int, default=30)
    parser.add_argument('--max-new-jobs', type=int)
    parser.add_argument('--smoke', action='store_true', help='tiny non-publication preflight, separate output')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.workers <= 0 or (args.max_new_jobs is not None and args.max_new_jobs <= 0):
        parser.error('workers and max-new-jobs must be positive')
    if args.smoke:
        if args.replications != 30 and args.replications != 1:
            parser.error('smoke uses exactly one replication')
        args.replications = 1
    default = (ROOT / '工作记录与备份/2026-10-13_后续实验预检/smoke' if args.smoke
               else CODE / 'results/followup_v1')
    output = (args.output or default).resolve()
    if args.smoke and not output.is_relative_to(ROOT / '工作记录与备份'):
        parser.error('smoke output must be under 工作记录与备份')
    if not args.smoke and not output.is_relative_to(CODE / 'results'):
        parser.error('supplementary results must be under code/results')
    if output == original.OUTPUT or output.is_relative_to(original.OUTPUT) or original.OUTPUT.is_relative_to(output):
        parser.error('cannot write into or above the main result archive')
    limiter = parallel.configure_threads()
    configs, migration = original.load_local_configs()
    tasks = build_tasks(configs, args.replications, args.smoke)
    plan = {'version': VERSION, 'replications': args.replications, 'smoke': args.smoke,
            'sources': source_hashes(), 'representative_settings': list(map(list, REPRESENTATIVE)),
            'depth_settings': list(map(list, ALL_SETTINGS)), 'hidden_layer_grid': [1, 2, 3, 4],
            'prediction_node_grid': [4, 8, 16] if args.smoke else [64, 128, 256],
            'predeclared_scope': 'depth all 12; other components five representative settings',
            'inference_note': 'paired supplementary analyses; no reselection from test outcomes',
            'tasks': {t.key: digest(t.specification()) for t in tasks}}
    with original.exclusive_lock(output):
        plan_path = output / 'protocol.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Frozen protocol changed; choose a new output directory, never mix results')
        if not plan_path.exists():
            original.atomic_json(plan_path, plan, overwrite=False)
        known_paths = {t.key + '.json' for t in tasks}
        actual_paths = {str(p.relative_to(output)) for p in output.glob('*/scenario*_rep*.json')}
        if actual_paths - known_paths:
            raise RuntimeError('Unexpected result files: ' + str(sorted(actual_paths-known_paths)))
        selected = [t for t in tasks if task_selected(t, args.suite)]
        missing, existing = [], {}
        for task in tasks:
            path = output / (task.key + '.json')
            if path.exists():
                verify_task_result(json.loads(path.read_text()), task)
                existing[task.key] = original.sha256(path)
            elif task in selected:
                missing.append(task)
        counts = {family: sum(t.family == family for t in selected) for family in sorted({t.family for t in selected})}
        print(f'Scope {args.suite}: {len(selected)-len(missing)}/{len(selected)} ready; {len(missing)} missing. '
              f'workers={args.workers}, float64, one CPU thread each.', flush=True)
        print('Planned records by family: ' + json.dumps(counts), flush=True)
        print('Output: ' + str(output), flush=True)
        if not args.run:
            print('CHECK ONLY; no fitting started.', flush=True)
            return
        if args.max_new_jobs:
            missing = missing[:args.max_new_jobs]
        if not missing:
            print('Nothing to run.', flush=True)
            return
        launch = output / 'launches' / (datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
        manifest = {'status': 'running', 'started_at': parallel.now(), 'protocol_sha256': original.sha256(plan_path),
                    'sources': source_hashes(), 'environment': original.environment(),
                    'suite': args.suite, 'workers': args.workers, 'completed': {},
                    'planned': [t.key for t in missing], 'migration': migration}
        original.atomic_json(launch / 'manifest.json', manifest)
        try:
            stopped = execute(missing, output, args.workers, launch, manifest)
            manifest['status'] = 'stopped_after_current_tasks' if stopped else 'batch_complete'
        except BaseException as error:
            manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            raise
        finally:
            unchanged = all(original.sha256(output/(key+'.json')) == value for key, value in existing.items())
            manifest.update(finished_at=parallel.now(), existing_results_unchanged=unchanged,
                            sources_unchanged=source_hashes() == manifest['sources'])
            if not unchanged or not manifest['sources_unchanged']:
                manifest['status'] = 'audit_failed'
            original.atomic_json(launch / 'manifest.json', manifest)
        if manifest['status'] == 'audit_failed':
            raise RuntimeError('Source/result audit failed')
        return 130 if stopped else 0


if __name__ == '__main__':
    raise SystemExit(main())
