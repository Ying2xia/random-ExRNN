#!/usr/bin/env python3
"""Application-specific, resumable selection followed by frozen held-out evaluation.

Default is a read-only plan. --run is required for fitting. All selected artifacts
live outside extremilefed, so this driver cannot change the main simulation hash.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from extremilefed.config import ERNNConfig, FederatedConfig, QuadratureConfig, J_GRID, LAMBDA_GRID, ETA_GRID
from extremilefed.development import DevelopmentProtocol, convergence, digest, exclusive_run, implementation_hash, minimum_candidate, score_fit, write_json
from extremilefed.baselines import train_fedavg, train_scaffold
from extremilefed.federated import partition_indices, train_extremilefed
from extremilefed.real_data import PreparedDataset, _preprocessor
from extremilefed.real_cli import evaluate_dataset
from extremilefed.experiments import ALL_METHODS
from extremilefed.reproducibility import load_frozen_config
from extremilefed.training import train_ernn
from extremilefed.tuning import select_rounds_one_se

torch.set_num_threads(1)
PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {'core': implementation_hash(), 'driver': sha(__file__)}


def read_required_csv(path, required):
    frame = pd.read_csv(path)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f'{Path(path).name}: missing required columns: {", ".join(missing)}')
    return frame


@dataclass
class RawSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    numeric: list
    categorical: list
    clients: int
    chronological: bool
    metadata: dict
    raw_hashes: dict
    train_times: np.ndarray | None = None


def load_raw(dataset, data_root):
    """Match the published fixed outer split before any fitted transformations."""
    seed = 20260903
    data_root = Path(data_root)
    if dataset == 'fremtpl2':
        paths = [data_root / 'freMTPL2freq.csv', data_root / 'freMTPL2sev.csv']
        frequency = read_required_csv(paths[0], {'IDpol', 'ClaimNb', 'Exposure', 'Area',
            'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region'})
        severity = read_required_csv(paths[1], {'IDpol', 'ClaimAmount'})
        if frequency['IDpol'].duplicated().any():
            raise ValueError('frequency table must have one row per policy')
        severity = severity.loc[np.isfinite(severity.ClaimAmount) & (severity.ClaimAmount > 0)]
        frame = frequency.merge(severity.groupby('IDpol', as_index=False).ClaimAmount.sum(), on='IDpol', how='inner')
        frame = frame.loc[np.isfinite(frame.ClaimAmount) & (frame.ClaimAmount > 0)].copy()
        y = frame.pop('ClaimAmount').to_numpy(float)
        frame = frame.drop(columns=['IDpol', 'ClaimNb'])
        bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
        tr, tmp, yt, yz, _, zb = train_test_split(frame, y, bins, test_size=.30, random_state=seed, stratify=bins)
        va, te, yv, ye = train_test_split(tmp, yz, test_size=.50, random_state=seed + 1, stratify=zb)
        categorical, clients, chronological, times = ['Area', 'VehBrand', 'VehGas', 'Region'], 10, False, None
        metadata = {'dataset': dataset, 'seed': seed, 'split': '70/15/15 response-decile stratified', 'row_indices': {'train': tr.index.tolist(), 'valid': va.index.tolist(), 'test': te.index.tolist()}}
    elif dataset == 'beijing':
        paths = sorted((data_root / 'beijing').glob('*.csv'))
        if not paths:
            raise FileNotFoundError('no Beijing CSV inputs')
        required = {'year', 'month', 'day', 'hour', 'PM2.5', 'TEMP', 'PRES', 'DEWP',
            'RAIN', 'wd', 'WSPM', 'station'}
        frame = pd.concat([read_required_csv(p, required) for p in paths], ignore_index=True)
        frame = frame.loc[np.isfinite(frame['PM2.5'])].copy()
        frame['timestamp'] = pd.to_datetime(frame[['year', 'month', 'day', 'hour']])
        frame = frame.sort_values('timestamp').reset_index(drop=True)
        y = frame.pop('PM2.5').to_numpy(float)
        timestamps = frame.timestamp.to_numpy()
        frame = frame[['year', 'month', 'day', 'hour', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd', 'station']]
        a = int(np.searchsorted(timestamps, timestamps[int(.70 * len(y))], side='left'))
        b = int(np.searchsorted(timestamps, timestamps[int(.85 * len(y))], side='left'))
        if not 0 < a < b < len(y):
            raise ValueError('insufficient distinct timestamps for the fixed outer split')
        tr, va, te, yt, yv, ye = frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:], y[:a], y[a:b], y[b:]
        categorical, clients, chronological, times = ['wd', 'station'], 20, True, timestamps[:a]
        metadata = {'dataset': dataset, 'seed': seed, 'split': 'chronological 70/15/15; common timestamps stay together', 'time_boundaries': [str(timestamps[a]), str(timestamps[b])]}
    else:
        raise ValueError('unknown application dataset: ' + str(dataset))
    metadata.update({'rows': len(y), 'split_sizes': [len(yt), len(yv), len(ye)], 'clients': clients})
    return RawSplits(tr, va, te, yt, yv, ye, [c for c in tr if c not in categorical], categorical, clients, chronological, metadata, {str(p.resolve()): sha(p) for p in paths}, times)


@dataclass(frozen=True)
class Protocol:
    smoke: bool = False
    seed: int = 20261013
    cv_rows: int = 5000
    eta_rows: int = 3000
    development_rows: int = 10000
    validation_rows: int = 1000
    repeats: int = 3
    j_grid: tuple = J_GRID
    lambda_grid: tuple = LAMBDA_GRID
    eta_grid: tuple = ETA_GRID
    eta_extension: float = 1000.
    local_epochs: tuple = (1, 5, 10)
    local_lr: tuple = (.0003, .001, .003, .01)
    prox_grid: tuple = (.0001, .001, .01, .1)
    epochs: int = 800
    refinement: int = 200
    surrogate_epochs: int = 120
    crossing_threshold: float = .005

    @classmethod
    def small(cls):
        return cls(smoke=True, cv_rows=90, eta_rows=80, development_rows=100,
            validation_rows=30, repeats=2, j_grid=(3,), lambda_grid=(.001,),
            eta_grid=(0., 1., 1000.), local_epochs=(1,), local_lr=(.001,),
            prox_grid=(.001,), epochs=2, refinement=1, surrogate_epochs=2)


def role_seed(p, role):
    return 1_000_000_000 + int(digest([p.seed, role])[:12], 16) % 1_000_000_000


def subset(n, count, seed):
    return np.sort(np.random.default_rng(seed).choice(n, min(n, count), replace=False))


def cv_indices(raw, p):
    rows = subset(len(raw.train), p.cv_rows, role_seed(p, 'cv_rows'))
    if not raw.chronological:
        return [(rows[a], rows[b]) for a, b in KFold(5, shuffle=True, random_state=role_seed(p, 'cv_folds')).split(rows)]
    # Split unique hours rather than rows: no station at a validation hour enters fitting.
    unique = np.unique(raw.train_times[rows])
    if len(unique) < 6:
        raise ValueError('five forward folds require at least six distinct training hours')
    blocks = np.array_split(unique, 6)
    return [(rows[np.isin(raw.train_times[rows], np.concatenate(blocks[:i]))],
             rows[np.isin(raw.train_times[rows], blocks[i])]) for i in range(1, 6)]


def transform_pair(raw, train_rows, valid_rows, *, cv=False):
    fitting = raw.train.iloc[train_rows]
    validation = raw.train.iloc[valid_rows] if cv else raw.valid.iloc[valid_rows]
    encoder = _preprocessor(fitting, raw.numeric, raw.categorical)
    return {'X': np.asarray(encoder.fit_transform(fitting)), 'y': raw.y_train[train_rows],
        'V': np.asarray(encoder.transform(validation)), 'z': (raw.y_train if cv else raw.y_valid)[valid_rows],
        'training_rows': np.asarray(train_rows), 'validation_rows': np.asarray(valid_rows)}


def full_prepared(raw):
    encoder = _preprocessor(raw.train, raw.numeric, raw.categorical)
    X = np.asarray(encoder.fit_transform(raw.train))
    seed = 20260905 if raw.metadata['dataset'] == 'fremtpl2' else 20260903
    return PreparedDataset(X, raw.y_train, np.asarray(encoder.transform(raw.valid)), raw.y_valid,
        np.asarray(encoder.transform(raw.test)), raw.y_test,
        partition_indices(len(X), raw.clients, seed), {**raw.metadata, 'features_after_encoding': X.shape[1]})


def base_config(p):
    neural = ERNNConfig(seed=role_seed(p, 'final_model'), max_epochs=p.epochs,
        min_epochs=min(30, p.epochs), patience=p.epochs + 1, refinement_epochs=p.refinement,
        quadrature=QuadratureConfig(6, 12) if p.smoke else QuadratureConfig(),
        batch_size=64 if p.smoke else 2048)
    return FederatedConfig(ernn=neural, rounds=1, surrogate_epochs=p.surrogate_epochs,
        surrogate_learning_rate=.001, surrogate_learning_rates=(.001, .001, .0003),
        surrogate_patience=20, host_epochs=p.epochs, host_patience=p.epochs + 1,
        host_learning_rate=.01, host_refinement_epochs=p.refinement,
        host_refinement_learning_rate=.001)


def decode_config(value):
    value = dict(value)
    neural = dict(value.pop('ernn'))
    neural['quadrature'] = QuadratureConfig(**neural['quadrature'])
    value['surrogate_learning_rates'] = tuple(value['surrogate_learning_rates'])
    return FederatedConfig(ernn=ERNNConfig(**neural), **value)


def fit_job(job):
    """Spawned workers only receive development blocks; no outer-test arrays."""
    torch.set_num_threads(1)
    conf = decode_config(job['config'])
    finite_policy = DevelopmentProtocol(numerical_acceptance='finite_step')
    with np.load(job['data_path'], allow_pickle=False) as block:
        X, y, V, z = [block[k] for k in ('X', 'y', 'V', 'z')]
    method = job['method']
    scores = None
    if method == 'centralized':
        fit = train_ernn(X, y, conf.ernn)
        checks = [convergence(fit.history, finite_policy)]
    else:
        parts = [(X[i], y[i]) for i in partition_indices(len(y), job['clients'], job['partition_seed'])]
        if method == 'extremilefed':
            result = train_extremilefed(parts, conf)
            checks = [convergence(result.history['host'], finite_policy)]
            for record in result.history['rounds']:
                check = convergence(record['solver'], finite_policy)
                check['passed'] = check['passed'] and record['finite_step_audit']['numerically_valid']
                checks.append(check)
        else:
            try:
                result = (train_scaffold(parts, conf.ernn, conf.rounds, job['local_epochs'], job['lr'])
                    if method == 'scaffold' else train_fedavg(parts, conf.ernn, conf.rounds, job['local_epochs'], job['lr'], job['mu']))
                scores = score_fit(result.ernn_result, V, z)
            except FloatingPointError as exc:
                return {'finite': False, 'passed': False, 'failure': str(exc)}
            checks = []
        fit = result.ernn_result
    if scores is None:
        scores = score_fit(fit, V, z)
    return {**scores, 'finite': True, 'checks': checks, 'passed': all(c['passed'] for c in checks),
        'model_seed': conf.ernn.seed, 'config_fingerprint': conf.fingerprint()}


class Store:
    def __init__(self, root, manifest, workers):
        self.root, self.workers = Path(root), workers
        self.manifest = json.loads(json.dumps(manifest))
        self.identity = digest(self.manifest)
        old = self.root / 'manifest.json'
        if old.exists() and json.loads(old.read_text()) != self.manifest:
            raise RuntimeError('raw data, protocol or source changed; use a NEW output directory')
        write_json(old, self.manifest)
        self.pool = None

    def data(self, key, arrays):
        path = self.root / 'development_blocks' / (key + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        content_hash = digest({k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()})
        proof = path.with_suffix('.json')
        if path.exists():
            metadata = json.loads(proof.read_text())
            if metadata != {'content': content_hash, 'sha256': sha(path)}:
                raise RuntimeError('development block changed: ' + str(path))
        else:
            np.savez(path, **arrays)
            write_json(proof, {'content': content_hash, 'sha256': sha(path)})
        return str(path.resolve())

    def batch(self, jobs):
        results, pending = {}, []
        for key, job in jobs:
            record_path = self.root / 'tasks' / (key + '.json')
            identity = digest([self.identity, job, sha(job['data_path'])])
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['identity'] != identity or record['checksum'] != digest(record['value']):
                    raise RuntimeError('task checkpoint changed: ' + str(record_path))
                results[key] = record['value']
            else:
                pending.append((key, job, record_path, identity))
        if pending:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'))
            futures = {self.pool.submit(fit_job, job): (key, path, identity, time.time()) for key, job, path, identity in pending}
            for future in as_completed(futures):
                key, path, identity, started = futures[future]
                value = future.result()
                write_json(path, {'identity': identity, 'checksum': digest(value), 'value': value,
                    'elapsed_including_queue_seconds': time.time() - started})
                results[key] = value
                print('[completed] ' + key, flush=True)
        return [results[key] for key, _ in jobs]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def develop(raw, p, root, workers):
    manifest = {'schema': 1, 'protocol': asdict(p), 'source_identity': source_identity(),
        'dataset': raw.metadata, 'raw_hashes': raw.raw_hashes,
        'selection_data': 'outer training/validation only; each CV encoder fit on fitting fold',
        'round_SE_scope': 'three paired subsample/model evaluations; heuristic variability, not independent inferential replicates'}
    store = Store(root, manifest, workers)
    fc = base_config(p)
    def job(path, config, method='centralized', rep=0, **extra):
        return {'data_path': path, 'config': asdict(config), 'method': method,
            'clients': min(raw.clients, 4) if p.smoke else raw.clients,
            'partition_seed': role_seed(p, 'partition_' + str(rep)), **extra}
    try:
        blocks = [store.data('cv_' + str(i), transform_pair(raw, tr, va, cv=True)) for i, (tr, va) in enumerate(cv_indices(raw, p))]
        cv_jobs, specs = [], []
        for j in p.j_grid:
            for lam in p.lambda_grid:
                keys = []
                for fold, path in enumerate(blocks):
                    conf = replace(fc, ernn=replace(fc.ernn, hidden_units=j, lambda_=lam, seed=role_seed(p, 'cv_model_' + str(fold))))
                    key = f'cv_J{j}_lambda{lam:g}_fold{fold}'
                    keys.append(key)
                    cv_jobs.append((key, job(path, conf)))
                specs.append((j, lam, keys))
        values = dict(zip([key for key, _ in cv_jobs], store.batch(cv_jobs)))
        records = [{'J': j, 'lambda': lam, 'score': float(np.mean([values[k]['score'] for k in keys])),
            'se': float(np.std([values[k]['score'] for k in keys], ddof=1) / np.sqrt(5)),
            'passed': all(values[k]['passed'] for k in keys)} for j, lam, keys in specs]
        if not p.smoke and not all(r['passed'] for r in records):
            raise RuntimeError('CV finite-step diagnostic failed; no selection frozen')
        chosen = minimum_candidate(records, lambda r: (r['J'], -r['lambda']))
        write_json(store.root / 'cv_selection.json', {'candidates': records, 'selected': chosen,
            'fold_design': 'five forward timestamp blocks' if raw.chronological else 'five shuffled training-only folds'})
        fc = replace(fc, ernn=replace(fc.ernn, hidden_units=chosen['J'], lambda_=chosen['lambda']))
        path = store.data('eta', transform_pair(raw, subset(len(raw.train), p.eta_rows, role_seed(p, 'eta_train')),
            subset(len(raw.valid), p.validation_rows, role_seed(p, 'eta_valid'))))
        records = []
        # The extension is reached only after every fixed-grid candidate fails CR.
        eta_candidates = tuple(p.eta_grid) + (() if p.eta_extension in p.eta_grid else (p.eta_extension,))
        for eta in eta_candidates:
            conf = replace(fc, ernn=replace(fc.ernn, eta=eta, seed=role_seed(p, 'eta_model')))
            result = store.batch([(f'eta_{eta:g}', job(path, conf))])[0]
            records.append({'eta': eta, **result})
            if result['crossing'] <= p.crossing_threshold:
                break
        write_json(store.root / 'eta_selection.json', {'fixed_grid': p.eta_grid,
            'extension_only_after_fixed_grid_failure': p.eta_extension,
            'candidates': records, 'selected': records[-1]})
        if not p.smoke and (records[-1]['crossing'] > p.crossing_threshold or not all(r['passed'] for r in records)):
            raise RuntimeError('eta crossing/finite-step gate failed; no selection frozen')
        fc = replace(fc, ernn=replace(fc.ernn, eta=records[-1]['eta']))
        dev = [store.data('development_' + str(rep), transform_pair(raw,
            subset(len(raw.train), p.development_rows, role_seed(p, f'dev_train_{rep}')),
            subset(len(raw.valid), p.validation_rows, role_seed(p, f'dev_valid_{rep}')))) for rep in range(p.repeats)]
        scores, round_records = {}, {}
        jobs = [(f'rounds_T{t}_rep{rep}', job(path, replace(fc, rounds=t,
            ernn=replace(fc.ernn, seed=role_seed(p, f'dev_model_{rep}'))), 'extremilefed', rep))
            for t in (1, 2, 3) for rep, path in enumerate(dev)]
        values = dict(zip([key for key, _ in jobs], store.batch(jobs)))
        for t in (1, 2, 3):
            round_records[t] = [values[f'rounds_T{t}_rep{rep}'] for rep in range(p.repeats)]
            scores[t] = [r['score'] for r in round_records[t]]
        selected_t = select_rounds_one_se(scores)
        write_json(store.root / 'round_selection.json', {'selected_T': selected_t, 'scores': scores, 'records': round_records})
        if not p.smoke and not all(r['passed'] for r in values.values()):
            raise RuntimeError('round finite-step gate failed; no selection frozen')
        fc = replace(fc, rounds=selected_t)
        for method in ('fedavg', 'fedprox', 'scaffold'):
            jobs, specs = [], []
            for epochs in p.local_epochs:
                for lr in p.local_lr:
                    for mu in (p.prox_grid if method == 'fedprox' else (0.,)):
                        keys = []
                        for rep, path in enumerate(dev):
                            key = f'{method}_epochs{epochs}_lr{lr:g}_mu{mu:g}_rep{rep}'
                            keys.append(key)
                            conf = replace(fc, ernn=replace(fc.ernn, seed=role_seed(p, f'dev_model_{rep}')))
                            jobs.append((key, job(path, conf, method, rep, local_epochs=epochs, lr=lr, mu=mu)))
                        specs.append((epochs, lr, mu, keys))
            values = dict(zip([key for key, _ in jobs], store.batch(jobs)))
            failed = [{'epochs': e, 'lr': lr, 'mu': mu, 'records': [values[k] for k in keys]}
                for e, lr, mu, keys in specs if not all(values[k].get('finite', True) for k in keys)]
            candidates = [{'epochs': e, 'lr': lr, 'mu': mu, 'score': float(np.mean([values[k]['score'] for k in keys]))}
                for e, lr, mu, keys in specs if all(values[k].get('finite', True) for k in keys)]
            chosen = minimum_candidate(candidates, lambda r: (r['epochs'], r['lr'], r['mu']))
            write_json(store.root / (method + '_selection.json'), {'candidates': candidates, 'failed_candidates': failed, 'selected': chosen})
            updates = {method + '_local_epochs': chosen['epochs'], method + '_learning_rate': chosen['lr']}
            if method == 'fedprox':
                updates['fedprox_mu'] = chosen['mu']
            fc = replace(fc, **updates)
        if source_identity() != manifest['source_identity']:
            raise RuntimeError('source changed during fitting; refusing to freeze')
        evidence = {p.name: sha(p) for p in store.root.glob('*_selection.json')}
        evidence['manifest.json'] = sha(store.root / 'manifest.json')
        tasks = {p.name: sha(p) for p in (store.root / 'tasks').glob('*.json')}
        output = {'config': asdict(fc), 'fingerprint': fc.fingerprint(), 'selection': {
            'complete': True, 'smoke': p.smoke, 'dataset': raw.metadata['dataset'],
            'source_identity': source_identity(), 'raw_hashes': raw.raw_hashes,
            'evidence': evidence, 'task_sha256': tasks,
            'test_used_for_selection': False, 'numerical_policy': 'finite-step diagnostic; no theory certification'}}
        selected_path = store.root / ('smoke_config.json' if p.smoke else 'selected_config.json')
        if selected_path.exists() and json.loads(selected_path.read_text()) != json.loads(json.dumps(output)):
            raise RuntimeError('different frozen config exists; choose new output')
        write_json(selected_path, output)
        print('Selection saved: ' + str(selected_path), flush=True)
        return selected_path
    finally:
        store.close()


def verify_selection(path, raw):
    payload = json.loads(Path(path).read_text())
    selection = payload.get('selection', {})
    if (not selection.get('complete') or selection.get('smoke', True) or
        selection.get('test_used_for_selection') is not False or selection.get('dataset') != raw.metadata['dataset']):
        raise ValueError('held-out evaluation requires application-specific, complete, non-smoke selection')
    if selection['source_identity'] != source_identity() or selection['raw_hashes'] != raw.raw_hashes:
        raise ValueError('source or raw data changed since selection')
    expected = {'cv_selection.json', 'eta_selection.json', 'round_selection.json',
        'fedavg_selection.json', 'fedprox_selection.json', 'scaffold_selection.json', 'manifest.json'}
    if set(selection['evidence']) != expected or not selection['task_sha256']:
        raise ValueError('selection evidence incomplete')
    for name, checksum in selection['evidence'].items():
        if sha(Path(path).parent / name) != checksum:
            raise ValueError('selection evidence changed: ' + name)
    for name, checksum in selection['task_sha256'].items():
        if sha(Path(path).parent / 'tasks' / name) != checksum:
            raise ValueError('selection task changed: ' + name)
    return load_frozen_config(path)


def validate_evaluation(method, metrics, predictions, conf, rows):
    """Do not mark a method complete when its numbers or CSL audit are invalid."""
    expected_shapes = {'observed_y': (rows,), method + '_extremiles': (rows, 7),
        method + '_quantiles': (rows, conf.ernn.quadrature.prediction_nodes)}
    for name, shape in expected_shapes.items():
        if name not in predictions or predictions[name].shape != shape:
            raise ValueError(f'invalid held-out prediction shape: {method}/{name}')
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError('nonfinite held-out predictions: ' + method)
    try:
        json.dumps(metrics, allow_nan=False)
    except ValueError as exc:
        raise FloatingPointError('nonfinite held-out metrics/diagnostics: ' + method) from exc
    record = metrics['methods'][method]
    if (record['integrated_pinball'] < 0 or not 0 <= record['coverage_error'] <= 1
            or not 0 <= record['crossing_rate'] <= 1):
        raise ValueError('out-of-range held-out scores: ' + method)
    if method == 'extremilefed':
        diagnostics = record['gradient_diagnostics']
        if (len(diagnostics) != conf.rounds or
                not all(r['finite_step_audit']['numerically_valid'] for r in diagnostics)):
            raise FloatingPointError('full-training CSL finite-step diagnostic failed; frozen selection retained')


def evaluate(raw, selected, output):
    conf = verify_selection(selected, raw)  # Gate before transforming/evaluating test data.
    data = full_prepared(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config_sha256': sha(selected), 'raw_hashes': raw.raw_hashes, 'source_identity': source_identity(),
        'split': raw.metadata, 'test_used_for_selection': False, 'evaluation_split': 'test'}
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError('evaluation identity changed; choose new output')
    write_json(path, identity)
    combined = {}
    for method in ALL_METHODS:
        target = output / method
        metrics_path, predictions_path = target / 'metrics.json', target / 'predictions.npz'
        marker = target / 'complete.json'
        if marker.exists():
            proof = json.loads(marker.read_text())
            if proof != {'identity': digest(identity), 'metrics': sha(metrics_path), 'predictions': sha(predictions_path)}:
                raise RuntimeError('evaluation checkpoint changed: ' + method)
            metrics = json.loads(metrics_path.read_text())
        else:
            metrics, predictions = evaluate_dataset(data, conf, (method,), split='test')
            validate_evaluation(method, metrics, predictions, conf, len(data.y_test))
            metrics.update({'selection_config_sha256': sha(selected), 'raw_file_sha256': raw.raw_hashes})
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(predictions_path, **predictions)
            write_json(metrics_path, metrics)
            write_json(marker, {'identity': digest(identity), 'metrics': sha(metrics_path), 'predictions': sha(predictions_path)})
            print('[test completed] ' + method, flush=True)
        combined[method] = metrics['methods'][method]
    with np.load(output / 'centralized' / 'predictions.npz') as f:
        central = f['centralized_extremiles']
    for method in ALL_METHODS:
        with np.load(output / method / 'predictions.npz') as f:
            combined[method]['mean_absolute_centralized_extremile_gap'] = np.mean(np.abs(f[method + '_extremiles'] - central), axis=0).tolist()
    write_json(output / 'metrics.json', {'methods': combined, 'metadata': data.metadata,
        'config': asdict(conf), 'identity': identity,
        'truth_note': 'Unknown conditional extremiles; no extremile truth MAE/RMSE reported.'})
    return output / 'metrics.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('fremtpl2', 'beijing'), required=True)
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'code' / 'data')
    parser.add_argument('--output', type=Path, default=PROJECT / 'code' / 'results' / 'real_followup_v1')
    parser.add_argument('--stage', choices=('develop', 'evaluate', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--unlock-stale', action='store_true', help='remove only a dead-process lock, without fitting')
    parser.add_argument('--smoke', action='store_true', help='development-only small-data pipeline check; never permits test evaluation')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.smoke and args.stage != 'develop':
        parser.error('--smoke requires --stage develop and cannot evaluate the held-out test')
    root = args.output.resolve() / args.dataset
    if args.unlock_stale:
        lock = root / 'run.lock'
        if lock.exists():
            pid = int(lock.read_text())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                lock.unlink()
                print('Removed dead-process lock only: ' + str(lock))
            else:
                raise RuntimeError(f'process {pid} still exists; refusing to remove lock')
        return 0
    raw = load_raw(args.dataset, args.data_root)
    p = Protocol.small() if args.smoke else Protocol()
    print(json.dumps({'dataset': args.dataset, 'split_sizes': raw.metadata['split_sizes'],
        'output': str(root), 'stage': args.stage, 'workers': args.workers,
        'core_threads_per_worker': 1, 'protocol': asdict(p), 'training_requested': args.run,
        'note': 'No main simulation configuration is reused. No test score selects any parameter.'}, indent=2), flush=True)
    if not args.run:
        return 0
    with exclusive_run(root):
        selected = root / 'development' / 'selected_config.json'
        if args.stage in ('develop', 'all'):
            selected = develop(raw, p, root / 'development', args.workers)
        if args.stage in ('evaluate', 'all'):
            print(evaluate(raw, selected, root / 'evaluation'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
