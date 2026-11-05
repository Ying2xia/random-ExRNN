from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from run_real_data_experiments import (Protocol, RawSplits, cv_indices, develop,
    evaluate, transform_pair, verify_selection, load_raw, validate_evaluation,
    base_config, fit_job)
from extremilefed.config import ETA_GRID
from dataclasses import asdict


def synthetic_raw(chronological=False):
    rng = np.random.default_rng(431)
    frame = pd.DataFrame({'x': rng.normal(size=180), 'category': np.tile(['a', 'b', 'c'], 60)})
    y = 2 + frame.x.to_numpy() + rng.normal(size=180) * .2
    return RawSplits(frame.iloc[:120], frame.iloc[120:150], frame.iloc[150:],
        y[:120], y[120:150], y[150:], ['x'], ['category'], 4, chronological,
        {'dataset': 'synthetic-test-only', 'seed': 431, 'split_sizes': [120, 30, 30]},
        {'synthetic': 'not-a-real-application'}, np.repeat(np.arange(40), 3) if chronological else None)


class RealFollowupTests(unittest.TestCase):
    def test_formal_eta_grid_and_extension_are_distinct(self):
        self.assertEqual(Protocol().eta_grid, ETA_GRID)
        self.assertEqual(Protocol().eta_extension, 1000.)

    def test_missing_csv_columns_fail_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'IDpol': [1]}).to_csv(root / 'freMTPL2freq.csv', index=False)
            pd.DataFrame({'IDpol': [1], 'ClaimAmount': [1.]}).to_csv(root / 'freMTPL2sev.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'missing required columns.*Area'):
                load_raw('fremtpl2', root)
            (root / 'beijing').mkdir()
            pd.DataFrame({'year': [2016]}).to_csv(root / 'beijing' / 'station.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'missing required columns.*PM2.5'):
                load_raw('beijing', root)

    def test_final_guard_rejects_nonfinite_and_failed_solver(self):
        conf = base_config(Protocol.small())
        pred = {'observed_y': np.ones(2), 'extremilefed_extremiles': np.ones((2, 7)),
            'extremilefed_quantiles': np.ones((2, conf.ernn.quadrature.prediction_nodes))}
        record = {'integrated_pinball': .5, 'coverage_error': .1, 'crossing_rate': 0.,
            'gradient_diagnostics': [{'finite_step_audit': {'numerically_valid': False}}]}
        metrics = {'methods': {'extremilefed': record}}
        with self.assertRaisesRegex(FloatingPointError, 'finite-step diagnostic failed'):
            validate_evaluation('extremilefed', metrics, pred, conf, 2)
        record['gradient_diagnostics'][0]['finite_step_audit']['numerically_valid'] = True
        record['integrated_pinball'] = float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'metrics/diagnostics'):
            validate_evaluation('extremilefed', metrics, pred, conf, 2)
        record['integrated_pinball'] = .5
        pred['extremilefed_quantiles'][0, 0] = float('inf')
        with self.assertRaisesRegex(FloatingPointError, 'predictions'):
            validate_evaluation('extremilefed', metrics, pred, conf, 2)

    def test_worker_only_catches_numerical_candidate_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'block.npz'
            np.savez(path, X=np.ones((12, 2)), y=np.arange(12.), V=np.ones((3, 2)), z=np.ones(3))
            job = {'config': asdict(base_config(Protocol.small())), 'data_path': str(path),
                'method': 'fedavg', 'clients': 2, 'partition_seed': 41, 'local_epochs': 1, 'lr': .001, 'mu': 0.}
            with patch('run_real_data_experiments.train_fedavg', side_effect=RuntimeError('unexpected bug')):
                with self.assertRaisesRegex(RuntimeError, 'unexpected bug'):
                    fit_job(job)
            with patch('run_real_data_experiments.train_fedavg', side_effect=FloatingPointError('nonfinite')):
                self.assertFalse(fit_job(job)['finite'])

    def test_fold_encoder_uses_fitting_only(self):
        raw = synthetic_raw()
        raw.train = raw.train.copy()
        raw.train.loc[0:2, 'x'] = [1., 2., 3.]
        raw.train.loc[3, 'x'] = 100000.
        raw.train.loc[3, 'category'] = 'unseen'
        block = transform_pair(raw, np.array([0, 1, 2]), np.array([3]), cv=True)
        self.assertAlmostEqual(block['X'][:, 0].mean(), 0.)
        self.assertGreater(block['V'][0, 0], 10000)
        self.assertEqual(block['V'][0, 1:].sum(), 0.)

    def test_forward_folds_never_split_hours(self):
        raw = synthetic_raw(chronological=True)
        for training, validation in cv_indices(raw, Protocol.small()):
            self.assertLess(raw.train_times[training].max(), raw.train_times[validation].min())
            self.assertFalse(set(raw.train_times[training]) & set(raw.train_times[validation]))

    def test_smoke_freeze_cannot_unlock_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'smoke.json'
            path.write_text(json.dumps({'selection': {'complete': True, 'smoke': True}}))
            with self.assertRaisesRegex(ValueError, 'non-smoke'):
                verify_selection(path, synthetic_raw())

    def test_tiny_complete_resume_test_isolation_and_integrity(self):
        # Synthetic-only tiny budgets are recorded and cannot be an application config.
        raw = synthetic_raw()
        p = replace(Protocol.small(), smoke=False, crossing_threshold=1.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = develop(raw, p, root / 'development', workers=2)
            frozen_bytes = selected.read_bytes()
            tasks_before = {p.name: p.read_bytes() for p in (selected.parent / 'tasks').glob('*.json')}
            raw.y_test[:] = 1e12
            repeated = develop(raw, p, root / 'development', workers=2)
            self.assertEqual(frozen_bytes, repeated.read_bytes())
            self.assertEqual(tasks_before, {p.name: p.read_bytes() for p in (selected.parent / 'tasks').glob('*.json')})
            raw.y_test[:] = 2.
            result = evaluate(raw, selected, root / 'evaluation')
            payload = json.loads(result.read_text())
            self.assertEqual(len(payload['methods']), 8)
            self.assertFalse(payload['identity']['test_used_for_selection'])
            self.assertFalse(any('extremile_rmse' in m for m in payload['methods'].values()))
            completions = {p.parent.name: p.read_bytes() for p in (root / 'evaluation').glob('*/complete.json')}
            evaluate(raw, selected, root / 'evaluation')
            self.assertEqual(completions, {p.parent.name: p.read_bytes() for p in (root / 'evaluation').glob('*/complete.json')})
            evidence = selected.parent / 'cv_selection.json'
            evidence.write_text(evidence.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                verify_selection(selected, raw)


if __name__ == '__main__':
    unittest.main()
