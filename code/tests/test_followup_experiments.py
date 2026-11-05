"""Supplementary protocols preserve pairing, provenance and resumability."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

import run_followup_experiments as followup


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if key not in ('elapsed_seconds', 'runtime_seconds')}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


class FollowupExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from release_utils import ROOT, configuration
        cls.configs = {}
        for path in (ROOT / 'configs/supplementary').glob('*.json'):
            scenario, error = path.stem.split('_', 1)
            cls.configs[(int(scenario.replace('scenario', '')), error)] = configuration('supplementary', path.stem)[0]
        cls.formal = followup.build_tasks(cls.configs, replications=2)
        cls.smoke = followup.build_tasks(cls.configs, replications=1, smoke=True)

    def task(self, family, *, depth=None):
        return next(task for task in self.smoke if task.family == family and
                    (depth is None or task.depth == depth))

    def run_small(self, task):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return followup.run_task(task)

    def test_grid_is_paired_unique_and_excludes_noop_contrasts(self):
        self.assertEqual(len(self.formal), len({task.key for task in self.formal}))
        for task in self.formal:
            base = self.configs[(task.job.scenario, task.job.error)]
            self.assertEqual(task.job.seed, 20260903)
            self.assertEqual(task.config.ernn.seed, base.ernn.seed)
            self.assertEqual((task.job.n_train, task.job.n_test), (100000, 5000))
            if task.family == 'penalty':
                self.assertNotEqual(base.ernn.eta, 0)
                self.assertEqual(task.config.ernn.eta, 0)
            elif task.family == 'rounds':
                self.assertNotEqual(task.config.rounds, base.rounds)
                self.assertEqual(len(task.config.surrogate_learning_rates), task.config.rounds)
            elif task.family == 'clients':
                self.assertNotEqual(task.job.clients, 20)
                self.assertEqual(task.methods, ('host', 'extremilefed'))
            if task.family not in ('reference', 'depth'):
                self.assertIn((task.job.scenario, task.job.error), followup.REPRESENTATIVE)
        for setting in followup.ALL_SETTINGS:
            selected = [task for task in self.formal if
                        (task.job.scenario, task.job.error) == setting and task.job.replication == 0]
            self.assertEqual({task.depth for task in selected if task.family in ('reference', 'depth')},
                             {1, 2, 3, 4})
        self.assertTrue(any(task.family == 'penalty' for task in self.formal))
        for replications in (0, 101):
            with self.assertRaises(ValueError):
                followup.build_tasks(self.configs, replications=replications)

    def test_reference_recomputed_without_historical_archive_and_reused(self):
        from reproduce_supplement import worker
        task = next(task for task in self.smoke if task.family == 'reference')
        with tempfile.TemporaryDirectory() as directory:
            worker((task.specification(), directory, 'test-release'))
            target = Path(directory) / 'records' / (task.key + '.json')
            result = json.loads(target.read_text())
            self.assertTrue(result['followup']['reference_recomputed'])
            self.assertEqual(set(result['methods']), {followup.METHOD_NAMES[m] for m in followup.CHE})
            before = target.read_bytes()
            with patch.object(followup.experiments, 'run_job', side_effect=AssertionError('unexpected fit')):
                self.assertIn('cached', worker((task.specification(), directory, 'test-release')))
            self.assertEqual(target.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'another task or source'):
                worker((task.specification(), directory, 'different-release'))

    def test_metadata_tampering_and_smoke_reuse_are_rejected(self):
        task = self.task('depth', depth=4)
        result = self.run_small(task)
        followup.verify_task_result(result, task)
        mutations = (
            lambda value: value['job'].__setitem__('seed', value['job']['seed'] + 1),
            lambda value: value['seeds'].__setitem__('model', value['seeds']['model'] + 1),
            lambda value: value['followup'].__setitem__('parameter_count', 1),
            lambda value: value['followup'].__setitem__('hidden_layers', 2),
            lambda value: value['quadrature'].__setitem__('training_nodes', 99),
            lambda value: value['quadrature']['evaluation_weights'].__setitem__(0, 0),
            lambda value: value['methods']['ExtremileFed'].__setitem__('rounds', 99),
            lambda value: value['methods']['ExtremileFed'].__setitem__('communication_mb', 0),
            lambda value: value['methods']['ExtremileFed']['metrics'].__setitem__('extremile_rmse', 0),
            lambda value: value['methods']['Host ERNN']['metrics'].__setitem__('crossing_rate', float('nan')),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                altered = deepcopy(result)
                mutate(altered)
                with self.assertRaises(ValueError):
                    followup.verify_task_result(altered, task)
        with self.assertRaisesRegex(ValueError, 'protocol/source mismatch'):
            followup.verify_task_result(result, replace(task, smoke=False))

    def test_spawn_matches_direct_result_and_does_not_leak_model_factory(self):
        task = self.task('depth', depth=3)
        old_factory = followup.training.ERNN
        direct = self.run_small(task)
        self.assertIs(followup.training.ERNN, old_factory)
        worker = followup.start_worker(task)
        try:
            self.assertTrue(worker.connection.poll(60), 'small spawned task did not finish')
            envelope = followup.parallel.receive_worker(worker)
            self.assertTrue(envelope['ok'], envelope.get('traceback', 'worker failed'))
            followup.verify_task_result(envelope['result'], task)
            self.assertEqual(without_timing(direct), without_timing(envelope['result']))
        finally:
            followup.parallel.close_worker(worker)

    def test_quadrature_uses_same_fit_and_validates_all_levels(self):
        task = self.task('quadrature')
        self.assertIsNone(task.config.ernn.second_hidden_units)
        calls = []
        actual_predict = followup.predict_extremiles

        def observed(fit, X, tau, *, quadrature):
            calls.append((fit, quadrature))
            return actual_predict(fit, X, tau, quadrature=quadrature)

        old_metrics = followup.experiments._neural_metrics
        with patch.object(followup, 'predict_extremiles', side_effect=observed):
            result = self.run_small(task)
        self.assertIs(followup.experiments._neural_metrics, old_metrics)
        self.assertEqual(len(calls), 9)
        for start in (0, 3, 6):
            self.assertEqual([nodes for _, nodes in calls[start:start+3]], [4, 8, 16])
            self.assertTrue(all(fit is calls[start][0] for fit, _ in calls[start:start+3]))
        for method in result['same_fit_quadrature'].values():
            self.assertEqual(method['8']['prediction_rmse_gap_to_G8'], 0)
        altered = deepcopy(result)
        altered['same_fit_quadrature']['ExtremileFed']['8']['prediction_rmse_gap_to_G8'] = 1
        with self.assertRaisesRegex(ValueError, 'reference quadrature mismatch'):
            followup.verify_task_result(altered, task)

    def test_budget_includes_terminal_gradient_and_records_mismatch(self):
        task = self.task('budget')
        old_loss = followup.training.data_loss
        result = self.run_small(task)
        self.assertIs(followup.training.data_loss, old_loss)
        budget = result['budget_matching']
        target = budget['target_gradient_observations']
        local_epochs = budget['baseline_local_epochs']
        block = task.job.n_train * task.config.rounds
        self.assertEqual(local_epochs, max(1, int(np.ceil(target / block)) - 1))
        self.assertEqual(budget['actual_work']['SCAFFOLD-ERNN']['gradient_observations'],
                         block * (local_epochs + 1))
        for method, actual in budget['actual_work'].items():
            self.assertEqual(budget['gradient_observation_difference'][method],
                             actual['gradient_observations'] - target)
        altered = deepcopy(result)
        altered['budget_matching']['baseline_local_epochs'] += 1
        with self.assertRaisesRegex(ValueError, 'budget/config mismatch'):
            followup.verify_task_result(altered, task)

    def test_frozen_global_protocol_is_independent_of_suite_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code = root / 'code'
            output = code / 'results' / 'followup'
            source_archive = code / 'results' / 'formal_v7'
            with patch.object(followup, 'ROOT', root), patch.object(followup, 'CODE', code), \
                    patch.object(followup.original, 'OUTPUT', source_archive), \
                    patch.object(followup.original, 'load_local_configs', return_value=(self.configs, [])), \
                    patch.object(followup, 'source_hashes', return_value={'core': 'fixture-v1'}), \
                    redirect_stdout(io.StringIO()):
                args = ['--output', str(output), '--replications', '1']
                followup.main(args + ['--suite', 'depth'])
                frozen = (output / 'protocol.json').read_bytes()
                followup.main(args + ['--suite', 'budget'])
                self.assertEqual((output / 'protocol.json').read_bytes(), frozen)
                plan = json.loads(frozen)
                self.assertTrue(any(key.startswith('depth_4/') for key in plan['tasks']))
                self.assertTrue(any(key.startswith('gradient_budget/') for key in plan['tasks']))
                with self.assertRaisesRegex(RuntimeError, 'Frozen protocol changed'):
                    followup.main(['--output', str(output), '--replications', '2'])
                with patch.object(followup, 'source_hashes', return_value={'core': 'fixture-v2'}):
                    with self.assertRaisesRegex(RuntimeError, 'Frozen protocol changed'):
                        followup.main(args)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    followup.main(['--smoke', '--output', str(output)])
                self.assertFalse(list(output.glob('*/scenario*_rep*.json')))


if __name__ == '__main__':
    unittest.main()
