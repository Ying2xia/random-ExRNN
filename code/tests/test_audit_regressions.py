"""Regression tests for the September 2026 manuscript/code audit."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from extremilefed.config import ERNNConfig, FederatedConfig, QuadratureConfig
from extremilefed.metrics import pinball_loss
from extremilefed.reporting import paired_mean_ci, holm_adjust
from extremilefed.reproducibility import freeze_config, load_frozen_config
from extremilefed.training import initialize_model, parameter_vector, fit_standardized, surrogate_gradient
from extremilefed.simulation import generate_data, innovation_extremile
from extremilefed.tuning import select_j_lambda, select_eta
from extremilefed.real_data import PreparedDataset
from extremilefed.real_cli import evaluate_dataset
from extremilefed.federated import partition_indices


class AuditRegressionTests(unittest.TestCase):
    def test_degenerate_paired_tests_and_invalid_inputs(self):
        self.assertEqual(paired_mean_ci([1, 1], [0, 0])["p_value"], 0)
        self.assertEqual(paired_mean_ci([1, 1], [1, 1])["p_value"], 1)
        with self.assertRaises(ValueError):
            paired_mean_ci([1, 2], [1])
        with self.assertRaises(ValueError):
            holm_adjust([np.nan])

    def test_integrated_pinball_uses_quadrature_weights(self):
        actual = pinball_loss(np.array([1.0]), np.array([[0., 0.]]), np.array([.1, .9]), np.array([.8, .2]))
        self.assertAlmostEqual(actual, .26)

    def test_frozen_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FederatedConfig()
            destination = Path(directory) / "config.json"
            freeze_config(config, destination)
            self.assertEqual(config, load_frozen_config(destination))

    def test_j_lambda_uses_minimum_not_one_se(self):
        base = ERNNConfig()
        X, y = np.zeros((10, 1)), np.zeros(10)
        # Width 3 is within one SE but must lose to the smaller mean at width 5.
        with patch("extremilefed.tuning.train_ernn"), patch("extremilefed.tuning.predict_quantiles", return_value=np.zeros((2, 128))), patch("extremilefed.tuning.pinball_loss", side_effect=[1.01]*5 + [.8, 1., 1., 1., 1.2]):
            selected, records = select_j_lambda(X, y, base, (3, 5), (0.,))
        self.assertEqual(selected.hidden_units, 5)

    def test_eta_failure_is_not_silently_accepted(self):
        X, y = np.zeros((10, 1)), np.zeros(10)
        with patch("extremilefed.tuning.train_ernn"), patch("extremilefed.tuning.predict_quantiles", return_value=np.tile(np.arange(99)[::-1], (10, 1))):
            with self.assertRaises(RuntimeError):
                select_eta(X, y, X, y, ERNNConfig(), (100.,))

    def test_scaffold_sgd_one_step_matches_corrected_gradient(self):
        X, y = generate_data(20, 1, "normal", 9301)
        config = ERNNConfig(hidden_units=3, batch_size=20, quadrature=QuadratureConfig(8, 16))
        model = initialize_model(3, config)
        initial = parameter_vector(model).detach().numpy().copy()
        correction = np.linspace(-.1, .1, len(initial))
        gradient = surrogate_gradient(model, X, y, config, correction)
        fitted, history = fit_standardized(X, y, config, init_vector=initial, linear_correction=correction, max_epochs=1, learning_rate=.001, fixed_sgd=True)
        np.testing.assert_allclose(parameter_vector(fitted).detach().numpy(), initial-.001*gradient, atol=1e-12)
        self.assertEqual(history["optimizer"], "SGD_fixed_steps")

    def test_adaptive_truth_symmetry(self):
        for error in ("normal", "t3"):
            self.assertAlmostEqual(innovation_extremile(error, .1), -innovation_extremile(error, .9), places=6)
        for error in ("normal", "t3", "lognormal", "contaminated"):
            self.assertEqual(innovation_extremile(error, .5), 0)

    def test_real_runner_reports_no_fabricated_extremile_truth(self):
        X, y = generate_data(100, 1, "normal", 9302)
        data = PreparedDataset(X[:60], y[:60], X[60:80], y[60:80], X[80:], y[80:], partition_indices(60, 4, 1), {"fixture": True})
        config = FederatedConfig(ernn=ERNNConfig(hidden_units=3, max_epochs=2, min_epochs=2, quadrature=QuadratureConfig(6, 12)), rounds=1, surrogate_epochs=2)
        results, predictions = evaluate_dataset(data, config, ("centralized", "extremilefed"))
        self.assertNotIn("extremile_mae", results["methods"]["extremilefed"])
        self.assertEqual(predictions["extremilefed_extremiles"].shape, (20, 7))


if __name__ == "__main__":
    unittest.main()
