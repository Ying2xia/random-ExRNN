from __future__ import annotations

import unittest
import numpy as np

from extremilefed.config import ERNNConfig, QuadratureConfig
from extremilefed.metrics import crossing_rate
from extremilefed.model import data_loss, noncrossing_penalty, predict_extremiles, predict_quantiles, smooth_pinball
from extremilefed.quadrature import extremile_weight, gauss_legendre_unit
from extremilefed.simulation import generate_data
from extremilefed.training import (
    data_gradient,
    initialize_model,
    load_parameter_vector,
    parameter_vector,
    train_ernn,
)
from extremilefed.tuning import select_rounds_one_se


class CoreTests(unittest.TestCase):
    def test_extremile_weights_integrate_to_one(self) -> None:
        nodes, weights = gauss_legendre_unit(256)
        for tau in (0.05, 0.3, 0.5, 0.7, 0.95):
            self.assertAlmostEqual(float(np.sum(weights * extremile_weight(nodes, tau))), 1.0, places=10)
        np.testing.assert_allclose(extremile_weight(nodes, 0.5), np.ones_like(nodes))

    def test_smooth_pinball_limit_and_quadrature_convergence(self) -> None:
        import torch

        residual = torch.tensor([[-2.0, 1.5]], dtype=torch.float64)
        levels = torch.tensor([0.2, 0.8], dtype=torch.float64)
        smooth = smooth_pinball(residual, levels, 1e-7).numpy()[0]
        exact = np.maximum(levels.numpy() * residual.numpy()[0], (levels.numpy() - 1.0) * residual.numpy()[0])
        np.testing.assert_allclose(smooth, exact, atol=1e-7)
        for count in (4, 8, 16):
            nodes, weights = gauss_legendre_unit(count)
            self.assertAlmostEqual(float(np.sum(weights * nodes**2)), 1.0 / 3.0, places=12)

    def test_noncrossing_penalty_orders_predictions(self) -> None:
        import torch

        ordered = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
        crossed = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float64)
        self.assertLess(float(noncrossing_penalty(ordered, 0.03)), float(noncrossing_penalty(crossed, 0.03)))

    def test_small_central_fit_and_predictions(self) -> None:
        X, y = generate_data(160, 1, "normal", 12)
        config = ERNNConfig(
            hidden_units=5,
            max_epochs=10,
            min_epochs=5,
            patience=3,
            batch_size=64,
            eta=1.0,
            dtype="float64",
            quadrature=QuadratureConfig(training_nodes=8, prediction_nodes=24),
        )
        fit = train_ernn(X, y, config)
        quantiles = predict_quantiles(fit, X[:12], (0.1, 0.5, 0.9))
        extremiles = predict_extremiles(fit, X[:12], (0.1, 0.5, 0.9))
        self.assertEqual(quantiles.shape, (12, 3))
        self.assertEqual(extremiles.shape, (12, 3))
        self.assertTrue(np.isfinite(quantiles).all())
        self.assertLessEqual(crossing_rate(quantiles), 1.0)

    def test_parameter_packing_and_finite_gradient(self) -> None:
        X, y = generate_data(40, 1, "normal", 99)
        config = ERNNConfig(
            hidden_units=3,
            max_epochs=3,
            min_epochs=2,
            patience=1,
            batch_size=20,
            quadrature=QuadratureConfig(training_nodes=6, prediction_nodes=12),
        )
        model = initialize_model(X.shape[1], config)
        original = parameter_vector(model).detach().cpu().numpy().copy()
        load_parameter_vector(model, original + 0.125)
        np.testing.assert_allclose(parameter_vector(model).detach().cpu().numpy(), original + 0.125)
        gradient = data_gradient(model, X, y, config)
        self.assertEqual(gradient.shape, original.shape)
        self.assertTrue(np.isfinite(gradient).all())
        import torch

        nodes, weights = gauss_legendre_unit(config.quadrature.training_nodes)
        X_t = torch.as_tensor(X, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        nodes_t = torch.as_tensor(nodes, dtype=torch.float64)
        weights_t = torch.as_tensor(weights, dtype=torch.float64)

        def objective(vector: np.ndarray) -> float:
            load_parameter_vector(model, vector)
            value, _ = data_loss(
                model, X_t, y_t, nodes_t, weights_t, config.h, config.eta
            )
            return float(value.detach())

        current = parameter_vector(model).detach().cpu().numpy().copy()
        epsilon = 1e-6
        plus, minus = current.copy(), current.copy()
        direction = np.random.default_rng(20).normal(size=current.size)
        direction /= np.linalg.norm(direction)
        plus += epsilon * direction
        minus -= epsilon * direction
        finite_difference = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        load_parameter_vector(model, current)
        self.assertAlmostEqual(float(gradient @ direction), finite_difference, places=6)

    def test_round_selection_uses_one_se_and_fewer_rounds(self) -> None:
        selected = select_rounds_one_se(
            {1: [1.02, 1.03, 1.01], 2: [1.00, 1.02, 1.01], 3: [0.95, 1.05, 1.00]}
        )
        self.assertEqual(selected, 1)


if __name__ == "__main__":
    unittest.main()
