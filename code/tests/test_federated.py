from __future__ import annotations

import unittest
import numpy as np

from extremilefed.config import ERNNConfig, FederatedConfig, QuadratureConfig
from extremilefed.federated import partition_indices, sample_size_weighted_average, train_extremilefed
from extremilefed.model import data_loss
from extremilefed.preprocessing import fit_standardizer
from extremilefed.quadrature import gauss_legendre_unit
from extremilefed.simulation import generate_data
from extremilefed.training import (
    data_gradient,
    initialize_model,
    load_parameter_vector,
    parameter_vector,
    surrogate_gradient,
)


class FederatedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = generate_data(120, 1, "normal", 42)
        self.config = ERNNConfig(
            hidden_units=3,
            max_epochs=8,
            min_epochs=4,
            patience=2,
            batch_size=40,
            dtype="float64",
            quadrature=QuadratureConfig(training_nodes=6, prediction_nodes=12),
        )

    def test_weighted_client_gradient_equals_pooled_gradient(self) -> None:
        scaler = fit_standardizer(self.X, self.y)
        Xs, ys = scaler.transform_X(self.X), scaler.transform_y(self.y)
        model = initialize_model(Xs.shape[1], self.config)
        parts = partition_indices(len(ys), 4, 7)
        local = [data_gradient(model, Xs[idx], ys[idx], self.config, batch_size=17) for idx in parts]
        aggregated = sample_size_weighted_average(local, [len(idx) for idx in parts])
        pooled = data_gradient(model, Xs, ys, self.config, batch_size=17)
        np.testing.assert_allclose(aggregated, pooled, rtol=1e-9, atol=1e-9)

    def test_surrogate_adds_the_common_penalty_once(self) -> None:
        import torch

        scaler = fit_standardizer(self.X, self.y)
        Xs, ys = scaler.transform_X(self.X), scaler.transform_y(self.y)
        model = initialize_model(Xs.shape[1], self.config)
        parts = partition_indices(len(ys), 4, 7)
        local = [data_gradient(model, Xs[idx], ys[idx], self.config) for idx in parts]
        global_gradient = sample_size_weighted_average(local, [len(idx) for idx in parts])
        correction = local[0] - global_gradient
        surrogate = surrogate_gradient(model, Xs[parts[0]], ys[parts[0]], self.config, correction)
        model.zero_grad(set_to_none=True)
        (self.config.lambda_ * model.l2_first_layer()).backward()
        penalty_gradient = np.concatenate(
            [
                np.zeros(parameter.numel())
                if parameter.grad is None
                else parameter.grad.detach().cpu().numpy().reshape(-1)
                for parameter in model.parameters()
            ]
        )
        np.testing.assert_allclose(surrogate, global_gradient + penalty_gradient, rtol=1e-9, atol=1e-9)

        nodes, weights = gauss_legendre_unit(self.config.quadrature.training_nodes)
        host_X = torch.as_tensor(Xs[parts[0]], dtype=torch.float64)
        host_y = torch.as_tensor(ys[parts[0]], dtype=torch.float64)
        nodes_t = torch.as_tensor(nodes, dtype=torch.float64)
        weights_t = torch.as_tensor(weights, dtype=torch.float64)
        current = parameter_vector(model).detach().cpu().numpy().copy()

        def objective(vector: np.ndarray) -> float:
            load_parameter_vector(model, vector)
            loss, _ = data_loss(
                model,
                host_X,
                host_y,
                nodes_t,
                weights_t,
                self.config.h,
                self.config.eta,
            )
            value = (
                loss
                + self.config.lambda_ * model.l2_first_layer()
                - torch.dot(torch.as_tensor(correction, dtype=torch.float64), parameter_vector(model))
            )
            return float(value.detach())

        epsilon = 1e-6
        plus, minus = current.copy(), current.copy()
        direction = np.random.default_rng(19).normal(size=current.size)
        direction /= np.linalg.norm(direction)
        plus += epsilon * direction
        minus -= epsilon * direction
        finite_difference = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        load_parameter_vector(model, current)
        self.assertAlmostEqual(float(surrogate @ direction), finite_difference, places=6)

    def test_federated_smoke_fit(self) -> None:
        parts = partition_indices(len(self.y), 4, 8)
        datasets = [(self.X[idx], self.y[idx]) for idx in parts]
        result = train_extremilefed(
            datasets,
            FederatedConfig(
                ernn=self.config,
                rounds=1,
                surrogate_epochs=5,
                surrogate_learning_rate=0.003,
            ),
        )
        self.assertEqual(len(result.history["rounds"]), 1)
        self.assertGreater(result.history["parameter_dim"], 0)
        self.assertFalse(result.history["raw_data_transmitted"])


if __name__ == "__main__":
    unittest.main()
