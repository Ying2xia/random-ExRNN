from __future__ import annotations

import unittest
import numpy as np

from extremilefed.baselines import train_fedavg, train_one_shot, train_scaffold
from extremilefed.config import ERNNConfig, QuadratureConfig
from extremilefed.federated import partition_indices
from extremilefed.linear import train_linear_extremile_sl
from extremilefed.simulation import generate_data


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = generate_data(100, 1, "normal", 19)
        idx = partition_indices(100, 4, 20)
        self.partitions = [(self.X[part], self.y[part]) for part in idx]
        self.config = ERNNConfig(
            hidden_units=3,
            max_epochs=5,
            min_epochs=3,
            patience=2,
            batch_size=32,
            dtype="float64",
            quadrature=QuadratureConfig(training_nodes=6, prediction_nodes=12),
        )

    def test_one_shot_and_fedavg(self) -> None:
        one_shot = train_one_shot(self.partitions, self.config)
        fedavg = train_fedavg(self.partitions, self.config, rounds=1, local_epochs=3)
        self.assertEqual(one_shot.method, "One-shot ERNN")
        self.assertEqual(fedavg.method, "FedAvg-ERNN")
        self.assertGreater(fedavg.communication_mb, 0.0)

    def test_linear_baseline(self) -> None:
        fit = train_linear_extremile_sl(self.X, self.y, nodes=8, max_epochs=20)
        prediction = fit.predict_extremiles(self.X[:5], np.array([0.1, 0.9]), n_nodes=16)
        self.assertEqual(prediction.shape, (5, 2))
        self.assertTrue(np.isfinite(prediction).all())

    def test_scaffold_and_two_layer_sensitivity(self) -> None:
        deep_config = ERNNConfig(
            hidden_units=3,
            second_hidden_units=2,
            max_epochs=4,
            min_epochs=2,
            patience=2,
            batch_size=32,
            dtype="float64",
            quadrature=QuadratureConfig(training_nodes=6, prediction_nodes=12),
        )
        fit = train_scaffold(
            self.partitions, deep_config, rounds=1, local_epochs=2, local_learning_rate=1e-3
        )
        self.assertEqual(fit.method, "SCAFFOLD-ERNN")
        self.assertIsNotNone(fit.ernn_result.model.hidden2)
        self.assertGreater(fit.communication_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
