from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from extremilefed import baselines, training
from extremilefed.config import ERNNConfig, FederatedConfig, QuadratureConfig
from extremilefed.federated import train_extremilefed
from extremilefed.model import ERNN, predict_extremiles
from extremilefed.training import initialize_model, load_parameter_vector, parameter_vector
from followup_depth import depth_architecture, parameter_count, validate_depth


class FollowupDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ERNNConfig(
            hidden_units=4, max_epochs=2, min_epochs=1, patience=1,
            refinement_epochs=1, batch_size=24, seed=91,
            quadrature=QuadratureConfig(training_nodes=4, prediction_nodes=8),
        )
        rng = np.random.default_rng(53)
        self.X = rng.normal(size=(24, 3))
        self.y = self.X[:, 0] + rng.normal(size=24) * 0.2

    def test_legacy_depths_are_exact_and_imported_initializer_sees_context(self) -> None:
        X = torch.as_tensor(self.X, dtype=torch.float64)
        u = torch.linspace(0.1, 0.9, len(X), dtype=torch.float64)
        for depth in (1, 2):
            with self.subTest(depth=depth):
                config = replace(self.config, second_hidden_units=None if depth == 1 else 4)
                reference = initialize_model(3, config)
                with depth_architecture(depth):
                    candidate = initialize_model(3, config)
                    imported_candidate = baselines.initialize_model(3, config)
                self.assertIs(type(candidate), ERNN)
                self.assertTrue(torch.equal(parameter_vector(reference), parameter_vector(candidate)))
                self.assertTrue(torch.equal(parameter_vector(reference), parameter_vector(imported_candidate)))
                self.assertTrue(torch.equal(reference(X, u), candidate(X, u)))
                self.assertEqual(parameter_vector(candidate).numel(), parameter_count(3, 4, depth))

    def test_all_deep_parameters_have_finite_gradients_and_round_trip(self) -> None:
        config = replace(self.config, second_hidden_units=4)
        for depth in (3, 4):
            with self.subTest(depth=depth), depth_architecture(depth):
                model = initialize_model(3, config)
                repeat = initialize_model(3, config)
                self.assertTrue(torch.equal(parameter_vector(model), parameter_vector(repeat)))
                self.assertEqual(parameter_vector(model).numel(), parameter_count(3, 4, depth))
                X = torch.as_tensor(self.X, dtype=torch.float64)
                nodes = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
                predictions = model.forward_grid(X, nodes)
                self.assertEqual(tuple(predictions.shape), (24, 3))
                predictions.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
                original = parameter_vector(model).detach().clone()
                load_parameter_vector(model, original + 0.125)
                torch.testing.assert_close(parameter_vector(model), original + 0.125, rtol=0, atol=0)
                model.to(dtype=torch.float64)
                torch.testing.assert_close(parameter_vector(model), original + 0.125, rtol=0, atol=0)

    def test_first_layer_penalty_does_not_penalize_added_layers(self) -> None:
        with depth_architecture(4):
            model = initialize_model(3, replace(self.config, second_hidden_units=4))
        torch.testing.assert_close(model.l2_first_layer(), model.hidden.weight.square().mean())
        model.l2_first_layer().backward()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.grad is not None, name == "hidden.weight", name)

    def test_additional_initialization_uses_requested_dtype_and_zero_bias(self) -> None:
        for dtype, torch_dtype in (("float64", torch.float64), ("float32", torch.float32)):
            with self.subTest(dtype=dtype), depth_architecture(4):
                model = initialize_model(3, replace(self.config, second_hidden_units=4, dtype=dtype))
                for layer in model.additional_hidden:
                    self.assertEqual(layer.weight.dtype, torch_dtype)
                    self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
                    self.assertLessEqual(float(layer.weight.detach().abs().max()), (6.0 / 8.0) ** 0.5)

    def test_context_restores_after_errors_and_nested_contexts(self) -> None:
        original = training.ERNN
        with depth_architecture(3):
            outer = training.ERNN
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with depth_architecture(1):
                    raise RuntimeError("intentional")
            self.assertIs(training.ERNN, outer)
        self.assertIs(training.ERNN, original)

    def test_depth_and_width_validation(self) -> None:
        for invalid in (0, 5, -1, 1.5, True, "2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_depth(invalid)
        for dimensions in ((0, 4, 1), (3, 0, 1), (3, 2.5, 1)):
            with self.assertRaises(ValueError):
                parameter_count(*dimensions)
        with depth_architecture(1), self.assertRaises(ValueError):
            initialize_model(3, replace(self.config, second_hidden_units=4))
        for depth in (2, 3, 4):
            with depth_architecture(depth), self.assertRaises(ValueError):
                initialize_model(3, replace(self.config, second_hidden_units=3))

    def test_tiny_csl_fits_preserve_count_communication_and_finite_predictions(self) -> None:
        partitions = [(self.X[:12], self.y[:12]), (self.X[12:], self.y[12:])]
        for depth in (3, 4):
            with self.subTest(depth=depth), depth_architecture(depth):
                result = train_extremilefed(partitions, FederatedConfig(
                    ernn=replace(self.config, second_hidden_units=4),
                    rounds=1, surrogate_epochs=2, host_epochs=2,
                    host_refinement_epochs=1,
                ))
                self.assertEqual(result.history["parameter_dim"], parameter_count(3, 4, depth))
                self.assertEqual(result.history["communication_bytes"], 2 * parameter_count(3, 4, depth) * 8)
                self.assertTrue(result.history["rounds"][0]["finite_step_audit"]["numerically_valid"])
            predictions = predict_extremiles(result.ernn_result, self.X[:5], (0.1, 0.5, 0.9))
            self.assertEqual(predictions.shape, (5, 3))
            self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
