from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from extremilefed.development import (DevelopmentProtocol, TaskStore, Paused, GateFailure,
    convergence, exclusive_run, implementation_hash, isolated_seed, minimum_candidate, run_case, write_json)
from extremilefed.reproducibility import load_frozen_config, verify_simulation_selection
from extremilefed.config import ERNNConfig, QuadratureConfig
from extremilefed.model import data_loss
from extremilefed.quadrature import gauss_legendre_unit
from extremilefed.simulation import generate_data
from extremilefed.training import fit_standardized, parameter_vector, surrogate_gradient


class DevelopmentTests(unittest.TestCase):
    def protocol(self):
        return replace(DevelopmentProtocol.small(), j_grid=(3,), lambda_grid=(.001,),
            eta_grid=(0.,), local_epochs_grid=(1,), local_lr_grid=(.001,), prox_grid=(.001,))

    def manifest(self, protocol):
        return {"protocol": asdict(protocol), "implementation_hash": implementation_hash()}

    def test_checkpoint_pause_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root, {"grid": (1, 2)}, 1)
            self.assertEqual(store.task("first", lambda: {"answer": 1}), {"answer": 1})
            with self.assertRaises(Paused):
                store.task("second", lambda: {})
            resumed = TaskStore(root, {"grid": (1, 2)})
            self.assertEqual(resumed.task("first", lambda: self.fail("cache was not used")), {"answer": 1})
            path = root / "tasks/first.json"
            payload = json.loads(path.read_text())
            payload["value"]["answer"] = 9
            write_json(path, payload)
            with self.assertRaises(RuntimeError):
                resumed.task("first", lambda: {})
            with self.assertRaises(RuntimeError):
                TaskStore(root, {"grid": (3, 4)})

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with exclusive_run(root):
                with self.assertRaises(RuntimeError):
                    with exclusive_run(root):
                        pass
            self.assertFalse((root / "run.lock").exists())

    def test_convergence_requires_gradient_and_plateau(self):
        history = {"loss": [1.]*5, "terminal_loss": 1., "terminal_gradient_norm": .001, "epochs": 5, "best_epoch": 5}
        p = self.protocol()
        self.assertTrue(convergence(history, p)["passed"])
        self.assertFalse(convergence(dict(history, terminal_gradient_norm=.1), p)["passed"])
        self.assertFalse(convergence(dict(history, loss=[5., 4., 3., 2., 1.]), p)["passed"])
        self.assertFalse(convergence(dict(history, loss=[1., 1.]), p)["passed"])

    def test_seeds_and_ties(self):
        p = self.protocol()
        values = [isolated_seed(p, 1, "normal", role) for role in ("pilot_train", "cv_data", "eta_train", "eta_validation", "dev_train_0", "dev_valid_0")]
        self.assertEqual(len(set(values)), len(values))
        self.assertTrue(all(1e9 <= v < 2e9 for v in values))
        self.assertEqual(values[0], isolated_seed(p, 1, "normal", "pilot_train"))
        records = [{"score": 1., "J": j, "lambda": l} for j,l in ((5,0.),(3,0.),(3,.01))]
        self.assertEqual(minimum_candidate(records, lambda r:(r["J"],-r["lambda"]))["lambda"], .01)

    def test_smoke_end_to_end_resume_and_formal_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, p = Path(tmp), self.protocol()
            store = TaskStore(root, self.manifest(p))
            result = run_case(store, p, 1, "normal")
            self.assertEqual(result["status"], "smoke_complete")
            self.assertFalse(result["eligible_for_formal_simulation"])
            load_frozen_config(result["config"])
            with self.assertRaises(ValueError):
                verify_simulation_selection(result["config"], 1, "normal")
            resumed = TaskStore(root, self.manifest(p))
            with patch("extremilefed.development.train_ernn", side_effect=AssertionError("retrained cache")):
                rerun = run_case(resumed, p, 1, "normal")
            self.assertEqual(rerun, result)
            self.assertEqual(resumed.new_tasks, 0)

    def test_failed_gate_does_not_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = replace(self.protocol(), smoke=False)
            store = TaskStore(tmp, self.manifest(p))
            with self.assertRaises(GateFailure):
                run_case(store, p, 1, "normal")
            self.assertFalse((Path(tmp) / "selected_config.json").exists())

    def test_formal_freeze_path_with_mocked_gates(self):
        # Integration test only: never use this artificial gate override for science.
        with tempfile.TemporaryDirectory() as tmp:
            p = replace(self.protocol(), smoke=False)
            store = TaskStore(tmp, self.manifest(p))
            with patch("extremilefed.development.convergence", return_value={"passed": True}), patch("extremilefed.development.score_fit", return_value={"score": 1., "crossing": 0.}):
                result = run_case(store, p, 1, "normal")
            verified = verify_simulation_selection(result["config"], 1, "normal")
            self.assertEqual(verified.rounds, 1)
            with self.assertRaises(ValueError):
                verify_simulation_selection(result["config"], 2, "normal")
            evidence = Path(tmp) / "eta_selection.json"
            write_json(evidence, {})
            with self.assertRaises(ValueError):
                verify_simulation_selection(result["config"], 1, "normal")

    def test_returned_objective_and_gradient_are_exact(self):
        X,y = generate_data(21, 1, "normal", 7)
        config = ERNNConfig(hidden_units=3, max_epochs=3, min_epochs=3, batch_size=8, quadrature=QuadratureConfig(6,12))
        model, history = fit_standardized(X,y,config)
        u,w = gauss_legendre_unit(6)
        loss,_ = data_loss(model, torch.tensor(X), torch.tensor(y), torch.tensor(u), torch.tensor(w), config.h, config.eta)
        objective = float(loss + config.lambda_ * model.l2_first_layer())
        self.assertAlmostEqual(history["terminal_loss"], objective, places=12)
        self.assertAlmostEqual(history["best_loss"], objective, places=12)
        gradient = surrogate_gradient(model,X,y,config,np.zeros(parameter_vector(model).numel()))
        self.assertAlmostEqual(history["terminal_gradient_norm"], np.linalg.norm(gradient), places=12)


if __name__ == "__main__":
    unittest.main()
