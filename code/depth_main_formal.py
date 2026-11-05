"""Single-method execution and numerical audit for a versioned depth study.

The outer runner owns the frozen protocol, source hashes, selection evidence,
independent seeds and result publication. This module does not use or modify
the old ready gate or original result archives. Architecture contexts and the
metric hook are process-local: invoke one task at a time in a spawned worker.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from torch import nn

from extremilefed import experiments
from extremilefed.config import ERNNConfig, FederatedConfig, QuadratureConfig
from extremilefed.experiments import SimulationJob
from extremilefed.training import parameter_vector
from followup_depth import depth_architecture, parameter_count, validate_depth


METHOD_NAMES = {
    "centralized": "Centralized ERNN",
    "linear": "Pooled linear Extremile-SL",
    "host": "Host ERNN",
    "one_shot": "One-shot ERNN",
    "fedavg": "FedAvg-ERNN",
    "fedprox": "FedProx-ERNN",
    "scaffold": "SCAFFOLD-ERNN",
    "extremilefed": "ExtremileFed",
}


def _finite_tree(value: Any, location: str = "result") -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Nonfinite number at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")
    elif isinstance(value, np.ndarray) and not np.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite array at {location}")


def _configuration(raw: dict) -> FederatedConfig:
    values = dict(raw)
    neural = dict(values.pop("ernn"))
    neural["quadrature"] = QuadratureConfig(**neural["quadrature"])
    if "surrogate_learning_rates" in values:
        values["surrogate_learning_rates"] = tuple(values["surrogate_learning_rates"])
    config = FederatedConfig(ernn=ERNNConfig(**neural), **values)
    config.validate()
    return config


def _architecture(model: nn.Module, depth: int, width: int) -> dict:
    hidden = [model.hidden]
    if model.hidden2 is not None:
        hidden.append(model.hidden2)
    hidden.extend(list(getattr(model, "additional_hidden", ())))
    if len(hidden) != depth or any(not isinstance(layer, nn.Linear) for layer in hidden):
        raise ValueError("Actual hidden-layer structure differs from the selected depth")
    input_dim = hidden[0].in_features - 1
    for index, layer in enumerate(hidden):
        expected_input = input_dim + 1 if index == 0 else width
        if (layer.in_features, layer.out_features) != (expected_input, width):
            raise ValueError("Actual hidden-layer width differs from the selected architecture")
    if not isinstance(model.output, nn.Linear) or (
            model.output.in_features, model.output.out_features) != (width, 1):
        raise ValueError("Unexpected neural output layer")
    dimension = sum(p.numel() for p in model.parameters())
    if dimension != parameter_count(input_dim, width, depth):
        raise ValueError("Actual parameter count differs from the selected architecture")
    return {"actual_depth": len(hidden), "hidden_layers": len(hidden),
            "width_each_layer": width, "input_dim": input_dim,
            "parameter_count": dimension}


def _objective_audit(history: dict, location: str) -> dict:
    """Audit returned finite-step objectives, never claim stationarity."""
    _finite_tree(history, location)
    initial = float(history["initial_loss"])
    terminal = float(history["terminal_loss"])
    gradient = float(history["terminal_gradient_norm"])
    epochs = history["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"Invalid training epoch count at {location}")
    if not history.get("loss") or gradient < 0:
        raise ValueError(f"Incomplete objective/gradient history at {location}")
    tolerance = 1e-10 * max(1.0, abs(initial), abs(terminal))
    if terminal > initial + tolerance:
        raise FloatingPointError(f"Returned objective increased at {location}")
    phases = {name: _objective_audit(value, f"{location}.phases.{name}")
              for name, value in history.get("phases", {}).items()}
    result = {"finite_step_valid": True, "epochs": epochs,
              "initial_loss": initial, "terminal_loss": terminal,
              "terminal_gradient_norm": gradient, "theory_certified": False}
    if phases:
        result["phases"] = phases
    return result


def _round_histories(history: dict, expected_rounds: int, clients: int,
                     local_epochs: int, location: str) -> list:
    """Federated optimizers need finite histories, not pooled descent."""
    rounds = history["rounds"]
    if len(rounds) != expected_rounds:
        raise ValueError(f"Incorrect round count at {location}")
    for index, row in enumerate(rounds, 1):
        if row["round"] != index or len(row["clients"]) != clients:
            raise ValueError(f"Incomplete client/round history at {location}")
        for client in row["clients"]:
            if not 0 < client["epochs"] <= local_epochs:
                raise ValueError(f"Invalid local epoch count at {location}")
            for key in ("initial_loss", "terminal_loss", "terminal_gradient_norm"):
                if not math.isfinite(float(client[key])):
                    raise FloatingPointError(f"Nonfinite local diagnostic at {location}")
    return [{"round": row["round"],
             "local_epochs": [client["epochs"] for client in row["clients"]]}
            for row in rounds]


def _training_audit(method: str, history: dict, config: FederatedConfig,
                    clients: int) -> dict:
    _finite_tree(history, "training_history")
    audit = {"finite": True, "passed": True, "theory_certified": False,
             "note": "Finite numerical checks; no convergence or performance threshold."}
    if method in ("centralized", "host"):
        audit["objective"] = _objective_audit(history, method)
    elif method == "one_shot":
        if len(history["clients"]) != clients:
            raise ValueError("One-shot client history is incomplete")
        audit["local_objectives"] = [
            _objective_audit(row, f"one_shot.client{index}")
            for index, row in enumerate(history["clients"])]
        audit["note"] += " Local descent does not certify descent after parameter averaging."
    elif method == "extremilefed":
        audit["host_objective"] = _objective_audit(history["host"], "csl.host")
        rounds = history["rounds"]
        if len(rounds) != config.rounds:
            raise ValueError("CSL round history is incomplete")
        checks = []
        for index, row in enumerate(rounds, 1):
            if row["round"] != index or not row["finite_step_audit"]["numerically_valid"]:
                raise FloatingPointError(f"CSL finite-step audit failed in round {index}")
            if not 0 < row["solver"]["epochs"] <= config.surrogate_epochs:
                raise ValueError(f"Invalid CSL epoch count in round {index}")
            checks.append({"round": index,
                           "objective": _objective_audit(row["solver"], f"csl.round{index}"),
                           "finite_step_audit": row["finite_step_audit"]})
        audit["rounds"] = checks
    elif method in ("fedavg", "fedprox", "scaffold"):
        epochs = getattr(config, f"{method}_local_epochs")
        audit["rounds"] = _round_histories(history, config.rounds, clients, epochs, method)
        audit["note"] += " No local or pooled objective-monotonicity requirement."
    else:
        raise ValueError(f"Unsupported neural method: {method}")
    return audit


def _save_checkpoint(output_root: Path, task_id: str, vector: np.ndarray) -> dict:
    relative = Path("checkpoints") / f"{task_id}.npy"
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, vector, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Refusing to replace a different model checkpoint: {relative}")
        else:
            # Atomic publication without overwriting a competing task.
            os.link(temporary, destination)
        return {"path": relative.as_posix(), "sha256": checksum,
                "dtype": vector.dtype.str, "shape": list(vector.shape)}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_formal(task: dict, output_root: Path, protocol: dict) -> dict:
    """Run one selected method without changing the frozen fitting routines."""
    if not isinstance(protocol, dict):
        raise TypeError("The outer runner must supply its frozen protocol")
    task_id = task["id"]
    if (not isinstance(task_id, str) or not task_id
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", component)
                   or component in (".", "..") for component in task_id.split("/"))):
        raise ValueError("Task id must be a safe relative path with nonempty components")
    if task["stage"] not in ("readiness", "formal"):
        raise ValueError("Expected a readiness or formal task")
    method = task["method"]
    if method not in experiments.ALL_METHODS:
        raise ValueError(f"Unknown method: {method}")
    depth = validate_depth(task["depth"])
    config = _configuration(task["config"])
    job = SimulationJob(**task["job"])
    if task["setting"] != f"scenario{job.scenario}_{job.error}":
        raise ValueError("Task setting and simulation job disagree")
    if min(job.n_train, job.n_test, job.clients) < 1 or job.clients > job.n_train:
        raise ValueError("Invalid simulation sample/client counts")
    expected_second = None if depth == 1 else config.ernn.hidden_units
    if config.ernn.second_hidden_units != expected_second:
        raise ValueError("Neural configuration is inconsistent with the requested depth")

    captured = []
    original_metrics = experiments._neural_metrics

    def capture_metrics(fit, *args, **kwargs):
        captured.append(fit)
        return original_metrics(fit, *args, **kwargs)

    try:
        with depth_architecture(depth):
            experiments._neural_metrics = capture_metrics
            result = experiments.run_job(job, config, (method,))
    finally:
        experiments._neural_metrics = original_metrics
    _finite_tree(result)
    name = METHOD_NAMES[method]
    if set(result["methods"]) != {name} or result["job"] != asdict(job):
        raise ValueError("Underlying runner returned an unexpected method or job")
    record = result["methods"][name]
    if record["elapsed_seconds"] < 0 or record["communication_mb"] < 0:
        raise ValueError("Negative resource measurement")
    for key, value in record["metrics"].items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Invalid metric: {key}")
    for key in ("coverage_error", "crossing_rate"):
        if not 0 <= record["metrics"][key] <= 1:
            raise ValueError(f"Invalid diagnostic proportion: {key}")

    if method == "linear":
        if captured or record["rounds"] != 0 or record["communication_mb"] != 0:
            raise ValueError("Unexpected neural capture or communication for linear reference")
        result.update(architecture=None, model_checkpoint=None,
                      numerical_audit={"finite": True, "passed": True, "theory_certified": False,
                                       "note": "Linear reference; no neural history or vector captured."})
    else:
        if len(captured) != 1:
            raise ValueError("Expected exactly one evaluated neural model")
        fit = captured[0]
        architecture = _architecture(fit.model, depth, config.ernn.hidden_units)
        vector = parameter_vector(fit.model).detach().cpu().numpy().copy()
        if vector.ndim != 1 or vector.size != architecture["parameter_count"] or not np.isfinite(vector).all():
            raise FloatingPointError("Invalid final model parameter vector")
        audit = _training_audit(method, fit.history, config, job.clients)
        expected_rounds = 0 if method in ("centralized", "host") else (1 if method == "one_shot" else config.rounds)
        if record["rounds"] != expected_rounds:
            raise ValueError("Reported communication-round count is inconsistent")
        factor = 0 if method in ("centralized", "host") else (
            1 if method == "one_shot" else (4 if method == "scaffold" else 2))
        expected_bytes = factor * expected_rounds * (job.clients - 1) * vector.size * vector.dtype.itemsize
        expected_mib = expected_bytes / 1024.0**2
        if not math.isclose(record["communication_mb"], expected_mib, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Communication payload disagrees with the actual parameter dimension")
        audit["communication_check"] = {"rounds": expected_rounds,
                                         "expected_bytes": expected_bytes, "passed": True}
        scaler = fit.scaler
        result.update(architecture=architecture, training_history=fit.history,
                      numerical_audit=audit,
                      model_scaler={"x_mean": scaler.x_mean.tolist(), "x_scale": scaler.x_scale.tolist(),
                                    "y_mean": float(scaler.y_mean), "y_scale": float(scaler.y_scale)})
        _finite_tree(result)
        result["model_checkpoint"] = _save_checkpoint(Path(output_root), task_id, vector)
    result.update(task_id=task_id, stage=task["stage"])
    return result
