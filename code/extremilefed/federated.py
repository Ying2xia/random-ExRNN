"""Communication-efficient surrogate refinement for ERNN."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np

from .config import FederatedConfig
from .solver_audit import audit_step
from .model import ERNNResult
from .preprocessing import Standardizer, fit_federated_standardizer
from .training import (
    data_gradient,
    fit_standardized,
    fit_standardized_staged,
    parameter_vector,
    surrogate_gradient,
)


@dataclass
class FederatedResult:
    ernn_result: ERNNResult
    config: FederatedConfig
    client_sizes: tuple[int, ...]
    history: dict[str, Any] = field(default_factory=dict)


def sample_size_weighted_average(
    values: list[np.ndarray], sample_sizes: list[int] | tuple[int, ...]
) -> np.ndarray:
    if not values:
        raise ValueError("values must not be empty")
    sizes = np.asarray(sample_sizes, dtype=float)
    if sizes.shape != (len(values),) or np.any(sizes <= 0):
        raise ValueError("one positive sample size is required per value")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("all arrays must have the same shape")
    return np.sum(np.stack(values) * (sizes / sizes.sum())[:, None], axis=0)


def partition_indices(n: int, clients: int, seed: int) -> list[np.ndarray]:
    if clients <= 0 or clients > n:
        raise ValueError("clients must lie between one and n")
    rng = np.random.default_rng(seed)
    return [part.astype(int) for part in np.array_split(rng.permutation(n), clients)]


def train_extremilefed(
    partitions: list[tuple[np.ndarray, np.ndarray]], config: FederatedConfig
) -> FederatedResult:
    config.validate()
    if not partitions:
        raise ValueError("partitions must not be empty")
    p = np.asarray(partitions[0][0]).shape[1]
    if any(np.asarray(X).ndim != 2 or np.asarray(X).shape[1] != p for X, _ in partitions):
        raise ValueError("all client feature matrices must have the same width")
    scaler = fit_federated_standardizer(partitions)
    standardized = [
        (scaler.transform_X(X), scaler.transform_y(y)) for X, y in partitions
    ]
    sizes = tuple(len(y) for _, y in standardized)
    host_X, host_y = standardized[0]
    started = perf_counter()
    model, host_history = fit_standardized_staged(host_X, host_y, config.host_config())
    round_history: list[dict[str, Any]] = []
    parameter_dim = int(parameter_vector(model).numel())
    for round_index in range(config.rounds):
        broadcast = parameter_vector(model).detach().cpu().numpy().copy()
        local_gradients = [
            data_gradient(model, X_client, y_client, config.ernn)
            for X_client, y_client in standardized
        ]
        global_gradient = sample_size_weighted_average(local_gradients, sizes)
        host_gradient = local_gradients[0]
        correction = host_gradient - global_gradient
        model, solve_history = fit_standardized(
            host_X,
            host_y,
            replace(config.ernn, patience=config.surrogate_patience or config.ernn.patience, refinement_epochs=0),
            init_vector=broadcast,
            linear_correction=correction,
            max_epochs=config.surrogate_epochs,
            learning_rate=(config.surrogate_learning_rates[round_index]
                if config.surrogate_learning_rates else config.surrogate_learning_rate),
            trust_center=broadcast,
            trust_radius=config.trust_radius,
        )
        residual = surrogate_gradient(model, host_X, host_y, config.ernn, correction)
        round_history.append(
            {
                "round": round_index + 1,
                "global_gradient_norm": float(np.linalg.norm(global_gradient)),
                "correction_norm": float(np.linalg.norm(correction)),
                "surrogate_gradient_norm": float(np.linalg.norm(residual)),
                "solver": solve_history,
                "finite_step_audit": audit_step(
                    solve_history["initial_loss"], solve_history["terminal_loss"],
                    broadcast, parameter_vector(model).detach().cpu().numpy(), residual,
                    radius=config.trust_radius,
                ),
            }
        )
    bytes_per_value = 8 if config.ernn.dtype == "float64" else 4
    communication_bytes = 2 * config.rounds * max(0, len(partitions) - 1) * parameter_dim * bytes_per_value
    result = ERNNResult(
        model=model,
        scaler=scaler,
        config=config.ernn,
        history={"host": host_history, "rounds": round_history},
    )
    history = {
        "host": host_history,
        "rounds": round_history,
        "runtime_seconds": perf_counter() - started,
        "parameter_dim": parameter_dim,
        "communication_bytes": communication_bytes,
        "communication_mb": communication_bytes / (1024.0**2),
        "protocol": "parameter broadcast + full local data-gradient upload",
        "raw_data_transmitted": False,
        "hessian_transmitted": False,
    }
    return FederatedResult(result, config, sizes, history)
