"""Centralized, host-only, averaging, and federated optimizer baselines."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .config import ERNNConfig
from .federated import sample_size_weighted_average
from .model import ERNNResult
from .preprocessing import fit_federated_standardizer
from .training import fit_standardized, fit_standardized_staged, initialize_model, load_parameter_vector, parameter_vector


@dataclass
class BaselineResult:
    ernn_result: ERNNResult
    method: str
    rounds: int
    runtime_seconds: float
    communication_mb: float
    history: dict[str, Any]


def _prepare(partitions: list[tuple[np.ndarray, np.ndarray]]):
    scaler = fit_federated_standardizer(partitions)
    standardized = [(scaler.transform_X(X), scaler.transform_y(y)) for X, y in partitions]
    sizes = [len(y) for _, y in standardized]
    return scaler, standardized, sizes


def train_host(partitions: list[tuple[np.ndarray, np.ndarray]], config: ERNNConfig) -> BaselineResult:
    scaler, standardized, _ = _prepare(partitions)
    started = perf_counter()
    model, history = fit_standardized_staged(*standardized[0], config)
    result = ERNNResult(model, scaler, config, history)
    return BaselineResult(result, "Host ERNN", 0, perf_counter() - started, 0.0, history)


def train_one_shot(partitions: list[tuple[np.ndarray, np.ndarray]], config: ERNNConfig) -> BaselineResult:
    scaler, standardized, sizes = _prepare(partitions)
    initial_model = initialize_model(standardized[0][0].shape[1], config)
    initial = parameter_vector(initial_model).detach().cpu().numpy()
    started = perf_counter()
    vectors = []
    local_histories = []
    for X_client, y_client in standardized:
        model, history = fit_standardized_staged(X_client, y_client, config, init_vector=initial)
        vectors.append(parameter_vector(model).detach().cpu().numpy())
        local_histories.append(history)
    averaged = sample_size_weighted_average(vectors, sizes)
    load_parameter_vector(initial_model, averaged)
    result = ERNNResult(initial_model, scaler, config, {"clients": local_histories})
    bytes_per = 8 if config.dtype == "float64" else 4
    communication = max(0, len(partitions) - 1) * averaged.size * bytes_per / (1024.0**2)
    return BaselineResult(
        result,
        "One-shot ERNN",
        1,
        perf_counter() - started,
        communication,
        {"clients": local_histories},
    )


def train_fedavg(
    partitions: list[tuple[np.ndarray, np.ndarray]],
    config: ERNNConfig,
    rounds: int,
    local_epochs: int = 5,
    local_learning_rate: float = 1e-3,
    prox_mu: float = 0.0,
) -> BaselineResult:
    scaler, standardized, sizes = _prepare(partitions)
    global_model = initialize_model(standardized[0][0].shape[1], config)
    started = perf_counter()
    round_history = []
    for round_index in range(int(rounds)):
        broadcast = parameter_vector(global_model).detach().cpu().numpy()
        vectors = []
        local_histories = []
        for X_client, y_client in standardized:
            local_model, history = fit_standardized(
                X_client,
                y_client,
                config,
                init_vector=broadcast,
                max_epochs=local_epochs,
                learning_rate=local_learning_rate,
                prox_center=broadcast if prox_mu else None,
                prox_mu=prox_mu,
            )
            vectors.append(parameter_vector(local_model).detach().cpu().numpy())
            local_histories.append(history)
        averaged = sample_size_weighted_average(vectors, sizes)
        load_parameter_vector(global_model, averaged)
        round_history.append({"round": round_index + 1, "clients": local_histories})
    parameter_dim = int(parameter_vector(global_model).numel())
    bytes_per = 8 if config.dtype == "float64" else 4
    communication = 2 * rounds * max(0, len(partitions) - 1) * parameter_dim * bytes_per / (1024.0**2)
    method = "FedProx-ERNN" if prox_mu else "FedAvg-ERNN"
    result = ERNNResult(global_model, scaler, config, {"rounds": round_history})
    return BaselineResult(
        result,
        method,
        rounds,
        perf_counter() - started,
        communication,
        {"rounds": round_history, "prox_mu": prox_mu},
    )


def train_scaffold(
    partitions: list[tuple[np.ndarray, np.ndarray]],
    config: ERNNConfig,
    rounds: int,
    local_epochs: int = 5,
    local_learning_rate: float = 1e-3,
) -> BaselineResult:
    """Train SCAFFOLD with full client participation and weighted controls."""

    if rounds <= 0 or local_epochs <= 0 or local_learning_rate <= 0:
        raise ValueError("rounds, local_epochs and local_learning_rate must be positive")
    scaler, standardized, sizes = _prepare(partitions)
    global_model = initialize_model(standardized[0][0].shape[1], config)
    dimension = int(parameter_vector(global_model).numel())
    client_controls = [np.zeros(dimension, dtype=float) for _ in standardized]
    server_control = np.zeros(dimension, dtype=float)
    started = perf_counter()
    round_history = []
    for round_index in range(int(rounds)):
        broadcast = parameter_vector(global_model).detach().cpu().numpy()
        local_vectors = []
        new_controls = []
        local_histories = []
        local_steps = []
        for client_index, (X_client, y_client) in enumerate(standardized):
            correction = client_controls[client_index] - server_control
            local_model, history = fit_standardized(
                X_client,
                y_client,
                config,
                init_vector=broadcast,
                linear_correction=correction,
                max_epochs=local_epochs,
                learning_rate=local_learning_rate,
                fixed_sgd=True,
            )
            local_vector = parameter_vector(local_model).detach().cpu().numpy()
            steps = max(1, int(history["epochs"]) * int(np.ceil(len(y_client) / config.batch_size)))
            updated_control = (
                client_controls[client_index]
                - server_control
                + (broadcast - local_vector) / (steps * local_learning_rate)
            )
            local_vectors.append(local_vector)
            new_controls.append(updated_control)
            local_histories.append(history)
            local_steps.append(steps)
        averaged = sample_size_weighted_average(local_vectors, sizes)
        server_control = sample_size_weighted_average(new_controls, sizes)
        client_controls = new_controls
        load_parameter_vector(global_model, averaged)
        round_history.append(
            {
                "round": round_index + 1,
                "client_steps": local_steps,
                "server_control_norm": float(np.linalg.norm(server_control)),
                "clients": local_histories,
            }
        )
    bytes_per = 8 if config.dtype == "float64" else 4
    # Each non-host client receives parameters/control and uploads parameters/control.
    communication = 4 * rounds * max(0, len(partitions) - 1) * dimension * bytes_per / (1024.0**2)
    history = {"rounds": round_history, "full_participation": True}
    result = ERNNResult(global_model, scaler, config, history)
    return BaselineResult(
        result,
        "SCAFFOLD-ERNN",
        rounds,
        perf_counter() - started,
        communication,
        history,
    )
