"""Centralized ERNN fitting and exact full-data gradient utilities."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .config import ERNNConfig
from .model import ERNN, ERNNResult, data_loss, torch_dtype
from .preprocessing import Standardizer, fit_standardizer
from .quadrature import gauss_legendre_unit


def initialize_model(input_dim: int, config: ERNNConfig) -> ERNN:
    config.validate()
    torch.manual_seed(int(config.seed))
    model = ERNN(input_dim, config.hidden_units, config.second_hidden_units)
    model.to(device=config.device, dtype=torch_dtype(config.dtype))
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.hidden.weight)
        torch.nn.init.zeros_(model.hidden.bias)
        if model.hidden2 is not None:
            torch.nn.init.xavier_uniform_(model.hidden2.weight)
            torch.nn.init.zeros_(model.hidden2.bias)
        torch.nn.init.xavier_uniform_(model.output.weight)
        torch.nn.init.zeros_(model.output.bias)
    return model


def _tensor_quadrature(model: ERNN, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = gauss_legendre_unit(count)
    parameter = next(model.parameters())
    return (
        torch.as_tensor(nodes, dtype=parameter.dtype, device=parameter.device),
        torch.as_tensor(weights, dtype=parameter.dtype, device=parameter.device),
    )


def parameter_vector(model: ERNN) -> torch.Tensor:
    return parameters_to_vector(model.parameters())


def load_parameter_vector(model: ERNN, vector: np.ndarray | torch.Tensor) -> None:
    reference = next(model.parameters())
    # vector_to_parameters installs views. Own the storage so local optimization
    # cannot mutate a caller's broadcast, trust center, or shared initializer.
    tensor = torch.as_tensor(vector, dtype=reference.dtype, device=reference.device).detach().clone()
    vector_to_parameters(tensor, model.parameters())


def _batches(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def fit_standardized(
    X_std: np.ndarray,
    y_std: np.ndarray,
    config: ERNNConfig,
    *,
    init_vector: np.ndarray | torch.Tensor | None = None,
    linear_correction: np.ndarray | torch.Tensor | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    trust_center: np.ndarray | torch.Tensor | None = None,
    trust_radius: float | None = None,
    prox_center: np.ndarray | torch.Tensor | None = None,
    prox_mu: float = 0.0,
    fixed_sgd: bool = False,
    full_batch: bool = False,
) -> tuple[ERNN, dict[str, object]]:
    model = initialize_model(X_std.shape[1], config)
    if init_vector is not None:
        load_parameter_vector(model, init_vector)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    nodes, weights = _tensor_quadrature(model, config.quadrature.training_nodes)
    X_t = torch.as_tensor(X_std, dtype=dtype, device=device)
    y_t = torch.as_tensor(y_std, dtype=dtype, device=device)
    optimizer_class = torch.optim.SGD if fixed_sgd else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=float(learning_rate or config.learning_rate))
    rng = np.random.default_rng(config.seed + 13)
    correction_t = None
    if linear_correction is not None:
        correction_t = torch.as_tensor(linear_correction, dtype=dtype, device=device)
    center_t = None
    if trust_center is not None:
        center_t = torch.as_tensor(trust_center, dtype=dtype, device=device)
    prox_center_t = None
    if prox_center is not None:
        prox_center_t = torch.as_tensor(prox_center, dtype=dtype, device=device)
    if prox_mu < 0:
        raise ValueError("prox_mu must be non-negative")
    if full_batch and fixed_sgd:
        raise ValueError("fixed-step baseline SGD must retain its original minibatch protocol")
    def take_step():
        if not fixed_sgd:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        if center_t is not None and trust_radius is not None:
            with torch.no_grad():
                vector = parameter_vector(model)
                displacement = vector - center_t
                norm = torch.linalg.vector_norm(displacement)
                if norm > float(trust_radius):
                    vector = center_t + displacement * (float(trust_radius) / norm)
                    vector_to_parameters(vector, model.parameters())
    def full_objective():
        with torch.no_grad():
            value = torch.zeros((), dtype=dtype, device=device)
            for start in range(0, len(y_std), config.batch_size):
                stop = min(start + config.batch_size, len(y_std))
                piece, _ = data_loss(model, X_t[start:stop], y_t[start:stop], nodes, weights, config.h, config.eta)
                value += piece * ((stop - start) / len(y_std))
            value += config.lambda_ * model.l2_first_layer()
            if correction_t is not None:
                value -= torch.dot(correction_t, parameter_vector(model))
            if prox_center_t is not None and prox_mu:
                value += .5 * prox_mu * torch.sum((parameter_vector(model) - prox_center_t).square())
            result = float(value.cpu())
            if not np.isfinite(result):
                raise FloatingPointError("non-finite complete training objective")
            return result
    epochs = int(max_epochs or config.max_epochs)
    initial_loss = full_objective()
    best_loss = initial_loss
    best_epoch = 0
    best_vector = parameter_vector(model).detach().clone()
    stale = 0
    losses: list[float] = []
    trajectory_losses: list[float] = []
    crossings: list[float] = []
    started = perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_crossing = 0.0
        model.train()
        if full_batch:
            optimizer.zero_grad(set_to_none=True)
        for idx in _batches(len(y_std), min(config.batch_size, len(y_std)), rng):
            if not full_batch:
                optimizer.zero_grad(set_to_none=True)
            loss, pieces = data_loss(
                model, X_t[idx], y_t[idx], nodes, weights, config.h, config.eta
            )
            objective = loss + float(config.lambda_) * model.l2_first_layer()
            if correction_t is not None:
                objective = objective - torch.dot(correction_t, parameter_vector(model))
            if prox_center_t is not None and prox_mu:
                objective = objective + 0.5 * float(prox_mu) * torch.sum(
                    (parameter_vector(model) - prox_center_t).square()
                )
            # Exact full-data mean gradient, accumulated without a giant n*K graph.
            weight = len(idx) / len(y_std)
            (objective * (weight if full_batch else 1.0)).backward()
            if not torch.isfinite(objective) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("non-finite objective or gradient during training")
            if not full_batch:
                take_step()
            epoch_loss += float(objective.detach().cpu()) * weight
            epoch_crossing += float(pieces["crossing"].detach().cpu()) * weight
        if full_batch:
            take_step()
        trajectory_losses.append(epoch_loss)
        epoch_loss = full_objective()
        losses.append(epoch_loss)
        crossings.append(epoch_crossing)
        if epoch_loss < best_loss - 1e-10:
            best_loss = epoch_loss
            best_vector = parameter_vector(model).detach().clone()
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if not fixed_sgd and epoch + 1 >= config.min_epochs and stale >= config.patience:
            break
    if not fixed_sgd:
        load_parameter_vector(model, best_vector)
    correction = np.zeros(int(parameter_vector(model).numel())) if correction_t is None else correction_t.detach().cpu().numpy()
    gradient = surrogate_gradient(model, X_std, y_std, config, correction)
    if prox_center_t is not None and prox_mu:
        gradient += prox_mu * (parameter_vector(model).detach().cpu().numpy() - prox_center_t.detach().cpu().numpy())
    history: dict[str, object] = {
        "loss": losses,
        "trajectory_loss": trajectory_losses,
        "loss_definition": "complete objective at the end of each epoch",
        "initial_loss": initial_loss,
        "terminal_loss": full_objective(),
        "terminal_gradient_norm": float(np.linalg.norm(gradient)),
        "best_epoch": best_epoch,
        "stopping_reason": "epoch_budget" if len(losses) == epochs else "patience",
        "crossing_penalty": crossings,
        "epochs": len(losses),
        "optimizer": "SGD_fixed_steps" if fixed_sgd else "Adam",
        "full_batch_updates": full_batch,
        "gradient_chunk_size": config.batch_size,
        "best_loss": best_loss,
        "runtime_seconds": perf_counter() - started,
        "config_fingerprint": config.fingerprint(),
    }
    return model, history


def fit_standardized_staged(X_std, y_std, config, *, init_vector=None):
    """Shared coarse/refinement schedule for centralized and host fitting."""
    config.validate()
    model, history = fit_standardized(X_std, y_std, config, init_vector=init_vector)
    if config.refinement_epochs:
        coarse = history
        fine_config = replace(config, max_epochs=config.refinement_epochs,
            learning_rate=config.refinement_learning_rate, refinement_epochs=0,
            min_epochs=config.refinement_epochs, patience=config.refinement_epochs + 1)
        model, fine = fit_standardized(X_std, y_std, fine_config,
            init_vector=parameter_vector(model).detach().clone(), full_batch=True)
        history = dict(fine)
        history.update(phases={"coarse": coarse, "refinement": fine},
            initial_loss=coarse["initial_loss"], epochs=coarse["epochs"] + fine["epochs"],
            best_epoch=(coarse["epochs"] + fine["best_epoch"] if fine["best_epoch"] else coarse["best_epoch"]),
            runtime_seconds=coarse["runtime_seconds"] + fine["runtime_seconds"],
            loss_definition="complete objective at each refinement epoch; coarse trajectory is in phases",
            optimizer="Adam minibatch then Adam full-data refinement", config_fingerprint=config.fingerprint())
    return model, history


def train_ernn(X: np.ndarray, y: np.ndarray, config: ERNNConfig) -> ERNNResult:
    config.validate()
    scaler = fit_standardizer(X, y)
    model, history = fit_standardized_staged(scaler.transform_X(X), scaler.transform_y(y), config)
    return ERNNResult(model=model, scaler=scaler, config=config, history=history)


def data_gradient(
    model: ERNN,
    X_std: np.ndarray,
    y_std: np.ndarray,
    config: ERNNConfig,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return the exact gradient of the average data loss, excluding L2."""

    model.zero_grad(set_to_none=True)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    nodes, weights = _tensor_quadrature(model, config.quadrature.training_nodes)
    X_t = torch.as_tensor(X_std, dtype=dtype, device=device)
    y_t = torch.as_tensor(y_std, dtype=dtype, device=device)
    step = int(batch_size or config.batch_size)
    n = len(y_std)
    for start in range(0, n, step):
        stop = min(start + step, n)
        loss, _ = data_loss(
            model, X_t[start:stop], y_t[start:stop], nodes, weights, config.h, config.eta
        )
        (loss * ((stop - start) / n)).backward()
    gradients = [parameter.grad.reshape(-1) for parameter in model.parameters()]
    return torch.cat(gradients).detach().cpu().numpy().astype(float)


def surrogate_gradient(
    model: ERNN,
    X_std: np.ndarray,
    y_std: np.ndarray,
    config: ERNNConfig,
    correction: np.ndarray,
) -> np.ndarray:
    data_grad = data_gradient(model, X_std, y_std, config)
    model.zero_grad(set_to_none=True)
    penalty = float(config.lambda_) * model.l2_first_layer()
    penalty.backward()
    penalty_grad = torch.cat(
        [
            p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
            for p in model.parameters()
        ]
    ).detach().cpu().numpy()
    return data_grad + penalty_grad - np.asarray(correction, dtype=float)
