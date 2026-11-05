"""Distributional, extremile, and communication metrics."""

from __future__ import annotations

import numpy as np


def pinball_loss(y: np.ndarray, predictions: np.ndarray, levels: np.ndarray, weights: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    levels = np.asarray(levels, dtype=float)
    residual = y[:, None] - predictions
    loss = np.where(residual >= 0.0, levels[None, :] * residual, (levels[None, :] - 1.0) * residual)
    if weights is None:
        return float(np.mean(loss))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != levels.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid quadrature weights")
    return float(np.mean(loss @ (weights / weights.sum())))


def crossing_rate(predictions: np.ndarray, tolerance: float = 0.0) -> float:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 2:
        raise ValueError("predictions must have shape (n, at least 2)")
    return float(np.mean(np.any(np.diff(predictions, axis=1) < -float(tolerance), axis=1)))


def coverage_error(y: np.ndarray, predictions: np.ndarray, levels: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    levels = np.asarray(levels, dtype=float)
    empirical = np.mean(y[:, None] <= predictions, axis=0)
    return float(np.mean(np.abs(empirical - levels)))


def extremile_errors(predictions: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    difference = np.asarray(predictions, dtype=float) - np.asarray(truth, dtype=float)
    return {
        "extremile_mae": float(np.mean(np.abs(difference))),
        "extremile_rmse": float(np.sqrt(np.mean(difference**2))),
    }


def communication_mb(rounds: int, clients: int, parameter_dim: int, bytes_per_value: int = 8) -> float:
    values = 2 * int(rounds) * max(0, int(clients) - 1) * int(parameter_dim)
    return float(values * int(bytes_per_value) / (1024.0**2))
