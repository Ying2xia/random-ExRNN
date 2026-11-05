"""Leakage-safe standardization for centralized and federated fits."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Standardizer:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.x_mean) / self.x_scale

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.y_mean) / self.y_scale

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * self.y_scale + self.y_mean


def fit_standardizer(X: np.ndarray, y: np.ndarray) -> Standardizer:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be 2D and y must be a conformable vector")
    x_scale = X.std(axis=0, ddof=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_scale = float(y.std(ddof=0))
    if y_scale <= 1e-12:
        y_scale = 1.0
    return Standardizer(X.mean(axis=0), x_scale, float(y.mean()), y_scale)


def fit_federated_standardizer(
    partitions: list[tuple[np.ndarray, np.ndarray]],
) -> Standardizer:
    if not partitions:
        raise ValueError("partitions must not be empty")
    sizes = np.array([len(y) for _, y in partitions], dtype=float)
    if np.any(sizes <= 0):
        raise ValueError("every client must contain at least one observation")
    n = float(sizes.sum())
    x_sum = sum(np.asarray(X, dtype=float).sum(axis=0) for X, _ in partitions)
    x_sq_sum = sum((np.asarray(X, dtype=float) ** 2).sum(axis=0) for X, _ in partitions)
    y_sum = sum(float(np.asarray(y, dtype=float).sum()) for _, y in partitions)
    y_sq_sum = sum(float((np.asarray(y, dtype=float) ** 2).sum()) for _, y in partitions)
    x_mean = x_sum / n
    x_var = np.maximum(x_sq_sum / n - x_mean**2, 0.0)
    x_scale = np.sqrt(x_var)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_mean = y_sum / n
    y_scale = float(np.sqrt(max(y_sq_sum / n - y_mean**2, 0.0)))
    if y_scale <= 1e-12:
        y_scale = 1.0
    return Standardizer(x_mean, x_scale, float(y_mean), y_scale)
