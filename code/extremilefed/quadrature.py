"""Quadrature and extremile distortion weights."""

from __future__ import annotations

import numpy as np


def gauss_legendre_unit(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted Gauss-Legendre nodes and weights on (0, 1)."""

    if int(n) < 2:
        raise ValueError("n must be at least two")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(int(n))
    nodes = 0.5 * (raw_nodes + 1.0)
    weights = 0.5 * raw_weights
    order = np.argsort(nodes)
    return nodes[order].astype(float), weights[order].astype(float)


def extremile_weight(u: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate J_tau(u) for tau and u strictly inside (0, 1)."""

    tau = float(tau)
    u = np.asarray(u, dtype=float)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie in (0, 1)")
    if np.any((u <= 0.0) | (u >= 1.0)):
        raise ValueError("u must lie strictly inside (0, 1)")
    if np.isclose(tau, 0.5):
        return np.ones_like(u)
    if tau < 0.5:
        s = np.log(0.5) / np.log1p(-tau)
        return s * np.power(1.0 - u, s - 1.0)
    r = np.log(0.5) / np.log(tau)
    return r * np.power(u, r - 1.0)


def normalized_extremile_weights(
    nodes: np.ndarray, weights: np.ndarray, tau: float
) -> np.ndarray:
    values = np.asarray(weights, dtype=float) * extremile_weight(nodes, tau)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("extremile quadrature weights are invalid")
    return values / total
