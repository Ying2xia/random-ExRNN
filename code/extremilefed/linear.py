"""Pooled supervised linear process with a cubic polynomial basis.

Uses (1,u,1.5*u**2-.5,2.5*u**3-1.5*u), spanning the alternative
(1,u,u**2,u**3) permitted in Section 2.1 of Jiang et al. (2026).
This is a numerical Adam implementation, not a call to the reference qrcm solver.
The legacy function name below is retained for API compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .quadrature import gauss_legendre_unit, normalized_extremile_weights
from .preprocessing import Standardizer, fit_standardizer


def shifted_legendre_basis(u: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.ones_like(u),
            u,
            1.5 * u.square() - 0.5,
            2.5 * u.pow(3) - 1.5 * u,
        ],
        dim=1,
    )


@dataclass
class LinearExtremileResult:
    alpha: np.ndarray
    scaler: Standardizer

    def predict_quantiles(self, X: np.ndarray, levels: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform_X(X)
        design = np.column_stack([np.ones(Xs.shape[0]), Xs])
        u = np.asarray(levels, dtype=float)
        basis = np.column_stack([np.ones_like(u), u, 1.5 * u**2 - 0.5, 2.5 * u**3 - 1.5 * u])
        prediction = design @ self.alpha @ basis.T
        return self.scaler.inverse_y(prediction)

    def predict_extremiles(self, X: np.ndarray, tau_levels: np.ndarray, n_nodes: int = 128) -> np.ndarray:
        nodes, weights = gauss_legendre_unit(n_nodes)
        quantiles = self.predict_quantiles(X, nodes)
        return np.column_stack(
            [quantiles @ normalized_extremile_weights(nodes, weights, float(tau)) for tau in tau_levels]
        )


def train_linear_extremile_sl(
    X: np.ndarray,
    y: np.ndarray,
    *,
    nodes: int = 32,
    max_epochs: int = 500,
    learning_rate: float = 0.03,
    seed: int = 20260903,
) -> LinearExtremileResult:
    scaler = fit_standardizer(X, y)
    Xs = scaler.transform_X(X)
    ys = scaler.transform_y(y)
    design = np.column_stack([np.ones(Xs.shape[0]), Xs])
    u_np, weights_np = gauss_legendre_unit(nodes)
    torch.manual_seed(seed)
    dtype = torch.float64
    design_t = torch.as_tensor(design, dtype=dtype)
    y_t = torch.as_tensor(ys, dtype=dtype)
    u_t = torch.as_tensor(u_np, dtype=dtype)
    weights_t = torch.as_tensor(weights_np, dtype=dtype)
    basis_t = shifted_legendre_basis(u_t)
    alpha = torch.zeros((design.shape[1], basis_t.shape[1]), dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=learning_rate)
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = design_t @ alpha @ basis_t.T
        residual = y_t[:, None] - predictions
        check = torch.maximum(u_t[None, :] * residual, (u_t[None, :] - 1.0) * residual)
        loss = torch.mean(torch.sum(check * weights_t[None, :], dim=1))
        loss.backward()
        optimizer.step()
    return LinearExtremileResult(alpha.detach().cpu().numpy(), scaler)
