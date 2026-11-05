"""ERNN architecture, loss components, and prediction interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ERNNConfig
from .preprocessing import Standardizer
from .quadrature import gauss_legendre_unit, normalized_extremile_weights


def torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


class ERNN(nn.Module):
    """Conditional quantile-process network; the second layer is sensitivity-only."""

    def __init__(
        self, input_dim: int, hidden_units: int, second_hidden_units: int | None = None
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_units = int(hidden_units)
        self.hidden = nn.Linear(self.input_dim + 1, self.hidden_units)
        self.hidden2 = (
            nn.Linear(self.hidden_units, int(second_hidden_units))
            if second_hidden_units is not None
            else None
        )
        output_width = int(second_hidden_units or self.hidden_units)
        self.output = nn.Linear(output_width, 1)

    def forward(self, X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u = u[:, None]
        z = torch.cat([X, u], dim=1)
        activation = torch.tanh(self.hidden(z))
        if self.hidden2 is not None:
            activation = torch.tanh(self.hidden2(activation))
        return self.output(activation).squeeze(-1)

    def forward_grid(self, X: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        n, k = X.shape[0], nodes.numel()
        X_rep = X[:, None, :].expand(n, k, X.shape[1]).reshape(n * k, X.shape[1])
        u_rep = nodes[None, :].expand(n, k).reshape(n * k)
        return self.forward(X_rep, u_rep).reshape(n, k)

    def l2_first_layer(self) -> torch.Tensor:
        return torch.mean(self.hidden.weight.square())


def smooth_pinball(
    residual: torch.Tensor, nodes: torch.Tensor, h: float
) -> torch.Tensor:
    """Gaussian convolution E[rho_u(residual - h * Z)], Z ~ N(0,1).

    h is the Gaussian standard deviation, not a logistic scale.
    Complementary error functions avoid subtracting a CDF near one.
    """
    z = residual / float(h)
    lower = 0.5 * torch.erfc(-z / (2.0 ** 0.5))
    upper = 0.5 * torch.erfc(z / (2.0 ** 0.5))
    coefficient = torch.where(z >= 0, nodes[None, :] - upper,
                              (nodes[None, :] - 1.0) + lower)
    density = torch.exp(-0.5 * z.square()) / (2.0 * np.pi) ** 0.5
    return residual * coefficient + float(h) * density


def noncrossing_penalty(predictions: torch.Tensor, delta: float) -> torch.Tensor:
    differences = predictions[:, :-1] - predictions[:, 1:]
    smooth_positive = float(delta) * F.softplus(differences / float(delta))
    return torch.mean(smooth_positive.square())


def data_loss(
    model: ERNN,
    X: torch.Tensor,
    y: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    h: float,
    eta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predictions = model.forward_grid(X, nodes)
    residual = y[:, None] - predictions
    pinball = torch.mean(torch.sum(smooth_pinball(residual, nodes, h) * weights[None, :], dim=1))
    crossing = noncrossing_penalty(predictions, h)
    total = pinball + float(eta) * crossing
    return total, {"pinball": pinball, "crossing": crossing}


@dataclass
class ERNNResult:
    model: ERNN
    scaler: Standardizer
    config: ERNNConfig
    history: dict[str, Any] = field(default_factory=dict)


def predict_quantiles(
    result: ERNNResult,
    X: np.ndarray,
    u_levels: np.ndarray | tuple[float, ...] | list[float],
    batch_size: int = 4096,
) -> np.ndarray:
    levels = np.asarray(u_levels, dtype=float)
    if np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("all u_levels must lie in (0, 1)")
    model = result.model
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    nodes = torch.as_tensor(levels, dtype=dtype, device=device)
    X_std = result.scaler.transform_X(X)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_std.shape[0], int(batch_size)):
            X_t = torch.as_tensor(X_std[start : start + batch_size], dtype=dtype, device=device)
            outputs.append(model.forward_grid(X_t, nodes).cpu().numpy())
    return result.scaler.inverse_y(np.vstack(outputs))


def predict_extremiles(
    result: ERNNResult,
    X: np.ndarray,
    tau_levels: np.ndarray | tuple[float, ...] | list[float],
    quadrature: int | None = None,
) -> np.ndarray:
    n_nodes = int(quadrature or result.config.quadrature.prediction_nodes)
    nodes, weights = gauss_legendre_unit(n_nodes)
    quantiles = predict_quantiles(result, X, nodes)
    columns = []
    for tau in np.asarray(tau_levels, dtype=float):
        normalized = normalized_extremile_weights(nodes, weights, float(tau))
        columns.append(quantiles @ normalized)
    return np.column_stack(columns)
