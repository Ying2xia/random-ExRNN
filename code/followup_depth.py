"""Isolated, fixed-width depth sensitivity without changing the frozen ERNN code.

Use ``with depth_architecture(depth):`` around a complete fit in a single
process. One and two hidden layers are the original ERNN class, bit for bit.
Depths three and four extend the same tanh network and retain its first-layer
penalty. This context changes a module global; use process workers, not threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral
from threading import current_thread, main_thread
from typing import Iterator

import torch
from torch import nn

from extremilefed import training
from extremilefed.model import ERNN


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_depth(depth: int) -> int:
    """Return an integer hidden-layer count in the prespecified grid 1--4."""
    depth = _positive_integer(depth, "depth")
    if depth not in (1, 2, 3, 4):
        raise ValueError("depth must be one of 1, 2, 3, 4 hidden layers")
    return depth


def parameter_count(input_dim: int, width: int, depth: int) -> int:
    """Count all weights and biases, including the quantile-level input."""
    input_dim = _positive_integer(input_dim, "input_dim")
    width = _positive_integer(width, "width")
    depth = validate_depth(depth)
    return (input_dim + 3) * width + 1 + (depth - 1) * width * (width + 1)


class _DeepERNN(ERNN):
    """Three/four-layer extension, constructed only by the depth context."""

    def __init__(self, input_dim: int, hidden_units: int, depth: int) -> None:
        nn.Module.__init__(self)
        self.input_dim = int(input_dim)
        self.hidden_units = int(hidden_units)
        self.depth = int(depth)
        self.hidden = nn.Linear(self.input_dim + 1, self.hidden_units)
        self.hidden2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.additional_hidden = nn.ModuleList(
            nn.Linear(self.hidden_units, self.hidden_units) for _ in range(depth - 2)
        )
        self.output = nn.Linear(self.hidden_units, 1)
        self._additional_initialized = False

    def to(self, *args: object, **kwargs: object) -> _DeepERNN:
        result = super().to(*args, **kwargs)
        # The frozen initialize_model converts to the requested dtype/device
        # before applying Xavier initialization to hidden, hidden2 and output.
        # Initialize the added layers at that same precision on this first
        # conversion. Subsequent .to() calls must never reset fitted parameters.
        if not self._additional_initialized:
            with torch.no_grad():
                for layer in self.additional_hidden:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            self._additional_initialized = True
        return result

    def forward(self, X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u = u[:, None]
        activation = torch.tanh(self.hidden(torch.cat([X, u], dim=1)))
        activation = torch.tanh(self.hidden2(activation))
        for layer in self.additional_hidden:
            activation = torch.tanh(layer(activation))
        return self.output(activation).squeeze(-1)


@contextmanager
def depth_architecture(depth: int) -> Iterator[None]:
    """Temporarily select a fixed-width network for all existing fit routines.

    Keep ``config.second_hidden_units=None`` for depth one and set it equal to
    ``config.hidden_units`` for depths two to four. The runner must separately
    record depth metadata: legacy configuration objects only describe up to
    two layers. The original factory is restored even if a fit raises.

    Contexts may nest on a process's main thread. Concurrent threaded fitting
    is intentionally unsupported because ``training.ERNN`` is process-global.
    """
    depth = validate_depth(depth)
    if current_thread() is not main_thread():
        raise RuntimeError("depth_architecture requires a process main thread; use process workers")

    def factory(
        input_dim: int, hidden_units: int, second_hidden_units: int | None = None
    ) -> ERNN:
        parameter_count(input_dim, hidden_units, depth)
        if depth == 1:
            if second_hidden_units is not None:
                raise ValueError("depth one requires second_hidden_units=None")
            return ERNN(input_dim, hidden_units, None)
        if second_hidden_units is None or _positive_integer(
            second_hidden_units, "second_hidden_units"
        ) != hidden_units:
            raise ValueError("depths two to four require second_hidden_units=hidden_units")
        if depth == 2:
            return ERNN(input_dim, hidden_units, second_hidden_units)
        return _DeepERNN(input_dim, hidden_units, depth)

    original_factory = training.ERNN
    training.ERNN = factory
    try:
        yield
    finally:
        training.ERNN = original_factory
