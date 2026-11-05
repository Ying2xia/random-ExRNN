"""Configuration objects and frozen experiment grids."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json


J_GRID = (3, 5, 10, 15, 20, 30, 40)
LAMBDA_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ETA_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
H_SENSITIVITY_GRID = (0.01, 0.03, 0.05, 0.10)
TAU_LEVELS = (0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95)


@dataclass(frozen=True)
class QuadratureConfig:
    training_nodes: int = 32
    prediction_nodes: int = 128


@dataclass(frozen=True)
class ERNNConfig:
    hidden_units: int = 10
    second_hidden_units: int | None = None
    lambda_: float = 1e-4
    h: float = 0.03
    smoothing_kernel: str = "gaussian"
    eta: float = 1.0
    max_epochs: int = 200
    batch_size: int = 2048
    learning_rate: float = 1e-2
    patience: int = 20
    min_epochs: int = 30
    refinement_epochs: int = 0
    refinement_learning_rate: float = 1e-4
    seed: int = 20260903
    dtype: str = "float64"
    device: str = "cpu"
    quadrature: QuadratureConfig = field(default_factory=QuadratureConfig)

    def validate(self) -> None:
        if self.smoothing_kernel != "gaussian":
            raise ValueError("ERNN uses Gaussian convolution smoothing; legacy kernels are not supported")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if self.second_hidden_units is not None and self.second_hidden_units <= 0:
            raise ValueError("second_hidden_units must be positive when provided")
        if self.lambda_ < 0 or self.h <= 0 or self.eta < 0:
            raise ValueError("lambda and eta must be non-negative; h must be positive")
        if self.max_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("max_epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.patience <= 0 or self.min_epochs <= 0:
            raise ValueError("learning rate, patience and minimum epochs must be positive")
        if self.refinement_epochs < 0 or self.refinement_learning_rate <= 0:
            raise ValueError("invalid refinement budget or learning rate")
        if self.quadrature.training_nodes < 2 or self.quadrature.prediction_nodes < 2:
            raise ValueError("quadrature node counts must be at least two")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FederatedConfig:
    ernn: ERNNConfig = field(default_factory=ERNNConfig)
    rounds: int = 2
    surrogate_epochs: int = 120
    surrogate_learning_rate: float = 3e-3
    trust_radius: float | None = 2.0
    host_epochs: int | None = None
    host_learning_rate: float | None = None
    host_patience: int | None = None
    surrogate_patience: int | None = None
    host_refinement_epochs: int = 0
    host_refinement_learning_rate: float = 1e-3
    surrogate_learning_rates: tuple[float, ...] = ()
    fedavg_local_epochs: int = 5
    fedavg_learning_rate: float = 1e-3
    fedprox_local_epochs: int = 5
    fedprox_learning_rate: float = 1e-3
    fedprox_mu: float = 1e-3
    scaffold_local_epochs: int = 5
    scaffold_learning_rate: float = 1e-3

    def validate(self) -> None:
        self.ernn.validate()
        if self.host_refinement_epochs < 0 or self.host_refinement_learning_rate <= 0:
            raise ValueError("invalid host refinement settings")
        if self.surrogate_learning_rates and (len(self.surrogate_learning_rates) < self.rounds
                or any(lr <= 0 for lr in self.surrogate_learning_rates)):
            raise ValueError("positive learning rates required for every requested round")
        for value in (self.host_epochs, self.host_learning_rate, self.host_patience, self.surrogate_patience):
            if value is not None and value <= 0:
                raise ValueError("host/surrogate solver overrides must be positive")
        if self.rounds <= 0 or self.surrogate_epochs <= 0:
            raise ValueError("rounds and surrogate_epochs must be positive")
        if self.surrogate_learning_rate <= 0:
            raise ValueError("surrogate_learning_rate must be positive")
        if self.trust_radius is not None and self.trust_radius <= 0:
            raise ValueError("trust_radius must be positive when provided")
        if min(self.fedavg_local_epochs, self.fedprox_local_epochs, self.scaffold_local_epochs) <= 0:
            raise ValueError("baseline local epochs must be positive")
        if min(self.fedavg_learning_rate, self.fedprox_learning_rate, self.scaffold_learning_rate, self.fedprox_mu) <= 0:
            raise ValueError("baseline learning rates and FedProx mu must be positive")

    def host_config(self) -> ERNNConfig:
        """Same statistical model, separate local-initializer numerical budget."""
        return replace(self.ernn, max_epochs=self.host_epochs or self.ernn.max_epochs,
            learning_rate=self.host_learning_rate or self.ernn.learning_rate,
            patience=self.host_patience or self.ernn.patience,
            refinement_epochs=self.host_refinement_epochs,
            refinement_learning_rate=self.host_refinement_learning_rate)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FormalExperimentConfig:
    n_train: int = 100_000
    n_test: int = 5_000
    clients: int = 20
    replications: int = 100
    scenarios: tuple[int, ...] = (1, 2, 3)
    errors: tuple[str, ...] = ("normal", "t3", "lognormal", "contaminated")
    tau_levels: tuple[float, ...] = TAU_LEVELS
    j_grid: tuple[int, ...] = J_GRID
    lambda_grid: tuple[float, ...] = LAMBDA_GRID
    eta_grid: tuple[float, ...] = ETA_GRID
    h: float = 0.03
    seed: int = 20260903
