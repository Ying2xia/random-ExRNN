"""Resumable simulation runner for formal and smoke studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .baselines import train_fedavg, train_host, train_one_shot, train_scaffold
from .config import FederatedConfig, TAU_LEVELS
from .federated import partition_indices, train_extremilefed
from .linear import train_linear_extremile_sl
from .metrics import coverage_error, crossing_rate, extremile_errors, pinball_loss
from .model import predict_extremiles, predict_quantiles
from .simulation import generate_data, true_extremiles
from .training import train_ernn
from .quadrature import gauss_legendre_unit

RESULT_SCHEMA = 7


ALL_METHODS = (
    "centralized",
    "linear",
    "host",
    "one_shot",
    "fedavg",
    "fedprox",
    "scaffold",
    "extremilefed",
)


@dataclass(frozen=True)
class SimulationJob:
    scenario: int
    error: str
    replication: int
    n_train: int
    n_test: int
    clients: int
    seed: int

    @property
    def key(self) -> str:
        return f"scenario{self.scenario}_{self.error}_rep{self.replication:03d}"


def _neural_metrics(fit, X_test, y_test, tau, levels, truth) -> dict[str, float]:
    extremile_prediction = predict_extremiles(fit, X_test, tau)
    quantile_prediction = predict_quantiles(fit, X_test, levels)
    return {
        **extremile_errors(extremile_prediction, truth),
        "integrated_pinball": pinball_loss(y_test, quantile_prediction, levels, gauss_legendre_unit(len(levels))[1]),
        "coverage_error": coverage_error(y_test, quantile_prediction, levels),
        "crossing_rate": crossing_rate(predict_quantiles(fit, X_test, np.linspace(0.01, 0.99, 99))),
        **{f"extremile_mae_tau_{t:.2f}": float(np.mean(np.abs(extremile_prediction[:, j] - truth[:, j]))) for j, t in enumerate(tau)},
        **{f"extremile_rmse_tau_{t:.2f}": float(np.sqrt(np.mean((extremile_prediction[:, j] - truth[:, j])**2))) for j, t in enumerate(tau)},
    }


def run_job(
    job: SimulationJob,
    config: FederatedConfig,
    methods: tuple[str, ...] = ALL_METHODS,
) -> dict[str, object]:
    unknown = set(methods).difference(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    train_seed = job.seed + 100_000 * job.scenario + 1_000 * job.replication
    X_train, y_train = generate_data(job.n_train, job.scenario, job.error, train_seed)
    X_test, y_test = generate_data(job.n_test, job.scenario, job.error, train_seed + 1)
    split = partition_indices(job.n_train, job.clients, train_seed + 2)
    partitions = [(X_train[index], y_train[index]) for index in split]
    tau = np.asarray(TAU_LEVELS)
    levels, evaluation_weights = gauss_legendre_unit(config.ernn.quadrature.prediction_nodes)
    extremile_truth = true_extremiles(X_test, job.scenario, job.error, tau)
    results: dict[str, object] = {}

    if "centralized" in methods:
        started = perf_counter()
        fit = train_ernn(X_train, y_train, config.ernn)
        training_elapsed = perf_counter() - started
        results["Centralized ERNN"] = {
            "metrics": _neural_metrics(fit, X_test, y_test, tau, levels, extremile_truth),
            "elapsed_seconds": training_elapsed,
            "rounds": 0,
            "communication_mb": 0.0,
        }
    if "linear" in methods:
        started = perf_counter()
        fit = train_linear_extremile_sl(
            X_train,
            y_train,
            nodes=config.ernn.quadrature.training_nodes,
            seed=config.ernn.seed,
        )
        training_elapsed = perf_counter() - started
        extremile_prediction = fit.predict_extremiles(
            X_test, tau, n_nodes=config.ernn.quadrature.prediction_nodes
        )
        quantile_prediction = fit.predict_quantiles(X_test, levels)
        results["Pooled linear Extremile-SL"] = {
            "metrics": {
                **extremile_errors(extremile_prediction, extremile_truth),
                "integrated_pinball": pinball_loss(y_test, quantile_prediction, levels, evaluation_weights),
                "coverage_error": coverage_error(y_test, quantile_prediction, levels),
                "crossing_rate": crossing_rate(fit.predict_quantiles(X_test, np.linspace(0.01, 0.99, 99))),
                **{f"extremile_mae_tau_{t:.2f}": float(np.mean(np.abs(extremile_prediction[:, j] - extremile_truth[:, j]))) for j, t in enumerate(tau)},
                **{f"extremile_rmse_tau_{t:.2f}": float(np.sqrt(np.mean((extremile_prediction[:, j] - extremile_truth[:, j])**2))) for j, t in enumerate(tau)},
            },
            "elapsed_seconds": training_elapsed,
            "rounds": 0,
            "communication_mb": 0.0,
        }

    baseline_builders = {
        "host": lambda: train_host(partitions, config.host_config()),
        "one_shot": lambda: train_one_shot(partitions, config.host_config()),
        "fedavg": lambda: train_fedavg(partitions, config.ernn, config.rounds, config.fedavg_local_epochs, config.fedavg_learning_rate),
        "fedprox": lambda: train_fedavg(
            partitions, config.ernn, config.rounds, config.fedprox_local_epochs, config.fedprox_learning_rate, prox_mu=config.fedprox_mu
        ),
        "scaffold": lambda: train_scaffold(partitions, config.ernn, config.rounds, config.scaffold_local_epochs, config.scaffold_learning_rate),
    }
    for method in methods:
        if method not in baseline_builders:
            continue
        fit = baseline_builders[method]()
        results[fit.method] = {
            "metrics": _neural_metrics(
                fit.ernn_result, X_test, y_test, tau, levels, extremile_truth
            ),
            "elapsed_seconds": fit.runtime_seconds,
            "rounds": fit.rounds,
            "communication_mb": fit.communication_mb,
        }
    if "extremilefed" in methods:
        fit = train_extremilefed(partitions, config)
        results["ExtremileFed"] = {
            "metrics": _neural_metrics(
                fit.ernn_result, X_test, y_test, tau, levels, extremile_truth
            ),
            "elapsed_seconds": fit.history["runtime_seconds"],
            "rounds": config.rounds,
            "communication_mb": fit.history["communication_mb"],
            "gradient_diagnostics": fit.history["rounds"],
        }
    if "Centralized ERNN" in results:
        central_metrics = results["Centralized ERNN"]["metrics"]
        for name, result in results.items():
            result["centralized_gap"] = {
                metric: float(value - central_metrics[metric])
                for metric, value in result["metrics"].items()
            }
    return {
        "schema_version": RESULT_SCHEMA,
        "config": asdict(config),
        "job": asdict(job),
        "seeds": {"training_data": train_seed, "test_data": train_seed + 1, "partition": train_seed + 2, "model": config.ernn.seed},
        "methods": results,
        "partition_sizes": [int(len(index)) for index in split],
        "quadrature": {
            "training_nodes": config.ernn.quadrature.training_nodes,
            "prediction_nodes": config.ernn.quadrature.prediction_nodes,
            "evaluation_levels": levels.tolist(),
            "evaluation_weights": evaluation_weights.tolist(),
        },
        "tau_levels": tau.tolist(),
        "config_fingerprint": config.fingerprint(),
    }


def run_jobs(
    jobs: list[SimulationJob],
    config: FederatedConfig,
    output_dir: Path,
    methods: tuple[str, ...] = ALL_METHODS,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for job in jobs:
        path = output_dir / f"{job.key}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("schema_version") != RESULT_SCHEMA:
                raise RuntimeError(f"legacy result uses different evaluation/solver definitions; choose a new output directory: {path}")
            if existing.get("config_fingerprint") != config.fingerprint():
                raise RuntimeError(f"refusing to overwrite result with a different configuration: {path}")
            if existing.get("job") != asdict(job):
                raise RuntimeError(f"result key collides with different job metadata: {path}")
            if not set(methods).issubset(
                {
                    "centralized" if name == "Centralized ERNN" else
                    "linear" if name == "Pooled linear Extremile-SL" else
                    "host" if name == "Host ERNN" else
                    "one_shot" if name == "One-shot ERNN" else
                    "fedavg" if name == "FedAvg-ERNN" else
                    "fedprox" if name == "FedProx-ERNN" else
                    "scaffold" if name == "SCAFFOLD-ERNN" else
                    "extremilefed" if name == "ExtremileFed" else ""
                    for name in existing.get("methods", {})
                }
            ):
                raise RuntimeError(f"existing result does not contain every requested method: {path}")
            completed.append(path)
            continue
        result = run_job(job, config, methods)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(path)
        completed.append(path)
    return completed
