"""Explicit, non-overwriting Chapter 5 evaluation on supplied local data.

No simulated truth is substituted for unknown real-data extremiles.
Configuration selection must be completed separately before a full test run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .baselines import train_fedavg, train_host, train_one_shot, train_scaffold
from .config import ERNNConfig, FederatedConfig, TAU_LEVELS
from .experiments import ALL_METHODS, RESULT_SCHEMA
from .federated import train_extremilefed
from .linear import train_linear_extremile_sl
from .metrics import pinball_loss, coverage_error, crossing_rate
from .model import predict_quantiles, predict_extremiles
from .quadrature import gauss_legendre_unit
from .real_data import prepare_fremtpl2, prepare_beijing_air_quality
from .reproducibility import load_frozen_config
from .training import train_ernn


def evaluate_dataset(data, config, methods=ALL_METHODS, *, split="test", local_epochs=None, local_lr=None, prox_mu=None):
    if split not in ("valid", "test") or set(methods).difference(ALL_METHODS):
        raise ValueError("invalid split or methods")
    X_eval, y_eval = getattr(data, f"X_{split}"), getattr(data, f"y_{split}")
    partitions = [(data.X_train[idx], data.y_train[idx]) for idx in data.client_indices]
    levels, weights = gauss_legendre_unit(config.ernn.quadrature.prediction_nodes)
    diagnostic = np.linspace(0.01, 0.99, 99)
    builders = {
        "host": lambda: train_host(partitions, config.host_config()),
        "one_shot": lambda: train_one_shot(partitions, config.host_config()),
        "fedavg": lambda: train_fedavg(partitions, config.ernn, config.rounds, local_epochs or config.fedavg_local_epochs, local_lr or config.fedavg_learning_rate),
        "fedprox": lambda: train_fedavg(partitions, config.ernn, config.rounds, local_epochs or config.fedprox_local_epochs, local_lr or config.fedprox_learning_rate, prox_mu or config.fedprox_mu),
        "scaffold": lambda: train_scaffold(partitions, config.ernn, config.rounds, local_epochs or config.scaffold_local_epochs, local_lr or config.scaffold_learning_rate),
    }
    results, predictions = {}, {}
    for method in methods:
        started = perf_counter()
        rounds, communication, diagnostics = 0, 0.0, {}
        if method == "centralized":
            fitted = train_ernn(data.X_train, data.y_train, config.ernn)
        elif method == "linear":
            fitted = train_linear_extremile_sl(data.X_train, data.y_train, nodes=config.ernn.quadrature.training_nodes, seed=config.ernn.seed)
        elif method == "extremilefed":
            fed = train_extremilefed(partitions, config)
            fitted, rounds = fed.ernn_result, config.rounds
            communication, diagnostics = fed.history["communication_mb"], fed.history["rounds"]
        else:
            baseline = builders[method]()
            fitted, rounds, communication = baseline.ernn_result, baseline.rounds, baseline.communication_mb
        elapsed = perf_counter() - started
        if method == "linear":
            q = fitted.predict_quantiles(X_eval, levels)
            q_diag = fitted.predict_quantiles(X_eval, diagnostic)
            ext = fitted.predict_extremiles(X_eval, np.asarray(TAU_LEVELS), config.ernn.quadrature.prediction_nodes)
        else:
            q = predict_quantiles(fitted, X_eval, levels)
            q_diag = predict_quantiles(fitted, X_eval, diagnostic)
            ext = predict_extremiles(fitted, X_eval, np.asarray(TAU_LEVELS))
        if not np.isfinite(q).all() or not np.isfinite(ext).all():
            raise RuntimeError(f"non-finite predictions: {method}")
        results[method] = {
            "integrated_pinball": pinball_loss(y_eval, q, levels, weights),
            "coverage_error": coverage_error(y_eval, q_diag, diagnostic),
            "crossing_rate": crossing_rate(q_diag),
            "training_seconds": elapsed,
            "rounds": rounds, "communication_mb": communication,
            "gradient_diagnostics": diagnostics,
        }
        predictions[f"{method}_extremiles"] = ext
        predictions[f"{method}_quantiles"] = q
    if "centralized_extremiles" in predictions:
        central = predictions["centralized_extremiles"]
        for method, result in results.items():
            result["mean_absolute_centralized_extremile_gap"] = np.mean(np.abs(predictions[f"{method}_extremiles"] - central), axis=0).tolist()
    metadata = {
        "schema_version": RESULT_SCHEMA, "dataset": data.metadata,
        "evaluation_split": split, "config": asdict(config), "config_fingerprint": config.fingerprint(),
        "baseline_optimizer": {"local_epochs": local_epochs, "local_lr": local_lr, "prox_mu": prox_mu},
        "split_sizes": [len(data.y_train), len(data.y_valid), len(data.y_test)],
        "client_indices": [idx.tolist() for idx in data.client_indices],
        "methods": results, "tau_levels": list(TAU_LEVELS), "quantile_levels": levels.tolist(),
        "evaluation_weights": weights.tolist(),
        "truth_note": "Real conditional extremiles are unknown; no extremile truth MAE/RMSE is reported.",
    }
    predictions["observed_y"] = y_eval
    return metadata, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("fremtpl2", "beijing"))
    parser.add_argument("--frequency", type=Path)
    parser.add_argument("--severity", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--local-lr", type=float)
    parser.add_argument("--prox-mu", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output directory already exists; choose a new name to preserve prior results")
    if not args.config and not args.smoke:
        parser.error("full runs require --config; a saved config does not establish that tuning was completed")
    if args.smoke and args.config:
        parser.error("smoke overrides may not be applied to a frozen configuration")
    if args.dataset == "fremtpl2":
        if not args.frequency or not args.severity:
            parser.error("fremtpl2 requires --frequency and --severity")
        data = prepare_fremtpl2(args.frequency, args.severity)
        sources = [args.frequency, args.severity]
    else:
        if not args.source:
            parser.error("beijing requires --source")
        data = prepare_beijing_air_quality(args.source)
        sources = sorted(args.source.glob("*.csv")) if args.source.is_dir() else [args.source]
    config = load_frozen_config(args.config) if args.config else FederatedConfig(
        ernn=ERNNConfig(hidden_units=5, max_epochs=3, min_epochs=3, batch_size=128), rounds=1, surrogate_epochs=3)
    if args.smoke:
        # Sample only from each already separated block; never repartition across time.
        rng = np.random.default_rng(90801)
        selected = np.sort(rng.choice(len(data.y_train), min(400, len(data.y_train)), replace=False))
        data.X_train, data.y_train = data.X_train[selected], data.y_train[selected]
        from .federated import partition_indices
        data.client_indices = partition_indices(len(selected), 4, 90802)
        for split in ("valid", "test"):
            indices = np.sort(rng.choice(len(getattr(data, f"y_{split}")), min(80, len(getattr(data, f"y_{split}"))), replace=False))
            setattr(data, f"X_{split}", getattr(data, f"X_{split}")[indices])
            setattr(data, f"y_{split}", getattr(data, f"y_{split}")[indices])
        data.metadata["smoke_subsampling"] = True
        data.metadata["actual_clients"] = 4
    metadata, predictions = evaluate_dataset(data, config, tuple(args.methods), split=args.split,
        local_epochs=args.local_epochs, local_lr=args.local_lr, prox_mu=args.prox_mu)
    metadata["smoke"] = args.smoke
    metadata["raw_file_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output / "metrics.json")


if __name__ == "__main__":
    main()
