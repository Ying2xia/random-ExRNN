"""Command-line entry points for smoke and sharded simulation runs."""

from __future__ import annotations

import argparse
import torch
from dataclasses import replace
from pathlib import Path

from .config import ERNNConfig, FederatedConfig, QuadratureConfig
from .experiments import ALL_METHODS, SimulationJob, run_jobs
from .reproducibility import freeze_config, load_frozen_config, verify_simulation_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ExtremileFed experiments")
    parser.add_argument("--scenario", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--error", choices=("normal", "t3", "lognormal", "contaminated"), default="normal")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-stop", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--hidden-units", type=int, default=10)
    parser.add_argument("--second-hidden-units", type=int)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-4)
    parser.add_argument("--h", type=float, default=0.03)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--no-crossing", action="store_true")
    parser.add_argument("--training-nodes", type=int, default=32)
    parser.add_argument("--prediction-nodes", type=int, default=128)
    parser.add_argument("--freeze-config", type=Path)
    parser.add_argument("--config", type=Path, help="load a previously frozen configuration; overrides model flags")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--development", action="store_true", help="explicit non-formal probe; output must be in 工作记录与备份")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "formal",
    )
    return parser


def main() -> None:
    torch.set_num_threads(1)
    args = build_parser().parse_args()
    if args.rep_start < 0 or args.rep_stop <= args.rep_start:
        raise ValueError("require 0 <= rep-start < rep-stop")
    if args.config and args.smoke:
        raise ValueError("do not combine a frozen configuration with smoke overrides")
    if not args.smoke and not args.development and not args.config:
        raise ValueError("run development_cli first; formal simulation requires --config with selection evidence")
    if args.development and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2] / "工作记录与备份"):
        raise ValueError("development probes must use an output directory under 工作记录与备份")
    ernn = ERNNConfig(
        hidden_units=args.hidden_units,
        second_hidden_units=args.second_hidden_units,
        lambda_=args.lambda_,
        h=args.h,
        eta=0.0 if args.no_crossing else args.eta,
        quadrature=QuadratureConfig(args.training_nodes, args.prediction_nodes),
    )
    if args.smoke:
        ernn = replace(
            ernn,
            hidden_units=5,
            max_epochs=12,
            min_epochs=6,
            patience=3,
            batch_size=128,
            dtype="float64",
        )
        args.n_train = 400
        args.n_test = 80
        args.clients = 4
        args.rounds = 1
        args.output = (
            Path(__file__).resolve().parents[2]
            / "工作记录与备份"
            / "代码中间结果"
            / "smoke_v7"
        )
        if args.methods is None:
            args.methods = ["extremilefed"]
    elif args.methods is None:
        args.methods = list(ALL_METHODS)
    config = FederatedConfig(
        ernn=ernn,
        rounds=args.rounds,
        surrogate_epochs=8 if args.smoke else 120,
        surrogate_learning_rate=3e-3,
    )
    if args.config:
        config = (load_frozen_config(args.config) if args.development else
            verify_simulation_selection(args.config, args.scenario, args.error))
    if args.freeze_config is not None:
        freeze_config(config, args.freeze_config)
    jobs = [
        SimulationJob(
            scenario=args.scenario,
            error=args.error,
            replication=replication,
            n_train=args.n_train,
            n_test=args.n_test,
            clients=args.clients,
            seed=20260903,
        )
        for replication in range(args.rep_start, args.rep_stop)
    ]
    for path in run_jobs(jobs, config, args.output, tuple(args.methods)):
        print(path)


if __name__ == "__main__":
    main()
