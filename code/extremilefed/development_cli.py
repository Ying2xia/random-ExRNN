"""Run: python -B -m extremilefed.development_cli --smoke

Then run --stage check for a real numerical pilot, followed by --stage all.
Repeating the same invocation resumes completed fits without changing the protocol.
"""
from dataclasses import asdict, replace
import argparse
import json
import os
import torch
from pathlib import Path

from .development import (DevelopmentProtocol, TaskStore, Paused, GateFailure,
    exclusive_run, implementation_hash, run_case, write_json)

PROJECT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finite-step", action="store_true", help="explicit finite-iteration policy; raw stationarity remains a diagnostic")
    p.add_argument("--stage", choices=("check", "all"), default="all")
    p.add_argument("--scenario", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--error", choices=("normal", "t3", "lognormal", "contaminated"), default="normal")
    p.add_argument("--all", action="store_true", help="develop all twelve settings sequentially")
    p.add_argument("--output", type=Path)
    p.add_argument("--max-new-tasks", type=int, help="pause after this many new fits, then resume with same protocol")
    p.add_argument("--status", action="store_true", help="read state only")
    p.add_argument("--unlock-stale", action="store_true", help="remove only a dead-process run.lock, without running training")
    p.add_argument("--epoch-budgets", type=int, nargs="+")
    p.add_argument("--surrogate-budgets", type=int, nargs="+")
    p.add_argument("--refinement-epochs", type=int)
    p.add_argument("--central-learning-rate", type=float)
    p.add_argument("--host-epochs", type=int)
    p.add_argument("--host-learning-rate", type=float)
    p.add_argument("--surrogate-learning-rate", type=float)
    p.add_argument("--seed", type=int)
    return p


def main(argv=None):
    torch.set_num_threads(1)
    args = parser().parse_args(argv)
    if args.max_new_tasks is not None and args.max_new_tasks < 1:
        raise ValueError("max-new-tasks must be positive")
    protocol = DevelopmentProtocol.small() if args.smoke else DevelopmentProtocol()
    if args.finite_step:
        protocol = replace(protocol, numerical_acceptance="finite_step",
            host_epochs=2 if args.smoke else 800, host_patience=3 if args.smoke else 801,
            host_refinement_epochs=2 if args.smoke else 200,
            surrogate_learning_rates=(.001,.001,.0003))
    if args.epoch_budgets:
        protocol = replace(protocol, epoch_budgets=tuple(args.epoch_budgets))
    if args.surrogate_budgets:
        protocol = replace(protocol, surrogate_budgets=tuple(args.surrogate_budgets))
    for field in ("refinement_epochs", "central_learning_rate", "host_epochs", "host_learning_rate", "surrogate_learning_rate", "seed"):
        if getattr(args, field) is not None:
            protocol = replace(protocol, **{field: getattr(args, field)})
    protocol.validate()
    root = (args.output or PROJECT / "工作记录与备份" /
        ("自动调参_有限步_v5" if args.finite_step else "自动调参_两阶段_v4") /
        ("smoke" if args.smoke else "development")).resolve()
    if not root.is_relative_to(PROJECT / "工作记录与备份"):
        raise ValueError("development outputs must be under the project's 工作记录与备份 directory")
    cases = [(s, e) for s in (1, 2, 3) for e in ("normal", "t3", "lognormal", "contaminated")] if args.all else [(args.scenario, args.error)]
    print("SMOKE: reduced grids; no formal configuration will be created." if args.smoke else
        "DEVELOPMENT: independent samples; all 49 J/lambda candidates, five folds; no formal Monte Carlo cells.", flush=True)
    for scenario, error in cases:
        case = root / f"scenario{scenario}_{error}"
        if args.status:
            status = case / "status.json"
            print(status.read_text() if status.exists() else f"not started: {case}")
            print(f"completed fit checkpoints: {len(list((case / 'tasks').glob('*.json')))}")
            continue
        if args.unlock_stale:
            lock = case / "run.lock"
            if lock.exists():
                pid = int(lock.read_text())
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    lock.unlink()
                    print(f"removed dead-process lock only: {lock}")
                else:
                    raise RuntimeError(f"process {pid} still exists; refusing to remove lock")
            continue
        manifest = {"pipeline_version": 3 if args.finite_step else 2, "protocol": asdict(protocol), "scope": {"scenario": scenario, "error": error},
            "implementation_hash": implementation_hash(), "seed_design": "role-keyed independent streams in [1e9,2e9)"}
        with exclusive_run(case):
            store = TaskStore(case, manifest, args.max_new_tasks)
            try:
                result = run_case(store, protocol, scenario, error, args.stage)
                if result.get("eligible_for_formal_simulation"):
                    selected = json.loads(Path(result["config"]).read_text())
                    destination = PROJECT / "code" / "configs" / "selected" / f"scenario{scenario}_{error}.json"
                    if destination.exists() and json.loads(destination.read_text()) != selected:
                        raise RuntimeError(f"different final configuration exists; archive/review it before replacing: {destination}")
                    write_json(destination, selected)
                    result["formal_config"] = str(destination)
            except Paused as exc:
                result = {"status": "paused", "message": str(exc)}
            except (GateFailure, FloatingPointError) as exc:
                result = {"status": "failed_gate", "message": str(exc), "eligible_for_formal_simulation": False}
                write_json(case / "status.json", result)
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                return 2
            except Exception as exc:
                write_json(case / "status.json", {"status": "error", "message": str(exc)})
                raise
            result.update({"new_tasks": store.new_tasks, "cached_tasks": store.cached_tasks})
            write_json(case / "status.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] == "paused":
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
