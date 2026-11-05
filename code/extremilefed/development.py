"""Independent simulation development, resumable selection, and audited freezing.

No formal Monte Carlo observations or test metrics are consumed here.
Numerical gates are engineering checks, not proofs of neural global optimality.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from time import time

import numpy as np
from sklearn.model_selection import KFold

from .baselines import train_fedavg, train_scaffold
from .config import ERNNConfig, FederatedConfig, QuadratureConfig, J_GRID, LAMBDA_GRID, ETA_GRID
from .federated import partition_indices, train_extremilefed
from .metrics import pinball_loss, crossing_rate
from .model import predict_quantiles
from .quadrature import gauss_legendre_unit
from .simulation import generate_data
from .training import train_ernn
from .tuning import select_rounds_one_se


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_hash():
    root = Path(__file__).parent
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Paused(RuntimeError):
    pass


class GateFailure(RuntimeError):
    pass


@contextmanager
def exclusive_run(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "run.lock"
    try:
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"development directory is locked: {lock}; do not run overlapping workers")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class TaskStore:
    def __init__(self, root, manifest, max_new_tasks=None):
        self.root = Path(root)
        manifest = json.loads(json.dumps(manifest))
        self.manifest = manifest
        self.fingerprint = digest(manifest)
        path = self.root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise RuntimeError("protocol or source code changed; choose a new development directory")
        else:
            write_json(path, manifest)
        self.max_new_tasks = max_new_tasks
        self.new_tasks = 0
        self.cached_tasks = 0

    def task(self, key, fn):
        path = self.root / "tasks" / (key + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record["manifest_fingerprint"] != self.fingerprint or record["checksum"] != digest(record["value"]):
                raise RuntimeError(f"checkpoint integrity failure: {path}")
            self.cached_tasks += 1
            return record["value"]
        if self.max_new_tasks is not None and self.new_tasks >= self.max_new_tasks:
            raise Paused("new-task budget reached; repeat the same command to continue")
        print(f"[run] {key}", flush=True)
        started = time()
        value = fn()
        write_json(path, {"manifest_fingerprint": self.fingerprint, "checksum": digest(value), "value": value, "seconds": time()-started})
        self.new_tasks += 1
        return value


@dataclass(frozen=True)
class DevelopmentProtocol:
    smoke: bool = False
    seed: int = 913051
    tune_n: int = 5000
    eta_train_n: int = 3000
    validation_n: int = 1000
    development_n: int = 10000
    clients: int = 20
    development_repeats: int = 3
    folds: int = 5
    epoch_budgets: tuple = (800,)
    surrogate_budgets: tuple = (120,)
    refinement_epochs: int = 200
    central_learning_rate: float = .01
    host_epochs: int = 200
    host_learning_rate: float = .01
    surrogate_learning_rate: float = .001
    host_patience: int = 20
    host_refinement_epochs: int = 0
    host_refinement_learning_rate: float = .001
    surrogate_learning_rates: tuple = ()
    numerical_acceptance: str = "stationarity"
    j_grid: tuple = J_GRID
    lambda_grid: tuple = LAMBDA_GRID
    eta_grid: tuple = ETA_GRID + (1000.0,)
    local_epochs_grid: tuple = (1, 5, 10)
    local_lr_grid: tuple = (0.0003, 0.001, 0.003, 0.01)
    prox_grid: tuple = (0.0001, 0.001, 0.01, 0.1)
    gradient_tolerance: float = 0.01
    plateau_tolerance: float = 0.001
    crossing_threshold: float = 0.005

    @classmethod
    def small(cls):
        return cls(smoke=True, tune_n=100, eta_train_n=80, validation_n=30,
            development_n=100, clients=4, development_repeats=2,
            epoch_budgets=(2,), surrogate_budgets=(2,), refinement_epochs=2, host_epochs=2, j_grid=(3, 5),
            lambda_grid=(0., .001), eta_grid=(0., 1., 1000.),
            local_epochs_grid=(1, 2), local_lr_grid=(.001, .003), prox_grid=(.001, .01))

    def validate(self):
        if self.numerical_acceptance not in ("stationarity", "finite_step"):
            raise ValueError("unknown numerical acceptance policy")
        if self.host_refinement_epochs < 0 or self.host_refinement_learning_rate <= 0 or self.host_patience < 1:
            raise ValueError("invalid host schedule")
        if self.surrogate_learning_rates and (len(self.surrogate_learning_rates) < 3 or any(x<=0 for x in self.surrogate_learning_rates)):
            raise ValueError("need a positive rate for all three development rounds")
        if self.folds != 5 or self.development_repeats < 2:
            raise ValueError("requires five CV folds and at least two development repetitions")
        if min(self.tune_n, self.eta_train_n, self.validation_n, self.development_n) < self.folds:
            raise ValueError("development samples are too small")
        if not 1 <= self.clients <= self.development_n:
            raise ValueError("invalid development client count")
        if not self.epoch_budgets or len(self.epoch_budgets) != len(self.surrogate_budgets):
            raise ValueError("paired epoch and surrogate budgets are required")
        if any(b <= 0 for b in self.epoch_budgets + self.surrogate_budgets):
            raise ValueError("epoch budgets must be positive")
        if min(self.gradient_tolerance, self.plateau_tolerance) <= 0:
            raise ValueError("numerical tolerances must be positive")
        if self.refinement_epochs < 0 or self.host_epochs <= 0 or min(self.central_learning_rate, self.host_learning_rate, self.surrogate_learning_rate) <= 0:
            raise ValueError("invalid two-stage/host/surrogate solver settings")


def isolated_seed(protocol, scenario, error, role):
    # Dedicated range [1e9,2e9); formal default data/model seeds are below this.
    return 1_000_000_000 + int(digest([protocol.seed, scenario, error, role])[:12], 16) % 1_000_000_000


def convergence(history, protocol):
    losses = np.asarray(history["loss"], dtype=float)
    terminal = float(history["terminal_loss"])
    gradient = float(history["terminal_gradient_norm"])
    plateau = float(np.ptp(losses[-5:]) / max(1., abs(terminal)))
    passed = bool(len(losses) >= 5 and np.isfinite(losses).all() and np.isfinite(gradient)
        and np.isfinite(terminal) and gradient <= protocol.gradient_tolerance
        and plateau <= protocol.plateau_tolerance)
    stationarity_passed = passed
    initial = float(history.get("initial_loss", float("nan")))
    finite_step_valid = bool(len(losses) > 0 and np.isfinite(losses).all()
        and np.isfinite([initial, terminal, gradient]).all()
        and terminal <= initial + 1e-10 * max(1., abs(initial), abs(terminal)))
    if "phases" in history:
        finite_step_valid = finite_step_valid and all(
            convergence(phase, replace(protocol, numerical_acceptance="finite_step"))["passed"]
            for phase in history["phases"].values())
    if protocol.numerical_acceptance == "finite_step":
        passed = finite_step_valid
    return {"passed": passed, "acceptance_policy": protocol.numerical_acceptance,
        "stationarity_passed": stationarity_passed, "finite_step_valid": finite_step_valid,
        "theory_certified": False, "gradient_norm": gradient, "plateau_range": plateau,
        "terminal_loss": terminal, "epochs": history["epochs"], "best_epoch": history["best_epoch"]}


def score_fit(fit, X, y):
    levels, weights = gauss_legendre_unit(fit.config.quadrature.prediction_nodes)
    predictions = predict_quantiles(fit, X, levels)
    crossing = predict_quantiles(fit, X, np.linspace(.01, .99, 99))
    value = {"score": pinball_loss(y, predictions, levels, weights), "crossing": crossing_rate(crossing)}
    if not np.isfinite(list(value.values())).all():
        raise FloatingPointError("non-finite validation prediction")
    return value


def minimum_candidate(records, tie_key):
    if not records:
        raise GateFailure("no valid candidates")
    best = min(r["score"] for r in records)
    eligible = [r for r in records if np.isclose(r["score"], best, rtol=1e-10, atol=1e-12)]
    return min(eligible, key=tie_key)


def run_case(store, protocol, scenario, error, stage="all"):
    """A task is one fit/fold/replicate; no formal test data are generated."""
    seed = lambda role: isolated_seed(protocol, scenario, error, role)
    data = lambda role, n: generate_data(n, scenario, error, seed(role))
    def partitions(X, y, role):
        return [(X[idx], y[idx]) for idx in partition_indices(len(y), protocol.clients, seed(role))]
    base = ERNNConfig(seed=seed("model"), refinement_epochs=protocol.refinement_epochs,
        learning_rate=protocol.central_learning_rate)
    def federated_config(ernn, surrogate_epochs, rounds=2):
        return FederatedConfig(ernn=ernn, rounds=rounds, surrogate_epochs=surrogate_epochs,
            surrogate_learning_rate=protocol.surrogate_learning_rate, surrogate_patience=20,
            host_epochs=protocol.host_epochs, host_learning_rate=protocol.host_learning_rate,
            host_patience=protocol.host_patience, host_refinement_epochs=protocol.host_refinement_epochs,
            host_refinement_learning_rate=protocol.host_refinement_learning_rate,
            surrogate_learning_rates=protocol.surrogate_learning_rates)
    if protocol.smoke:
        base = replace(base, hidden_units=3, batch_size=64, min_epochs=2, patience=2,
            quadrature=QuadratureConfig(6, 12))

    # Stage 1: budget pilot. Independent of CV, eta, and T development samples.
    Xp, yp = data("pilot_train", protocol.development_n)
    pilot_parts = partitions(Xp, yp, "pilot_partition")
    pilots = []
    for epochs, surrogate_epochs in zip(protocol.epoch_budgets, protocol.surrogate_budgets):
        ernn = replace(base, max_epochs=epochs, patience=epochs + 1)
        fc = federated_config(ernn, surrogate_epochs, rounds=1)
        def pilot():
            central = train_ernn(Xp, yp, ernn)
            fed = train_extremilefed(pilot_parts, fc)
            checks = {"centralized": convergence(central.history, protocol),
                "host": convergence(fed.history["host"], protocol),
                "surrogate": convergence(fed.history["rounds"][0]["solver"], protocol)}
            if protocol.numerical_acceptance == "finite_step":
                checks["surrogate"]["passed"] = checks["surrogate"]["passed"] and fed.history["rounds"][0]["finite_step_audit"]["numerically_valid"]
            return {"epochs": epochs, "surrogate_epochs": surrogate_epochs, "checks": checks,
                "passed": all(v["passed"] for v in checks.values()),
                "histories": {"centralized": central.history, "federated": fed.history}}
        record = store.task(f"pilot_e{epochs}_s{surrogate_epochs}", pilot)
        pilots.append(record)
        if record["passed"] or protocol.smoke:
            break
    chosen_pilot = pilots[-1]
    write_json(store.root / "convergence_report.json", {"pilots": pilots, "smoke": protocol.smoke})
    if not chosen_pilot["passed"] and not protocol.smoke:
        raise GateFailure("pilot convergence failed at every budget; inspect convergence_report.json; no config frozen")
    if stage == "check":
        return {"status": "check_complete", "passed": chosen_pilot["passed"], "smoke": protocol.smoke}
    base = replace(base, max_epochs=chosen_pilot["epochs"], patience=chosen_pilot["epochs"] + 1)

    # Stage 2: all 49 model candidates share identical five folds and model seeds.
    Xt, yt = data("cv_data", protocol.tune_n)
    folds = list(KFold(5, shuffle=True, random_state=seed("cv_folds")).split(Xt))
    candidates = []
    for width in protocol.j_grid:
        for regularization in protocol.lambda_grid:
            scores = []
            for fold, (tr, va) in enumerate(folds):
                conf = replace(base, hidden_units=width, lambda_=regularization, seed=seed(f"cv_model_{fold}"))
                def cv_fit():
                    fit = train_ernn(Xt[tr], yt[tr], conf)
                    return {**score_fit(fit, Xt[va], yt[va]), "convergence": convergence(fit.history, protocol),
                        "train_indices": tr.tolist(), "validation_indices": va.tolist(), "model_seed": conf.seed}
                scores.append(store.task(f"cv_J{width}_l{regularization:g}_fold{fold}", cv_fit))
            candidates.append({"J": width, "lambda": regularization,
                "score": float(np.mean([r["score"] for r in scores])),
                "se": float(np.std([r["score"] for r in scores], ddof=1)/np.sqrt(5)),
                "folds_converged": all(r["convergence"]["passed"] for r in scores)})
    chosen = minimum_candidate(candidates, lambda r: (r["J"], -r["lambda"]))
    write_json(store.root / "cv_selection.json", {"candidates": candidates, "selected": chosen,
        "rule": "minimum mean; numerical ties: smaller J, larger lambda"})
    if not all(r["folds_converged"] for r in candidates) and not protocol.smoke:
        raise GateFailure("CV comparison contains unconverged folds; increase training budgets in a NEW run directory")
    base = replace(base, hidden_units=chosen["J"], lambda_=chosen["lambda"])

    # Stage 3: eta uses distinct fitting/validation data, ascending candidates.
    Xe, ye = data("eta_train", protocol.eta_train_n)
    Xv, yv = data("eta_validation", protocol.validation_n)
    eta_records = []
    for eta in sorted(set(protocol.eta_grid)):
        conf = replace(base, eta=eta, seed=seed("eta_model"))
        def eta_fit():
            fit = train_ernn(Xe, ye, conf)
            return {"eta": eta, **score_fit(fit, Xv, yv), "convergence": convergence(fit.history, protocol)}
        result = store.task(f"eta_{eta:g}", eta_fit)
        eta_records.append(result)
        if result["crossing"] <= protocol.crossing_threshold:
            break
    selected_eta = eta_records[-1]
    write_json(store.root / "eta_selection.json", {"candidates": eta_records, "selected": selected_eta})
    if not protocol.smoke and (selected_eta["crossing"] > protocol.crossing_threshold or not all(r["convergence"]["passed"] for r in eta_records)):
        raise GateFailure("eta crossing or convergence gate failed; no config frozen")
    base = replace(base, eta=selected_eta["eta"])

    # Stage 4: independent repeated development data, shared by every T/method.
    def dev_repeat(rep):
        X, y = data(f"dev_train_{rep}", protocol.development_n)
        V, z = data(f"dev_valid_{rep}", protocol.validation_n)
        return X, y, partitions(X, y, f"dev_partition_{rep}"), V, z
    fc = federated_config(base, chosen_pilot["surrogate_epochs"])
    central_records = []
    for rep in range(protocol.development_repeats):
        def central_development():
            X, y, _, V, z = dev_repeat(rep)
            central = train_ernn(X, y, replace(base, seed=seed(f"dev_model_{rep}")))
            return {**score_fit(central, V, z), "check": convergence(central.history, protocol)}
        central_records.append(store.task(f"dev_central_rep{rep}", central_development))
    round_scores, round_records = {}, {}
    for rounds in (1, 2, 3):
        records = []
        for rep in range(protocol.development_repeats):
            def evaluate_rounds():
                X, y, parts, V, z = dev_repeat(rep)
                conf = replace(fc, rounds=rounds, ernn=replace(base, seed=seed(f"dev_model_{rep}")))
                fit = train_extremilefed(parts, conf)
                checks = [central_records[rep]["check"], convergence(fit.history["host"], protocol)]
                for r in fit.history["rounds"]:
                    check = convergence(r["solver"], protocol)
                    if protocol.numerical_acceptance == "finite_step":
                        check["passed"] = check["passed"] and r["finite_step_audit"]["numerically_valid"]
                    checks.append(check)
                return {**score_fit(fit.ernn_result, V, z), "checks": checks,
                    "centralized_score": central_records[rep]["score"],
                    "passed": all(c["passed"] for c in checks), "round_history": fit.history["rounds"]}
            records.append(store.task(f"rounds_T{rounds}_rep{rep}", evaluate_rounds))
        round_records[rounds] = records
        round_scores[rounds] = [r["score"] for r in records]
    rounds = select_rounds_one_se(round_scores)
    write_json(store.root / "round_selection.json", {"selected_T": rounds, "scores": round_scores, "records": round_records})
    if not protocol.smoke and not all(r["passed"] for records in round_records.values() for r in records):
        raise GateFailure("selected-model development convergence failed; compare round_selection.json; no config frozen")
    fc = replace(fc, rounds=rounds)

    # Stage 5: each baseline gets its own optimizer search at the chosen common T.
    for method in ("fedavg", "fedprox", "scaffold"):
        candidates = []
        for epochs in protocol.local_epochs_grid:
            for lr in protocol.local_lr_grid:
                for mu in (protocol.prox_grid if method == "fedprox" else (0.,)):
                    records = []
                    for rep in range(protocol.development_repeats):
                        def fit_baseline():
                            _, _, parts, V, z = dev_repeat(rep)
                            conf = replace(base, seed=seed(f"dev_model_{rep}"))
                            try:
                                fitted = (train_scaffold(parts, conf, rounds, epochs, lr) if method == "scaffold"
                                    else train_fedavg(parts, conf, rounds, epochs, lr, mu))
                                return {"finite": True, **score_fit(fitted.ernn_result, V, z)}
                            except FloatingPointError as exc:
                                return {"finite": False, "failure": str(exc)}
                        records.append(store.task(f"{method}_e{epochs}_lr{lr:g}_mu{mu:g}_rep{rep}", fit_baseline))
                    if all(r["finite"] for r in records):
                        candidates.append({"epochs": epochs, "lr": lr, "mu": mu,
                            "score": float(np.mean([r["score"] for r in records])), "replicates": records})
        selected = minimum_candidate(candidates, lambda r: (r["epochs"], r["lr"], r["mu"]))
        write_json(store.root / f"{method}_selection.json", {"selected": selected, "candidates": candidates})
        updates = {f"{method}_local_epochs": selected["epochs"], f"{method}_learning_rate": selected["lr"]}
        if method == "fedprox":
            updates["fedprox_mu"] = selected["mu"]
        fc = replace(fc, **updates)
    fc.validate()
    if implementation_hash() != store.manifest["implementation_hash"]:
        raise GateFailure("implementation changed during this run; no configuration frozen")
    scope = {"scenario": scenario, "error": error}
    evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(store.root.glob("*_selection.json"))}
    evidence["convergence_report.json"] = hashlib.sha256((store.root / "convergence_report.json").read_bytes()).hexdigest()
    payload = {"config": asdict(fc), "fingerprint": fc.fingerprint(), "selection": {
        "pipeline_version": 3 if protocol.numerical_acceptance == "finite_step" else 2,
        "numerical_acceptance": protocol.numerical_acceptance,
        "scope": scope,
        "selection_complete": True,
        "formal_size_verified": False,
        "eligible_for_formal_simulation": not protocol.smoke and protocol.numerical_acceptance != "finite_step",
        "smoke": protocol.smoke, "manifest_fingerprint": store.fingerprint,
        "implementation_hash": implementation_hash(), "evidence_files_sha256": evidence,
        "evidence_root": str(store.root.resolve()),
        "numerical_gate_note": ("finite-step feasibility and objective checks; raw stationarity retained separately; no optimality or equivalence certification"
            if protocol.numerical_acceptance == "finite_step" else
            "engineering stationarity/plateau checks; not a global optimality or statistical equivalence proof"),
        "not_certified": ["linear qrcm reference agreement", "formal-size solver convergence", "real-data tuning", "publication table completeness"]}}
    payload = json.loads(json.dumps(payload))
    destination = store.root / ("smoke_config.json" if protocol.smoke else "selected_config.json")
    if destination.exists() and json.loads(destination.read_text()) != payload:
        raise RuntimeError("refusing to replace a different selected configuration")
    write_json(destination, payload)
    return {"status": "smoke_complete" if protocol.smoke else "selection_complete", "config": str(destination),
        "eligible_for_formal_simulation": not protocol.smoke and protocol.numerical_acceptance != "finite_step",
        "formal_size_check_required": not protocol.smoke and protocol.numerical_acceptance == "finite_step", "J": base.hidden_units,
        "lambda": base.lambda_, "eta": base.eta, "T": rounds}
