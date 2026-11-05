"""Immutable configuration snapshots for formal experiment launches."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import hashlib

from .config import ERNNConfig, FederatedConfig, QuadratureConfig


def verify_simulation_selection(path, scenario, error, *, allow_size_pending=False):
    """CLI safety gate: smoke/manual defaults cannot masquerade as selected settings."""
    from .development import implementation_hash
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selection = payload.get("selection", {})
    pending_allowed = (allow_size_pending and selection.get("numerical_acceptance") == "finite_step"
        and selection.get("selection_complete") is True)
    if (not selection.get("eligible_for_formal_simulation") and not pending_allowed) or selection.get("smoke", True):
        raise ValueError("formal simulation requires a passed development selection, not a smoke/manual config")
    if selection.get("scope") != {"scenario": scenario, "error": error}:
        raise ValueError("selected configuration belongs to a different scenario/error")
    if selection.get("implementation_hash") != implementation_hash():
        raise ValueError("implementation changed after selection; rerun development in a new directory")
    root = Path(selection["evidence_root"])
    evidence = selection.get("evidence_files_sha256", {})
    required = {"cv_selection.json", "eta_selection.json", "round_selection.json", "fedavg_selection.json", "fedprox_selection.json", "scaffold_selection.json", "convergence_report.json"}
    if set(evidence) != required:
        raise ValueError("selection evidence is incomplete")
    for name, expected in evidence.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"selection evidence changed: {name}")
    config = load_frozen_config(path)
    if selection.get("numerical_acceptance") == "finite_step" and not allow_size_pending:
        if not selection.get("formal_size_verified"):
            raise ValueError("finite-step configuration requires independent formal-size validation")
        proof_path = Path(selection["readiness_evidence"])
        if hashlib.sha256(proof_path.read_bytes()).hexdigest() != selection["readiness_sha256"]:
            raise ValueError("formal-size evidence checksum mismatch")
        proof = json.loads(proof_path.read_text())
        from .experiments import ALL_METHODS
        if (proof.get("passed") is not True or proof.get("config_fingerprint") != config.fingerprint()
            or proof.get("implementation_hash") != implementation_hash()
            or proof.get("scope") != {"scenario":scenario,"error":error}
            or proof.get("n_train") != 100000 or proof.get("n_eval") != 5000
            or proof.get("clients") != 20 or set(proof.get("methods",[])) != set(ALL_METHODS)):
            raise ValueError("incomplete or mismatched formal-size evidence")
        method_evidence=proof.get("method_evidence_sha256",{})
        if set(method_evidence)!=set(ALL_METHODS):
            raise ValueError("missing full-size method checkpoints")
        for method,expected in method_evidence.items():
            if hashlib.sha256((proof_path.parent/"tasks"/(method+".json")).read_bytes()).hexdigest()!=expected:
                raise ValueError(f"full-size method evidence changed: {method}")
    return config


def load_frozen_config(path: str | Path) -> FederatedConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = dict(payload["config"])
    if "surrogate_learning_rates" in values:
        values["surrogate_learning_rates"] = tuple(values["surrogate_learning_rates"])
    neural = dict(values.pop("ernn"))
    if neural.get("smoothing_kernel") != "gaussian":
        raise ValueError("legacy/non-Gaussian configuration: rerun development with Gaussian convolution")
    if "h" not in neural:
        raise ValueError("legacy bandwidth field: rerun development with the explicit h configuration")
    if "refinement_epochs" not in neural or "host_epochs" not in values:
        raise ValueError("legacy solver configuration: rerun the two-stage development pipeline")
    neural["quadrature"] = QuadratureConfig(**neural["quadrature"])
    config = FederatedConfig(ernn=ERNNConfig(**neural), **values)
    config.validate()
    if config.fingerprint() != payload["fingerprint"]:
        raise ValueError("frozen configuration checksum mismatch")
    return config


def freeze_config(config: FederatedConfig, path: str | Path) -> Path:
    """Create a config snapshot or verify that an existing snapshot is identical."""

    config.validate()
    destination = Path(path)
    payload = {
        "config": asdict(config),
        "fingerprint": config.fingerprint(),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"frozen configuration differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)
    return destination
