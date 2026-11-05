"""Coverage checks for sharded formal outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from .config import TAU_LEVELS
from .experiments import RESULT_SCHEMA


FORMAL_METHODS = {
    "Centralized ERNN",
    "Pooled linear Extremile-SL",
    "Host ERNN",
    "One-shot ERNN",
    "FedAvg-ERNN",
    "FedProx-ERNN",
    "SCAFFOLD-ERNN",
    "ExtremileFed",
}


def validate_formal_coverage(
    output_dir: str | Path,
    *,
    replications: int = 100,
    scenarios: tuple[int, ...] = (1, 2, 3),
    errors: tuple[str, ...] = ("normal", "t3", "lognormal", "contaminated"),
    required_methods: set[str] | None = None,
) -> dict[str, object]:
    root = Path(output_dir)
    expected = {
        f"scenario{scenario}_{error}_rep{replication:03d}.json"
        for scenario in scenarios
        for error in errors
        for replication in range(replications)
    }
    existing = {path.name for path in root.glob("*.json")}
    missing = sorted(expected - existing)
    unexpected = sorted(existing - expected)
    invalid = []
    fingerprints = set()
    fingerprints_by_setting = {}
    methods_required = FORMAL_METHODS if required_methods is None else required_methods
    for name in sorted(expected & existing):
        try:
            payload = json.loads((root / name).read_text(encoding="utf-8"))
            job = payload["job"]
            if (payload.get("schema_version") != RESULT_SCHEMA
                    or (job["n_train"], job["n_test"], job["clients"]) != (100000, 5000, 20)
                    or name != f"scenario{job['scenario']}_{job['error']}_rep{job['replication']:03d}.json"
                    or payload.get("tau_levels") != list(TAU_LEVELS)):
                invalid.append(name)
            required = {"extremile_mae", "extremile_rmse", "integrated_pinball", "coverage_error", "crossing_rate"}
            methods = payload["methods"]
            if not methods_required.issubset(methods) or any(
                not required.issubset(methods[method]["metrics"])
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in methods[method]["metrics"].values())
                for method in methods_required
            ):
                invalid.append(name)
            fingerprints.add(payload["config_fingerprint"])
            setting = (job["scenario"], job["error"])
            fingerprints_by_setting.setdefault(setting, set()).add(payload["config_fingerprint"])
        except (KeyError, TypeError, ValueError, OSError):
            invalid.append(name)
    return {
        "complete": not missing and not unexpected and not invalid and bool(fingerprints) and all(len(v) == 1 for v in fingerprints_by_setting.values()),
        "expected_count": len(expected),
        "existing_count": len(existing),
        "missing": missing,
        "unexpected": unexpected,
        "invalid": sorted(set(invalid)),
        "config_fingerprints": sorted(fingerprints),
    }
