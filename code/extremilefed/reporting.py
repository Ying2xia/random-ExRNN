"""Paired uncertainty summaries and multiplicity correction for formal results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from .validation import validate_formal_coverage


def holm_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_mean_ci(
    first: np.ndarray | list[float],
    second: np.ndarray | list[float],
    confidence: float = 0.95,
) -> dict[str, float]:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or not 0 < confidence < 1:
        raise ValueError("paired shapes must match and confidence must be in (0,1)")
    difference = first - second
    if difference.ndim != 1 or difference.size < 2 or not np.isfinite(difference).all():
        raise ValueError("paired inputs require at least two finite differences")
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / np.sqrt(difference.size))
    critical = float(t.ppf(0.5 + confidence / 2.0, difference.size - 1))
    statistic = mean / se if se > 0 else (0.0 if mean == 0 else np.inf)
    p_value = float(2.0 * t.sf(abs(statistic), difference.size - 1)) if se > 0 else float(mean == 0)
    return {
        "mean_difference": mean,
        "lower": mean - critical * se,
        "upper": mean + critical * se,
        "p_value": p_value,
        "pairs": int(difference.size),
    }


def load_validated_results(output_dir: str | Path) -> list[dict[str, object]]:
    validation = validate_formal_coverage(output_dir)
    if not validation["complete"]:
        raise RuntimeError("formal results are incomplete or inconsistent; export is blocked")
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(output_dir).glob("*.json"))
    ]
