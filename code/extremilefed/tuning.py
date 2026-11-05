"""Sequential J/lambda and noncrossing-penalty selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from sklearn.model_selection import KFold

from .config import ERNNConfig
from .metrics import crossing_rate, pinball_loss
from .model import predict_quantiles
from .training import train_ernn
from .quadrature import gauss_legendre_unit


@dataclass(frozen=True)
class SelectionRecord:
    hidden_units: int
    lambda_: float
    eta: float
    mean_pinball: float
    se_pinball: float
    mean_crossing_rate: float


def select_j_lambda(
    X: np.ndarray,
    y: np.ndarray,
    base: ERNNConfig,
    j_grid: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    folds: int = 5,
) -> tuple[ERNNConfig, list[SelectionRecord]]:
    levels, weights = gauss_legendre_unit(base.quadrature.prediction_nodes)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=base.seed)
    records: list[SelectionRecord] = []
    for hidden_units in j_grid:
        for lambda_ in lambda_grid:
            scores = []
            crossings = []
            for fold, (train_idx, valid_idx) in enumerate(splitter.split(X)):
                config = replace(base, hidden_units=int(hidden_units), lambda_=float(lambda_), seed=base.seed + fold)
                result = train_ernn(X[train_idx], y[train_idx], config)
                pred = predict_quantiles(result, X[valid_idx], levels)
                scores.append(pinball_loss(y[valid_idx], pred, levels, weights))
                crossings.append(crossing_rate(pred))
            records.append(
                SelectionRecord(
                    hidden_units=int(hidden_units),
                    lambda_=float(lambda_),
                    eta=float(base.eta),
                    mean_pinball=float(np.mean(scores)),
                    se_pinball=float(np.std(scores, ddof=1) / np.sqrt(folds)),
                    mean_crossing_rate=float(np.mean(crossings)),
                )
            )
    best_mean = min(record.mean_pinball for record in records)
    eligible = [record for record in records if np.isclose(record.mean_pinball, best_mean, rtol=1e-10, atol=1e-12)]
    selected = sorted(eligible, key=lambda record: (record.hidden_units, -record.lambda_))[0]
    return replace(base, hidden_units=selected.hidden_units, lambda_=selected.lambda_), records


def select_eta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    base: ERNNConfig,
    eta_grid: tuple[float, ...],
    crossing_threshold: float = 0.005,
) -> tuple[ERNNConfig, list[SelectionRecord]]:
    levels = np.linspace(0.01, 0.99, 99)
    records = []
    candidate_grid = sorted(set(eta_grid))
    if 100.0 in candidate_grid and 1000.0 not in candidate_grid:
        candidate_grid.append(1000.0)
    for eta in candidate_grid:
        config = replace(base, eta=float(eta))
        result = train_ernn(X_train, y_train, config)
        pred = predict_quantiles(result, X_valid, levels)
        record = SelectionRecord(
            hidden_units=config.hidden_units,
            lambda_=config.lambda_,
            eta=float(eta),
            mean_pinball=pinball_loss(y_valid, pred, levels),
            se_pinball=0.0,
            mean_crossing_rate=crossing_rate(pred),
        )
        records.append(record)
        if record.mean_crossing_rate <= crossing_threshold:
            return config, records
    raise RuntimeError("No eta candidate meets the crossing threshold, including the extension; do not freeze this configuration.")


def select_rounds_one_se(scores: dict[int, list[float]]) -> int:
    """Select the smallest round count within one SE of the best mean score."""

    if not scores or any(rounds not in (1, 2, 3) for rounds in scores):
        raise ValueError("scores must contain one or more of the round counts 1, 2, 3")
    if any(len(values) < 2 or not np.isfinite(values).all() for values in map(np.asarray, scores.values())):
        raise ValueError("each round count requires at least two finite development scores")
    means = {rounds: float(np.mean(values)) for rounds, values in scores.items()}
    best_rounds = min(means, key=means.get)
    best_values = np.asarray(scores[best_rounds], dtype=float)
    best_se = float(np.std(best_values, ddof=1) / np.sqrt(best_values.size))
    eligible = [rounds for rounds, mean in means.items() if mean <= means[best_rounds] + best_se]
    return min(eligible)
