"""Location-scale simulation mechanisms with known quantile processes."""

from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import norm, t

from .quadrature import gauss_legendre_unit, normalized_extremile_weights, extremile_weight


@lru_cache(maxsize=128)
def innovation_extremile(error: str, tau: float) -> float:
    """Adaptive probability-space reference integration, independent of ERNN nodes."""
    if tau == 0.5:
        return 0.0  # All four innovations are centered by construction.
    def integrand(u):
        return float(error_ppf(np.array([u]), error)[0] * extremile_weight(np.array([u]), tau)[0])
    parts = [quad(integrand, a, b, epsabs=1e-8, epsrel=1e-8, limit=300) for a, b in ((0.0, 0.5), (0.5, 1.0))]
    value, estimated_error = sum(v for v, e in parts), sum(e for v, e in parts)
    if not np.isfinite(value) or estimated_error > 1e-6:
        raise RuntimeError(f"truth quadrature failed tolerance: {error}, {tau}, {estimated_error}")
    return value


SCENARIO_DIMENSIONS = {1: 3, 2: 5, 3: 8}
ERRORS = ("normal", "t3", "lognormal", "contaminated")


def location_scale(X: np.ndarray, scenario: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if scenario == 1:
        location = 2.0 * np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2]
        scale = 0.3 * (1.0 + 0.5 * np.abs(X[:, 0]))
    elif scenario == 2:
        location = 1.5 * X[:, 0] * X[:, 1] + np.sin(np.pi * X[:, 2] * X[:, 3]) + 0.5 * X[:, 4] ** 2
        scale = 0.25 * (1.0 + 0.5 * X[:, 0] ** 2)
    elif scenario == 3:
        location = (
            1.4 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + (X[:, 2] + 0.5 * X[:, 3]) ** 2
            + 0.8 * np.sin(np.pi * X[:, 4] * X[:, 5])
            + 0.7 * X[:, 6] * X[:, 7]
        )
        scale = 0.25 * (1.0 + 0.2 * X[:, 0] ** 2 + 0.15 * X[:, 4] ** 2)
    else:
        raise ValueError("scenario must be 1, 2, or 3")
    return location, scale


def _mixture_cdf(x: float) -> float:
    raw_mean = 0.6
    raw_sd = np.sqrt(1.0 + 0.9 * 0.1 * 6.0**2)
    raw = x * raw_sd + raw_mean
    return float(0.9 * norm.cdf(raw) + 0.1 * norm.cdf(raw - 6.0))


@lru_cache(maxsize=8192)
def _mixture_ppf_cached(level_rounded: float) -> float:
    level = float(level_rounded)
    return float(brentq(lambda x: _mixture_cdf(x) - level, -20.0, 20.0))


def error_ppf(levels: np.ndarray, error: str) -> np.ndarray:
    levels = np.asarray(levels, dtype=float)
    if error == "normal":
        return norm.ppf(levels)
    if error == "t3":
        return t.ppf(levels, df=3) / np.sqrt(3.0)
    if error == "lognormal":
        raw = np.exp(norm.ppf(levels))
        return (raw - np.exp(0.5)) / np.sqrt((np.e - 1.0) * np.e)
    if error == "contaminated":
        return np.array([_mixture_ppf_cached(round(float(level), 14)) for level in levels])
    raise ValueError(f"unsupported error: {error}")


def sample_error(n: int, error: str, rng: np.random.Generator) -> np.ndarray:
    if error == "normal":
        return rng.normal(size=n)
    if error == "t3":
        return rng.standard_t(df=3, size=n) / np.sqrt(3.0)
    if error == "lognormal":
        raw = np.exp(rng.normal(size=n))
        return (raw - np.exp(0.5)) / np.sqrt((np.e - 1.0) * np.e)
    if error == "contaminated":
        component = rng.random(n) >= 0.9
        raw = rng.normal(size=n) + 6.0 * component
        return (raw - 0.6) / np.sqrt(1.0 + 0.9 * 0.1 * 6.0**2)
    raise ValueError(f"unsupported error: {error}")


def generate_data(n: int, scenario: int, error: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = SCENARIO_DIMENSIONS[int(scenario)]
    X = rng.uniform(-1.0, 1.0, size=(int(n), p))
    location, scale = location_scale(X, int(scenario))
    y = location + scale * sample_error(int(n), error, rng)
    return X, y


def true_quantiles(X: np.ndarray, scenario: int, error: str, levels: np.ndarray) -> np.ndarray:
    location, scale = location_scale(X, scenario)
    return location[:, None] + scale[:, None] * error_ppf(np.asarray(levels), error)[None, :]


def true_extremiles(
    X: np.ndarray,
    scenario: int,
    error: str,
    tau_levels: np.ndarray,
    quadrature_nodes: int | None = None,
) -> np.ndarray:
    if quadrature_nodes is None:
        location, scale = location_scale(X, scenario)
        constants = np.array([innovation_extremile(error, float(tau)) for tau in tau_levels])
        return location[:, None] + scale[:, None] * constants[None, :]
    nodes, weights = gauss_legendre_unit(quadrature_nodes)
    quantiles = true_quantiles(X, scenario, error, nodes)
    columns = [quantiles @ normalized_extremile_weights(nodes, weights, float(tau)) for tau in tau_levels]
    return np.column_stack(columns)
