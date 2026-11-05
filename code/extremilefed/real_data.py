"""Leakage-safe preprocessing for the two prespecified real-data studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .federated import partition_indices


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    client_indices: list[np.ndarray]
    metadata: dict[str, Any]


def _preprocessor(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def prepare_fremtpl2(
    frequency_csv: str | Path,
    severity_csv: str | Path,
    *,
    seed: int = 20260903,
    clients: int = 10,
) -> PreparedDataset:
    frequency = pd.read_csv(frequency_csv)
    severity = pd.read_csv(severity_csv)
    required_frequency = {
        "IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge",
        "BonusMalus", "VehBrand", "VehGas", "Density", "Region",
    }
    if not required_frequency.issubset(frequency.columns) or not {"IDpol", "ClaimAmount"}.issubset(severity.columns):
        raise ValueError("freMTPL2 files do not contain the documented columns")
    severity = severity.loc[np.isfinite(severity["ClaimAmount"]) & (severity["ClaimAmount"] > 0)]
    if frequency["IDpol"].duplicated().any():
        raise ValueError("frequency table must have one row per policy")
    severity_by_policy = severity.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    frame = frequency.merge(severity_by_policy, on="IDpol", how="inner")
    frame = frame.loc[np.isfinite(frame["ClaimAmount"]) & (frame["ClaimAmount"] > 0)].copy()
    target = frame.pop("ClaimAmount").to_numpy(dtype=float)
    frame = frame.drop(columns=["IDpol", "ClaimNb"])
    bins = pd.qcut(target, q=10, labels=False, duplicates="drop")
    train_frame, temp_frame, train_y, temp_y, _, temp_bins = train_test_split(
        frame, target, bins, test_size=0.30, random_state=seed, stratify=bins
    )
    valid_frame, test_frame, valid_y, test_y = train_test_split(
        temp_frame, temp_y, test_size=0.50, random_state=seed + 1, stratify=temp_bins
    )
    categorical = ["Area", "VehBrand", "VehGas", "Region"]
    numeric = [column for column in train_frame.columns if column not in categorical]
    transformer = _preprocessor(train_frame, numeric, categorical)
    X_train = transformer.fit_transform(train_frame)
    X_valid = transformer.transform(valid_frame)
    X_test = transformer.transform(test_frame)
    client_indices = partition_indices(len(train_y), clients, seed + 2)
    return PreparedDataset(
        np.asarray(X_train), np.asarray(train_y), np.asarray(X_valid), np.asarray(valid_y),
        np.asarray(X_test), np.asarray(test_y), client_indices,
        {
            "dataset": "freMTPL2 positive policy-level claim severity",
            "split": "70/15/15 stratified by response decile",
            "clients": clients,
            "client_allocation": "random near-equal",
            "rows": len(target),
            "features_after_encoding": int(X_train.shape[1]),
            "seed": seed,
            "excluded_predictors": ["IDpol", "ClaimNb"],
            "row_indices": {"train": train_frame.index.tolist(), "valid": valid_frame.index.tolist(), "test": test_frame.index.tolist()},
        },
    )


def prepare_beijing_air_quality(
    csv_or_directory: str | Path,
    *,
    seed: int = 20260903,
    clients: int = 20,
) -> PreparedDataset:
    source = Path(csv_or_directory)
    if source.is_dir():
        files = sorted(source.glob("*.csv"))
        if not files:
            raise FileNotFoundError("no Beijing station CSV files were found")
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    else:
        frame = pd.read_csv(source)
    required = {
        "year", "month", "day", "hour", "PM2.5", "TEMP", "PRES", "DEWP",
        "RAIN", "wd", "WSPM", "station",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Beijing file does not contain the documented variables")
    frame = frame.loc[np.isfinite(frame["PM2.5"])].copy()
    frame["timestamp"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    target = frame.pop("PM2.5").to_numpy(dtype=float)
    timestamps = frame["timestamp"].to_numpy()
    frame = frame[["year", "month", "day", "hour", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd", "station"]]
    n = len(frame)
    train_stop = int(np.searchsorted(timestamps, timestamps[int(np.floor(0.70 * n))], side="left"))
    valid_stop = int(np.searchsorted(timestamps, timestamps[int(np.floor(0.85 * n))], side="left"))
    if not 0 < train_stop < valid_stop < n:
        raise ValueError("not enough distinct times for three disjoint splits")
    train_frame, valid_frame, test_frame = frame.iloc[:train_stop], frame.iloc[train_stop:valid_stop], frame.iloc[valid_stop:]
    train_y, valid_y, test_y = target[:train_stop], target[train_stop:valid_stop], target[valid_stop:]
    categorical = ["wd", "station"]
    numeric = [column for column in train_frame.columns if column not in categorical]
    transformer = _preprocessor(train_frame, numeric, categorical)
    X_train = transformer.fit_transform(train_frame)
    X_valid = transformer.transform(valid_frame)
    X_test = transformer.transform(test_frame)
    client_indices = partition_indices(len(train_y), clients, seed)
    return PreparedDataset(
        np.asarray(X_train), np.asarray(train_y), np.asarray(X_valid), np.asarray(valid_y),
        np.asarray(X_test), np.asarray(test_y), client_indices,
        {
            "dataset": "UCI Beijing Multi-Site Air Quality",
            "response": "PM2.5",
            "split": "chronological 70/15/15",
            "time_boundaries": [str(timestamps[train_stop]), str(timestamps[valid_stop])],
            "split_sizes": [train_stop, valid_stop - train_stop, n - valid_stop],
            "predictors": frame.columns.tolist(),
            "seed": seed,
            "clients": clients,
            "client_allocation": "random near-equal within training period",
            "rows": n,
            "features_after_encoding": int(X_train.shape[1]),
        },
    )
