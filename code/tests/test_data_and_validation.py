from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from extremilefed.config import ERNNConfig, FederatedConfig
from extremilefed.real_data import prepare_beijing_air_quality, prepare_fremtpl2
from extremilefed.reporting import holm_adjust, paired_mean_ci
from extremilefed.reproducibility import freeze_config
from extremilefed.validation import validate_formal_coverage


class DataAndValidationTests(unittest.TestCase):
    def test_fremtpl2_preparation_uses_expected_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n = 120
            frequency = pd.DataFrame(
                {
                    "IDpol": np.arange(n),
                    "ClaimNb": np.ones(n),
                    "Exposure": np.ones(n),
                    "Area": np.resize(list("ABC"), n),
                    "VehPower": np.resize(np.arange(4), n),
                    "VehAge": np.resize(np.arange(10), n),
                    "DrivAge": 18 + np.resize(np.arange(50), n),
                    "BonusMalus": 50 + np.resize(np.arange(20), n),
                    "VehBrand": np.resize(["B1", "B2"], n),
                    "VehGas": np.resize(["Diesel", "Regular"], n),
                    "Density": 100 + np.arange(n),
                    "Region": np.resize(["R1", "R2", "R3"], n),
                }
            )
            severity = pd.DataFrame({"IDpol": np.arange(n), "ClaimAmount": 100.0 + np.arange(n)})
            freq_path, sev_path = root / "freq.csv", root / "sev.csv"
            frequency.to_csv(freq_path, index=False)
            severity.to_csv(sev_path, index=False)
            prepared = prepare_fremtpl2(freq_path, sev_path, clients=4)
            self.assertEqual(len(prepared.y_train), 84)
            self.assertEqual(sum(map(len, prepared.client_indices)), 84)
            self.assertEqual(len(prepared.client_indices), 4)
            from extremilefed.real_cli import main
            output = root / "fremtpl_run"
            with patch("sys.argv", ["real_cli", "--dataset", "fremtpl2", "--frequency", str(freq_path), "--severity", str(sev_path), "--smoke", "--output", str(output)]):
                main()
            self.assertEqual(len(json.loads((output / "metrics.json").read_text())["methods"]), 8)

    def test_beijing_chronological_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beijing.csv"
            n = 100
            frame = pd.DataFrame(
                {
                    "year": 2013,
                    "month": 3,
                    "day": 1 + np.arange(n) // 24,
                    "hour": np.arange(n) % 24,
                    "PM2.5": np.linspace(10.0, 80.0, n),
                    "TEMP": np.linspace(-2.0, 10.0, n),
                    "PRES": 1010.0,
                    "DEWP": -5.0,
                    "RAIN": 0.0,
                    "wd": np.resize(["N", "S"], n),
                    "WSPM": 1.0,
                    "station": np.resize(["A", "B"], n),
                }
            )
            frame.to_csv(path, index=False)
            prepared = prepare_beijing_air_quality(path, clients=5)
            self.assertEqual((len(prepared.y_train), len(prepared.y_valid), len(prepared.y_test)), (70, 15, 15))
            self.assertLess(prepared.y_train.max(), prepared.y_test.min())
            from extremilefed.real_cli import main
            output = Path(directory) / "beijing_run"
            with patch("sys.argv", ["real_cli", "--dataset", "beijing", "--source", str(path), "--smoke", "--output", str(output)]):
                main()
            self.assertTrue((output / "predictions.npz").exists())

    def test_formal_coverage_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "schema_version": 7,
                "job": {"scenario": 1, "error": "normal", "replication": 0, "n_train": 100000, "n_test": 5000, "clients": 20},
                "tau_levels": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                "methods": {
                    "ExtremileFed": {
                        "metrics": {
                            "extremile_mae": 1.0,
                            "extremile_rmse": 1.0,
                            "integrated_pinball": 1.0,
                            "coverage_error": 0.1,
                            "crossing_rate": 0.0,
                        }
                    }
                },
                "config_fingerprint": "abc",
            }
            (root / "scenario1_normal_rep000.json").write_text(json.dumps(payload), encoding="utf-8")
            result = validate_formal_coverage(
                root,
                replications=1,
                scenarios=(1,),
                errors=("normal",),
                required_methods={"ExtremileFed"},
            )
            self.assertTrue(result["complete"])

    def test_config_freeze_and_paired_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal.json"
            config = FederatedConfig(ernn=ERNNConfig(hidden_units=5), rounds=2)
            freeze_config(config, path)
            freeze_config(config, path)
            with self.assertRaises(RuntimeError):
                freeze_config(FederatedConfig(ernn=ERNNConfig(hidden_units=6), rounds=2), path)
        summary = paired_mean_ci([1.0, 1.2, 1.1], [0.9, 1.0, 1.0])
        self.assertEqual(summary["pairs"], 3)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(np.all((adjusted >= 0.0) & (adjusted <= 1.0)))


if __name__ == "__main__":
    unittest.main()
