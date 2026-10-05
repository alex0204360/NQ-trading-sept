"""Standalone calibration tests; run with unittest or the repository's pytest."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "calibrate_nq", Path(__file__).parents[1] / "examples" / "calibrate_nq.py"
)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def training_bars():
    close = 16000 + (np.arange(1200) % 17) * 0.25
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-02T14:30:00Z", periods=1200, freq="min"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 200 + np.arange(1200) % 31,
    })


class CalibrationTests(unittest.TestCase):
    def test_training_profile_is_finite_and_records_fit_interval(self):
        frame = training_bars()
        result = calibration.calibrate(frame, "fixture")
        self.assertEqual(result["training_rows"], 1200)
        self.assertTrue(result["calibrated"])
        self.assertGreater(result["sigma_log_return"], 0)
        self.assertEqual(result["fit_end_utc"], frame.timestamp.iloc[-1].isoformat())

    def test_holdout_and_naive_timestamps_are_rejected(self):
        for value in ["2025-01-01T05:00:00Z", "2024-01-01 14:30:00"]:
            frame = training_bars()
            frame["timestamp"] = frame.timestamp.astype(str)
            frame.loc[1199, "timestamp"] = value
            with self.assertRaises(ValueError):
                calibration.calibrate(frame, "fixture")

    def test_gaps_are_not_used_as_one_minute_returns(self):
        frame = training_bars()
        frame.loc[600:, "timestamp"] += pd.Timedelta(minutes=20)
        result = calibration.calibrate(frame, "fixture")
        self.assertEqual(result["contiguous_returns"], 1198)

    def test_tick_violations_and_flat_history_are_rejected(self):
        frame = training_bars()
        frame.loc[10, "close"] += 0.01
        with self.assertRaises(ValueError):
            calibration.calibrate(frame, "fixture")
        frame = training_bars()
        frame["close"] = frame["open"] = 16000
        with self.assertRaises(ValueError):
            calibration.calibrate(frame, "fixture")


if __name__ == "__main__":
    unittest.main()
