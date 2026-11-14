"""Checks for leakage, persistence, missing data and model reuse."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from failure_alerts import apply_alarm_rule, predict, predict_stream, train


class DetectorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame({
            "time_stamp": pd.date_range("2024-01-01", periods=2000, freq="10min"),
            "asset_id": 1, "status_type_id": 0,
            "train_test": ["train"] * 1700 + ["prediction"] * 300,
            "sensor_a_avg": rng.normal(size=2000),
            "sensor_b_avg": rng.normal(size=2000),
        })
        self.features = ["sensor_a_avg", "sensor_b_avg"]
        with threadpool_limits(limits=1):
            self.bundle = train(self.frame, "A", self.features, k=3, hours=24)
        self.pred = self.frame.iloc[1700:].copy()

    def test_prediction_values_do_not_change_training(self):
        changed = self.frame.copy()
        changed.loc[1700:, self.features] = 1e9
        with threadpool_limits(limits=1):
            other = train(changed, "A", self.features, k=3, hours=24)
        np.testing.assert_array_equal(self.bundle["model"].cluster_centers_, other["model"].cluster_centers_)
        self.assertEqual(self.bundle["threshold"], other["threshold"])
        gap = (pd.Timestamp(self.bundle["calibration_start"]).to_pydatetime()
               - pd.Timestamp(self.bundle["fit_end"]).to_pydatetime())
        self.assertGreater(gap.total_seconds(), 86400)

    def test_persistence_is_causal_and_requires_coverage(self):
        self.pred[self.features] = 1000.
        scores = predict(self.bundle, self.pred, "A")
        self.assertFalse(scores.alarm.iloc[:71].any())
        self.assertTrue(scores.alarm.iloc[71:].all())
        prefix = predict(self.bundle, self.pred.iloc[:150], "A")
        pd.testing.assert_frame_equal(scores.iloc[:150], prefix)

    def test_missing_sensors_do_not_create_healthy_predictions(self):
        self.pred[self.features] = np.nan
        scores = predict(self.bundle, self.pred, "A")
        self.assertFalse(scores.eligible.any())
        self.assertTrue(scores.score.isna().all())
        self.assertFalse(scores.alarm.any())
        with self.assertRaisesRegex(ValueError, "Sensores ausentes"):
            predict(self.bundle, self.pred.drop(columns=self.features[0]), "A")

    def test_reject_wrong_asset_farm_and_training_overlap(self):
        with self.assertRaises(ValueError):
            predict(self.bundle, self.pred.assign(asset_id=2), "A")
        with self.assertRaises(ValueError):
            predict(self.bundle, self.pred, "B")
        with self.assertRaisesRegex(ValueError, "posterior"):
            predict(self.bundle, self.frame.iloc[:100], "A")

    def test_serialization_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(self.bundle, path)
            pd.testing.assert_frame_equal(predict(self.bundle, self.pred, "A"),
                                          predict(joblib.load(path), self.pred, "A"))

    def test_status_filter_and_cadence(self):
        bundle = dict(self.bundle, farm="B")
        scores = predict(bundle, self.pred.assign(status_type_id=4), "B")
        self.assertFalse(scores.eligible.any())
        with self.assertRaisesRegex(ValueError, "10 minutos"):
            predict(self.bundle, self.pred.assign(time_stamp=pd.date_range("2025-01-01", periods=300, freq="1min")), "A")

    def test_stricter_fraction_removes_intermittent_alerts(self):
        scores = pd.DataFrame({"time_stamp": self.pred.time_stamp.to_numpy(),
                               "eligible": True,
                               "score": np.where(np.arange(len(self.pred)) % 5 < 2, 2., 0.)})
        loose = apply_alarm_rule(scores, 1., 24, .25)
        strict = apply_alarm_rule(scores, 1., 24, .5)
        self.assertTrue(loose.alarm.any())
        self.assertFalse(strict.alarm.any())
        self.assertFalse((strict.alarm & ~loose.alarm).any())

    def test_configurable_training_retains_centers(self):
        with threadpool_limits(limits=1):
            strict = train(self.frame, "A", self.features, k=3, quantile=.99,
                           hours=48, alarm_fraction=.5)
        np.testing.assert_array_equal(self.bundle["model"].cluster_centers_, strict["model"].cluster_centers_)
        self.assertGreaterEqual(strict["threshold"], self.bundle["threshold"])
        self.pred[self.features] = 1000.
        scores = predict(strict, self.pred, "A")
        self.assertFalse(scores.alarm.iloc[:143].any())
        self.assertTrue(scores.alarm.iloc[143:].all())
        with self.assertRaises(ValueError):
            train(self.frame, "A", self.features, quantile=1.)

    def test_primary_profile_defaults_to_six_hour_detection(self):
        with threadpool_limits(limits=1):
            primary = train(self.frame, "A", self.features, k=3)
        self.assertEqual(primary["hours"], 6)
        self.assertEqual(primary["threshold_quantile"], .95)
        self.assertEqual(primary["alarm_fraction"], .25)
        self.assertEqual(primary["recovery_hours"], 6)
        self.pred[self.features] = 1000.
        scores = predict(primary, self.pred, "A")
        self.assertFalse(scores.alarm.iloc[:17].any())
        self.assertTrue(scores.alarm.iloc[17:].all())

    def test_streaming_context_matches_single_batch_without_duplicate_notice(self):
        self.pred[self.features] = 1000.
        full = predict(self.bundle, self.pred, "A")
        first, state = predict_stream(self.bundle, self.pred.iloc[:100], "A")
        state = json.loads(json.dumps(state, allow_nan=False))
        second, _ = predict_stream(self.bundle, self.pred.iloc[100:], "A", state)
        pd.testing.assert_frame_equal(full, pd.concat([first, second], ignore_index=True))
        self.assertEqual(full.notification.sum(), 1)
        self.assertEqual(second.notification.sum(), 0)
        with self.assertRaisesRegex(ValueError, "sem repetir"):
            predict_stream(self.bundle, self.pred.iloc[:100], "A", state)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
