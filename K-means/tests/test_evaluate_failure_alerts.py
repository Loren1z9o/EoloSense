import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_failure_alerts import event_result, metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_from_known_confusion_matrix(self):
        frame = pd.DataFrame({"anomaly": [True] * 15 + [False] * 15,
                              "detected": [True] * 7 + [False] * 8 + [True] * 8 + [False] * 7,
                              "window_with_rolling_coverage": [100] * 30})
        result = metrics(frame)
        self.assertEqual((result["tp"], result["fn"], result["tn"], result["fp"]), (7, 8, 7, 8))
        for key in ["precision", "sensitivity", "specificity", "accuracy", "f1", "balanced_accuracy"]:
            self.assertAlmostEqual(result[key], 7 / 15)
        frame.detected = False
        self.assertIsNone(metrics(frame)["precision"])

    def test_window_boundary_and_normal_false_alert(self):
        frame = pd.DataFrame({"time_stamp": pd.date_range("2024-01-01", periods=3, freq="10min"),
                              "alarm": [True, False, False], "eligible": True,
                              "rolling_exceedance_fraction": .3})
        meta = SimpleNamespace(event_label="anomaly", event_start=frame.time_stamp.iloc[1], event_end=frame.time_stamp.iloc[2])
        self.assertFalse(event_result(frame, meta)["detected"])
        frame.loc[2, "alarm"] = True
        self.assertTrue(event_result(frame, meta)["detected"])
        frame.loc[2, "alarm"] = False
        meta.event_label = "normal"
        self.assertTrue(event_result(frame, meta)["detected"])


if __name__ == "__main__":
    unittest.main()
