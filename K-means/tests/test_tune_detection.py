import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tune_detection import BASELINE, CONFIGS, GRIDS, measure, rank_candidates


class DetectionSelectionTests(unittest.TestCase):
    def test_recall_priority_and_always_active_rejected(self):
        rows = []
        for name, tp, fp, duty in [("sensitive", 9, 3, .2), ("precise", 6, 1, .1), ("always_on", 10, 10, 1.)]:
            actual = [True] * 10 + [False] * 10
            detected = [True] * tp + [False] * (10 - tp) + [True] * fp + [False] * (10 - fp)
            rows.extend(dict(config=name, quantile=.95, hours=6, fraction=.1,
                             anomaly=a, detected=d, signal_detected=d, notifications=int(d),
                             monitoring_days=30., monitored_rows=4320, active_monitored_rows=int(duty * 4320),
                             window_with_rolling_coverage=100) for a, d in zip(actual, detected))
        ranking = rank_candidates(rows)
        self.assertEqual(ranking.iloc[0].config, "sensitive")
        self.assertFalse(ranking.loc[ranking.config.eq("always_on"), "within_budget"].iloc[0])
        failed = rank_candidates(rows, max_false_per_30_days=.0001, max_active_fraction=.001)
        self.assertFalse(failed.within_budget.any())

    def test_balanced_criterion_weighs_specificity(self):
        """Com o critério de sensibilidade, quem avisa em todo evento normal vence; com acurácia balanceada, não."""
        rows = []
        for name, tp, fp in [("noisy", 9, 9), ("selective", 7, 2)]:
            actual = [True] * 10 + [False] * 10
            detected = [True] * tp + [False] * (10 - tp) + [True] * fp + [False] * (10 - fp)
            rows.extend(dict(config=name, quantile=.95, hours=6, fraction=.1,
                             anomaly=a, detected=d, signal_detected=d, notifications=int(d),
                             monitoring_days=30., monitored_rows=4320, active_monitored_rows=432,
                             window_with_rolling_coverage=100) for a, d in zip(actual, detected))
        self.assertEqual(rank_candidates(rows, 5, .25).iloc[0].config, "noisy")
        balanced = rank_candidates(rows, 5, .25, criterion="balanced")
        self.assertEqual(balanced.iloc[0].config, "selective")
        self.assertAlmostEqual(balanced.iloc[0].balanced_accuracy, (.7 + .8) / 2)
        with self.assertRaises(ValueError):
            rank_candidates(rows, criterion="desconhecido")

    def test_wide_grid_contains_standard_grid(self):
        quantiles, wide = GRIDS["wide"]
        self.assertTrue(set(CONFIGS) <= set(wide))
        self.assertIn(BASELINE, wide)
        self.assertEqual(len(wide), 75)
        self.assertTrue({c[0] for c in wide} <= set(quantiles))

    def test_old_open_episode_not_counted_as_new_detection(self):
        times = pd.date_range("2025-01-01", periods=3, freq="10min")
        scores = pd.DataFrame({"time_stamp": times, "eligible": True,
                               "rolling_exceedance_fraction": .5, "alarm": True,
                               "incident_active": True, "notification": [True, False, False]})
        meta = SimpleNamespace(event_label="anomaly", event_start=times[1], event_end=times[2])
        result = measure(scores, meta)
        self.assertTrue(result["signal_detected"])
        self.assertFalse(result["detected"])


if __name__ == "__main__":
    unittest.main()
