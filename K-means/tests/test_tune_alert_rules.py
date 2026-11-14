import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tune_alert_rules import rank_configurations


class RuleSelectionTests(unittest.TestCase):
    def test_precision_priority_requires_minimum_sensitivity(self):
        rows = []
        # A single correct alarm has perfect precision but insufficient recall.
        for name, tp, fp in [("rare", 1, 0), ("chosen", 3, 1), ("baseline", 5, 4)]:
            actual = [True] * 6 + [False] * 4
            detected = [True] * tp + [False] * (6 - tp) + [True] * fp + [False] * (4 - fp)
            rows.extend(dict(config=name, quantile=.95, hours=24, fraction=.5,
                             anomaly=a, detected=d, window_with_rolling_coverage=100)
                        for a, d in zip(actual, detected))
        ranking = rank_configurations(rows)
        self.assertEqual(ranking.iloc[0].config, "chosen")
        self.assertFalse(ranking.loc[ranking.config.eq("rare"), "eligible_for_selection"].iloc[0])


if __name__ == "__main__":
    unittest.main()
