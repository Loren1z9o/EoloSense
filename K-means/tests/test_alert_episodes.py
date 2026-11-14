import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alert_episodes import group_alerts


def sample(alarms):
    return pd.DataFrame({"time_stamp": pd.date_range("2025-01-01", periods=len(alarms), freq="10min"),
                         "eligible": True, "rolling_exceedance_fraction": .5, "alarm": alarms})


class EpisodeTests(unittest.TestCase):
    def test_one_notification_for_thousands_of_alarm_rows(self):
        scores, state = group_alerts(sample([True] * 3000))
        self.assertEqual(scores.notification.sum(), 1)
        self.assertTrue(scores.incident_active.all())
        self.assertTrue(state["active"])

    def test_recovery_then_new_alarm_rearms(self):
        frame = sample([True] + [False] * 36 + [True])
        scores, _ = group_alerts(frame)
        self.assertEqual(scores.notification.sum(), 2)
        self.assertEqual(scores.recovered.sum(), 1)
        self.assertTrue(scores.incident_active.iloc[35])
        self.assertFalse(scores.incident_active.iloc[36])

    def test_missing_data_and_gaps_cannot_confirm_recovery(self):
        frame = sample([True] + [False] * 80)
        frame.loc[30:50, "eligible"] = False
        frame.loc[30:50, "rolling_exceedance_fraction"] = np.nan
        scores, state = group_alerts(frame)
        self.assertFalse(scores.recovered.any())
        self.assertTrue(state["active"])
        frame = sample([True] + [False] * 40)
        frame.loc[30:, "time_stamp"] = frame.loc[30:, "time_stamp"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(1, "D")
        scores, state = group_alerts(frame)
        self.assertFalse(scores.recovered.any())

    def test_chunk_state_does_not_duplicate_notifications(self):
        frame = sample([True] + [False] * 36 + [True] * 80)
        full, expected = group_alerts(frame)
        first, state = group_alerts(frame.iloc[:20])
        second, state = group_alerts(frame.iloc[20:], state=state)
        pd.testing.assert_frame_equal(full, pd.concat([first, second]))
        self.assertEqual(state, expected)
        with self.assertRaises(ValueError):
            group_alerts(frame, state=state)
