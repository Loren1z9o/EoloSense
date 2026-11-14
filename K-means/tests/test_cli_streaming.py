"""Exercise persisted state through the public CLI, including failed requests."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from failure_alerts import train


class StreamingCliTests(unittest.TestCase):
    def test_cli_persists_episode_and_rejects_overlap_or_wrong_model(self):
        rng = np.random.default_rng(10)
        frame = pd.DataFrame({"time_stamp": pd.date_range("2025-01-01", periods=2000, freq="10min"),
                              "asset_id": 1, "status_type_id": 0,
                              "train_test": ["train"] * 1700 + ["prediction"] * 300,
                              "sensor_a_avg": rng.normal(size=2000),
                              "sensor_b_avg": rng.normal(size=2000)})
        features = ["sensor_a_avg", "sensor_b_avg"]
        with threadpool_limits(limits=1):
            bundle = train(frame, "A", features, k=3)
        frame.loc[1700:, features] = 1000.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, state = root / "model.joblib", root / "state.json"
            joblib.dump(bundle, model)
            frame.iloc[1700:1800].to_csv(root / "first.csv", sep=";", index=False)
            frame.iloc[1800:].to_csv(root / "second.csv", sep=";", index=False)

            def invoke(csv, output, selected_model=model):
                return subprocess.run([sys.executable, str(SOURCE / "failure_alerts.py"), "predict",
                                       "--model", str(selected_model), "--csv", str(root / csv),
                                       "--farm", "A", "--output-dir", str(root / output),
                                       "--state-file", str(state)], capture_output=True, text=True, timeout=30)

            first = invoke("first.csv", "out1")
            self.assertEqual(first.returncode, 0, first.stderr)
            second = invoke("second.csv", "out2")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(len(pd.read_csv(root / "out1/notifications.csv")), 1)
            self.assertEqual(len(pd.read_csv(root / "out2/notifications.csv")), 0)
            previous_state = state.read_bytes()
            repeated = invoke("second.csv", "overlap")
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("sem repetir", repeated.stderr)
            self.assertEqual(state.read_bytes(), previous_state)
            other = root / "other.joblib"
            joblib.dump(dict(bundle, recovery_hours=12), other)
            wrong_model = invoke("second.csv", "wrong_model", other)
            self.assertNotEqual(wrong_model.returncode, 0)
            self.assertIn("outro modelo", wrong_model.stderr)
            self.assertEqual(state.read_bytes(), previous_state)


if __name__ == "__main__":
    unittest.main()
