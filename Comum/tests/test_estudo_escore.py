"""Testes das variantes de escore do 3º experimento (Comum/src/estudo_escore.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Comum/src"))

import estudo_escore as ee  # noqa: E402


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2024-01-01", periods=300, freq="10min").to_numpy()
        self.smooth = np.zeros((300, 3))
        self.smooth[:, 0] = -4.0   # alvo mais FRIO que o esperado
        self.smooth[:, 1] = 2.0    # alvo mais quente
        self.smooth[:, 2] = 1.0
        self.eligible = np.ones(300, dtype=bool)

    def test_sinal_e_agregacao(self):
        table = ee.variant_scores(self.times, self.eligible, self.smooth)
        self.assertEqual(table.abs_max_1h.iloc[0], 4.0)      # |−4|
        self.assertEqual(table.pos_max_1h.iloc[0], 2.0)      # resfriamento não conta
        self.assertEqual(table.abs_top2_1h.iloc[0], 3.0)     # (4 + 2) / 2
        self.assertEqual(table.pos_top2_1h.iloc[0], 1.5)     # (2 + 1) / 2
        self.assertEqual(set(ee.VARIANTS) | {"time_stamp", "eligible"}, set(table.columns))

    def test_mediana_diaria_e_causal_e_exige_historico(self):
        table = ee.variant_scores(self.times, self.eligible, self.smooth)
        self.assertTrue(table.abs_max_med24h.iloc[:ee.MEDIAN_MIN_ROWS - 1].isna().all())
        changed = self.smooth.copy()
        changed[200:, :] = 50.0  # muda só o futuro
        other = ee.variant_scores(self.times, self.eligible, changed)
        pd.testing.assert_series_equal(table.abs_max_med24h.iloc[:200], other.abs_max_med24h.iloc[:200])

    def test_inelegivel_fica_sem_escore(self):
        eligible = self.eligible.copy()
        eligible[10] = False
        table = ee.variant_scores(self.times, eligible, self.smooth)
        self.assertTrue(table.loc[10, ee.VARIANTS].isna().all())
        self.assertFalse(table.eligible.iloc[10])

    def test_um_alvo_so(self):
        table = ee.variant_scores(self.times, self.eligible, self.smooth[:, [1]])
        self.assertEqual(table.abs_top2_1h.iloc[0], 2.0)


if __name__ == "__main__":
    unittest.main()
