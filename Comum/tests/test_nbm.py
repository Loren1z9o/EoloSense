"""Testes das garantias metodológicas dos modelos de comportamento normal.

Rodar na raiz do projeto:
    python -m unittest discover -s Comum/tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Comum/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nbm  # noqa: E402
import sintetico  # noqa: E402


class NBMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        sintetico.build(Path(cls.tmp.name), assets=2, events_per_asset=1, train_days=30, pred_days=8)
        cls.farm = Path(cls.tmp.name) / "CARE_To_Compare/Wind Farm A"
        cls.frame, cls.features, cls.descriptions = nbm.load_event(cls.farm / "datasets/0.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_papeis_dos_sensores(self):
        self.assertEqual(nbm.sensor_role("sensor_0_avg", "Ambient temperature"), "ambiente")
        self.assertEqual(nbm.sensor_role("power_29_avg", "Active power"), "potencia")
        self.assertIsNone(nbm.sensor_role("reactive_power_27_avg", "Reactive power"))
        self.assertEqual(nbm.sensor_role("sensor_1_avg", "Rotor speed"), "rotacao")
        self.assertEqual(nbm.sensor_role("wind_speed_3_avg", ""), "vento")
        self.assertIsNone(nbm.sensor_role("sensor_11_avg", "Gearbox bearing temperature"))

    def test_angulos_e_contadores_ficam_fora(self):
        self.assertNotIn("sensor_2_avg", self.features)
        self.assertNotIn("sensor_5_avg", self.features)
        self.assertTrue(all(f.endswith("_avg") for f in self.features))

    def test_divisao_temporal_sem_sobreposicao(self):
        parts = nbm.split_event(self.frame)
        self.assertGreaterEqual(parts["fit_val"].time_stamp.min() - parts["fit_train"].time_stamp.max(), nbm.GAP)
        self.assertLess(parts["fit"].time_stamp.max(), parts["calibration"].time_stamp.min() - nbm.GAP)
        self.assertLess(parts["calibration"].time_stamp.max(), parts["prediction"].time_stamp.min())
        for name in ("fit", "calibration"):
            self.assertTrue(parts[name].status_type_id.eq(0).all())

    def test_atributos_derivados_sao_causais(self):
        layout = nbm.plan_layout(nbm.split_event(self.frame)["fit_train"], self.features, self.descriptions)
        changed = self.frame.copy()
        cut = len(changed) // 2
        changed.loc[cut:, layout.engineered[0][1]] += 100  # muda só o futuro
        a = nbm.add_engineered(self.frame, layout)
        b = nbm.add_engineered(changed, layout)
        columns = [e[0] for e in layout.engineered]
        pd.testing.assert_frame_equal(a.loc[:cut - 1, columns], b.loc[:cut - 1, columns])

    def test_previsao_nao_influencia_o_modelo(self):
        """Alterar a seção de previsão (onde está a falha) não pode mudar modelo, alvos nem limiares."""
        changed = self.frame.copy()
        prediction = nbm.split_rows(changed)[1]
        changed.loc[prediction, "sensor_11_avg"] += 50
        first = nbm.train_nbm(self.frame, self.features, self.descriptions, "arvore", "A", "0")[0].bundle
        second = nbm.train_nbm(changed, self.features, self.descriptions, "arvore", "A", "0")[0].bundle
        self.assertEqual(first["targets"], second["targets"])
        self.assertEqual(first["chosen"], second["chosen"])
        self.assertEqual(first["calibration_thresholds"], second["calibration_thresholds"])

    def test_escore_sobe_com_falha_injetada(self):
        trained, full = nbm.train_nbm(self.frame, self.features, self.descriptions, "linear", "A", "0")
        prediction = full.loc[nbm.split_rows(full)[1]]
        scores = nbm.score_frame(trained.bundle, prediction, "A")
        info = pd.read_csv(self.farm / "event_info.csv", sep=";").iloc[0]
        self.assertEqual(info.event_label, "anomaly")
        late = scores.time_stamp >= pd.Timestamp(info.event_end)
        early = scores.time_stamp < pd.Timestamp(info.event_start)
        self.assertGreater(scores.score[late].median(), 2 * scores.score[early].median())
        self.assertEqual(scores.loc[late, "top_sensor"].mode().iloc[0], "sensor_11_avg")

    def test_todas_as_familias_treinam(self):
        for family in ("arvore", "ensembles", "svm", "linear"):
            with self.subTest(family=family):
                trained = nbm.train_nbm(self.frame, self.features, self.descriptions, family, "A", "0")[0]
                self.assertGreater(trained.regression["mean_r2"], .5)
                thresholds = list(trained.bundle["calibration_thresholds"].values())
                self.assertEqual(thresholds, sorted(thresholds))

    def test_poda_da_arvore_registrada(self):
        bundle = nbm.train_nbm(self.frame, self.features, self.descriptions, "arvore", "A", "0")[0].bundle
        self.assertIn("ccp_alpha", bundle["chosen"])
        self.assertGreater(len(bundle["pruning"]), 1)

    def test_fazenda_bc_exige_status_zero_na_previsao(self):
        trained, full = nbm.train_nbm(self.frame, self.features, self.descriptions, "linear", "A", "0")
        prediction = full.loc[nbm.split_rows(full)[1]].copy()
        prediction["status_type_id"] = 4
        self.assertFalse(nbm.score_frame(trained.bundle, prediction, "B").eligible.any())
        self.assertTrue(nbm.score_frame(trained.bundle, prediction, "A").eligible.any())


if __name__ == "__main__":
    unittest.main()
