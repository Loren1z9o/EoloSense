"""Checks for the Farm C multisensor experiment."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Comum/src'))
import nbm
from sensores_c import plan_layout,target_selector,sensor_group,score_profiles
from multisensor_c import fit_profiles

class FarmCTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        n=2200
        power=rng.uniform(.1,1,n)
        self.frame=pd.DataFrame(dict(time_stamp=pd.date_range('2024-01-01',periods=n,freq='10min'),asset_id=1,status_type_id=0,train_test=['train']*1800+['prediction']*400,
            wind_speed_235_avg=power*10,sensor_8_avg=power*100,power_2_avg=power,sensor_7_avg=rng.normal(15,1,n),
            sensor_151_avg=20+power*5+rng.normal(0,.1,n),sensor_90_avg=rng.normal(0,1,n),
            sensor_48_avg=80+power*2+rng.normal(0,.1,n),sensor_127_avg=10+power*4+rng.normal(0,.1,n)))
        self.desc={'wind_speed_235_avg':'Wind speed 1','sensor_8_avg':'Generator angle speed','power_2_avg':'Active power','sensor_7_avg':'Ambient temperature',
            'sensor_151_avg':'Temperature planetary bearing','sensor_90_avg':'Nacelle vibration longitudinal',
            'sensor_48_avg':'Hydraulic pressure Aggregate System pressure','sensor_127_avg':'RMS current phase 1 HV grid'}
        self.features=list(self.desc)
    def test_groups_and_disjoint_roles(self):
        parts=nbm.split_event(self.frame)
        layout=plan_layout(parts['fit_train'],self.features,self.desc)
        self.assertEqual(set(sensor_group(self.desc[t]) for t in layout.candidates),{'temperatura','vibracao','pressao','eletrica'})
        self.assertFalse(set(layout.inputs)&set(layout.candidates))
        self.assertIn('sensor_8_avg',layout.inputs)
        self.assertIsNone(sensor_group('Nacelle external air pressure'))
    def test_unpredictable_vibration_is_not_silently_removed(self):
        parts=nbm.split_event(self.frame)
        layout=plan_layout(parts['fit_train'],self.features,self.desc)
        full=nbm.add_engineered(self.frame,layout)
        selected,_=target_selector(self.desc)(full.loc[parts['fit_train'].index],full.loc[parts['fit_val'].index],layout.x_columns,layout.candidates)
        self.assertIn('sensor_90_avg',selected)
    def trained(self,frame):
        return nbm.train_nbm(frame,self.features,self.desc,'arvore','C','1',layout_planner=plan_layout,target_selector=target_selector(self.desc),candidate_options=[('tree',make_pipeline(SimpleImputer(),DecisionTreeRegressor(max_depth=3,random_state=42)),None)],max_fit_rows=1000)
    def test_future_prediction_does_not_change_training_and_scores_are_causal(self):
        with threadpool_limits(limits=1):
            original,full=self.trained(self.frame)
            changed=self.frame.copy()
            changed.loc[1800:,'sensor_48_avg']+=10000
            other,_=self.trained(changed)
        self.assertEqual(original.bundle['targets'],other.bundle['targets'])
        self.assertEqual(original.bundle['calibration_thresholds'],other.bundle['calibration_thresholds'])
        prediction=full.iloc[1800:]
        a=score_profiles(original.bundle,prediction,self.desc)
        b=score_profiles(original.bundle,prediction.iloc[:200],self.desc)
        for profile in a:
            pd.testing.assert_frame_equal(a[profile].iloc[:200],b[profile])
    def test_component_quotas_prevent_cabinet_only_temperature_bank(self):
        from unittest.mock import patch
        descriptions={'cabinet':'Temperature electrical cabinet','stator':'Temperature stator winding U1',
                      'gear':'Temperature planetary bearing','hydraulic':'Hydraulic oil tank temperature',
                      'transformer':'Oil temperature transformer'}
        values={'cabinet':.99,'stator':.7,'gear':.6,'hydraulic':.4,'transformer':.5}
        def fake(*args):
            return [],{name:values[name] for name in args[-1]}
        with patch.object(nbm,'choose_targets',side_effect=fake):
            selected,_=target_selector(descriptions)(None,None,[],list(descriptions))
        self.assertEqual(set(selected),{'stator','gear','hydraulic','transformer'})
    def test_kmeans_retrained_per_profile_with_correct_features(self):
        parts=nbm.split_event(self.frame)
        layout=plan_layout(parts['fit_train'],self.features,self.desc)
        groups={t:sensor_group(self.desc[t]) for t in layout.candidates}
        with threadpool_limits(limits=1):
            bundles=fit_profiles(parts,layout.inputs,layout.candidates,groups,max_fit_rows=1000,k=3)
        self.assertIn('sensor_90_avg',bundles['vibracao']['features'])
        self.assertNotIn('sensor_48_avg',bundles['vibracao']['features'])
        self.assertIn('sensor_48_avg',bundles['combinado']['features'])

if __name__=='__main__':
    unittest.main()
