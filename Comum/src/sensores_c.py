"""Sensor banks and causal scores for the Farm C study; no failure labels used."""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import nbm

GROUPS = ('temperatura', 'vibracao', 'pressao', 'eletrica')
PROFILES = (*GROUPS, 'combinado')
MAX_PER_GROUP = 4
MAX_FIT_ROWS = 6000

def sensor_group(description):
    text = description.lower()
    if 'temp' in text:
        return 'temperatura'
    if 'vibration' in text:
        return 'vibracao'
    if re.search(r'hydraulic pressure|gearbox oil pressure|water pressure|pollution indicator', text):
        return 'pressao'
    if re.search(r'current|voltage|reactive power|mains frequency|grid frequency', text):
        return 'eletrica'
    return None

def plan_layout(fit, features, descriptions):
    values = fit[features].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
    usable = [f for f in features if values[f].isna().mean() <= .3 and values[f].nunique() > 1]
    roles = {f: nbm.sensor_role(f,descriptions.get(f,'')) for f in usable}
    for f in usable:
        if 'generator angle speed' in descriptions.get(f,'').lower():
            roles[f]='rotacao'
    inputs, engineered = [], []
    for role in ('vento','potencia','rotacao','ambiente'):
        members=sorted((f for f in usable if roles[f]==role), key=lambda f:(
            0 if role=='rotacao' and 'generator' in descriptions.get(f,'').lower() else 1,
            values[f].isna().mean(),f))
        inputs.extend(members[:2])
        if members and role!='ambiente':
            for window in (('1h',) if role=='vento' else ('1h','6h')):
                engineered.append((f'{members[0]}__media_{window}',members[0],window))
    candidates=[f for f in usable if f not in inputs and roles[f] is None and sensor_group(descriptions.get(f,''))]
    if not inputs or not candidates:
        raise ValueError('Sem condições de operação ou sensores monitorados utilizáveis.')
    return nbm.Layout(inputs,engineered,candidates,{f:roles[f] for f in inputs})

def component(group,description):
    text=description.lower()
    if group=='temperatura':
        if 'hydraulic' in text: return 'hydraulic'
        if 'transformer' in text: return 'transformer'
        if re.search(r'gearbox|planetary|axial bearing|rotor bearing',text): return 'gearbox'
        if re.search(r'generator|stator',text): return 'generator'
    if group=='pressao':
        if 'aggregate system' in text: return 'system'
        if 'rotor brake' in text: return 'brake'
        if 'gearbox oil' in text: return 'gearbox'
        if 'water' in text: return 'cooling'
    if group=='eletrica':
        if 'current' in text and 'pump' in text: return 'pump'
        if 'reactive power' in text: return 'reactive'
        if 'voltage' in text: return 'voltage'
        if 'current' in text: return 'current'
    return 'other'

COMPONENT_ORDER={'temperatura':('generator','gearbox','hydraulic','transformer'),
                 'pressao':('system','brake','gearbox','cooling'),
                 'eletrica':('current','voltage','reactive','pump')}

def target_selector(descriptions):
    def choose(train,val,x_columns,candidates):
        selected, scores = [], {}
        for group in GROUPS:
            members=[f for f in candidates if sensor_group(descriptions.get(f,''))==group]
            if not members:
                continue
            try:
                _, ranking=nbm.choose_targets(train,val,x_columns,members)
            except ValueError:
                continue
            scores.update(ranking)
            def key(f):
                text=descriptions.get(f,'').lower()
                direct=group=='temperatura' and bool(re.search(r'winding|bearing|oil',text))
                return (not direct,-ranking[f],f)
            ordered=sorted(ranking,key=key)
            picked=[]
            for part in COMPONENT_ORDER.get(group,()):
                members=[f for f in ordered if component(group,descriptions[f])==part]
                if members:
                    picked.append(members[0])
            picked.extend(f for f in ordered if f not in picked)
            selected.extend(picked[:MAX_PER_GROUP])
        if not selected:
            raise ValueError('Nenhum sensor monitorado com dados de validação suficientes.')
        return selected,scores
    return choose

def compact_grid(family):
    pipe=lambda model,scale=False:make_pipeline(SimpleImputer(strategy='median'),*([StandardScaler()] if scale else []),model)
    if family=='linear':
        return [(f'ridge alpha={a}',pipe(Ridge(alpha=a),True),None) for a in (1.,10.)]
    if family=='arvore':
        return [(f'arvore depth={d} leaf={leaf}',pipe(DecisionTreeRegressor(max_depth=d,min_samples_leaf=leaf,random_state=42)),None)
                for d in (8,14) for leaf in (20,80)]
    if family=='ensembles':
        return [('random_forest leaf=20',pipe(RandomForestRegressor(n_estimators=100,min_samples_leaf=20,max_features=.7,max_samples=.7,random_state=42,n_jobs=1)),None),
                ('extra_trees leaf=5',pipe(ExtraTreesRegressor(n_estimators=100,min_samples_leaf=5,max_features=.7,random_state=42,n_jobs=1)),None)]
    if family=='svm':
        return [(f'svr C={c} epsilon=.1',pipe(MultiOutputRegressor(SVR(C=c,epsilon=.1,cache_size=200)),True),1500) for c in (1.,10.)]
    raise ValueError(family)

def score_profiles(bundle,frame,descriptions,require_status=True):
    observed,predicted,raw=nbm.predict_z(bundle,frame)
    residual=(observed-predicted)/bundle['sigma']
    input_values=frame[bundle['inputs']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    input_ok=input_values.isna().mean(axis=1).le(.3).to_numpy()
    if require_status:
        input_ok &= frame.status_type_id.eq(0).to_numpy()
    results={}
    for profile in PROFILES:
        positions=[j for j,t in enumerate(bundle['targets']) if profile=='combinado' or sensor_group(descriptions[t])==profile]
        if not positions:
            results[profile]=pd.DataFrame(dict(time_stamp=frame.time_stamp.to_numpy(),eligible=False,score=np.nan,top_sensor=None))
            continue
        values=raw[:,positions]
        eligible=input_ok & (np.isnan(values).mean(axis=1)<=.3)
        r=residual[:,positions].copy()
        r[~eligible]=np.nan
        smooth=pd.DataFrame(r,index=pd.DatetimeIndex(frame.time_stamp)).rolling('1h',min_periods=3).mean().to_numpy()
        absolute=np.where(np.isnan(smooth),-np.inf,np.abs(smooth))
        top=absolute.argmax(axis=1)
        valid=eligible & np.isfinite(absolute.max(axis=1))
        names=np.array(bundle['targets'],dtype=object)[positions]
        results[profile]=pd.DataFrame(dict(time_stamp=frame.time_stamp.to_numpy(),eligible=valid,
            score=np.where(valid,absolute.max(axis=1),np.nan),top_sensor=np.where(valid,names[top],None)))
    return results
