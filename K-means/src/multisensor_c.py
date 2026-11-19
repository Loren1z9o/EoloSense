"""K-means retraining for each sensor profile in the Farm C study."""
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_profiles(parts,inputs,targets,groups,max_fit_rows=6000,k=32):
    fit=parts['fit'].sample(min(max_fit_rows,len(parts['fit'])),random_state=42)
    bundles={}
    for profile in (*sorted(set(groups.values())),'combinado'):
        selected=[t for t in targets if profile=='combinado' or groups[t]==profile]
        columns=list(dict.fromkeys(inputs+selected))
        values=fit[columns].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        columns=values.columns[values.nunique().gt(1) & values.isna().mean().le(.3)].tolist()
        imputer,scaler=SimpleImputer(strategy='median'),StandardScaler()
        x=scaler.fit_transform(imputer.fit_transform(values[columns]))
        model=MiniBatchKMeans(n_clusters=k,n_init=3,batch_size=1024,max_iter=100,random_state=42).fit(x)
        bundles[profile]=dict(version=1,profile=profile,features=columns,imputer=imputer,scaler=scaler,model=model,fit_rows=len(fit))
    return bundles


def score(bundle,frame,require_status=True):
    values=frame[bundle['features']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    eligible=values.isna().mean(axis=1).le(.3).to_numpy()
    if require_status:
        eligible &= frame.status_type_id.eq(0).to_numpy()
    x=bundle['scaler'].transform(bundle['imputer'].transform(values))
    distances=bundle['model'].transform(x).min(axis=1)
    return pd.DataFrame(dict(time_stamp=frame.time_stamp.to_numpy(),eligible=eligible,score=np.where(eligible,distances,np.nan),top_sensor=None))
