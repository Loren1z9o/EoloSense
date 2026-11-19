"""Retrain Farm C normal-behavior models and compare sensor-group diagnostics."""
import argparse
import hashlib
import itertools
import json
import platform
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits
import nbm
from sensores_c import GROUPS,PROFILES,MAX_FIT_ROWS,plan_layout,target_selector,compact_grid,score_profiles,sensor_group
from treinar_modelos import FAMILIES,load_manifest,event_meta
from failure_alerts import apply_alarm_rule
from alert_episodes import group_alerts
from tune_detection import measure,summarize
from multisensor_c import fit_profiles,score as km_score

ROOT=nbm.PROJECT_ROOT
RUN='fazenda_C_componentes'
FOLDERS={**{f:v[0] for f,v in FAMILIES.items()},'kmeans':'K-means'}
GRID=list(itertools.product([.90,.925,.95,.975,.99],[6,12,24,48,72],[.1,.25,.5]))

def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def directory(family,event):
    return ROOT/FOLDERS[family]/'outputs'/RUN/'eventos'/f'farm_C_event_{event}'

def pack_scores(pred,cal,output):
    combined=None
    thresholds={}
    for profile in PROFILES:
        if profile not in pred or profile not in cal:
            continue
        scores=cal[profile].loc[cal[profile].eligible,'score'].dropna()
        thresholds[profile]={str(q):float(np.quantile(scores,q)) for q in sorted({x[0] for x in GRID})} if len(scores)>=50 else None
        data=pred[profile].rename(columns={c:f'{c}__{profile}' for c in ('eligible','score','top_sensor')})
        combined=data if combined is None else combined.merge(data,on='time_stamp',validate='one_to_one')
    combined.to_csv(output/'scores.csv',index=False)
    return thresholds

def train_event(entry):
    logs=[]
    with threadpool_limits(limits=1):
        todo=[f for f in FOLDERS if not (directory(f,entry.event)/'report.json').exists()]
        if not todo:
            return [f'C/{entry.event}: já concluído']
        frame,features,descriptions=nbm.load_event(Path(entry.path))
        if str(frame.asset_id.iloc[0])!=entry.asset:
            raise ValueError('Identidade diferente do manifesto')
        parts=nbm.split_event(frame)
        layout=plan_layout(parts['fit_train'],features,descriptions)
        full=nbm.add_engineered(frame,layout)
        views={k:full.loc[v.index] for k,v in parts.items() if isinstance(v,pd.DataFrame)}
        targets,ranks=target_selector(descriptions)(nbm.sample(views['fit_train'],4800),nbm.sample(views['fit_val'],2000),layout.x_columns,layout.candidates)
        groups={t:sensor_group(descriptions[t]) for t in targets}
        for family in todo:
            out=directory(family,entry.event)
            out.mkdir(parents=True,exist_ok=True)
            start=time.perf_counter()
            try:
                if family=='kmeans':
                    bundles=fit_profiles(views,layout.inputs,targets,groups,max_fit_rows=MAX_FIT_ROWS)
                    pred={p:km_score(b,views['prediction']) for p,b in bundles.items()}
                    cal={p:km_score(b,views['calibration'],False) for p,b in bundles.items()}
                    joblib.dump(bundles,out/'model.joblib',compress=3)
                    chosen='MiniBatchKMeans k=32 n_init=3'
                    regression=None
                else:
                    # All families share the exact same sensor bank, selected only on normal validation data.
                    selector=lambda *args:(targets,ranks)
                    trained,model_frame=nbm.train_nbm(frame,features,descriptions,family,'C',entry.asset,
                        layout_planner=plan_layout,target_selector=selector,candidate_options=compact_grid(family),max_fit_rows=MAX_FIT_ROWS)
                    bundle=trained.bundle
                    bundle['target_groups']=groups
                    pred=score_profiles(bundle,model_frame.loc[parts['prediction'].index],descriptions)
                    cal=score_profiles(bundle,model_frame.loc[parts['calibration'].index],descriptions,False)
                    joblib.dump(bundle,out/'model.joblib',compress=3)
                    chosen=bundle['chosen']
                    regression=trained.regression
                thresholds=pack_scores(pred,cal,out)
                if thresholds.get('combinado') is None:
                    raise ValueError('Calibração combinada insuficiente')
                report=dict(family=family,event=entry.event,asset=entry.asset,split=entry.split,inputs=layout.inputs,
                    targets=targets,target_groups=groups,target_descriptions={t:descriptions[t] for t in targets},
                    chosen=chosen,thresholds=thresholds,regression=regression,
                    fit_end=str(parts['fit'].time_stamp.max()),calibration_start=str(parts['calibration'].time_stamp.min()),
                    prediction_start=str(parts['prediction'].time_stamp.min()),seconds=round(time.perf_counter()-start,1))
                dump(out/'report.json',report)
                (out/'error.json').unlink(missing_ok=True)
                logs.append(f'C/{entry.event} {family}: {chosen} ({report["seconds"]}s)')
            except Exception as exc:
                dump(out/'error.json',dict(error=str(exc),trace=traceback.format_exc()))
                logs.append(f'C/{entry.event} {family}: ERRO {exc}')
    return logs

def config_id(c):
    return f'q{c[0]}_h{c[1]}_f{c[2]}'

def apply_rule(scores,threshold,config):
    return group_alerts(apply_alarm_rule(scores,threshold,config[1],config[2]),6)[0]

def rank(rows):
    summaries=[]
    for name,group in pd.DataFrame(rows).groupby('config'):
        first=group.iloc[0]
        summaries.append(dict(config=name,quantile=float(first['quantile']),hours=int(first.hours),fraction=float(first.fraction),**summarize(group)))
    result=pd.DataFrame(summaries)
    result['within_budget']=result.false_notifications_per_30_days.le(5)&result.normal_incident_active_fraction.le(.25)
    result['violation']=((result.false_notifications_per_30_days/5-1).clip(lower=0)+(result.normal_incident_active_fraction/.25-1).clip(lower=0))
    return result.sort_values(['within_budget','violation','balanced_accuracy','sensitivity','false_notifications_per_30_days','config'],ascending=[False,True,False,False,True,True]).reset_index(drop=True)

def evaluate_family(family,manifest):
    root=ROOT/FOLDERS[family]/'outputs'/RUN
    cache={}
    for entry in manifest.itertuples():
        out=directory(family,entry.event)
        if not(out/'report.json').exists():
            raise ValueError(f'{family}/{entry.event}: sem treino concluído')
        cache[entry.event]=(entry,json.loads((out/'report.json').read_text(encoding='utf-8')),pd.read_csv(out/'scores.csv',parse_dates=['time_stamp']),event_meta(entry.path,entry.event))
    rows=[]
    primary_config=None
    for profile in ('combinado',*GROUPS):
        profile_out=root/profile
        profile_out.mkdir(exist_ok=True)
        selection=[]
        def event_scores(record):
            _,report,wide,_=record
            thresholds=report['thresholds'].get(profile)
            if not thresholds:
                return None,None
            scores=wide[['time_stamp',f'eligible__{profile}',f'score__{profile}']].rename(columns={f'eligible__{profile}':'eligible',f'score__{profile}':'score'})
            return scores,thresholds
        if profile=='combinado':
            for entry in manifest.loc[manifest.split.eq('selection')].itertuples():
                record=cache[entry.event]
                scores,thresholds=event_scores(record)
                if scores is None:
                    raise ValueError(f'{family}/{profile}/{entry.event}: grupo sem calibra??o; n?o excluir silenciosamente')
                for c in GRID:
                    result=measure(apply_rule(scores,thresholds[str(c[0])],c),record[3])
                    selection.append(dict(event=entry.event,config=config_id(c),quantile=c[0],hours=c[1],fraction=c[2],**result))
            ranking=rank(selection)
            ranking.to_csv(profile_out/'selection_ranking.csv',index=False)
            pd.DataFrame(selection).to_csv(profile_out/'selection_events.csv',index=False)
            winner=ranking.iloc[0]
            chosen=(float(winner['quantile']),int(winner.hours),float(winner.fraction))
            primary_config=chosen
            within_budget=bool(winner.within_budget)
        else:
            chosen=primary_config
            within_budget=None
        frozen=dict(config=config_id(chosen),quantile=chosen[0],hours=chosen[1],fraction=chosen[2],criterion='balanced',primary=profile=='combinado',within_budget=within_budget,selection_source='combinado')
        dump(profile_out/'frozen_config.json',frozen)
        print(f'FROZEN {family}/{profile}: {config_id(chosen)}',flush=True)
        results=[]
        for entry in manifest.itertuples():
            record=cache[entry.event]
            scores,thresholds=event_scores(record)
            if scores is None:
                raise ValueError(f'{family}/{profile}/{entry.event}: grupo sem calibração na avaliação')
            predicted=apply_rule(scores,thresholds[str(chosen[0])],chosen)
            result=measure(predicted,record[3])
            results.append(dict(family=family,profile=profile,split=entry.split,farm='C',asset=entry.asset,event=entry.event,label=record[3].event_label,**result))
            if profile=='combinado':
                predicted.to_csv(directory(family,entry.event)/'predictions.csv',index=False)
        data=pd.DataFrame(results)
        data.to_csv(profile_out/'events.csv',index=False)
        for split,group in data.groupby('split'):
            rows.append(dict(family=family,profile=profile,split=split,config=config_id(chosen),**summarize(group)))
    pd.DataFrame(rows).to_csv(root/'profile_metrics.csv',index=False)
    return rows

def write_comparison():
    rows=[]
    for family,folder in FOLDERS.items():
        path=ROOT/folder/'outputs'/RUN/'profile_metrics.csv'
        if not path.exists():
            raise ValueError(f'Métricas ausentes: {family}')
        rows.extend(pd.read_csv(path).to_dict('records'))
    data=pd.DataFrame(rows)
    output=ROOT/'Comum/outputs'/RUN
    data.to_csv(output/'all_metrics.csv',index=False)
    primary=data.loc[data.split.eq('evaluation') & data.profile.eq('combinado')]
    primary.to_csv(output/'comparacao.csv',index=False)
    text=['# Fazenda C — novo estudo multissensores','',
          'Resultado principal: conjunto combinado. Modelos e banco de sensores retreinados por evento; regras escolhidas somente nos 42 eventos de desenvolvimento e congeladas antes dos 16 reservados. Avaliação interna reutilizada, não teste externo intocado.','',
          '| Modelo | Sensibilidade | Especificidade | Acurácia balanceada | Precisão | VP | FN | VN | FP | Avisos falsos / 30 dias | Sem cobertura |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in primary.itertuples():
        text.append(f'| {row.family} | {row.sensitivity:.1%} | {row.specificity:.1%} | {row.balanced_accuracy:.1%} | {row.precision:.1%} | {row.tp} | {row.fn} | {row.tn} | {row.fp} | {row.false_notifications_per_30_days:.2f} | {row.events_without_rolling_coverage} |')
    text+=['','## Diagnóstico por grupo','',
           'Os modelos supervisionados compartilham um banco multissaída de até quatro sensores por grupo. Os escores por grupo são diagnósticos desse mesmo modelo, não ablações com retreinamento independente. O K-means é ajustado separadamente por perfil. Os diagn?sticos usam a mesma regra escolhida no combinado, com limiares calibrados por grupo. Nenhum perfil foi escolhido usando os resultados reservados.','',
           '| Modelo | Perfil | Sensibilidade | Especificidade | Acurácia balanceada |','| --- | --- | ---: | ---: | ---: |']
    for row in data.loc[data.split.eq('evaluation')].itertuples():
        text.append(f'| {row.family} | {row.profile} | {row.sensitivity:.1%} | {row.specificity:.1%} | {row.balanced_accuracy:.1%} |')
    text+=['','Os escores medem desvios dos sensores, não probabilidade ou data de falha. É avaliação simplificada de notificações, não CARE oficial. A comparação com estudos anteriores também muda orçamento de ajuste/grade de modelos e banco de alvos; não isola causalmente somente o efeito de acrescentar sensores.']
    (output/'COMPARACAO.md').write_text('\n'.join(text),encoding='utf-8')
    print(primary.to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--family',choices=list(FOLDERS))
    args=parser.parse_args()
    output=ROOT/'Comum/outputs'/RUN
    output.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest(ROOT/'data','C')
    if len(manifest)!=58 or manifest.split.eq('evaluation').sum()!=16:
        raise ValueError('Partição inesperada da fazenda C')
    manifest.to_csv(output/'manifest.csv',index=False)
    if not args.evaluate_only:
        code_paths=[Path(__file__),Path(nbm.__file__),Path(__file__).with_name('sensores_c.py'),nbm.KMEANS_SRC/'multisensor_c.py']
        protocol=dict(farm='C',seed=42,fit_rows_cap=MAX_FIT_ROWS,max_per_group=4,k=32,
            profiles=list(PROFILES),component_quotas=True,primary='combinado',grid=[list(c) for c in GRID],criterion='balanced',
            budgets=dict(false_per_30_days=5,active_fraction=.25),fit_only_normal=True,
            split='42 development / 16 evaluation, by turbine, existing partition',
            diagnostic='same multioutput model for supervised group scores; K-means refitted per group',
            compact_model_grid=True,svr_rows_cap=1500,diagnostic_rule='same as combined; group-specific calibration quantiles',
            original_data='read only',python=platform.python_version(),
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths})
        if (output/'protocol.json').exists():
            if json.loads((output/'protocol.json').read_text(encoding='utf-8'))!=protocol:
                raise ValueError('Protocolo ou código mudou; não misturar com checkpoints anteriores')
        else:
            dump(output/'protocol.json',protocol)
        for completed,logs in enumerate(Parallel(n_jobs=args.jobs,return_as='generator_unordered')(delayed(train_event)(entry) for entry in manifest.itertuples()),1):
            for line in logs:
                print(f'[{completed}/58] {line}',flush=True)
        errors=[str(directory(f,e.event)/'error.json') for e in manifest.itertuples() for f in FOLDERS if not(directory(f,e.event)/'report.json').exists()]
        if errors:
            dump(output/'errors.json',errors)
            raise ValueError(f'{len(errors)} treinos sem resultado; não publicar comparação parcial')
        print('TRAINING_COMPLETE',flush=True)
        return
    if args.family:
        evaluate_family(args.family,manifest)
    else:
        write_comparison()

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
