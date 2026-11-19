"""Descriptive event-level ROC AUC using existing held-out prediction files only."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {'kmeans':'K-means','arvore':'Arvore-de-Decisao','ensembles':'Ensembles','svm':'SVM','linear':'Regressao-Linear'}

def event_summary(path, entry):
    frame = pd.read_csv(path, usecols=['time_stamp','eligible','score','threshold'], parse_dates=['time_stamp'])
    eligible = frame.eligible.astype(str).str.lower().eq('true')
    valid = eligible & np.isfinite(frame.score)
    thresholds = frame.threshold.dropna().unique()
    if len(thresholds) != 1 or not np.isfinite(thresholds[0]) or thresholds[0] <= 0:
        raise ValueError(f'Invalid normal-calibration threshold: {path}')
    if not valid.any():
        raise ValueError(f'No eligible scores: {path}')
    info = pd.read_csv(Path(entry.path).parent.parent/'event_info.csv', sep=';')
    meta = info.loc[info.event_id.eq(int(entry.event))].iloc[0]
    assert meta.event_label == entry.label
    window = frame.time_stamp.between(pd.Timestamp(meta.event_start),pd.Timestamp(meta.event_end)) if entry.label=='anomaly' else pd.Series(True,index=frame.index)
    window_valid = valid & window
    if not window_valid.any():
        raise ValueError(f'No scores in annotated window: {path}')
    maximum = float(frame.loc[valid,'score'].max())
    annotated = float(frame.loc[window_valid,'score'].max())
    threshold = float(thresholds[0])
    return dict(farm=entry.farm,event=entry.event,asset=entry.asset,label=entry.label,
                anomaly=entry.label=='anomaly',eligible_rows=int(valid.sum()),
                annotated_eligible_rows=int(window_valid.sum()),threshold=threshold,
                max_score=maximum,max_over_normal_threshold=maximum/threshold,
                annotated_max_score=annotated,annotated_max_over_normal_threshold=annotated/threshold,
                source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    out = ROOT/'Comum/outputs/diagnostico_auc_eventos'
    out.mkdir(parents=True,exist_ok=True)
    all_events=[]
    for study in ('acuracia_balanceada','fazenda_C_componentes'):
        manifest_path = ROOT/('Ensembles/outputs/acuracia_balanceada/manifest.csv' if study=='acuracia_balanceada' else 'Comum/outputs/fazenda_C_componentes/manifest.csv')
        manifest=pd.read_csv(manifest_path,dtype=str)
        held=manifest.loc[manifest.split.eq('evaluation')]
        assert len(held)==(30 if study=='acuracia_balanceada' else 16)
        for family,folder in FOLDERS.items():
            run=('detection_acuracia_balanceada' if family=='kmeans' else study) if study=='acuracia_balanceada' else study
            base=ROOT/folder/'outputs'/run
            for entry in held.itertuples():
                name=f'farm_{entry.farm}_event_{entry.event}'
                directory=base/name if family=='kmeans' and study=='acuracia_balanceada' else base/'eventos'/name
                all_events.append(dict(study=study,family=family,**event_summary(directory/'predictions.csv',entry)))
    events=pd.DataFrame(all_events)
    events.to_csv(out/'eventos.csv',index=False)
    rows=[]
    columns=('max_score','max_over_normal_threshold','annotated_max_score','annotated_max_over_normal_threshold')
    for (study,family),group in events.groupby(['study','family'],sort=False):
        row=dict(study=study,family=family,events=len(group),anomaly=int(group.anomaly.sum()),normal=int((~group.anomaly).sum()))
        row.update({f'auc_{column}':float(roc_auc_score(group.anomaly,group[column])) for column in columns})
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'auc.csv',index=False)
    text=['# Diagnóstico de separação por evento','',
          'Calculado somente com previsões já salvas dos eventos reservados. Não houve retreino, alteração de limiares ou escolha de novas regras. Escores maiores significam maior desvio. O tempo de apresentação informado pela equipe é **10 minutos**.','',
          '| Estudo | Modelo | Eventos (anomalia/normal) | AUC máximo bruto | AUC máximo/limiar normal |',
          '| --- | --- | ---: | ---: | ---: |']
    for r in summary.itertuples():
        text.append(f'| {r.study} | {r.family} | {r.events} ({r.anomaly}/{r.normal}) | {r.auc_max_score:.3f} | {r.auc_max_over_normal_threshold:.3f} |')
    text+=['','## Como interpretar','',
           'Cada evento é uma observação. O resumo principal é o maior escore finito e elegível de toda a seção de previsão, tanto nos eventos normais como nos anômalos. AUC é a fração de pares anômalo/normal em que o anômalo tem maior valor, contando empates pela metade. 0,5 indica ausência de ordenação útil nesse resumo; 1 representa ordenação perfeita nesta amostra; valores menores que 0,5 indicam ordenação invertida. Não são porcentagens de acerto de uma regra de alerta.','',
           'Os modelos são ajustados por evento, e suas escalas podem diferir. Por isso também mostramos o máximo dividido pelo limiar da calibração normal já congelado. Essa normalização não usa rótulos da previsão e não modifica os avisos. A coluna bruta atende ao diagnóstico solicitado; a normalizada ajuda a verificar dependência de escala.','',
           'O CSV inclui ainda máximos restritos à janela anotada das anomalias, mantendo toda a previsão nos normais. Essa análise secundária tem exposições desiguais e usa a anotação para recortar a janela; não equivale a um detector executável sem conhecer o evento.','',
           'Máximos são sensíveis a picos isolados, duração e quantidade de registros elegíveis. Uma AUC baixa enfraquece a hipótese de resolver o problema com um único limiar sobre o máximo, mas não prova que toda combinação temporal, seleção de sensores ou modelo falhará. AUC alta também não garante avisos novos dentro da janela anotada nem baixa taxa de falsos avisos.','',
           'São avaliações internas pequenas e já reutilizadas, com eventos correlacionados por turbina. O estudo anterior reúne 30 eventos das três fazendas; o novo reúne 16 da C. Não comparar esses totais como se fossem a mesma amostra, nem escolher grupos ou configurações pelo melhor resultado reservado.','',
           'Artefatos: `auc.csv` contém as quatro AUCs; `eventos.csv` contém os valores, exposições, origem e SHA-256 de cada previsão. Reprodução: `python -X utf8 Comum/src/diagnostico_auc_eventos.py`.']
    (out/'DIAGNOSTICO.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__':
    main()
