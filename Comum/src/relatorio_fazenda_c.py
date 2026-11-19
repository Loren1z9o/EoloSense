"""Audit and visualize the completed Farm C experiment; no model selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from estudo_fazenda_c import ROOT,RUN,FOLDERS
from evaluate_failure_alerts import metrics

NAMES={'arvore':'Árvore','ensembles':'Ensembles','svm':'SVR','linear':'Ridge','kmeans':'K-means'}

def main():
    out=ROOT/'Comum/outputs'/RUN
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    source_paths={name:ROOT/('K-means/src' if name=='multisensor_c.py' else 'Comum/src')/name for name in protocol['source_sha256']}
    for name,path in source_paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['source_sha256'][name]:
            raise ValueError(f'Código mudou desde o treino: {name}')
    manifest=pd.read_csv(out/'manifest.csv',dtype=str)
    assert len(manifest)==58 and manifest.groupby('asset').split.nunique().max()==1
    primary=pd.read_csv(out/'comparacao.csv')
    assert len(primary)==5 and set(primary.family)==set(FOLDERS)
    for family,folder in FOLDERS.items():
        base=ROOT/folder/'outputs'/RUN
        reports=list((base/'eventos').glob('*/report.json'))
        assert len(reports)==58,family
        assert not list((base/'eventos').glob('*/error.json')),family
        selection=pd.read_csv(base/'combinado/selection_events.csv')
        assert len(selection)==42*75 and selection.config.nunique()==75,family
        events=pd.read_csv(base/'combinado/events.csv')
        held=events.loc[events.split.eq('evaluation')]
        assert len(held)==16 and held.anomaly.sum()==7,family
        actual=metrics(held)
        row=primary.loc[primary.family.eq(family)].iloc[0]
        for key in ('tp','fn','tn','fp','sensitivity','specificity','balanced_accuracy'):
            assert np.isclose(actual[key],row[key]),(family,key)
    previous=pd.read_csv(out/'referencia_anterior_C.csv')
    inverse={v:k for k,v in FOLDERS.items()}
    previous['family']=previous.family.map(inverse)
    comparison=primary.merge(previous[['family','balanced_accuracy']],on='family',suffixes=('','_anterior'),validate='one_to_one')
    comparison.to_csv(out/'antes_depois_C.csv',index=False)
    order=list(FOLDERS)
    comparison=comparison.set_index('family').loc[order]
    plt.rcParams.update({'font.size':12})
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(5)
    a=ax.bar(x-.18,100*comparison.balanced_accuracy_anterior,.36,label='Estudo anterior — C',color='#8A9EAD')
    b=ax.bar(x+.18,100*comparison.balanced_accuracy,.36,label='Novo conjunto combinado',color='#008C87')
    ax.bar_label(a,fmt='%.1f',padding=3,fontsize=10)
    ax.bar_label(b,fmt='%.1f',padding=3,fontsize=10)
    ax.set_xticks(x,[NAMES[f] for f in order])
    ax.set_ylabel('Acurácia balanceada (%)')
    ax.set_ylim(0,105)
    ax.axhline(50,ls=':',color='#6F7479',lw=1)
    ax.legend(loc='upper left')
    ax.set_title('Fazenda C — mesmos 16 eventos de avaliação interna')
    fig.text(.5,.015,'Protocolos diferem em sensores, orçamento de ajuste e seleção da regra; comparação descritiva.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(out/'comparacao.png',dpi=150)
    plt.close(fig)
    data=pd.read_csv(out/'all_metrics.csv')
    pivot=data.loc[data.split.eq('evaluation')].pivot(index='family',columns='profile',values='balanced_accuracy').loc[order,['temperatura','vibracao','pressao','eletrica','combinado']]
    fig,ax=plt.subplots(figsize=(12,5.5))
    im=ax.imshow(pivot.values,vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(5),['Temperatura','Vibração','Pressão','Elétricos','Combinado'],fontsize=11)
    ax.set_yticks(range(5),[NAMES[f] for f in order])
    for i in range(5):
        for j in range(5):
            value=pivot.iloc[i,j]
            ax.text(j,i,f'{value:.1%}',ha='center',va='center',color='white' if value>.6 else 'black')
    ax.set_title('Diagnóstico por grupo — acurácia balanceada nos reservados')
    fig.colorbar(im,ax=ax,label='Acurácia balanceada')
    fig.tight_layout()
    fig.savefig(out/'diagnostico_grupos.png',dpi=150)
    plt.close(fig)
    audit={'events':58,'development':42,'evaluation':16,'evaluation_anomaly':7,'evaluation_normal':9,
           'model_families':5,'models_saved':290,'rules_per_family':75,'primary':'combinado',
           'group_diagnostics':'shared primary rule; supervised models share the same output bank',
           'source_hashes_match':True,'confusion_matrices_recomputed':True,
           'accepted_on_small_internal_evaluation':primary.loc[primary.sensitivity.ge(.7)&primary.specificity.ge(.8),'family'].tolist()}
    (out/'AUDITORIA.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(comparison[['balanced_accuracy_anterior','balanced_accuracy','sensitivity','specificity']].to_string())

if __name__=='__main__':
    main()
