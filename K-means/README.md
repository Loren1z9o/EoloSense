# K-means — primeira exploração

**Versão principal adotada:** K=32, P95, janela de 6h, excedências de 25% e recuperação de 6h, com avisos agrupados e estado entre blocos. Guia em [DETECCAO_E_ALERTAS.md](DETECCAO_E_ALERTAS.md); modelos em [outputs/detection_balanced/](outputs/detection_balanced/). O comando principal `failure_alerts.py train` usa esse perfil por padrão. As seções seguintes registram experimentos históricos.

Fluxo que prioriza detecção com controle de episódios e avisos repetidos: [DETECCAO_E_ALERTAS.md](DETECCAO_E_ALERTAS.md). Inclui estado entre blocos e comparação de regras por sensibilidade e carga de notificações.

**Resumo atualizado incluindo o estudo com agentes:** [REVISAO_ATUAL.md](REVISAO_ATUAL.md). O melhor equilíbrio medido nos mesmos 30 eventos reservados continua sendo K64/P99/6h (46,7% de acurácia balanceada); o novo fluxo K32 reproduziu 43,3%, com mais detecção e mais falsos alertas.

Fluxo executável para **treinar, salvar e reutilizar um detector de anomalias**: [ALERTAS.md](ALERTAS.md). Inclui comandos PowerShell, interpretação dos alertas e testes. É um baseline experimental; o desempenho anterior ainda não sustenta previsão confiável de falhas.

Busca posterior de **108 configurações, incluindo K=3**: protocolo, comandos e resultados em [BUSCA_CONFIGURACOES.md](BUSCA_CONFIGURACOES.md). A configuração escolhida alcançou 46,7% de sensibilidade e 46,7% de especificidade em 30 eventos reservados por turbina; a grade não resolveu o problema de generalização.

> **Revisão metodológica de 14/11/2026:** os experimentos históricos abaixo têm limitações que mudam sua interpretação. `event_start` é o início estimado da janela de anomalia que antecede a falha; avaliar somente o trecho anterior a ele não mede corretamente a antecipação do benchmark. A anonimização por arquivo impede inferir a cronologia real entre eventos; timestamps iguais também não comprovam medições duplicadas. Portanto, os testes que juntaram eventos por data, excluíram o evento 72 só pela coincidência de timestamps ou interpretaram estados 3/4 da fazenda A como estados operacionais observados não sustentam as conclusões anteriores de desempenho. Consulte [DECISAO_KMEANS.md](DECISAO_KMEANS.md) e o teste `src/benchmark_decision.py` para a revisão. Os registros abaixo foram preservados como histórico, não como validação final.

Este modelo explora grupos naturais nos dados SCADA do [CARE to Compare v6](https://zenodo.org/records/15846963). Ele ainda não classifica falhas. O projeto também terá ensembles, árvore de decisão e SVM; cada modelo terá sua própria pasta.

## O que fizemos

Usamos o evento `0` da fazenda A. Cada linha do CSV representa um intervalo de 10 minutos; as colunas `_avg` contêm as médias medidas pelos sensores nesse intervalo. Dos 54.986 registros, 47.252 são linhas de treino com `status_type_id = 0` (operação normal) e 2.838 são linhas de previsão. O rótulo de falha do evento não foi usado para treinar o K-means.

Selecionamos as médias dos sensores, retiramos sensores angulares e contadores conforme `feature_description.csv`, removemos colunas com muitos valores ausentes ou constantes, preenchemos os demais valores ausentes com a mediana e padronizamos as escalas. O ajuste foi feito com uma amostra de 20 mil linhas de treino normal. Depois atribuímos grupos às outras linhas.

O **`k`** é o número de grupos solicitado ao K-means. Testamos `k=2` a `k=6`; `k=2` teve a melhor **silhouette**, `0,366`. Essa medida vai de `−1` a `+1`: valores próximos de `+1` indicam grupos bem separados, próximos de `0` indicam sobreposição e negativos indicam que pontos podem estar mais próximos de outro grupo. `0,366` sugere separação razoável, ainda não forte.

| Grupo | Registros de treino normal | Vento médio (`wind_speed_3_avg`) | Potência média (`power_29_avg`) |
| --- | ---: | ---: | ---: |
| 0 | 17.070 | 10,16 | 0,6632 |
| 1 | 30.182 | 4,13 | 0,0893 |

Até aqui, os grupos parecem representar regimes de **maior e menor vento/geração**. Os números 0 e 1 são identificadores arbitrários, não significam "normal" e "falha". O gráfico PCA mostra uma projeção em duas dimensões; o agrupamento foi calculado com todos os sensores selecionados. Este resultado usa apenas um evento de uma turbina e ainda precisa ser comparado com outros eventos e fazendas.

## Comandos, em sequência

No Git Bash, a partir da **raiz do projeto**. O download e a extração já foram feitos; os comandos ficam aqui para reproduzir o trabalho:

```bash
mkdir -p data
curl.exe --ssl-no-revoke -L -C - --retry 8 --retry-delay 10 \
  -o data/CARE_To_Compare.zip \
  'https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1'
md5sum data/CARE_To_Compare.zip
unzip data/CARE_To_Compare.zip -d data/CARE_To_Compare
```

O MD5 esperado é `2547b58c21ac8c242d13232860cf500c`. Os dados extraídos devem ficar em `data/CARE_To_Compare/Wind Farm A`, `Wind Farm B` e `Wind Farm C`.

```bash
python --version
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python K-means/src/kmeans_baseline.py --list-events
python K-means/src/kmeans_baseline.py --farm A --event-id 0
```

Para repetir com outro evento, troque `--farm` e `--event-id`, por exemplo:

```bash
python K-means/src/kmeans_baseline.py --farm B --event-id 53
```

O programa usa `data/` e `K-means/outputs/kmeans/` por padrão, independentemente da pasta atual. É possível alterar esses caminhos com `--data-dir` e `--output-dir`. Os resultados do exemplo estão em `K-means/outputs/kmeans/farm_A_event_0/`: `report.json`, `clusters_pca.png`, `cluster_profile.csv`, `train_clusters.csv` e `prediction_clusters.csv`.

## Segundo evento e limite da comparação

O evento `24` da fazenda A é rotulado `normal` e pertence à mesma turbina (`asset_id = 0`) do evento `0`, rotulado `anomaly`. Ele também escolheu `k=2`, com silhouette `0,370`. As médias de vento/potência foram `10,43`/`0,6913` no grupo de maior geração e `4,25`/`0,0980` no grupo de menor geração. Esses perfis são próximos dos obtidos no evento `0`.

Entretanto, **37.681 timestamps de treino aparecem nos dois CSVs**, aproximadamente 72% dos timestamps de treino de cada evento. Por isso, a semelhança não constitui uma validação independente. Os números dos grupos são arbitrários e cada evento ajusta seu próprio modelo; compare as características dos grupos, não os IDs 0 e 1. O rótulo `normal`/`anomaly` descreve a seção de previsão do evento e não os grupos de treino.

O evento `3` é normal e pertence a outra turbina (`asset_id = 10`). Ele pode ser usado mais tarde para verificar se os regimes de vento e geração se repetem em outro equipamento, com um modelo próprio.

## Primeira investigação da seção de previsão

Cada registro em `prediction_clusters.csv` traz a distância ao centroide mais próximo. Usando o percentil 95 das distâncias das linhas de treino normal **do próprio evento** como referência exploratória:

| Evento | Rótulo do evento | Limiar de treino | Registros de previsão acima do limiar |
| --- | --- | ---: | ---: |
| 0 | `anomaly` | 7,198 | 26,3% (de 2.838) |
| 24 | `normal` | 7,149 | 0,6% (de 2.714) |

No evento 0, 148 dos 288 registros de previsão **anteriores ao intervalo da falha** ficaram acima do limiar (51,4%). Durante o intervalo marcado como falha, foram 21,3%; depois, 31,4%. Os 288 registros anteriores têm `status_type_id = 0`. Na seção de previsão completa do evento 0, porém, a maioria dos registros tem estado `4` (parada) ou `3` (serviço), o que pode aumentar as distâncias por motivos operacionais óbvios.

Esse contraste é uma **pista para investigar**, não evidência de detecção antecipada. Os modelos e limiares foram ajustados separadamente por evento, os períodos de treino dos eventos 0 e 24 se sobrepõem, e uma distância alta também pode resultar de condições de vento, sazonalidade ou qualidade dos sensores. Antes de reportar desempenho, precisamos definir um protocolo sem sobreposição e avaliar vários eventos normais e de falha.

## Um único modelo para os eventos 0 e 24

O script `src/shared_reference.py` aplica o **mesmo K-means e o mesmo limiar** aos dois eventos da turbina 0. Ele lê os arquivos originais em `data/` sem editá-los. A previsão do evento 24 aparece inteira no trecho de treino do CSV do evento 0; por isso, o script corta a referência **antes de 24/04/2023 15:00**, início da primeira previsão. Só usa linhas anteriores com `status_type_id = 0`. A divisão é cronológica, com um dia de intervalo: 20.000 linhas amostradas até 05/03/2023 para ajuste e 7.047 linhas de 06/03 a 24/04 para calibrar o limiar.

Na raiz do projeto, use explicitamente o Python do ambiente virtual:

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe K-means/src/shared_reference.py
```

O modelo escolheu `k=2` usando apenas o conjunto de ajuste. O percentil 95 das distâncias de calibração definiu o limiar **6,982**. Esse mesmo limiar foi aplicado às seções de previsão:

| Evento | Rótulo | Registros acima do limiar | Antes do intervalo do evento |
| --- | --- | ---: | ---: |
| 0 | `anomaly` | 1.207 / 2.838 (42,5%) | 199 / 288 (69,1%) |
| 24 | `normal` | 55 / 2.714 (2,0%) | 4 / 431 (0,9%) |

Os arquivos ficam em `K-means/outputs/shared_reference/farm_A_asset_0_events_0_24/`: `report.json`, `model.joblib`, `scores_timeline.png` e um CSV de escores para cada evento. A linha vermelha do gráfico é o limiar; a faixa laranja marca o intervalo informado pelo dataset. Um alerta significa **distância acima do limiar**, não diagnóstico de falha.

Este resultado é mais comparável que o anterior porque usa uma referência comum e não põe previsões no ajuste ou na calibração. Ainda é **exploratório**: só há um evento normal e um de falha nesse teste, e o evento de falha ocorre meses após a calibração. Sazonalidade, mudanças de regime e estados de parada/serviço podem explicar parte dos alertas.

## Avaliação ampliada: 21 eventos da fazenda A

Para verificar se o sinal se repete, rodamos um modelo e um limiar **por turbina**, sempre com o corte de treino anterior à primeira previsão daquela turbina. Usamos 21 dos 22 eventos da fazenda A: 10 normais e 11 com anomalia. O evento `72` ficou fora desta comparação porque 1.918 dos seus 2.017 timestamps de previsão também estão no evento `51`; contá-los como casos independentes duplicaria quase toda a evidência.

Comandos executados na raiz do projeto:

```bash
./.venv/Scripts/python.exe K-means/src/shared_reference.py --farm A --reference-event 71 --events 71 24 73 0 26
./.venv/Scripts/python.exe K-means/src/shared_reference.py --farm A --reference-event 40 --events 40 3 42 10 17
./.venv/Scripts/python.exe K-means/src/shared_reference.py --farm A --reference-event 92 --events 92 25 68 69
./.venv/Scripts/python.exe K-means/src/shared_reference.py --farm A --reference-event 14 --events 14 45 38 84
./.venv/Scripts/python.exe K-means/src/shared_reference.py --farm A --reference-event 13 --events 13 22 51
./.venv/Scripts/python.exe K-means/src/summarize_farm_a.py
```

O resumo reproduzível está em `K-means/outputs/farm_A_summary/`. A taxa de alertas na previsão inteira teve **AUC 0,745** ao ordenar eventos com falha acima dos normais. Porém, esse período inclui parada e serviço, que podem ser detectados sem antecipar uma falha. Para a taxa de alertas **antes** do intervalo do evento, a AUC foi **0,670** entre os 15 eventos que têm registros anteriores (8 normais, 7 com anomalia). AUC 0,5 equivale a ordenação sem discriminação; 1,0 seria ordenação perfeita. Esses números são exploratórios, com amostra pequena e eventos correlacionados.

O evento `0` teve 71,5% de alertas antes da falha, mas o `84` teve só 0,7%, enquanto o evento normal `69` teve 12,5%. Portanto, o contraste do evento 0 **não se generaliza de forma confiável**. Isso motivou o experimento por regime descrito abaixo.

## Desvio de sensores dentro de cada regime

O script `src/regime_residual.py` usa K-means apenas para separar regimes de **vento e potência**. Na fazenda A, entram quatro medidas `_avg` de velocidade do vento e potência. Dentro de cada regime, o script calcula uma referência para os outros sensores (38 no experimento da turbina 0) e mede o quanto cada registro se afasta dela. Os limites são calibrados separadamente por regime, com as mesmas divisões temporais do experimento anterior. Dados em `data/` são apenas lidos.

Comandos para reproduzir a avaliação de 21 eventos:

```bash
./.venv/Scripts/python.exe K-means/src/regime_residual.py --farm A --reference-event 71 --events 71 24 73 0 26
./.venv/Scripts/python.exe K-means/src/regime_residual.py --farm A --reference-event 40 --events 40 3 42 10 17
./.venv/Scripts/python.exe K-means/src/regime_residual.py --farm A --reference-event 92 --events 92 25 68 69
./.venv/Scripts/python.exe K-means/src/regime_residual.py --farm A --reference-event 14 --events 14 45 38 84
./.venv/Scripts/python.exe K-means/src/regime_residual.py --farm A --reference-event 13 --events 13 22 51
./.venv/Scripts/python.exe K-means/src/summarize_farm_a.py --experiment regime_residual
```

| Medida por evento | Distância ao centro original | Desvio dentro do regime |
| --- | ---: | ---: |
| AUC antes do intervalo (15 eventos) | 0,670 | 0,696 |
| AUC na previsão inteira (21 eventos) | 0,745 | 0,818 |

O ganho **antes** do intervalo é pequeno. Por exemplo, a taxa de alertas antes do evento `73` com falha caiu de 36,8% para 3,5%, e o evento `84` continuou sem alertas antes da falha. A melhora da AUC da previsão inteira pode refletir estados de parada ou serviço já ocorridos. Assim, este método também **não está validado como detector antecipado**. O resultado útil do K-means até agora é caracterizar os regimes de operação; a detecção de falhas deve ser avaliada separadamente, com rótulos e métricas definidos para os próximos modelos.

Resultados detalhados: `K-means/outputs/regime_residual/` por turbina e `K-means/outputs/farm_A_summary_regime_residual/` para a tabela de eventos e métricas.
