# Detectar falhas sem repetir milhares de avisos

**Este é o fluxo principal do K-means no projeto.** O comando [failure_alerts.py](src/failure_alerts.py) usa por padrão K=32, P95, janela de 6h, fração de excedências de 25% e recuperação de 6h. O agrupamento e o estado persistente são implementados com [alert_episodes.py](src/alert_episodes.py).

Os 95 modelos treinados estão em [outputs/detection_balanced/](outputs/detection_balanced/), um por evento CARE. Por exemplo, [farm_A_event_0/model.joblib](outputs/detection_balanced/farm_A_event_0/model.joblib) contém o modelo da fazenda A/evento 0, incluindo sensores, imputador, padronizador, centros, limiar e política de alerta. Eles compartilham a configuração, mas cada evento tem sua própria referência; não há um único modelo global para todas as turbinas.

O fluxo separa a decisão do detector da abertura de episódios. A seleção em [tune_detection.py](src/tune_detection.py) prioriza detectar anomalias e acompanha o volume de avisos falsos e o tempo que os incidentes permanecem abertos. Os centros K-means e o pré-processamento são reutilizados; os limiares são calculados somente com a calibração normal de cada arquivo.

## Como os avisos funcionam

- `alarm`: sinal por intervalo de 10 minutos, calculado por distância e persistência.
- `notification`: aviso emitido uma única vez ao abrir um episódio. Os avisos ficam em `notifications.csv` para acompanhamento ou integração.
- `incident_active`: episódio ainda aberto, atualizado em cada leitura.
- `recovered`: recuperação observada após seis horas consecutivas de dados elegíveis, com cobertura da janela, sem alarme.
- `episode_id`: identificador do episódio dentro do fluxo/modelo.

Um episódio com 3.000 registros de alarme produz um aviso de abertura. Depois da recuperação, um novo sinal pode abrir outro episódio. Não há bloqueio por um prazo fixo após a recuperação. Dados ausentes, leituras inelegíveis e lacunas não confirmam recuperação; interrompem a contagem e mantêm o episódio pendente. Isso representa a recuperação do sinal do modelo, não a comprovação física de reparo.

## Usar o modelo com continuidade entre blocos

Na raiz do projeto, para reproduzir a seção de previsão do evento 0 com o modelo experimental selecionado:

```powershell
& ./.venv/Scripts/python.exe K-means/src/failure_alerts.py predict --model K-means/outputs/detection_balanced/farm_A_event_0/model.joblib --csv "data/CARE_To_Compare/Wind Farm A/datasets/0.csv" --farm A --output-dir K-means/outputs/detection_replay/event_0 --state-file K-means/outputs/detection_replay/state_A_0.json
```

Os próximos blocos do mesmo fluxo devem usar o mesmo modelo e `--state-file`, um CSV com **somente leituras novas**, e outro diretório de saída. O arquivo de estado guarda o contexto da janela móvel e o episódio aberto; assim uma nova execução não perde a persistência nem reenvia o aviso do episódio anterior. A comparação do hash do modelo impede usar esse estado com outro modelo. Registros repetidos ou anteriores ao último bloco são rejeitados.

Sem `--state-file`, cada execução começa independente. Para repetir o exemplo desde o início, use outro arquivo de estado. Não execute simultaneamente dois processos escrevendo o mesmo estado. Não transfira o estado entre arquivos CARE anonimizados diferentes, mesmo quando pertencem à mesma turbina.

Cada diretório de saída contém `predictions.csv`, `notifications.csv` e `report.json`. O relatório distingue a quantidade de registros em alarme da quantidade de notificações.

## Protocolo de seleção

São 24 combinações: percentis 90/92,5/95/97,5, janelas de 6/12/24 horas e proporções de excedências de 10%/25%. O K permanece 32, e o rearmamento exige seis horas de recuperação. A janela é causal e exige pelo menos metade dos registros esperados na cadência de 10 minutos.

A regra é registrada antes de calcular resultados:

1. Exigir, nos trechos normais de desenvolvimento, até **dois avisos falsos por 30 dias monitorados** e incidente ativo em até **25% do tempo monitorado normal**. São hipóteses experimentais ajustáveis, não limites operacionais confirmados pelo usuário.
2. Entre as regras que respeitam os dois limites, maximizar a sensibilidade de **novas notificações dentro da janela anotada**. Desempatar por F2, menor frequência de avisos falsos, menor tempo ativo normal e identificador. F2 dá mais peso à sensibilidade do que à precisão.
3. Se nenhuma respeitar os limites, salvar a candidata de menor violação, explicitamente **não aprovada no desenvolvimento**. Os limites não são relaxados silenciosamente.
4. Congelar a configuração antes de avaliar somente ela e o baseline nos 30 eventos reservados.

Uma ocorrência antiga ainda aberta dentro de uma nova janela não conta como nova detecção. O relatório também mostra a sensibilidade do sinal bruto, comparável à medição anterior. Os 30 eventos já foram usados no projeto: a comparação é interna e reutilizada, não teste externo intocado.

```powershell
& ./.venv/Scripts/python.exe K-means/src/tune_detection.py
```

Para repetir, use um diretório novo com `--output-dir`. Os limites podem ser explicitamente alterados por `--max-false-per-30-days` e `--max-active-fraction`; isso define outro experimento, a ser registrado antes da avaliação.

Resultados completos em [RESULTADOS.md](outputs/detection_first/RESULTADOS.md), [ranking de desenvolvimento](outputs/detection_first/selection_ranking.csv), [configuração congelada](outputs/detection_first/frozen_config.json) e [métricas](outputs/detection_first/metrics.csv).

## Resultado e perfil com prioridade em detecção

A hipótese inicial de dois avisos falsos por 30 dias manteve o baseline de 24 horas. Antes de avaliar uma nova candidata nos reservados, foi definido um segundo orçamento de **cinco avisos por 30 dias normais monitorados**, mantendo o teto de 25% de tempo ativo. Esse orçamento foi motivado pelo resultado de desenvolvimento: a janela de seis horas detectava mais anomalias com cerca de um aviso falso por semana.

```powershell
& ./.venv/Scripts/python.exe K-means/src/tune_detection.py --max-false-per-30-days 5 --output-dir K-means/outputs/detection_balanced
```

Nos 65 eventos de desenvolvimento, a configuração congelada **P95 / 6h / 25%, com recuperação de 6h**, notificou 27/30 anomalias (90%), com 4,21 avisos falsos por 30 dias normais e 21,8% de tempo ativo normal. Atendeu aos limites de seleção. Esse perfil passou a ser o padrão do comando `train`. O avaliador histórico [evaluate_failure_alerts.py](src/evaluate_failure_alerts.py) continua solicitando explicitamente a janela de 24h para reproduzir seu baseline.

Comparação nos mesmos 30 eventos reservados, sem outra escolha de configuração depois de consultá-los:

| Medida | Baseline 24h | Perfil 6h |
| --- | ---: | ---: |
| Anomalias com sinal de alarme na janela | 8/15 (53,3%) | **11/15 (73,3%)** |
| Anomalias com novo aviso na janela | 6/15 (40,0%) | **9/15 (60,0%)** |
| Eventos normais com algum aviso | 10/15 | 14/15 |
| Episódios falsos nos trechos normais | 23 | 47 |
| Dias monitorados nos trechos normais | 274,51 | 280,13 |
| Episódios falsos por 30 dias normais monitorados | 2,51 | **5,03** |
| Tempo ativo nos trechos normais monitorados | 18,3% | 19,8% |
| Precisão por evento, usando novos avisos | 37,5% | 39,1% |

O perfil de seis horas melhorou a detecção nesta comparação, mas aumentou os falsos avisos. A taxa de 5,03 ficou ligeiramente acima do teto de 5 definido no desenvolvimento; não é uma garantia operacional. Ainda ficaram seis anomalias sem novo aviso dentro da janela, e quatro sem sinal bruto nessa janela. Um aviso anterior à janela pode já estar aberto; por isso as duas medidas são distintas. A melhora não torna o modelo um previsor confiável, e requer confirmação com dados novos.

Os 95 modelos e suas previsões foram salvos. O [relatório do perfil de detecção](outputs/detection_balanced/RESULTADOS.md), as [métricas](outputs/detection_balanced/metrics.csv) e a [configuração congelada](outputs/detection_balanced/frozen_config.json) registram os resultados. Ambos os experimentos (orçamentos de dois e cinco) permanecem disponíveis, sem ocultar o primeiro resultado.

As taxas mensais são normalizações dos registros monitorados nos arquivos avaliados. Não garantem uma frequência mensal numa turbina real. Não se une a cronologia entre arquivos anonimizados. O mecanismo agrupa avisos, mas não garante detectar todas as falhas nem prevê a data exata de uma falha.

## 2º experimento: escolher pela acurácia balanceada

Com o critério acima, a especificidade não entra na escolha: 14 dos 15 eventos normais reservados receberam aviso. O 2º experimento reaproveita os mesmos modelos e muda só a escolha da regra. Ela passa a ser a de maior acurácia balanceada por evento ((sensibilidade + especificidade) / 2) nos 65 eventos de desenvolvimento, com o mesmo teto de avisos falsos, numa grade de 75 regras que inclui P99, janelas de 48/72 h e fração de 50%. A regra continua sendo congelada antes dos reservados.

```powershell
& ./.venv/Scripts/python.exe K-means/src/tune_detection.py --max-false-per-30-days 5 --criterion balanced --grid wide --output-dir K-means/outputs/detection_acuracia_balanceada
```

Sem `--criterion` e `--grid`, o comando reproduz exatamente os experimentos anteriores.

## Verificação

```powershell
& ./.venv/Scripts/python.exe -m unittest discover -s K-means/tests -v
```

Os testes verificam separação de treino/previsão, causalidade, persistência, escolha de parâmetros, recuperação, lacunas, um aviso para milhares de linhas de alarme e reprodução das saídas entre blocos com estado persistente.

Os 19 testes passaram, incluindo o comando real com arquivo de estado, a rejeição de blocos repetidos e a proteção do estado em caso de erro ou troca de modelo. O comando de previsão também foi executado no evento 0 com o modelo de seis horas para conferir a reprodução dos avisos reais.
