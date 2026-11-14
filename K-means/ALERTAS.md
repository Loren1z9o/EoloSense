# K-means para alertas de possíveis falhas

O fluxo em [failure_alerts.py](src/failure_alerts.py) treina e salva um detector experimental de anomalias. Ele aprende referências dos sensores durante operação normal e sinaliza desvios persistentes. O número do cluster representa um regime de operação; não é um rótulo de falha. O escore não é uma probabilidade de falha.

## Treinar e avaliar um evento

Na raiz do projeto, em PowerShell, com as dependências de [requirements.txt](../requirements.txt) instaladas:

```powershell
& ./.venv/Scripts/python.exe K-means/src/failure_alerts.py train --farm A --event-id 0 --output-dir K-means/outputs/failure_alerts/farm_A_event_0
```

O treinamento usa somente linhas `train` com `status_type_id = 0`, no arquivo selecionado. Os últimos 20% dessas linhas são reservados para calibração; há um intervalo de um dia antes da calibração. Até 12 mil linhas do trecho anterior ajustam a imputação por mediana, a padronização e o MiniBatchKMeans com 32 clusters. São utilizadas médias `_avg`, excluindo ângulos, contadores, sensores constantes e sensores com mais de 30% de ausências no ajuste. A semente é 42.

O percentil 95 das distâncias da calibração define o limiar. Um registro recebe alerta quando pelo menos 25% dos registros elegíveis nas últimas 24 horas excedem o limiar, com no mínimo 72 registros elegíveis. A janela usa apenas passado e presente. As primeiras 71 observações não têm cobertura suficiente. São permitidas lacunas, mas os registros devem seguir a cadência de 10 minutos.

Na previsão, linhas com mais de 30% dos sensores selecionados ausentes ficam inelegíveis, com escore vazio. Nas fazendas B/C, somente status 0 é elegível. Na A o status não filtra a previsão, conforme a revisão metodológica existente. Ausência de alerta em linha inelegível **não significa operação saudável**; consulte sempre `eligible`.

O diretório de saída contém:

- `model.joblib`: sensores, imputador, padronizador, centros e limiar para reutilização.
- `predictions.csv`: timestamps, elegibilidade, cluster, distância (`score`), excedência e alerta persistente (`alarm`).
- `report.json`: parâmetros, contagens e avaliação do evento. Nos eventos de anomalia, verifica alerta dentro da janela anotada; nos normais, verifica falso alerta na previsão inteira. Isso não implementa a métrica CARE oficial.

`event_info.csv` só é consultado depois de ajustar e aplicar o modelo. Os rótulos da seção de previsão não escolhem sensores, clusters ou limiares.

## Aplicar o modelo salvo

Para verificar a reprodução na seção de previsão do mesmo arquivo:

```powershell
& ./.venv/Scripts/python.exe K-means/src/failure_alerts.py predict --model K-means/outputs/failure_alerts/farm_A_event_0/model.joblib --csv "data/CARE_To_Compare/Wind Farm A/datasets/0.csv" --farm A --output-dir K-means/outputs/failure_alerts/replay_event_0
```

Se houver `train_test`, somente a seção de previsão será pontuada. Um CSV novo sem essa coluna terá todas as linhas pontuadas. Ele deve usar `;`, conter `time_stamp`, `asset_id` e todos os sensores do modelo, ser da mesma fazenda/turbina e ter timestamps posteriores ao treino. B/C também exigem `status_type_id`. Carregue apenas modelos joblib de origem confiável.

Use o modelo em dados futuros que compartilhem a mesma definição dos sensores e o mesmo eixo temporal. **Não aplique automaticamente a outro evento CARE**: os arquivos foram anonimizados separadamente, e datas comparáveis não garantem continuidade entre eventos. Cada evento CARE deve treinar sua própria referência. A checagem de identidade/data pelo programa não comprova continuidade física entre arquivos.

Cada execução de `predict` reinicia a janela de persistência. Ao processar blocos sucessivos, inclua as últimas 24 horas como contexto e descarte as saídas repetidas do contexto.

## Interpretação e validação

Este fluxo usa o baseline fixo K32/P95/24h já estudado no projeto. A [busca anterior](BUSCA_CONFIGURACOES.md) escolheu K64/P99/6h na seleção, mas obteve apenas 46,7% de sensibilidade e 46,7% de especificidade em 30 eventos de avaliação. O baseline K32 detectou 8/15 anomalias e não alertou em 5/15 normais. A avaliação ampliada abaixo retreina o novo fluxo, que também rejeita registros com sensores insuficientes, sem realizar nova busca de parâmetros.

Um evento isolado não valida previsão de falhas. Distância elevada pode resultar de mudança de regime, ambiente ou defeito de sensor. Para avaliar uma nova hipótese, fixe o protocolo antes dos resultados, separe turbinas entre desenvolvimento e avaliação e meça sensibilidade, falsos alertas e antecipação na janela anotada. Os eventos já explorados não constituem teste externo intocado. O modelo não estima causa, vida útil restante ou data exata da falha.

Verificações de separação de dados, causalidade, cobertura, identidade e serialização:

```powershell
& ./.venv/Scripts/python.exe -m unittest discover -s K-means/tests -v
```

## Execução local em 14/11/2026

Seis testes automatizados passaram. O treinamento e a reutilização foram executados nos dados locais:

| Evento da fazenda A | Rótulo | Linhas de previsão | Linhas com alerta | Avaliação do evento |
| --- | --- | ---: | ---: | --- |
| 0 | anomaly | 2.838 | 1.059 | Houve alerta dentro da janela anotada |
| 24 | normal | 2.714 | 0 | Sem falso alerta |

Cada evento treinou sua própria referência. São dois exemplos de funcionamento, não uma estimativa de desempenho geral. No evento 0 foram selecionados 42 sensores, com 12.000 linhas de ajuste e 9.451 de calibração. A reutilização do modelo salvo reproduziu as previsões do treinamento. Artefatos locais em [outputs/failure_alerts/](outputs/failure_alerts/), ignorados pelo Git.

## Retreinamento e avaliação ampliada

O avaliador [evaluate_failure_alerts.py](src/evaluate_failure_alerts.py) retreina cada um dos 95 eventos, salva os modelos e as previsões, e calcula métricas por evento. Reutiliza a mesma separação anterior de 65 eventos de desenvolvimento e 30 de avaliação, agrupados por fazenda/turbina. Os parâmetros permanecem fixos. O conjunto de avaliação já foi consultado no projeto, portanto os resultados representam avaliação interna reutilizada.

```powershell
& ./.venv/Scripts/python.exe K-means/src/evaluate_failure_alerts.py
```

A saída padrão deve estar vazia antes da execução. Para repetir, use `--output-dir` com outro diretório. O programa registra erros por evento e só publica as métricas finais se todos os 95 eventos forem processados. Os resultados ficam em [RESULTADOS.md](outputs/failure_alerts_evaluation/RESULTADOS.md), com [decisões por evento](outputs/failure_alerts_evaluation/events.csv), [métricas](outputs/failure_alerts_evaluation/metrics.csv) e [protocolo](outputs/failure_alerts_evaluation/protocol.json).

As métricas incluem precisão, sensibilidade, especificidade, F1, acurácia e acurácia balanceada, separadas por conjunto e fazenda. Precisão significa a proporção de eventos realmente anômalos entre os sinalizados segundo a janela de avaliação. Uma anomalia só é detectada se houver alerta dentro da janela anotada; um evento normal é falso positivo se houver qualquer alerta na previsão inteira. A matriz de confusão principal usa os 30 eventos reservados. Os oito testes automatizados incluem cálculos de métricas e limites dessas janelas.

### Resultados do retreinamento de 14/11/2026

Os 95 eventos foram processados sem erros, gerando 95 modelos e 95 arquivos de previsão. Os rótulos efetivamente encontrados nos arquivos locais somam 45 anomalias e 50 normais. Os oito testes passaram.

| Métrica | 30 eventos reservados | Todos os 95 eventos (descritivo) |
| --- | ---: | ---: |
| Precisão | **44,4%** | 51,7% |
| Sensibilidade | 53,3% | 68,9% |
| Especificidade | 33,3% | 42,0% |
| F1 | 48,5% | 59,0% |
| Acurácia | 43,3% | 54,7% |
| Acurácia balanceada | 43,3% | 55,4% |

Nos reservados: **8 verdadeiros positivos, 7 falsos negativos, 5 verdadeiros negativos e 10 falsos positivos**. A comparação por fazenda/evento confirmou que nenhuma das 30 decisões mudou em relação ao baseline histórico K32/P95/24h. A organização do fluxo e a proteção contra registros com sensores insuficientes não produziram melhora de desempenho nessa avaliação.

Os 46,7% informados anteriormente pertencem à configuração histórica K64/P99/6h; este fluxo executa K32/P95/24h. A comparação direta com a mesma configuração K32 demonstra reprodução do resultado anterior. A maior dificuldade nos reservados é a fazenda C: 8 dos seus 9 eventos normais geraram falso alerta. O detector continua experimental e não demonstrou previsão confiável de falhas.
