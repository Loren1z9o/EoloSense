# Revisão consolidada dos resultados — 14/11/2026

## Melhor desempenho comparável

Entre as duas configurações medidas nos mesmos 30 eventos reservados por turbina, o maior resultado de **acurácia balanceada** continua sendo o K64/euclidiano/P99/6h, com **46,7%**. O novo estudo `failure_alerts` reproduz o baseline K32/P95/24h: **43,3%**. A diferença é um único acerto em 30 eventos, insuficiente para afirmar melhora confiável.

| Métrica por evento nos 30 reservados | K64 escolhido na busca | K32 do novo estudo |
| --- | ---: | ---: |
| Verdadeiros positivos | 7 | 8 |
| Falsos negativos | 8 | 7 |
| Verdadeiros negativos | 7 | 5 |
| Falsos positivos | 8 | 10 |
| Precisão dos alertas | 46,7% | 44,4% |
| Sensibilidade | 46,7% | 53,3% |
| Especificidade | 46,7% | 33,3% |
| F1 | 46,7% | 48,5% |
| Acurácia / acurácia balanceada | 46,7% | 43,3% |

Portanto, K64 é melhor pelo equilíbrio entre detecção e falsos alertas; K32 detecta uma anomalia adicional e tem F1 maior, ao custo de dois falsos alertas adicionais. Não existe uma única configuração vencedora para todas as métricas. Nenhuma satisfaz o critério experimental proposto de sensibilidade >=70% e especificidade >=80%.

## O que o estudo novo acrescenta

Os arquivos `src/failure_alerts.py`, `src/evaluate_failure_alerts.py`, `ALERTAS.md` e os dois arquivos de testes acrescentam um detector reutilizável, validação de cadência/identidade, tratamento explícito de registros inelegíveis por ausências, modelo serializado e avaliação reproduzível.

Revisei o fluxo de ajuste, calibração, previsão, janelas e métricas. Executei novamente os **oito testes**, todos aprovados. Conferi os 95 modelos e 95 arquivos de previsão; os hashes dos quatro scripts registrados no protocolo coincidem com os arquivos atuais. Comparei as decisões dos 30 eventos com o K32 histórico: **zero decisões diferentes**. Logo, o estudo novo melhora a implementação e a auditabilidade, mas não demonstra ganho de desempenho nesses eventos. Os testes validam propriedades de software, não capacidade preditiva por si só.

## Números altos que não devem ser confundidos com o resultado principal

- AUC **0,818** do estudo histórico de desvios por regime: ordenação de 21 eventos da fazenda A, com referência entre arquivos cuja cronologia anonimizada foi posteriormente questionada. Não significa 81,8% de acurácia e não é comparável aos 30 eventos reservados.
- Acurácia balanceada **66,7%** da configuração K64 na seleção: 65 eventos usados para escolher entre 108 configurações; caiu para 46,7% na avaliação.
- Acurácia balanceada **69,2%** do K32 nos 22 eventos da fazenda A: resultado descritivo, inclui eventos de seleção e reflete uma fazenda específica. Nos 95 eventos, o K32 fica em 55,4%.
- Precisão **100%** do K32 nos nove reservados da fazenda A: dois alertas corretos, nenhum falso positivo, mas quatro das seis anomalias foram perdidas. Não representa 100% de detecção nem o desempenho global.

## Situação do projeto

O projeto tem experimentos e um fluxo operacional experimental para K-means. Árvore de decisão, SVM e ensembles ainda não têm implementação nos arquivos revisados. Dataset original mantido em leitura. A avaliação atual é interna reutilizada: as três fazendas e os eventos reservados já foram examinados; há poucos grupos e correlação entre eventos. Não há validação externa nem comprovação da data de uma futura falha. As regras de avaliação por evento são simplificadas e não implementam CARE oficial.

Fontes locais: `outputs/tuning_search/evaluation_metrics.csv`, `outputs/tuning_search/selection_ranking.csv`, `outputs/failure_alerts_evaluation/metrics.csv`, `outputs/failure_alerts_evaluation/events.csv`, `outputs/failure_alerts_evaluation/protocol.json` e `outputs/failure_alerts_evaluation/RESULTADOS.md`.
