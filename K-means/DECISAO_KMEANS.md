# Decisão sobre K-means — 14/11/2026

**Decisão para o projeto: não aprovar o K-means como previsor confiável de falhas com a evidência atual.** Ele pode ser apresentado como modelo não supervisionado de regimes e como baseline de alertas de anomalia, com os resultados negativos documentados. Não afirmar que K-means é matematicamente incapaz de detectar qualquer falha: os testes não cobrem todas as configurações possíveis.

Atualização após busca solicitada pelo usuário: foram executadas 108 configurações, incluindo K=3, com seleção em 65 eventos e avaliação em outros 30 de turbinas diferentes. A configuração escolhida (K64/euclidiano/P99/6h) detectou 7/15 anomalias e acertou 7/15 normais. A decisão permanece restrita às abordagens avaliadas. Detalhes em [BUSCA_CONFIGURACOES.md](BUSCA_CONFIGURACOES.md).

## Correções que mudam a interpretação

A [FAQ dos autores](https://aefdi.github.io/EnergyFaultDetector/care2compare_faq.html) esclarece que a janela `event_start`–`event_end` é uma estimativa de anomalia que antecede a falha. Portanto, detectar dentro dessa janela pode ser o objetivo de antecipação; o trecho anterior a `event_start` não é uma medida universal de antecipação. As AUCs anteriores de 0,670/0,696 nesse trecho não são uma avaliação oficial de early fault detection.

Os timestamps foram anonimizados por arquivo e não preservam a ordem cronológica real entre eventos. Coincidência de timestamps não comprova medições duplicadas. Assim, os modelos de referência construídos com cortes de data entre vários arquivos e a exclusão do evento 72 somente por sobreposição de timestamps não constituem um protocolo válido de validação temporal real. Na fazenda A, os status também derivam de informação de falhas e não devem ser interpretados como observações operacionais independentes na previsão.

Os scripts anteriores e seus resultados ficam como histórico exploratório. A revisão atual inclui todos os 95 eventos, inclusive o 72, e usa o split de treino/previsão fornecido dentro de cada arquivo.

## Solução testada e critério de decisão

Implementação: `src/benchmark_decision.py`. Cada evento recebe um modelo de comportamento normal treinado somente na sua seção de treino com status 0. Ordenamos esse trecho internamente, reservamos os últimos 20% para calibração e deixamos um dia de intervalo. Ajuste com até 12 mil registros; medianas e padronização calculadas no ajuste; médias dos sensores, sem ângulos/contadores. O MiniBatchKMeans usa 32 protótipos, semente 42 e três inicializações. Essa é uma implementação de K-means para representar vários modos normais, sem selecionar k pelos rótulos de falha.

O escore é a distância ao protótipo mais próximo. O limite é o percentil 95 da calibração. Um alerta exige pelo menos 25% de excedências em uma janela causal de 24 horas, com no mínimo 72 registros elegíveis. Na previsão de B/C, somente status 0 é elegível; na A, ignoramos status para avaliação, conforme a orientação dos autores. Eventos com anomalia são detectados quando há alerta dentro da janela anotada; eventos normais são falsos positivos se houver alerta em qualquer parte de sua previsão.

Os parâmetros foram fixados no código antes da execução, sem otimização pelos rótulos de B/C. O critério proposto para aceitar o método nesta investigação foi sensibilidade >=70% e especificidade >=80%. É um requisito de triagem escolhido para o experimento, não uma exigência do professor ou um padrão universal. É avaliação simplificada por evento, **não o CARE oficial**. Não mede a data exata de uma futura falha nem produz previsão de vida útil restante.

## Resultados

| Conjunto | Eventos de anomalia detectados | Eventos normais sem alerta | Acurácia balanceada |
| --- | ---: | ---: | ---: |
| A | 7/12 (58,3%) | 8/10 (80,0%) | 69,2% |
| B | 5/6 (83,3%) | 4/9 (44,4%) | 63,9% |
| C | 19/27 (70,4%) | 9/31 (29,0%) | 49,7% |
| B e C | 24/33 (72,7%) | 13/40 (32,5%) | 52,6% |
| Todos | 31/45 (68,9%) | 21/50 (42,0%) | 55,4% |

Nenhuma fazenda satisfez simultaneamente os dois critérios. Em B/C, 67,5% dos eventos normais receberam alerta. Esse custo de falso alarme impede recomendar o método como solução confiável, mesmo quando ele encontra várias anomalias. Os eventos da mesma turbina podem ser correlacionados; as contagens não devem ser interpretadas como 95 ensaios independentes. Os dados já foram explorados durante o projeto, portanto B/C são uma extensão sem ajuste de parâmetros aos rótulos nesta execução, não um conjunto de teste intocado para sempre.

## Como continuar o projeto

Manter K-means como um dos quatro modelos: demonstrar os grupos naturais, implementar o baseline de alerta e reportar suas limitações. Para árvore, ensemble e SVM, definir o alvo e um protocolo comum antes de treinar; eles também precisam provar desempenho, não se presume que funcionarão. Qualquer nova variante de K-means deve ser tratada como uma nova hipótese e validada em dados reservados, não ajustada repetidamente até melhorar nesses mesmos rótulos.

## Reprodução

Na raiz do projeto, Git Bash:

```bash
./.venv/Scripts/python.exe K-means/src/benchmark_decision.py
```

Saídas em `K-means/outputs/benchmark_decision/`: `events.csv`, `metrics.json` e escores/alertas por evento. Os CSVs originais são somente lidos. Fontes: [FAQ dos autores](https://aefdi.github.io/EnergyFaultDetector/care2compare_faq.html), [publicação do dataset](https://www.mdpi.com/2306-5729/9/12/138). Há propostas publicadas com K-means integrado a outros mecanismos, como [K-means com ANFIS](https://www.mdpi.com/1996-1073/10/7/898); isso demonstra possibilidade de uso em sistemas de alerta, não valida automaticamente este projeto.
