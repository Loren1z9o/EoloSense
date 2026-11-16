# Árvore de Decisão — regressão do comportamento normal

**Papel no projeto:** prever as temperaturas dos componentes a partir das condições de operação com um modelo que dá para *ler*. A árvore divide o espaço em regiões ("vento acima de 9 m/s **e** potência média das últimas 6 h acima de 70%") e prevê a média de cada região.

Protocolo, entradas, alvos, escore e regra de alerta são comuns aos modelos supervisionados: [Comum/README.md](../Comum/README.md).

## Como foi treinada

- `DecisionTreeRegressor` **multi-saída**: uma única árvore prevê todos os alvos de uma vez. Os alvos são padronizados antes (média 0, desvio 1), para que uma temperatura com mais variação não domine o critério de divisão.
- **Grade** escolhida na validação cronológica (sem rótulos de falha): `max_depth` ∈ {6, 10, 14, sem limite} × `min_samples_leaf` ∈ {10, 40, 120}.
- **Poda por custo-complexidade**: na árvore vencedora, calcula-se o caminho de `ccp_alpha` e testam-se 12 valores. Fica o de menor erro na validação e, em empate, a poda maior (árvore mais simples). O alfa escolhido e a curva testada ficam em `report.json` (`pruning`).
- Sem escalonamento das entradas: divisões por limiar não dependem de escala. Ausentes nas entradas são imputados pela mediana do ajuste.

## Por que esperar que ela perca para o ensemble

Uma árvore isolada tem **alta variância** (pequenas mudanças no treino mudam muito a árvore) e prevê em degraus. Para temperatura, que varia de forma suave com a carga, os degraus viram resíduo, e resíduo é exatamente o que o escore mede. A árvore entra como o modelo interpretável e como ponto de comparação direto para os ensembles, que são várias árvores combinadas.

## Executar

```bash
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --models arvore --jobs 4
```

Saídas em `outputs/principal/`: `RESULTADOS.md`, `metrics.csv`, `regression.csv`, `figuras/linha_do_tempo_reservados.png` e um `model.joblib` por evento. A importância de cada entrada fica em `report.json` (`importances`).
