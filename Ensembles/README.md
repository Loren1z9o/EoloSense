# Ensembles — Random Forest, Extra Trees e Gradient Boosting

**Papel no projeto:** o mesmo modelo de comportamento normal da árvore, mas combinando muitas árvores para reduzir o erro e, principalmente, o ruído do resíduo que vira escore de anomalia.

Protocolo, entradas, alvos, escore e regra de alerta são comuns aos modelos supervisionados: [Comum/README.md](../Comum/README.md).

## Como foi treinado

Em cada evento, três famílias competem na validação cronológica (sem rótulos de falha) e fica a de menor RMSE padronizada:

| Família | Ideia (guia de Ensemble Learning) | Grade testada |
| --- | --- | --- |
| **Random Forest** | *Bagging* de árvores com amostra aleatória de linhas (70%) e de atributos por divisão: as árvores erram de forma diferente e a média cancela parte do erro | `min_samples_leaf` ∈ {5, 20} × `max_features` ∈ {0,5; 1,0} |
| **Extra Trees** | Como a floresta, mas com limiares de corte sorteados: ainda menos variância, previsão mais suave | `min_samples_leaf` ∈ {5, 20} × `max_features` ∈ {0,5; 1,0} |
| **Gradient Boosting** (`HistGradientBoostingRegressor`) | *Boosting*: cada árvore corrige o erro das anteriores. Mesma família do XGBoost/LightGBM, já incluída no scikit-learn | `max_leaf_nodes` ∈ {15, 31}, taxa 0,08, até 400 árvores com *early stopping* |

- Na busca, as florestas usam 100 árvores; o modelo final da configuração vencedora usa **300** (mais árvores só reduzem variância, não mudam o viés).
- Random Forest e Extra Trees são multi-saída nativos. O boosting treina um modelo por alvo (`MultiOutputRegressor`).
- A família vencedora por evento aparece em `RESULTADOS.md` ("Configuração escolhida na validação interna") e em `regression.csv` (coluna `chosen`). Isso responde, com dados, qual estratégia de ensemble funciona melhor para cada turbina.

## Executar

```bash
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --models ensembles --jobs 4
```

Saídas em `outputs/principal/`. Cada `report.json` traz a grade completa com o erro de validação de cada candidato (`tuning`) e as importâncias das entradas (`importances`), úteis para a seção de interpretação do relatório.
