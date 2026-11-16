# EoloSense — projeto final de aprendizado de máquina

Projeto com quatro famílias de modelos: K-means, ensembles, árvore de decisão e SVM. Objetivo: detectar, a partir dos sensores SCADA, o comportamento anormal que antecede falhas de turbinas eólicas.

O dataset [CARE to Compare v6](https://zenodo.org/records/15846963) está em `data/CARE_To_Compare/`. Os dados brutos não entram no Git. Cada modelo tem código, anotações e resultados em sua própria pasta.

| Modelo | Abordagem | Documentação |
| --- | --- | --- |
| K-means | Não supervisionado: distância aos protótipos de operação normal | [K-means/DETECCAO_E_ALERTAS.md](K-means/DETECCAO_E_ALERTAS.md) · [histórico](K-means/README.md) |
| Árvore de Decisão | Comportamento normal: prevê as temperaturas e mede o resíduo | [Arvore-de-Decisao/README.md](Arvore-de-Decisao/README.md) |
| Ensembles | Idem, com Random Forest, Extra Trees e Gradient Boosting | [Ensembles/README.md](Ensembles/README.md) |
| SVM | Idem, com SVR de kernel RBF | [SVM/README.md](SVM/README.md) |
| Regressão Linear | Referência mínima para os supervisionados | [Regressao-Linear/README.md](Regressao-Linear/README.md) |

Todos usam a mesma partição (65 eventos de desenvolvimento, 30 reservados por turbina), a mesma regra de alerta com episódios e as mesmas métricas, então os resultados são comparáveis. Protocolo dos supervisionados e comando de treino: [Comum/README.md](Comum/README.md).

```bash
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --jobs 4
```

As dependências Python compartilhadas ficam em `requirements.txt` na raiz do projeto.
