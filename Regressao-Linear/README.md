# Regressão Linear (Ridge) — referência mínima

Não é um dos quatro modelos do projeto; é a régua. Se a árvore, o ensemble ou o SVM não superarem uma regressão linear com as mesmas entradas, a complexidade extra não se justifica.

- `Ridge` com entradas padronizadas; `alpha` ∈ {0,1; 1; 10; 100} escolhido na validação cronológica.
- Mesmas entradas (inclusive as médias móveis de carga), alvos, escore e regra de alerta dos outros modelos: [Comum/README.md](../Comum/README.md).

```bash
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --models linear
```
