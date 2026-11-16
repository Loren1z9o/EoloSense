# SVM — regressão por vetores de suporte (SVR, kernel RBF)

**Papel no projeto:** o mesmo modelo de comportamento normal, com uma fronteira suave em vez de degraus. O SVR ajusta uma função que ignora erros menores que `epsilon` (um "tubo" de tolerância) e penaliza com `C` o que sai dele. O kernel RBF permite relações não lineares, como a curva de potência.

Protocolo, entradas, alvos, escore e regra de alerta são comuns aos modelos supervisionados: [Comum/README.md](../Comum/README.md).

## Como foi treinado

- `SVR(kernel="rbf", gamma="scale")`, um por alvo (`MultiOutputRegressor`). Entradas **padronizadas**: ao contrário das árvores, o SVM mede distâncias, então vento em m/s e potência em kW precisam estar na mesma escala. Alvos também padronizados, então `epsilon` está em desvios-padrão.
- **Grade** escolhida na validação cronológica (sem rótulos de falha): `C` ∈ {1, 10, 100} × `epsilon` ∈ {0,05; 0,2}.
- **Amostra de 3.000 linhas** do ajuste. O custo do SVR cresce aproximadamente com o quadrado do número de linhas; com 12 mil linhas e 10 alvos por evento, os 95 eventos ficariam inviáveis num notebook. A amostra é aleatória (semente 42) e cobre todo o período de ajuste. Essa é uma limitação a declarar no relatório.

## Executar

```bash
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --models svm --jobs 4
```

Saídas em `outputs/principal/`. É o modelo mais lento por evento.
