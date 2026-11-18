# Modelos de comportamento normal — Árvore, Ensembles, SVM e baseline linear

Novo estudo somente na fazenda C, com retreinamento e sensores de temperatura, vibra??o, press?o e grandezas el?tricas: [FAZENDA_C_MULTISSENSORES.md](FAZENDA_C_MULTISSENSORES.md). Resultados separados em `outputs/fazenda_C_componentes/`.

Código compartilhado pelos modelos supervisionados do EoloSense. O K-means (pasta `K-means/`) agrupa os sensores sem alvo; estes modelos fazem outra pergunta: **"dadas as condições de operação de agora, qual temperatura cada componente deveria ter se a turbina estivesse saudável?"** Quando a temperatura real se afasta da prevista de forma persistente, sai um aviso.

| Pasta | Modelo | Resultados |
| --- | --- | --- |
| [Arvore-de-Decisao/](../Arvore-de-Decisao/README.md) | Árvore de regressão com poda `ccp_alpha` | `Arvore-de-Decisao/outputs/principal/RESULTADOS.md` |
| [Ensembles/](../Ensembles/README.md) | Random Forest, Extra Trees e Gradient Boosting | `Ensembles/outputs/principal/RESULTADOS.md` |
| [SVM/](../SVM/README.md) | SVR com kernel RBF | `SVM/outputs/principal/RESULTADOS.md` |
| [Regressao-Linear/](../Regressao-Linear/README.md) | Ridge (referência mínima) | `Regressao-Linear/outputs/principal/RESULTADOS.md` |

Comparação dos quatro (e do K-means, se seus resultados existirem): `Comum/outputs/principal/COMPARACAO.md` e `comparacao.png`. Se os modelos forem treinados em comandos separados, `treinar_modelos.py --compare-only` refaz a comparação com tudo o que já existe. Para explorar os resultados evento a evento: `streamlit run app_streamlit.py` (veja o [README principal](../README.md#6-resultados)).

## Como rodar

Pré-requisito: o CARE extraído em `data/CARE_To_Compare/` (mesmos comandos do [README do K-means](../K-means/README.md#comandos-em-sequência)). Na raiz do projeto, com o ambiente virtual:

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Teste rápido do código (sem o dataset real, ~1 min)
./.venv/Scripts/python.exe -m unittest discover -s Comum/tests -v

# Treino completo: 4 modelos x 95 eventos. Use --jobs com o número de núcleos livres.
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --jobs 4
```

Variações úteis:

```bash
# Só a fazenda A (22 eventos), para conferir antes da execução longa
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --farms A --jobs 4 --run fazenda_A

# Só alguns modelos
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --models arvore ensembles --jobs 4
```

A execução é **retomável**: se parar no meio, rode o mesmo comando e os eventos prontos são pulados. Para refazer do zero, apague a pasta `outputs/<run>` do modelo ou use outro `--run`. Tempo esperado: alguns minutos por evento somando os quatro modelos (o SVM e os ensembles são os mais lentos), então a execução completa leva algumas horas em um núcleo e cai proporcionalmente com `--jobs`.

Para incluir o K-means na comparação, os resultados do perfil principal dele precisam existir em `K-means/outputs/detection_balanced/` (comando em [DETECCAO_E_ALERTAS.md](../K-means/DETECCAO_E_ALERTAS.md)).

## O que acontece em cada evento

```
seção de treino, status 0 (operação normal), em ordem cronológica
├── ajuste (80%) ──────────────────────────────┤1 dia├── calibração (20%) ──┤ previsão (onde pode estar a falha)
├── ajuste-treino (80%) ──┤1 dia├── validação ─┤
```

1. **Entradas (condições de operação)**: sensores de vento, potência ativa, rotação e temperatura ambiente, identificados pela descrição em `feature_description.csv` (ou pelos prefixos `wind_speed_`/`power_`). Somam-se médias móveis **causais** de 1 h e 6 h da potência e da rotação (e 1 h do vento), porque temperatura tem inércia: depende da carga das últimas horas, não só do instante.
2. **Alvos**: sensores de temperatura dos componentes (até 10). São mantidos os mais previsíveis a partir das entradas, medidos por R² de uma Ridge na validação. Sensores que as condições de operação não explicam não servem para comparar "real × esperado".
3. **Hiperparâmetros**: cada família testa sua grade no ajuste-treino e escolhe pela menor RMSE padronizada na validação. Depois o modelo é reajustado com todo o ajuste.
4. **Escore**: resíduo padronizado de cada alvo → média móvel causal de 1 h → maior valor absoluto entre os alvos. Usamos o maior e não a média, porque uma falha costuma esquentar **um** componente e a média entre dez sensores diluiria o sinal. O sensor que mais desviou fica salvo (`top_sensor`), o que permite dizer *qual* componente motivou o aviso.
5. **Limiar**: percentis 90 a 99 do escore na calibração (dados normais não vistos).
6. **Alerta**: exatamente a regra e os episódios do K-means (`K-means/src/failure_alerts.py` e `alert_episodes.py`): fração de excedências numa janela causal, um aviso por episódio, rearme após 6 h de recuperação. Na fazenda A, todos os registros da previsão são elegíveis (orientação dos autores do CARE); em B e C, só status 0.

## Seleção e avaliação (sem vazamento)

- **Mesma partição do K-means**: 30% das turbinas de cada fazenda, por SHA-256 com semente 42 → 65 eventos de desenvolvimento e 30 reservados. Se `K-means/outputs/tuning_search/manifest.csv` existir, o script confere que a partição é idêntica e para se não for.
- **Nada no treino usa rótulo de falha**: entradas, alvos, hiperparâmetros e limiares vêm só de dados normais. Um teste automatizado altera a seção de previsão e confere que modelo, alvos e limiares não mudam.
- **Regra de alerta** (24 combinações de percentil × janela × fração, a mesma grade do K-means): escolhida só nos 65 eventos de desenvolvimento, pela maior sensibilidade de avisos dentro do orçamento de **até 5 avisos falsos por 30 dias** e **até 25% de tempo em incidente** em trechos normais. É congelada antes de abrir os 30 reservados.
- **Duas camadas de métrica**: (1) regressão — MAE, RMSE e R² por alvo na calibração; (2) detecção por evento — sensibilidade (aviso novo dentro da janela anotada), especificidade (evento normal sem aviso), precisão, avisos falsos por 30 dias.

## 2º experimento: regra escolhida pela acurácia balanceada

Na execução `principal` (16/11/2026), a regra foi escolhida pela **maior sensibilidade** dentro do orçamento de 5 avisos falsos por 30 dias. A especificidade não entrava na escolha, e o orçamento praticamente a zera: nos reservados, cada evento normal tem cerca de 18,7 dias monitorados, o que dá uns 3 avisos falsos por evento. Se os avisos caíssem ao acaso, só uns 5% dos eventos normais passariam sem nenhum. Saiu 1 de 15 (6,7%), e a acurácia balanceada ficou entre 30% e 53%, perto ou abaixo dos 50% de um sorteio. A regressão estava boa (R² 0,91–0,95); o problema era a regra de alerta.

O 2º experimento muda só a escolha da regra, sem retreinar nada:

- **Critério** (`--criterio acuracia-balanceada`): maior acurácia balanceada por evento, (sensibilidade + especificidade) / 2, nos 65 eventos de desenvolvimento. Desempate por sensibilidade e depois por menos avisos falsos. O orçamento (5 avisos falsos / 30 dias, 25% de tempo ativo) continua valendo como teto.
- **Grade ampla** (`--grade ampla`): 75 regras em vez de 24. Acrescenta o percentil 99, janelas de 48 e 72 h e fração de excedências de 50%, regras mais exigentes que só fazem sentido quando a especificidade conta. A grade padrão está toda contida nela.
- **Sem retreino** (`--reusar principal`): lê `scores.csv` e os limiares salvos em `outputs/principal/` e grava a nova regra, as métricas e os `predictions.csv` em outra pasta. A execução `principal` fica intacta, e os dois experimentos continuam visíveis.

Como antes, a regra é escolhida só no desenvolvimento e congelada antes de abrir os 30 reservados. O registro fica em `frozen_config.json` (`criterion`, `grid`, `scores_from`).

```bash
# K-means com o mesmo critério e a mesma grade (precisa de K-means/outputs/failure_alerts_evaluation)
./.venv/Scripts/python.exe K-means/src/tune_detection.py --max-false-per-30-days 5 --criterion balanced --grid wide \
    --output-dir K-means/outputs/detection_acuracia_balanceada

# Árvore, Ensembles, SVM e Ridge: só a regra de alerta, a partir da execução principal (minutos)
./.venv/Scripts/python.exe Comum/src/treinar_modelos.py --reusar principal --run acuracia_balanceada \
    --criterio acuracia-balanceada --grade ampla --kmeans-run detection_acuracia_balanceada
```

A comparação sai em `Comum/outputs/acuracia_balanceada/COMPARACAO.md`, e o painel mostra a execução `acuracia_balanceada` no seletor "Resultados".

Limites que valem para todos os modelos: os 30 reservados já foram consultados nos experimentos do K-means (avaliação interna reutilizada, não teste externo intocado); eventos da mesma turbina são correlacionados; não se prevê a data da falha nem se calcula o CARE score oficial.

## Arquivos

- `src/nbm.py`: entradas, alvos, divisão temporal, grades de hiperparâmetros, poda, escore.
- `src/treinar_modelos.py`: partição, treino paralelo e retomável, seleção da regra, avaliação, relatórios e figuras.
- `tests/sintetico.py`: gera um mini-CARE com o mesmo formato de arquivos e uma falha injetada (aquecimento gradual do rolamento da caixa multiplicadora).
- `tests/test_nbm.py`: 9 testes (papéis dos sensores, ângulos/contadores fora, divisão sem sobreposição, atributos causais, previsão não influencia o modelo, escore sobe com a falha e aponta o sensor certo, poda registrada, status em B/C).
