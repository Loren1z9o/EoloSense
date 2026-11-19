# Material para o Claude preparar os slides

**Tempo de apresentação: 10 minutos.**

Este arquivo reúne o relatório completo solicitado e o diagnóstico de AUC. O estudo `acuracia_balanceada` é anterior ao retreinamento multissensores da fazenda C. Não misturar os números: o primeiro avalia 30 eventos de A/B/C; o segundo, 16 eventos da C. Em ambos, os reservados já foram examinados em estudos anteriores.

## Orientação para a narrativa

O projeto testou detecção de desvios do comportamento normal. Os escores não são probabilidade de falha. Os modelos supervisionados são regressores, incluindo SVR: aprendem sensores em condições normais, e resíduos geram escores. K-means usa distância aos centros. A regra temporal converte esses escores em notificações. Não apresentar isso como classificação supervisionada direta de falha/não falha.

Nenhum conjunto principal atingiu simultaneamente sensibilidade de 70% e especificidade de 80%. AUC baixa do máximo por evento não demonstra impossibilidade de melhorar o pipeline. A exceção da Ridge anterior normalizada merece menção: AUC 0,747, contra 0,596 bruta, indicando dependência de escala e alguma ordenação nessa amostra. Isso não garante boa notificação temporal nem autoriza escolher um novo limiar usando os reservados.

Sugestão de distribuição dos 10 minutos: problema e objetivo (1 min); dataset e sensores (1 min); metodologia e separação dos dados (2 min); resultados anteriores (1,5 min); novo estudo da C (1,5 min); diagnóstico de AUC (1 min); limitações e próximos passos (1 min).

O gráfico mais útil para o diagnóstico deve separar as duas execuções e identificar claramente máximo bruto e máximo/limiar normal. Os detalhes completos abaixo permitem selecionar o essencial para os slides.

---

## Relatório anterior solicitado — conteúdo completo

# Comparação dos modelos — eventos reservados

Fazendas: ALL. Cada modelo teve sua regra de alerta escolhida apenas no desenvolvimento.

| Modelo | R² normal (mediana) | Sensibilidade | Especificidade | Precisão | Acurácia balanceada | Avisos falsos / 30 dias | Regra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K-means | — | 40.0% | 46.7% | 42.9% | 43.3% | 1.40 | q0.99_h12_f0.1 |
| Árvore de Decisão | 0.911 | 26.7% | 46.7% | 33.3% | 36.7% | 2.47 | q0.99_h6_f0.5 |
| Ensembles | 0.950 | 20.0% | 66.7% | 37.5% | 43.3% | 0.92 | q0.99_h72_f0.1 |
| SVM (SVR) | 0.939 | 13.3% | 60.0% | 25.0% | 36.7% | 1.20 | q0.99_h24_f0.5 |
| Regressão Linear (baseline) | 0.920 | 40.0% | 73.3% | 60.0% | 56.7% | 0.87 | q0.99_h12_f0.5 |

K-means: resultados de `K-means/outputs/detection_acuracia_balanceada`.

Critério experimental do projeto para aceitar um detector: sensibilidade ≥ 70% e especificidade ≥ 80%.


---

## Novo estudo da fazenda C

# Resultado do retreinamento — fazenda C

Foram processados os 58 eventos da fazenda C, com 42 para desenvolvimento e 16 para avaliação interna: 7 com anomalia e 9 normais. Os cinco modelos foram retreinados, incluindo o K-means. Foram salvos 290 pacotes por evento/família; cada pacote do K-means contém cinco modelos, um por perfil. Dados originais e resultados anteriores foram preservados.

O conjunto principal inclui temperatura, vibração, pressão e grandezas elétricas, condicionado por vento, potência ativa, rotação e temperatura ambiente. Uma regra por família foi escolhida entre 75 candidatas somente nos 42 eventos de desenvolvimento e congelada para avaliação.

| Modelo | Anomalias detectadas | Normais sem aviso | Sensibilidade | Especificidade | Acurácia balanceada |
| --- | ---: | ---: | ---: | ---: | ---: |
| Árvore | 1/7 | 5/9 | 14,3% | 55,6% | 34,9% |
| Ensembles | 2/7 | 4/9 | 28,6% | 44,4% | 36,5% |
| SVM (SVR) | 2/7 | 4/9 | 28,6% | 44,4% | 36,5% |
| Ridge | 0/7 | 7/9 | 0,0% | 77,8% | 38,9% |
| K-means | 1/7 | 6/9 | 14,3% | 66,7% | 40,5% |

Sensibilidade mede quantas anomalias receberam um aviso novo dentro da janela anotada. Especificidade mede quantos eventos normais ficaram sem aviso. Acurácia balanceada é a média das duas taxas. Assim, o K-means teve a maior média neste conjunto, mas deixou passar 6 das 7 anomalias. Nenhum modelo atingiu a meta experimental de sensibilidade de 70% e especificidade de 80%. Estes resultados não sustentam o uso desta configuração para antecipação confiável de falhas.

## Comparação e interpretação

Nos mesmos 16 eventos da C, a acurácia balanceada anterior era 42,1% para árvore, 47,6% para ensembles, 34,9% para SVR, 67,5% para Ridge e 45,2% para K-means. O novo conjunto piorou em quatro famílias; SVR melhorou ligeiramente. A comparação é descritiva: além dos sensores, mudaram a grade, o orçamento de ajuste e a escolha das regras. Não permite atribuir a diferença apenas à inclusão de sensores.

No diagnóstico por grupo, ensembles com escore de temperatura marcou 61,9% de acurácia balanceada (4/7 anomalias e 6/9 normais). É um diagnóstico observado após a avaliação, não um vencedor validado para substituir o conjunto principal. Os supervisionados compartilham o mesmo modelo multissaída; os grupos não foram retreinados como ablações independentes.

Os resultados sugerem que simplesmente combinar mais canais por um escore de maior desvio pode acrescentar ruído e diluir sinais úteis. Os canais elétricos tiveram taxas baixas em todos os diagnósticos; isso justifica investigá-los no desenvolvimento, mas não demonstra que sejam inúteis. Vibração é agregada em SCADA, sem os sinais brutos de alta frequência. As hipóteses ainda precisam de experimentos controlados.

O próximo experimento indicado é comparar formas de combinar os grupos e selecionar sensores por componente usando validação agrupada por turbina dentro do desenvolvimento. A decisão de remover grupos, alterar a regra ou escolher modelos deve ocorrer nessa validação. Os 16 eventos já foram examinados antes; não são um teste externo intocado, e a amostra é pequena. Um resultado positivo futuro exigirá confirmação independente. As janelas anotadas representam anomalias; estes escores não estimam a data exata da falha nem sua probabilidade.

## Arquivos e verificação

- [Protocolo e comandos](FAZENDA_C_MULTISSENSORES.md).
- [Tabelas completas e diagnósticos](outputs/fazenda_C_componentes/COMPARACAO.md).
- [Comparação anterior/novo](outputs/fazenda_C_componentes/antes_depois_C.csv).
- [Gráfico de comparação](outputs/fazenda_C_componentes/comparacao.png).
- [Gráfico por grupo](outputs/fazenda_C_componentes/diagnostico_grupos.png).
- [Auditoria](outputs/fazenda_C_componentes/AUDITORIA.json): 58 eventos, cinco famílias, nenhuma falha de treino, hashes dos códigos de treino conferidos e matrizes de confusão recalculadas.

Os 14 testes de `Comum/tests` passaram com `python -X utf8 -m unittest discover -s Comum/tests -v`. Saídas novas ficam em `outputs/fazenda_C_componentes` de cada família; código específico de K-means fica em `K-means/src/multisensor_c.py`. A grade de K deste experimento foi fixa em 32, não uma nova busca de K.


---

## Diagnóstico adicional solicitado

# Diagnóstico de separação por evento

Calculado somente com previsões já salvas dos eventos reservados. Não houve retreino, alteração de limiares ou escolha de novas regras. Escores maiores significam maior desvio. O tempo de apresentação informado pela equipe é **10 minutos**.

| Estudo | Modelo | Eventos (anomalia/normal) | AUC máximo bruto | AUC máximo/limiar normal |
| --- | --- | ---: | ---: | ---: |
| acuracia_balanceada | kmeans | 30 (15/15) | 0.516 | 0.538 |
| acuracia_balanceada | arvore | 30 (15/15) | 0.493 | 0.453 |
| acuracia_balanceada | ensembles | 30 (15/15) | 0.524 | 0.489 |
| acuracia_balanceada | svm | 30 (15/15) | 0.409 | 0.453 |
| acuracia_balanceada | linear | 30 (15/15) | 0.596 | 0.747 |
| fazenda_C_componentes | kmeans | 16 (7/9) | 0.413 | 0.286 |
| fazenda_C_componentes | arvore | 16 (7/9) | 0.492 | 0.508 |
| fazenda_C_componentes | ensembles | 16 (7/9) | 0.508 | 0.444 |
| fazenda_C_componentes | svm | 16 (7/9) | 0.302 | 0.270 |
| fazenda_C_componentes | linear | 16 (7/9) | 0.397 | 0.476 |

## Como interpretar

Cada evento é uma observação. O resumo principal é o maior escore finito e elegível de toda a seção de previsão, tanto nos eventos normais como nos anômalos. AUC é a fração de pares anômalo/normal em que o anômalo tem maior valor, contando empates pela metade. 0,5 indica ausência de ordenação útil nesse resumo; 1 representa ordenação perfeita nesta amostra; valores menores que 0,5 indicam ordenação invertida. Não são porcentagens de acerto de uma regra de alerta.

Os modelos são ajustados por evento, e suas escalas podem diferir. Por isso também mostramos o máximo dividido pelo limiar da calibração normal já congelado. Essa normalização não usa rótulos da previsão e não modifica os avisos. A coluna bruta atende ao diagnóstico solicitado; a normalizada ajuda a verificar dependência de escala.

O CSV inclui ainda máximos restritos à janela anotada das anomalias, mantendo toda a previsão nos normais. Essa análise secundária tem exposições desiguais e usa a anotação para recortar a janela; não equivale a um detector executável sem conhecer o evento.

Máximos são sensíveis a picos isolados, duração e quantidade de registros elegíveis. Uma AUC baixa enfraquece a hipótese de resolver o problema com um único limiar sobre o máximo, mas não prova que toda combinação temporal, seleção de sensores ou modelo falhará. AUC alta também não garante avisos novos dentro da janela anotada nem baixa taxa de falsos avisos.

São avaliações internas pequenas e já reutilizadas, com eventos correlacionados por turbina. O estudo anterior reúne 30 eventos das três fazendas; o novo reúne 16 da C. Não comparar esses totais como se fossem a mesma amostra, nem escolher grupos ou configurações pelo melhor resultado reservado.

Artefatos: `auc.csv` contém as quatro AUCs; `eventos.csv` contém os valores, exposições, origem e SHA-256 de cada previsão. Reprodução: `python -X utf8 Comum/src/diagnostico_auc_eventos.py`.
