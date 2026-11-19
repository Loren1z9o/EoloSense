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
