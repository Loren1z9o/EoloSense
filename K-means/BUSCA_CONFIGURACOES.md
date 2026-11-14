# Busca de configurações do K-means

Esta investigação testa a hipótese de reduzir falsos alertas alterando a quantidade de protótipos, o escore e a persistência. Os CSVs originais são apenas lidos. Resultados anteriores não provam impossibilidade de melhorar o K-means.

## Protocolo fixado antes da escolha

- K: **3, 8, 16, 32, 64, 128**.
- Escore: distância euclidiana ao centro mais próximo, ou distância ajustada pela dispersão de cada sensor dentro do grupo atribuído (escala diagonal; não é covariância completa).
- Limiar: percentil **95, 97,5 ou 99** dos escores de calibração.
- Persistência: **6, 12 ou 24 horas**, com pelo menos metade dos registros esperados elegíveis e 25% de excedências para disparar alerta.

Total: **108 combinações**, sendo 18 com K=3. MiniBatchKMeans, n_init=3, semente 42, até 12 mil registros normais de ajuste por arquivo. Imputação e padronização usam somente ajuste. O trecho normal de treino de cada arquivo é separado internamente em ajuste e calibração (últimos 20%), com intervalo de um dia. Não se infere cronologia real entre arquivos anonimizados.

Turbinas inteiras são separadas dentro de cada fazenda: aproximadamente 70% para seleção e 30% para avaliação, por ordenação SHA256 determinística com semente textual 42. A identidade inclui fazenda e turbina. Os eventos da mesma turbina ficam no mesmo conjunto, evitando compartilhar suas medições entre seleção e avaliação. O modelo normal por evento pode usar a sua própria seção de treino fornecida pelo benchmark; os rótulos de sua previsão são usados somente para medir desempenho.

A seleção prioriza configurações com especificidade >=80% e, entre elas, maior sensibilidade; desempates por especificidade, menor K e identificador. Se nenhuma atingir 80% de especificidade, escolhe-se a maior acurácia balanceada. A configuração é congelada antes de calcular os resultados no conjunto de avaliação. Somente ela e o baseline fixo K=32/euclidiana/P95/24h são avaliados nesse conjunto. Não se escolhe outra configuração depois de olhar a avaliação. O critério de aceitação continua sensibilidade >=70% e especificidade >=80%, como requisito experimental proposto.

Na fazenda A, status é ignorado na previsão; em B/C, status 0 determina a elegibilidade. Um evento com anomalia conta como detectado se há alerta dentro da janela anotada; um normal recebe falso positivo se há alerta em qualquer parte da previsão. É avaliação simplificada, não CARE oficial, nem previsão da data exata de falha. Os eventos normais são avaliados por períodos completos que podem ter durações diferentes, uma limitação desse critério.

## Reprodução e artefatos

Na raiz do projeto:

```bash
./.venv/Scripts/python.exe K-means/src/tuning_search.py
```

O programa salva checkpoints por evento e retoma a seleção já calculada. Para mudar o protocolo, use outro diretório de saída ou remova manualmente os checkpoints antigos; eles só podem ser reutilizados com o mesmo protocolo e os mesmos dados.

Saídas em `K-means/outputs/tuning_search/`: `manifest.csv`, `protocol.json`, `selection_events.csv`, `selection_ranking.csv`, `frozen_winner.json`, `evaluation_events.csv` e `evaluation_metrics.csv`.

**Limitações:** as três fazendas já foram exploradas anteriormente, por isso esta é uma avaliação interna agrupada, não uma validação externa intocada. Há poucas turbinas, eventos correlacionados e apenas uma semente de aleatoriedade. Buscar 108 configurações aumenta o risco de sobreajuste na seleção; o resultado separado de avaliação tem prioridade sobre o ranking de seleção.

## Resultados executados

Foram concluídas 7.020 avaliações de evento/configuração: 65 eventos de seleção vezes 108 configurações. Nenhuma combinação satisfez simultaneamente sensibilidade >=70% e especificidade >=80% na seleção. Os 30 eventos reservados contêm 15 normais e 15 com anomalia. A identidade fazenda/turbina nunca aparece nos dois conjuntos.

Melhor sensibilidade por K entre configurações com especificidade >=80%, exclusivamente na seleção:

| K | Escore | Percentil | Horas | Sensibilidade | Especificidade |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | Euclidiano | 99 | 6 | 40,0% | 80,0% |
| 8 | Euclidiano | 99 | 6 | 36,7% | 82,9% |
| 16 | Euclidiano | 99 | 6 | 36,7% | 82,9% |
| 32 | Euclidiano | 99 | 12 | 43,3% | 80,0% |
| 64 | Euclidiano | 99 | 6 | 53,3% | 80,0% |
| 128 | Euclidiano | 99 | 12 | 40,0% | 80,0% |

A vencedora congelada foi K=64/euclidiano/P99/6h. O escore ajustado pela dispersão também foi testado em toda a grade, mas não venceu pela regra fixada. K=3 recebeu os mesmos 18 testes dos outros valores de K; não foi avaliado novamente nos eventos reservados porque não venceu na seleção.

Resultados nos mesmos 30 eventos reservados:

| Configuração | Anomalias detectadas | Normais sem falso alerta | Acurácia balanceada |
| --- | ---: | ---: | ---: |
| Baseline K32/euclidiano/P95/24h | 8/15 (53,3%) | 5/15 (33,3%) | 43,3% |
| Selecionada K64/euclidiano/P99/6h | 7/15 (46,7%) | 7/15 (46,7%) | 46,7% |

A escolhida acertou 14/30 eventos, contra 13/30 do baseline: um único acerto adicional, com menor detecção de anomalias e menos falsos alertas. Esse ganho não é evidência de melhora confiável. A diferença entre seleção e avaliação mostra instabilidade de generalização entre turbinas. O conjunto pequeno exige cautela; não se conclui impossibilidade de toda variante futura de K-means.

**Conclusão desta busca:** alterar K, dispersão, percentil e persistência dentro desta grade não produziu um previsor confiável. A recomendação de manter K-means como baseline experimental e modelo de regimes continua. Uma nova hipótese deve ser avaliada com nova validação; não se deve escolher outra configuração usando estes 30 resultados como nova seleção e continuar chamando-os de teste final.
