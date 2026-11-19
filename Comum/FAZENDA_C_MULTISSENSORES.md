# Fazenda C — experimento multissensores

Novo estudo autorizado em 18/11/2026. Todos os modelos são retreinados; dados originais e estudos anteriores são preservados. O objetivo é verificar se medir desvios em vibração, pressão e grandezas elétricas acrescenta informação às temperaturas.

## Protocolo

- Apenas os 58 eventos da fazenda C; mesma separação por turbina: 42 de desenvolvimento e 16 de avaliação (7 com anomalia e 9 normais).
- Dentro de cada arquivo: somente treino normal, divisão cronológica em ajuste, validação e calibração com intervalos de um dia; não se reconstrói cronologia entre arquivos.
- Condições de operação: vento, potência ativa, rotação e temperatura ambiente, com atributos móveis causais.
- Banco monitorado: até quatro sensores por grupo de temperatura, vibração, pressão hidráulica/óleo/água e grandezas elétricas. Os grupos são definidos pelas descrições dos sensores, não pelos rótulos de falha. Dentro de cada grupo, a ordenação usa previsibilidade em validação normal; um grupo inteiro não é descartado por R² baixo.
- **Conjunto combinado é a análise principal.** Os quatro grupos isolados são diagnósticos. Os supervisionados compartilham um banco multissaída; seus diagnósticos não são ablações com retreinamentos independentes. O K-means é retreinado por perfil, usando condições de operação e o grupo monitorado.
- Uma regra de alerta por modelo é escolhida apenas no combinado dos 42 eventos de desenvolvimento: maior acurácia balanceada dentro dos tetos de 5 falsos avisos/30 dias e 25% de tempo em incidente normal. Grade de 75 regras: P90/P92,5/P95/P97,5/P99 × 6/12/24/48/72h × fração 10/25/50%. Se nenhuma atende aos tetos, a candidata é explicitamente não aprovada e se prioriza a menor violação.
- A regra congelada é aplicada aos 16 reservados e aos grupos diagnósticos; os valores dos limiares são calibrados por grupo, somente em dados normais. Nenhum perfil será eleito usando os reservados.
- Status 0 define elegibilidade na previsão da C. Uma detecção exige notificação nova dentro da janela anotada. Episódios rearmam após 6h de recuperação. Métricas por evento e taxas por exposição; não é CARE oficial nem previsão da data exata da falha.

## Orçamento de cálculo

Até 6.000 linhas normais no ajuste; SVR limitado a 1.500. Árvore testa quatro configurações e poda na validação; ensembles comparam Random Forest e Extra Trees; SVR compara C=1/10; Ridge compara alpha=1/10. Florestas finais têm 300 árvores. K-means usa K=32, três inicializações e semente 42. Essa grade é menor que a do estudo anterior e não inclui Gradient Boosting; comparações históricas não isolam apenas o efeito dos novos sensores.

Sensores de vibração são agregados SCADA, não sinais brutos de alta frequência. Não há promessa de ganho de desempenho. Os reservados já foram examinados em estudos anteriores, portanto a avaliação é interna reutilizada.

## Reprodução — Git Bash, raiz do projeto

```bash
./.venv/Scripts/python.exe -X utf8 -m unittest discover -s Comum/tests -v
./.venv/Scripts/python.exe -X utf8 Comum/src/estudo_fazenda_c.py --jobs 4
for modelo in arvore ensembles svm linear kmeans; do
  ./.venv/Scripts/python.exe -X utf8 Comum/src/estudo_fazenda_c.py --evaluate-only --family "$modelo"
done
./.venv/Scripts/python.exe -X utf8 Comum/src/estudo_fazenda_c.py --evaluate-only
./.venv/Scripts/python.exe -X utf8 Comum/src/relatorio_fazenda_c.py
```

Saídas separadas em `outputs/fazenda_C_componentes` de cada família; comparação em `Comum/outputs/fazenda_C_componentes/COMPARACAO.md`. Modelos e escores por evento são salvos para retomada. O protocolo registra hashes e bloqueia mistura de código alterado com checkpoints do mesmo estudo. O piloto usa pastas `fazenda_C_multissensores_piloto`, fora dos resultados finais.

O primeiro desenho foi interrompido antes da avaliação porque o piloto concentrou os alvos térmicos em gabinetes e não incluiu pressão hidr?ulica. Seus arquivos ficam nas pastas piloto/multissensores, sem entrar na comparação final. A execução com cotas por componente usa o nome `fazenda_C_componentes`.

O critério de desempenho experimental já usado no projeto continua sendo sensibilidade de pelo menos 70% e especificidade de pelo menos 80%. Nos 16 reservados da C, isso exige pelo menos 5/7 eventos com anomalia detectados e 8/9 normais sem falso aviso. Não é uma exigência universal nem validação industrial.


## Resultado concluído

Consulte [RESULTADOS_FAZENDA_C.md](RESULTADOS_FAZENDA_C.md). Os cinco modelos foram retreinados e avaliados; nenhum atingiu o critério experimental.
