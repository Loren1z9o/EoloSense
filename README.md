# EoloSense — início da análise CARE to Compare

Este repositório começa com uma exploração de agrupamentos naturais usando **K-means**. As decisões sobre o objetivo final dos quatro modelos (K-means, ensembles, árvore de decisão e SVM) ficam para a próxima etapa. O K-means atual é **exploratório**: os números de cluster não representam tipos de falha.

## Dados

Fonte: [CARE to Compare, versão v6 no Zenodo](https://zenodo.org/records/15846963). O ZIP tem 5.503.439.673 bytes e MD5 `2547b58c21ac8c242d13232860cf500c`. Ele já foi validado e extraído em `data/CARE_To_Compare` (22 eventos na fazenda A, 15 na B e 58 na C). Para conferir novamente o ZIP, no Git Bash, a partir da pasta `data`:

```bash
md5sum CARE_To_Compare.zip
```

O programa procura os arquivos `Wind Farm A/B/C/datasets/<event_id>.csv` recursivamente, mesmo que o ZIP acrescente pastas intermediárias. Dados brutos e resultados estão no `.gitignore` por causa do tamanho.

## Ambiente (Git Bash, na raiz do projeto)

Use Python 3.10 ou superior:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Primeira execução

```bash
python src/kmeans_baseline.py --list-events
python src/kmeans_baseline.py --farm A --event-id 0
```

O segundo comando usa o evento 0 da fazenda A. Para outro evento específico:

```bash
python src/kmeans_baseline.py --farm B --event-id 53
```

Se os CSVs estiverem fora de `data`, informe `--data-dir CAMINHO`. Para testar outros números de grupos, use `--k-values 2 3 4 5 6 7 8`.

Cada execução cria uma pasta em `outputs/kmeans/` com:

- `report.json`: sensores usados, tamanhos das amostras, inércia e silhouette para cada k, e k selecionado;
- `clusters_pca.png`: projeção 2D dos grupos de treino;
- `train_clusters.csv` e `prediction_clusters.csv`: grupo e distância ao centroide por linha.

O programa usa somente sensores com sufixo `_avg`, porque o [registro do Zenodo](https://zenodo.org/records/15846963) documenta inconsistências nas estatísticas mínimo, máximo e desvio padrão. O arquivo `feature_description.csv` também é usado para excluir sensores angulares e contadores nesta primeira análise. Remove colunas muito incompletas ou constantes, imputa medianas e padroniza os sensores. O ajuste dessas etapas e do K-means usa **apenas linhas de treino com `status_type_id == 0`**. A parte de previsão recebe atribuições de grupo após o ajuste. O limite padrão de 20 mil linhas para o ajuste evita um primeiro experimento muito pesado; o arquivo completo do evento ainda é lido para produzir as atribuições.

O [guia dos autores](https://aefdi.github.io/EnergyFaultDetector/care2compare_guide.html) distingue o estado operacional por timestamp (`status_type_id`) do rótulo de anomalia do evento (`event_label`). Por isso, este início não calcula acurácia de falhas nem a métrica CARE. Também não interpreta a silhouette, por si só, como evidência de defeitos.

**Pendente:** rodar um evento de cada fazenda, revisar os resultados com o grupo e decidir a tarefa e a avaliação comuns aos quatro modelos.
