# K-means no estudo da fazenda C

Implementação específica em `src/multisensor_c.py`. O orquestrador comum é `Comum/src/estudo_fazenda_c.py`; o protocolo está em [Comum/FAZENDA_C_MULTISSENSORES.md](../Comum/FAZENDA_C_MULTISSENSORES.md).

O K-means é retreinado por evento e por perfil (temperatura, vibração, pressão, elétrico e combinado). Usa as condições de operação e os sensores monitorados do perfil, com imputação e padronização ajustadas somente no treino normal, até 6.000 registros, K=32 e três inicializações. Os sensores monitorados são os mesmos selecionados para os modelos supervisionados, sem consultar rótulos de falha.

O perfil **combinado** é principal. Uma regra de alerta é escolhida nesse perfil nos 42 eventos de desenvolvimento e congelada antes dos 16 eventos de avaliação. Os perfis diagnósticos usam a mesma regra, com limites calculados em sua própria calibração normal. Não se escolhe o perfil com melhor resultado nos reservados para anunciá-lo como teste final.

Resultados em `K-means/outputs/fazenda_C_componentes/`. Os dados e estudos antigos são preservados.
