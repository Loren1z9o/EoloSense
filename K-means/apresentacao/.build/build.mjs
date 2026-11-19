import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const rt='C:/Users/migue/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=rt+'/node/node_modules';
const req=createRequire(rt+'/node/node_modules/__runtime__.cjs');
const {Presentation,PresentationFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const SKILL='C:/Users/migue/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const root='C:/Users/migue/EoloSense/K-means/apresentacao';
const pres=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#122C3E',teal='#007F82',muted='#526573',white='#FFFFFF',paper='#F6F8FA';
const sources='Fontes do projeto: K-means/REVISAO_ATUAL.md; BUSCA_CONFIGURACOES.md; DECISAO_KMEANS.md; ALERTAS.md. Recorte: trabalho discutido neste chat até 14/11/2026. Apresentação preparada em 18/11/2026.';
function text(slide,value,x,y,w,h,size=26,color=navy,bold=false){let sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});sh.text=value;sh.text.style={fontFamily:'Arial',fontSize:size,color,bold};return sh;}
function page(title,note=''){let s=pres.slides.add();s.background.fill=paper;text(s,title,64,48,1152,100,44,navy,true);text(s,String(pres.slides.items.length).padStart(2,'0'),1160,666,55,26,16,muted);s.speakerNotes.text=sources+'\n'+note;return s;}
function para(s,lines,y=190,w=1120,size=28){lines.forEach((v,i)=>text(s,v,72,y+i*100,w,86,size));}
function table(s,values,y=185,height=330,widths){let t=s.tables.add({rows:values.length,columns:values[0].length,left:72,top:y,width:1136,height,values,columnWidths:widths});t.borders.assign({fill:'#D5DEE3',width:1,style:'solid'});for(let r=0;r<values.length;r++){t.rows[r].height=height/values.length;for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?navy:(r%2?white:'#EBF1F4');cell.text.style={fontFamily:'Arial',fontSize:24,color:r===0?white:navy,bold:r===0};}}return t;}
function note(s,v,y=600){text(s,v,72,y,1130,56,23,teal,true);}
function chart(s,cats,series){let c=s.charts.add('bar',{position:{left:90,top:175,width:1090,height:390},categories:cats,series,hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:22,typeface:'Arial'}},barOptions:{direction:'column',grouping:'clustered'},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:28,typeface:'Arial',bold:true}},xAxis:{textStyle:{fontSize:22,typeface:'Arial'}},yAxis:{min:0,max:15,majorUnit:5,textStyle:{fontSize:22,typeface:'Arial'}},chartFill:'none',plotAreaFill:'none'});applyPresentationChartFont(c,{fontFamily:'Arial'});return c;}
// 01
let s=pres.slides.add();s.background.fill=navy;text(s,'EoloSense',72,90,1100,90,70,white,true);text(s,'Exploração de dados e alertas\ncom K-means',72,220,1120,180,54,white,true);text(s,'Projeto final de Machine Learning\nDesenvolvimento, experimentos e aprendizados',76,460,1100,100,29,'#9CD5D0');text(s,'Histórico deste chat • novembro de 2026',76,636,1000,35,21,'#C4D2DA');s.speakerNotes.text=sources;
//02
s=page('Objetivo do projeto');para(s,['Investigar anomalias nos sensores de turbinas eólicas que possam anteceder falhas.','O projeto prevê K-means, árvore de decisão, ensembles e SVM.','O professor sugeriu começar pelos grupos naturais; este histórico se concentra no K-means.']);note(s,'Alerta de anomalia não estima a data exata de uma futura quebra.');
//03
s=page('CARE to Compare v6','Fonte externa: https://zenodo.org/records/15846963 — 95 eventos e 36 turbinas. Contagens locais v6: 45 anomalias e 50 eventos normais.');table(s,[['Conjunto','Fazenda A','Fazenda B','Fazenda C'],['Eventos','22','15','58'],['Turbinas','5','9','22'],['Colunas por arquivo','86','257','957']],185,300,[410,242,242,242]);text(s,'Medições SCADA em intervalos de 10 minutos; um CSV por evento, com seções de treino e previsão.',72,525,1130,85,27);note(s,'Os sensores diferem entre fazendas: não juntamos as colunas indiscriminadamente.',635);
//04
s=page('Preparação dos dados');para(s,['1. Baixar o ZIP pelo terminal, conferir o MD5 e extrair os 95 eventos.','2. Ler os CSVs com separador “;” e selecionar médias dos sensores (_avg).','3. Excluir ângulos, contadores e colunas inadequadas; imputar e padronizar no treino.','4. Reservar previsão e salvar todos os resultados fora do dataset original.'],175,1120,27);
//05
s=page('O que o K-means aprende');para(s,['Cada registro é uma combinação das leituras de vários sensores.','K define quantos grupos o algoritmo forma; os centros resumem os padrões desses grupos.','A silhouette varia de −1 a +1 e mede a separação entre os grupos.'],185,1120,29);note(s,'Um grupo não recebe automaticamente o significado “falha” ou “normal”.');
//06
s=page('Primeiro experimento: evento 0 da fazenda A','Treino com 20.000 registros amostrados entre 47.252 linhas normais, 42 sensores selecionados. Teste de k=2 a 6; silhouette de 0,3656415 para k=2.');table(s,[['Grupo','Registros de treino','Vento médio','Potência média'],['0','17.070','10,16','0,6632'],['1','30.182','4,13','0,0893']],200,225,[180,350,303,303]);text(s,'K=2 teve a melhor silhouette: 0,366. Os perfis sugerem regimes de maior e menor vento e geração.',72,470,1125,110,30);note(s,'São grupos de funcionamento, não duas classes supervisionadas de falha.');
//07
s=page('Da clusterização ao alerta');para(s,['Distância ao centro: mede quanto uma leitura se afasta dos padrões aprendidos.','Calibração: dados normais separados definem um limite para essa distância.','Persistência: o alerta exige desvios recorrentes em uma janela móvel causal.'],180,1120,29);note(s,'Distância alta é evidência de comportamento incomum; a causa precisa ser investigada.');
//08
s=page('Correções metodológicas importantes','Fonte externa: https://aefdi.github.io/EnergyFaultDetector/care2compare_faq.html, seções de semântica e anonimização. Fontes locais: DECISAO_KMEANS.md e REVISAO_ATUAL.md.');para(s,['A janela event_start–event_end descreve anomalias que antecedem a falha.','A anonimização por arquivo impede reconstruir a cronologia real entre eventos.','Timestamps coincidentes não comprovam medições duplicadas entre arquivos.','As AUCs históricas de estudos entre arquivos não são a validação final.'],168,1120,27);
//09
s=page('Baseline fixo nos 95 eventos','Fonte: outputs/benchmark_decision/metrics.json e outputs/failure_alerts_evaluation/metrics.csv; os resultados gerais do baseline K32 foram reproduzidos.');table(s,[['Decisão por evento','Anomalias (45)','Normais (50)'],['Com alerta','31 detectadas','29 falsos alertas'],['Sem alerta','14 não detectadas','21 corretos']],195,240,[470,333,333]);text(s,'K=32 • P95 da calibração • janela de 24h\nSensibilidade: 68,9% • Especificidade: 42,0%',72,480,1120,100,29);note(s,'O volume de falsos alertas motivou uma busca controlada de configurações.');
//10
s=page('Busca de 108 configurações','Fonte: BUSCA_CONFIGURACOES.md e outputs/tuning_search/protocol.json. Todas as combinações foram testadas na seleção; K=3 teve 18 combinações.');table(s,[['Parâmetro','Valores testados'],['K','3, 8, 16, 32, 64 e 128'],['Escore','Euclidiano; ajustado pela dispersão do grupo'],['Percentil de calibração','95; 97,5; 99'],['Janela de persistência','6; 12; 24 horas']],172,340,[375,761]);note(s,'Meta experimental: detectar ≥70% das anomalias e acertar ≥80% dos normais.',580);
//11
s=page('Separação por turbina');text(s,'65',90,195,380,120,90,teal,true);text(s,'eventos para seleção',94,328,480,55,30);text(s,'30',700,195,380,120,90,teal,true);text(s,'eventos para avaliação',704,328,480,55,30);text(s,'Turbinas inteiras em conjuntos distintos. A configuração foi congelada antes da avaliação dos 30 eventos.',72,445,1130,100,30);note(s,'Avaliação interna: as três fazendas já haviam sido exploradas no projeto.');
//12
s=page('Melhores configurações na seleção','Fonte: outputs/tuning_search/selection_ranking.csv. Para cada K, maior sensibilidade entre configurações com especificidade >=80%; não são resultados de teste final.');table(s,[['K','Percentil / horas','Detecção de anomalias','Acerto nos normais'],['3','99 / 6h','40,0%','80,0%'],['8','99 / 6h','36,7%','82,9%'],['16','99 / 6h','36,7%','82,9%'],['32','99 / 12h','43,3%','80,0%'],['64','99 / 6h','53,3%','80,0%'],['128','99 / 12h','40,0%','80,0%']],165,385,[100,280,378,378]);note(s,'K=64 foi escolhido; nenhuma das 108 combinações atingiu as duas metas.',590);
//13
s=page('Avaliação nos mesmos 30 eventos','Fonte: outputs/tuning_search/evaluation_metrics.csv e outputs/failure_alerts_evaluation/metrics.csv. Há 15 eventos de cada classe; métricas por evento.');chart(s,['Anomalias detectadas','Normais sem falso alerta'],[{name:'K32 / P95 / 24h',values:[8,5],fill:'#8093A3'},{name:'K64 / P99 / 6h',values:[7,7],fill:teal}]);note(s,'K64: 14 acertos em 30 • K32: 13 acertos em 30. Ganho de apenas um evento.',600);
//14
s=page('Revisão do estudo com agentes');para(s,['O novo fluxo salva e reutiliza modelos e explicita registros com sensores insuficientes.','Foram gerados 95 modelos e 95 arquivos de previsão; os oito testes passaram.','As 30 decisões do K32 coincidiram com o baseline histórico: a engenharia melhorou, mas o desempenho foi reproduzido.'],170,1120,28);note(s,'Melhor acurácia balanceada comparável: K64, 46,7%; K32, 43,3%.');
//15
s=page('Conclusão e continuidade');text(s,'O K-means foi útil para entender regimes de operação e construir um baseline de alertas.',72,190,1130,120,36,teal,true);para(s,['As abordagens avaliadas ainda não sustentam previsão confiável de falhas.','O resultado negativo é parte do trabalho: os protocolos e limitações estão documentados.','A continuidade exige alvo e avaliação comuns para comparar árvore, ensembles e SVM.'],350,1120,27);
//16
s=page('Reprodução e referências','Fontes principais: https://zenodo.org/records/15846963; https://aefdi.github.io/EnergyFaultDetector/care2compare_faq.html; https://doi.org/10.3390/data9120138. Documentos locais: K-means/REVISAO_ATUAL.md, BUSCA_CONFIGURACOES.md, ALERTAS.md.');text(s,'Git Bash • comandos na raiz do projeto',72,174,1130,48,27,teal,true);text(s,'./.venv/Scripts/python.exe K-means/src/tuning_search.py\n\n./.venv/Scripts/python.exe -m unittest discover -s K-means/tests -v',72,245,1135,155,23);text(s,'Documentação: REVISAO_ATUAL.md • BUSCA_CONFIGURACOES.md • ALERTAS.md\n\nDados: CARE to Compare v6, Fraunhofer IEE / Zenodo\nMétodo e avaliação: publicação e FAQ dos autores',72,435,1120,160,25);
await fs.mkdir(root+'/.build/previews',{recursive:true});
const draft=root+'/.build/candidate.pptx';
await (await PresentationFile.exportPptx(pres)).save(draft);
for(let i=0;i<pres.slides.items.length;i++){const blob=await pres.export({slide:pres.slides.items[i],format:'png',scale:1});await fs.writeFile(root+`/.build/previews/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await blob.arrayBuffer()));}
const result=await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:root+'/output/EoloSense_Kmeans_Apresentacao.pptx',pythonExecutable:rt+'/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[3,6,9,10,12].flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'design',families:['Arial']},explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[3,6,9,10,12],requiredNativeChartOwnerSlides:[13],materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:root+'/.build/validation-v2.json'});
console.log(JSON.stringify(result));



