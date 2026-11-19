import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire('C:/Users/migue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/__runtime__.cjs');
const {PresentationFile,FileBlob}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const root='C:/Users/migue/EoloSense/K-means/apresentacao';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/output/EoloSense_Kmeans_Apresentacao.pptx'));
await fs.mkdir(root+'/.build/final-previews',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(root+`/.build/final-previews/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(p.slides.items.length+' slides rendered from final PPTX');

