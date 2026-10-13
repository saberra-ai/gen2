import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import ort from 'onnxruntime-node';
const [model,pcmPath,nativePath,output]=process.argv.slice(2).map(p=>path.resolve(p));
const {parsePack}=await import(pathToFileURL(path.join(model,'js/pack.js')));
const {Tokenizer,Engram,transcribe}=await import(pathToFileURL(path.join(model,'js/whistle.js')));
const {OrtBackend}=await import(pathToFileURL(path.join(model,'js/backend.js')));
const bytes=fs.readFileSync(path.join(model,'whistle.pack'));const pack=parsePack(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const ctx={tok:new Tokenizer(pack.man.tok),engram:new Engram(pack),fb:new Float32Array(pack.buf.slice(pack.base+pack.man.melfb.src,pack.base+pack.man.melfb.src+257*80*4))};
const sessionOptions={executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,enableCpuMemArena:true};
const backend=new OrtBackend(ort,await ort.InferenceSession.create(path.join(model,'onnx/encoder.onnx'),sessionOptions),await ort.InferenceSession.create(path.join(model,'onnx/decoder.onnx'),sessionOptions));
const raw=fs.readFileSync(pcmPath);const base=new Float32Array(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
const native=JSON.parse(fs.readFileSync(nativePath));const sweep=[];
const p95=a=>[...a].sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];
const median=a=>[...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
for (const entry of native.sweep) {
  const audio=Float32Array.from({length:entry.seconds*16000},(_,i)=>base[i%base.length]);
  const expected=await transcribe(backend,ctx,audio,{maxTokens:1024});
  if(expected.text!==entry.text || JSON.stringify(expected.ids)!==JSON.stringify(entry.token_ids))throw new Error('sweep token/text parity');
  const samples=[];for(let i=0;i<5;i++){const started=performance.now();const actual=await transcribe(backend,ctx,audio,{maxTokens:1024});samples.push(performance.now()-started);if(actual.text!==expected.text)throw new Error('oracle state contamination');}
  const nativeMs=entry.samples.map(s=>Number(s.wall_micros)/1000);const ratio=median(nativeMs)/median(samples),rtf=median(nativeMs)/(entry.seconds*1000);
  sweep.push({seconds:entry.seconds,native_median_ms:median(nativeMs),native_p95_ms:p95(nativeMs),node_median_ms:median(samples),node_p95_ms:p95(samples),node_samples_ms:samples,ratio,rtf,passed:ratio<=1.25&&rtf<1});
}
backend.release();
const reservationBytes=2048*1024*1024;
const soakTail=native.soak_rss.slice(10);
const memory={reservation_bytes:reservationBytes,peak_rss:native.peak_rss,soak_growth_bytes:Math.max(...soakTail)-soakTail[0],passed:native.peak_rss<reservationBytes&&Math.max(...soakTail)-soakTail[0]<=64*1024*1024};
const receipt={environment:{node:process.version,ort:ort.env.versions,session_options:sessionOptions},sweep,memory,passed:sweep.every(s=>s.passed)&&memory.passed};
fs.writeFileSync(output,JSON.stringify(receipt,null,2));if(!receipt.passed)throw new Error('performance gate failed');
