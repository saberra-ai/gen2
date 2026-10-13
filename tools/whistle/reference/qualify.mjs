// The pinned upstream pipeline is the oracle; this wrapper only records results.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import ort from 'onnxruntime-node';

const [model, corpus, output] = process.argv.slice(2).map(p => path.resolve(p));
const {parsePack} = await import(pathToFileURL(path.join(model, 'js/pack.js')));
const {Tokenizer, Engram, transcribe, melFeatures} = await import(pathToFileURL(path.join(model, 'js/whistle.js')));
const {OrtBackend} = await import(pathToFileURL(path.join(model, 'js/backend.js')));
const buf = fs.readFileSync(path.join(model, 'whistle.pack'));
const pack = parsePack(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const ctx = {tok: new Tokenizer(pack.man.tok), engram: new Engram(pack),
  fb: new Float32Array(pack.buf.slice(pack.base + pack.man.melfb.src, pack.base + pack.man.melfb.src + 257*80*4))};
const options = {executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,enableCpuMemArena:true};
const backend = new OrtBackend(ort,
  await ort.InferenceSession.create(path.join(model, 'onnx/encoder.onnx'),options),
  await ort.InferenceSession.create(path.join(model, 'onnx/decoder.onnx'),options));
const manifest = JSON.parse(fs.readFileSync(path.join(corpus,'corpus.json')));
fs.mkdirSync(output,{recursive:true});
const records=[];
const probes=path.join(output,'probes');fs.mkdirSync(probes,{recursive:true});
const save=(name,tensor)=>fs.writeFileSync(path.join(probes,`${name}.f32`),Buffer.from(tensor.buffer,tensor.byteOffset,tensor.byteLength));
const originalEncode=backend.encode.bind(backend),originalStep=backend.step.bind(backend);
let capture=false;
backend.encode=async(mel,T)=>{const S=await originalEncode(mel,T);if(capture){save('mel',mel);save('cross_k',backend.ck.data);save('cross_v',backend.cv.data);}return S;};
backend.step=async(tok,pos,features,attention)=>{const r=await originalStep(tok,pos,features,attention);if(capture&&pos<3){save(`step${pos}-efeat`,features);save(`step${pos}-logits`,r.logits);save(`step${pos}-prev`,backend.prev.data);save(`step${pos}-k`,backend.pk.data);save(`step${pos}-v`,backend.pv.data);}return r;};
save('engram',ctx.engram.feat([2,8192,42,110,77,12,400],6));
for (const record of manifest.records) {
  const buf = fs.readFileSync(path.join(corpus,record.pcm));
  if (crypto.createHash('sha256').update(buf).digest('hex') !== record.pcm_sha256) throw new Error('PCM checksum');
  const pcm = new Float32Array(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));
  capture=records.length===0;
  const result = await transcribe(backend,ctx,pcm);
  capture=false;
  const forced = await transcribe(backend,ctx,pcm,{language:record.language});
  // Observe EOS vs limit using the same upstream decision rule, without changing it.
  let lastToken=null;
  const step=backend.step.bind(backend);
  backend.step=async(...args)=>{const r=await step(...args);if(args[1]>0){lastToken=0;for(let i=1;i<8192;i++)if(r.logits[i]>r.logits[lastToken])lastToken=i;}return r;};
  const observed=await transcribe(backend,ctx,pcm);
  const complete=lastToken===1;
  const observedForced=await transcribe(backend,ctx,pcm,{language:record.language});
  const forcedComplete=lastToken===1;
  backend.step=step;
  if (observed.text!==result.text || JSON.stringify(observed.ids)!==JSON.stringify(result.ids)) throw new Error('instrumentation changed oracle');
  if(observedForced.text!==forced.text || JSON.stringify(observedForced.ids)!==JSON.stringify(forced.ids))throw new Error("forced instrumentation changed oracle");
  records.push({...record,text:result.text,token_ids:result.ids,detected_language:result.language,
    forced_text:forced.text,forced_token_ids:forced.ids,complete,forced_complete:forcedComplete,timing:result.timing});
  if (records.length<=2) {
    const {mel}=melFeatures(pcm,ctx.fb);
    fs.writeFileSync(path.join(output,`${record.id}.mel.f32`),Buffer.from(mel.buffer));
    const features=ctx.engram.feat([2,8192,42,110,77,12,400],6);
    fs.writeFileSync(path.join(output,'engram.f32'),Buffer.from(features.buffer));
  }
  fs.writeFileSync(path.join(output,'reference.json'),JSON.stringify({environment:{node:process.version,ort:ort.env.versions,session_options:options},corpus:manifest,records},null,2));
  console.log(record.id,result.language,result.ids.length,Math.round(result.timing.total_ms));
}
backend.release();
