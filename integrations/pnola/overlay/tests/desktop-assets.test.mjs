import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm, mkdir, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { checkModelBundle, readSpeakerRuntimeLock, speakerBuildConfiguration, speakerInputs } from '../scripts/speaker-assets.mjs';

test('speaker staging selects the exact native profile lock and rejects identity mismatches',async()=>{
  const root=await mkdtemp(join(tmpdir(),'pnola-native-profile-'));
  try{
    const directory=join(root,'tools/nemotron-diarization');
    await mkdir(directory,{recursive:true});
    for(const [profile,filename,version] of [
      ['gen2-valid-mel-v1','runtime-valid-mel.lock.json','nemo-speech-asr 0.2.0+gen2-valid-mel-v1'],
      ['gen2-valid-mel-pool-v1','runtime-valid-mel-pool.lock.json','nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1'],
      ['gen2-valid-mel-pool-cpu-v2','runtime-valid-mel-pool-cpu.lock.json','nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2'],
    ]){
      await writeFile(join(directory,filename),JSON.stringify({profile,model:{selected:profile},runtime:{reported_version:version}}));
      const manifest={runtime_profile:profile,runtime_reported_version:version};
      assert.equal((await readSpeakerRuntimeLock(root,manifest)).model.selected,profile);
      await assert.rejects(readSpeakerRuntimeLock(root,{...manifest,runtime_reported_version:version+'.extra'}),/profile\/version/);
      await assert.rejects(readSpeakerRuntimeLock(root,{runtime_profile:profile}),/profile\/version/);
    }
    for(const profile of ['../runtime','gen2-valid-mel-pool-v2','gen2-valid-mel-pool-v1.extra','gen2-valid-mel-pool-cpu-v3','gen2-valid-mel-pool-cpu-v2.extra','__proto__'])await assert.rejects(readSpeakerRuntimeLock(root,{runtime_profile:profile}),/Unsupported/);
    await writeFile(join(directory,'runtime-valid-mel-pool.lock.json'),JSON.stringify({profile:'gen2-valid-mel-v1',runtime:{reported_version:'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1'}}));
    await assert.rejects(readSpeakerRuntimeLock(root,{runtime_profile:'gen2-valid-mel-pool-v1',runtime_reported_version:'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1'}),/profile\/version/);
  }finally{await rm(root,{recursive:true,force:true});}
});

test('speaker tooling follows the actual Cargo checkout and requires an explicit ASR-only selection',async()=>{
  const root=await mkdtemp(join(tmpdir(),'pnola-build-path-'));
  try{
    await mkdir(join(root,'src-tauri'));
    await writeFile(join(root,'src-tauri/Cargo.toml'),'gen2 = { path = "../../selected-gen2", default-features = false }\n');
    const inputs=speakerInputs(root,{GEN2_NEMO_PACKAGE:root,GEN2_NEMOTRON_ASSETS:root});
    assert.equal(inputs.gen2Root,join(root,'../selected-gen2'));
    assert.equal(speakerInputs(root,{PNOLA_SPEAKERS:'0'}),null);
    assert.throws(()=>speakerInputs(root,{}),/requires/);
  }finally{await rm(root,{recursive:true,force:true});}
});

test('desktop staging rejects changed model bytes and missing notice files',async()=>{
  const root=await mkdtemp(join(tmpdir(),'pnola-model-staging-'));
  try {
    const card=Buffer.from('pinned model card');
    const model=Buffer.from('model');
    const lock={artifacts:[{name:'model.gguf',role:'model',size:model.length,sha256:createHash('sha256').update(model).digest('hex')},{name:'README.md',role:'notice',size:card.length,sha256:createHash('sha256').update(card).digest('hex')}]};
    await writeFile(join(root,'model.gguf'),model);
    await assert.rejects(checkModelBundle(root,lock),/README/);
    await writeFile(join(root,'README.md'),card);
    await checkModelBundle(root,lock);
    await writeFile(join(root,'model.gguf'),'other');
    await assert.rejects(checkModelBundle(root,lock),/checksum/);
    assert.equal(await readFile(join(root,'README.md'),'utf8'),'pinned model card');
  }finally{await rm(root,{recursive:true,force:true});}
});

test('speaker bundle declares the actual native macOS floor and preserves base resources',()=>{
  const base={bundle:{resources:{'../.desktop-assets/bark/':'bark/'},macOS:{minimumSystemVersion:'13.0',frameworks:['ort.dylib']}}};
  const full=speakerBuildConfiguration(base,{minimum_macos:'26.0',libraries:{'libgen2_nemo_native.dylib':{}}});
  assert.equal(full.bundle.macOS.minimumSystemVersion,'26.0');
  assert.equal(full.bundle.resources['../.desktop-assets/nemo/'],'nemo/');
  assert.equal(full.bundle.resources['../.desktop-assets/nemotron/'],'nemotron/');
  assert.equal(full.bundle.resources['../.desktop-assets/bark/'],'bark/');
  assert.deepEqual(full.bundle.macOS.frameworks,['ort.dylib','../.desktop-assets/nemo/lib/libgen2_nemo_native.dylib','../.desktop-assets/nemo/lib/libnemo_speech_asr_c.dylib']);
  assert.equal(base.bundle.macOS.minimumSystemVersion,'13.0');
  assert.equal(speakerBuildConfiguration(base,null).bundle.resources['../.desktop-assets/nemo/'],undefined);
  assert.equal(speakerBuildConfiguration(base,{minimum_macos:'12.0'}).bundle.macOS.minimumSystemVersion,'13.0');
  assert.throws(()=>speakerBuildConfiguration(base,{minimum_macos:'unknown'}),/macOS/);
});


test('desktop preparation uses explicit artifacts in an isolated checkout and preserves staged assets on invalid input',async()=>{
  const root=await mkdtemp(join(tmpdir(),'pnola-portable-desktop-'));
  try {
    await cp(new URL('../scripts/',import.meta.url),join(root,'scripts'),{recursive:true});
    await mkdir(join(root,'inputs/bark'),{recursive:true});
    await writeFile(join(root,'inputs/bark/model.onnx'),'fixture model');
    await writeFile(join(root,'inputs/ort.dylib'),'fixture runtime');
    await writeFile(join(root,'inputs/LICENSE'),'fixture runtime license');
    await writeFile(join(root,'inputs/sample.wav'),'fixture audio');
    const {execFile}=await import('node:child_process');
    const {promisify}=await import('node:util');
    const run=promisify(execFile);
    const env={PATH:process.env.PATH,PNOLA_SPEAKERS:'0',GEN2_BARK_BUNDLE:join(root,'inputs/bark'),GEN2_ORT_LIBRARY:join(root,'inputs/ort.dylib'),GEN2_ORT_LICENSE:join(root,'inputs/LICENSE'),PNOLA_SAMPLE_AUDIO:join(root,'inputs/sample.wav')};
    await run(process.execPath,[join(root,'scripts/prepare-desktop.mjs')],{env});
    assert.equal(await readFile(join(root,'.desktop-assets/runtime/LICENSE'),'utf8'),'fixture runtime license');
    assert.equal(await readFile(join(root,'.desktop-assets/bark/model.onnx'),'utf8'),'fixture model');
    assert.equal(await readFile(join(root,'.desktop-assets/sample.wav'),'utf8'),'fixture audio');
    const prior=await readFile(join(root,'.desktop-assets/runtime/LICENSE'),'utf8');
    await assert.rejects(run(process.execPath,[join(root,'scripts/prepare-desktop.mjs')],{env:{...env,GEN2_ORT_LICENSE:join(root,'missing')}}),/GEN2_ORT_LICENSE/);
    assert.equal(await readFile(join(root,'.desktop-assets/runtime/LICENSE'),'utf8'),prior);
    await assert.rejects(run(process.execPath,[join(root,'scripts/prepare-desktop.mjs')],{env:{...env,PNOLA_SAMPLE_AUDIO:''}}),/PNOLA_SAMPLE_AUDIO/);
    assert.equal(await readFile(join(root,'.desktop-assets/runtime/LICENSE'),'utf8'),prior);
  }finally{await rm(root,{recursive:true,force:true});}
});
