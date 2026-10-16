import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { checkModelBundle, speakerBuildConfiguration, speakerInputs } from '../scripts/speaker-assets.mjs';

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
