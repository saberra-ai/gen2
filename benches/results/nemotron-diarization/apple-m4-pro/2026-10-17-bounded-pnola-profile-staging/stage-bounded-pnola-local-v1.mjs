// Copy and statically verify real package/model assets. No app or SDK load.
import assert from 'node:assert/strict';
import {readFile, writeFile, lstat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';

const root='/Users/victor/workspace/gen2-nemotron3-n0';
const cache='/Users/victor/.cache/gen2-nemotron3-n0';
const expected=JSON.parse(await readFile(`${cache}/bounded-pnola-staging-inputs-v1.json`,'utf8'));
for(const [name,value] of Object.entries(expected.sources)){
  const raw=await readFile(`${root}/${name}`);
  assert.deepEqual({bytes:raw.length,sha256:createHash('sha256').update(raw).digest('hex')},value);
}
const destination=`${cache}/bounded-pnola-staging-local-v1`;
await assert.rejects(lstat(destination),{code:'ENOENT'});
const {stageSpeakers}=await import(pathToFileURL(`${root}/integrations/pnola/overlay/scripts/speaker-assets.mjs`));
const manifest=await stageSpeakers(destination,{
  packageRoot:`${cache}/bounded-mixed-local-v1/work/nemo`,
  modelRoot:`${cache}/bounded-mixed-local-v1/work/models/nemotron`,
  gen2Root:root,
  python:`${cache}/python-reference/bin/python`,
});
assert.equal(manifest.runtime_profile,'gen2-valid-mel-pool-cpu-v2');
assert.equal(manifest.runtime_reported_version,'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2');
const lock=JSON.parse(await readFile(`${root}/tools/nemotron-diarization/runtime-valid-mel-pool-cpu.lock.json`,'utf8'));
const license=JSON.parse(await readFile(`${root}/tools/nemotron-diarization/notices/openmdw-1.1/sources.json`,'utf8'));
assert.deepEqual(JSON.parse(await readFile(`${destination}/.desktop-assets/nemotron/bundle.json`,'utf8')),
  {model:lock.model,runtime_profile:lock.profile,frame_contract:lock.runtime.frame_contract,
    files:lock.artifacts.filter(item=>['model','notice'].includes(item.role)),license});
await writeFile(`${destination}/result.json`,JSON.stringify({passed:true,profile:manifest.runtime_profile,
  version:manifest.runtime_reported_version,private_pnola_modified:false,
  app_or_native_library_loads:0,model_inference_calls:0,scope:'Actual isolated source-package/model staging and static metadata only; no application build, signing, execution or release qualification.'},null,2)+'\n');
console.log('Actual bounded Pnola staging and bundle/lock provenance passed.');
