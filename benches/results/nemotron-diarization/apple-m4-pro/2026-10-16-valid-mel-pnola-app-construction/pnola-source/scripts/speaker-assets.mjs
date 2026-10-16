import { cp, lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { createReadStream, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { join, resolve, basename } from 'node:path';

const execute=promisify(execFile);
const loadJson=async path=>JSON.parse(await readFile(path,'utf8'));
async function sha256(path){const hash=createHash('sha256');for await(const block of createReadStream(path))hash.update(block);return hash.digest('hex');}

export async function checkModelBundle(directory,lock){
  const files=lock.artifacts.filter(item=>item.role==='model'||item.role==='notice');
  if(!files.some(file=>file.role==='model')||!files.some(file=>file.role==='notice'))throw new Error('Missing model or model-card records in the native lock.');
  for(const file of files){
    if(basename(file.name)!==file.name)throw new Error('Invalid model artifact name.');
    const path=join(directory,file.name),info=await lstat(path);
    if(!info.isFile()||info.isSymbolicLink()||info.size!==file.size||await sha256(path)!==file.sha256)throw new Error(`Model checksum/size mismatch: ${file.name}`);
  }
  return files;
}

function version(value){if(!/^\d+\.\d+(\.\d+)?$/.test(value))throw new Error(`Invalid minimum macOS version: ${value}`);return value.split('.').map(Number);}
function highestVersion(a,b){const left=version(a),right=version(b);for(let i=0;i<3;i++){if((left[i]||0)!==(right[i]||0))return (left[i]||0)>(right[i]||0)?a:b;}return a;}

export function speakerBuildConfiguration(base,manifest){
  const config=structuredClone(base);
  if(manifest){
    config.bundle.resources['../.desktop-assets/nemo/']='nemo/';
    config.bundle.resources['../.desktop-assets/nemotron/']='nemotron/';
    config.bundle.macOS.minimumSystemVersion=highestVersion(config.bundle.macOS.minimumSystemVersion,manifest.minimum_macos);
    // Tauri signs native code under Frameworks with the same selected identity
    // as the app. Retain the original package under Resources for provenance.
    const native=Object.keys(manifest.libraries||{}).map(name=>`../.desktop-assets/nemo/lib/${name}`);
    native.push('../.desktop-assets/nemo/lib/libnemo_speech_asr_c.dylib');
    config.bundle.macOS.frameworks=[...(config.bundle.macOS.frameworks||[]),...native];
  }
  return {bundle:config.bundle};
}

export function speakerInputs(root,environment=process.env){
  if(environment.PNOLA_SPEAKERS==='0')return null;
  if(!environment.GEN2_NEMO_PACKAGE||!environment.GEN2_NEMOTRON_ASSETS)throw new Error('A speaker-enabled build requires GEN2_NEMO_PACKAGE and GEN2_NEMOTRON_ASSETS. Use PNOLA_SPEAKERS=0 explicitly for an ASR-only build.');
  const cargoPath=environment.GEN2_ROOT?null:readFileSync(join(root,'src-tauri/Cargo.toml'),'utf8').match(/^gen2 = \{ path = "([^"]+)"/m)?.[1];
  if(!environment.GEN2_ROOT&&!cargoPath)throw new Error('Set GEN2_ROOT to the Gen2 checkout containing the native packaging tools.');
  return {packageRoot:resolve(environment.GEN2_NEMO_PACKAGE),modelRoot:resolve(environment.GEN2_NEMOTRON_ASSETS),gen2Root:environment.GEN2_ROOT?resolve(environment.GEN2_ROOT):resolve(root,'src-tauri',cargoPath),python:environment.PNOLA_PYTHON||'python3'};
}

export async function inspectSpeakerPackage(inputs){
  if(process.platform!=='darwin'||process.arch!=='arm64')throw new Error('This native speaker package is currently qualified for macOS arm64. Other platforms remain under qualification.');
  await execute(inputs.python,[join(inputs.gen2Root,'tools/nemotron-diarization/package_macos.py'),'--verify',inputs.packageRoot]);
  return loadJson(join(inputs.packageRoot,'package.json'));
}

export async function stageSpeakers(root,inputs){
  const assets=join(root,'.desktop-assets');
  await mkdir(assets,{recursive:true});
  if(!inputs){
    // These are derived build resources, never user history or source models.
    for(const name of ['nemo','nemotron'])await rm(join(assets,name),{recursive:true,force:true});
    return null;
  }
  const manifest=await inspectSpeakerPackage(inputs);
  const lock=await loadJson(join(inputs.gen2Root,'tools/nemotron-diarization/runtime-valid-mel.lock.json'));
  const files=await checkModelBundle(inputs.modelRoot,lock);
  const licenseRoot=join(inputs.gen2Root,'tools/nemotron-diarization/notices/openmdw-1.1');
  const license=await loadJson(join(licenseRoot,'sources.json'));
  if(await sha256(join(licenseRoot,'LICENSE.OpenMDW-1.1'))!==license.sha256)throw new Error('OpenMDW license checksum mismatch.');
  const stage=await mkdtemp(join(assets,'.speakers-stage-'));
  const published=[],backups=[];
  try {
    await cp(inputs.packageRoot,join(stage,'nemo'),{recursive:true,verbatimSymlinks:true,errorOnExist:true,force:false});
    await inspectSpeakerPackage({...inputs,packageRoot:join(stage,'nemo')});
    await mkdir(join(stage,'nemotron'));
    for(const file of files)await cp(join(inputs.modelRoot,file.name),join(stage,'nemotron',file.name));
    await checkModelBundle(join(stage,'nemotron'),lock);
    await cp(join(licenseRoot,'LICENSE.OpenMDW-1.1'),join(stage,'nemotron/LICENSE.OpenMDW-1.1'));
    if(await sha256(join(stage,'nemotron/LICENSE.OpenMDW-1.1'))!==license.sha256)throw new Error('Staged OpenMDW license checksum mismatch.');
    await writeFile(join(stage,'nemotron/bundle.json'),JSON.stringify({model:lock.model,runtime_profile:lock.profile,frame_contract:lock.runtime.frame_contract,files,license},null,2)+'\n');
    for(const name of ['nemo','nemotron']){
      try{await rename(join(assets,name),join(stage,`old-${name}`));backups.push(name);}catch(error){if(error.code!=='ENOENT')throw error;}
      await rename(join(stage,name),join(assets,name));published.push(name);
    }
    return manifest;
  }catch(error){
    for(const name of published.reverse())await rm(join(assets,name),{recursive:true,force:true});
    for(const name of backups.reverse())await rename(join(stage,`old-${name}`),join(assets,name));
    throw error;
  }finally{await rm(stage,{recursive:true,force:true});}
}
