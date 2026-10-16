import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { speakerInputs, inspectSpeakerPackage, speakerBuildConfiguration } from './speaker-assets.mjs';
const root=fileURLToPath(new URL('../',import.meta.url));
// A local build should not inherit an unrelated Developer ID from the host.
// Set PNOLA_SIGNING_IDENTITY explicitly when preparing a distributable build.
const environment={...process.env};
// A local app build must not inherit the host's notarization credentials.
for(const key of Object.keys(environment))if(key.startsWith('APPLE_'))delete environment[key];
environment.APPLE_SIGNING_IDENTITY=process.env.PNOLA_SIGNING_IDENTITY||'-';
const arguments_=[`${root}node_modules/@tauri-apps/cli/tauri.js`,'build','--bundles','app'];
const inputs=speakerInputs(root,environment);
const manifest=inputs?await inspectSpeakerPackage(inputs):null;
const base=JSON.parse(await readFile(`${root}src-tauri/tauri.conf.json`,'utf8'));
const configuration=speakerBuildConfiguration(base,manifest);
if(environment.APPLE_SIGNING_IDENTITY==='-')configuration.bundle.macOS.entitlements='Entitlements.local.plist';
if(manifest)environment.MACOSX_DEPLOYMENT_TARGET=configuration.bundle.macOS.minimumSystemVersion;
arguments_.push('--config',JSON.stringify(configuration));
const child=spawn(process.execPath,arguments_,{
  cwd:root,stdio:'inherit',env:environment,
});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',code=>{process.exitCode=code??1;});
