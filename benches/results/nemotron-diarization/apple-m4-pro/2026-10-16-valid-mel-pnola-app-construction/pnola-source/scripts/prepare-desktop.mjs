import { cp, mkdir, stat, mkdtemp, rename, rm } from 'node:fs/promises';
import { speakerInputs, stageSpeakers } from './speaker-assets.mjs';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
async function input(name, directory = false) {
  const value = process.env[name];
  if (!value) throw new Error(`Desktop preparation requires ${name}.`);
  const path = resolve(value);
  let info;
  try { info = await stat(path); }
  catch (error) { throw new Error(`Cannot read ${name}: ${path}`, { cause: error }); }
  if (directory ? !info.isDirectory() : !info.isFile() || info.size === 0)
    throw new Error(`${name} must be ${directory ? 'a directory' : 'a nonempty file'}: ${path}`);
  return path;
}

// Validate every explicit artifact before changing previously staged resources.
// Neither the ORT notice nor the demo audio depends on a developer checkout.
const bundle = await input('GEN2_BARK_BUNDLE', true);
const library = await input('GEN2_ORT_LIBRARY');
const license = await input('GEN2_ORT_LICENSE');
const sample = await input('PNOLA_SAMPLE_AUDIO');
const inputs = speakerInputs(root);
const stage = await mkdtemp(join(root, '.desktop-stage-'));
const assets = join(root, '.desktop-assets');
let saved = false;
try {
  const speakers = await stageSpeakers(stage, inputs);
  const next = join(stage, '.desktop-assets');
  await mkdir(join(next, 'runtime'), { recursive: true });
  await cp(bundle, join(next, 'bark'), { recursive: true, force: false, errorOnExist: true });
  const libraryName = process.platform === 'darwin' ? 'libonnxruntime.dylib'
    : process.platform === 'win32' ? 'onnxruntime.dll' : 'libonnxruntime.so';
  await cp(library, join(next, 'runtime', libraryName));
  await cp(license, join(next, 'runtime/LICENSE'));
  await cp(sample, join(next, 'sample.wav'));
  try { await rename(assets, join(stage, 'previous')); saved = true; }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  try { await rename(next, assets); }
  catch (error) { if (saved) await rename(join(stage, 'previous'), assets); throw error; }
  console.log('Desktop model, runtime, license, and sample staged locally.');
  if (speakers) console.log(`Speaker model and ${Object.keys(speakers.libraries).length} private native libraries staged; minimum macOS ${speakers.minimum_macos}.`);
} finally { await rm(stage, { recursive: true, force: true }); }
