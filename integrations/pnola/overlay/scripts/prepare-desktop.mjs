import { cp, mkdir, access } from "node:fs/promises";
import { speakerInputs, stageSpeakers } from './speaker-assets.mjs';
import { fileURLToPath } from "node:url";
import { resolve, join } from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url));
const gen2 = resolve(root, "../gen2-bark-onnx");
const bundle =
  process.env.GEN2_BARK_BUNDLE || join(gen2, "target/bark/model/onnx");
const capi = join(
  gen2,
  "target/bark/reference124/lib/python3.12/site-packages/onnxruntime",
);
const library =
  process.env.GEN2_ORT_LIBRARY ||
  join(capi, "capi/libonnxruntime.1.24.4.dylib");
await access(bundle);
await access(library);
const speakers=await stageSpeakers(root,speakerInputs(root));
const assets = join(root, ".desktop-assets");
await mkdir(join(assets, "runtime"), { recursive: true });
await cp(bundle, join(assets, "bark"), { recursive: true, force: true });
const libraryName =
  process.platform === "darwin"
    ? "libonnxruntime.dylib"
    : process.platform === "win32"
      ? "onnxruntime.dll"
      : "libonnxruntime.so";
await cp(library, join(assets, "runtime", libraryName));
await cp(join(capi, "LICENSE"), join(assets, "runtime/LICENSE"));
await cp(
  join(gen2, "target/bark/oracle124/000.wav"),
  join(assets, "sample.wav"),
);
console.log("Desktop model, runtime, license, and sample staged locally.");
if(speakers)console.log(`Speaker model and ${Object.keys(speakers.libraries).length} private native libraries staged; minimum macOS ${speakers.minimum_macos}.`);
