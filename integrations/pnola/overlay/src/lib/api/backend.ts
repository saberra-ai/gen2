import { Channel, convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { readMessages } from "../../stream.js";
import { validateHistory } from "../studio/state";
import type {
  AudioSelection,
  CaptureState,
  EngineStatus,
  Progress,
  Result,
  Transcript,
  LiveSnapshot,
} from "../studio/types";

export const desktop = isTauri();
interface NativeAudio {
  path: string;
  name: string;
  bytes: number;
  duration: number | null;
  sha256?:string;
}
function nativeSelection(file: NativeAudio): AudioSelection {
  return {
    ...file,
    duration: file.duration || 0,
    url: convertFileSrc(file.path),
  };
}
async function json<T>(response: Response): Promise<T> {
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Local request failed.");
  return value;
}
export async function importPath(path: string): Promise<AudioSelection> {
  return nativeSelection(await invoke<NativeAudio>("file_info", { path }));
}
export async function chooseAudio(): Promise<AudioSelection | null> {
  const path = await open({
    multiple: false,
    directory: false,
    filters: [
      {
        name: "Audio",
        extensions: ["wav", "mp3", "m4a", "ogg", "webm", "flac", "aac", "mp4"],
      },
    ],
  });
  return typeof path === "string" ? importPath(path) : null;
}
export async function importBlob(
  blob: Blob,
  name: string,
): Promise<AudioSelection> {
  if (blob.size > 256 * 1024 * 1024)
    throw new Error("Choose an audio file under 256 MB.");
  const url = URL.createObjectURL(blob);
  return { blob, name, url, bytes: blob.size, duration: 0 };
}
export async function engineStatus(): Promise<EngineStatus> {
  return desktop ? invoke("engine_status") : json(await fetch("/api/status"));
}
export async function sampleAudio(): Promise<AudioSelection> {
  if (desktop) return nativeSelection(await invoke<NativeAudio>("sample_file"));
  const response = await fetch("/api/sample");
  if (!response.ok) throw new Error("The sample recording is unavailable.");
  return importBlob(await response.blob(), "A sample thought.wav");
}
export async function transcribe(
  audio: AudioSelection,
  job: string,
  onEvent: (event: Progress) => void,
  signal: AbortSignal,
): Promise<Result> {
  if (desktop) {
    const channel = new Channel<Progress>();
    channel.onmessage = onEvent;
    return invoke("transcribe_file", {
      path: audio.path,
      jobId: job,
      onEvent: channel,
    });
  }
  const response = await fetch("/api/transcribe-file", {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Audio-Name": encodeURIComponent(audio.name),
    },
    body: audio.blob,
    signal,
  });
  if (!response.ok)
    throw new Error(
      (await response.json()).error || "Could not import the recording.",
    );
  let result: Result | undefined;
  for await (const event of readMessages(response.body!)) {
    if (event.type === "error") throw new Error(event.error);
    if (event.type === "complete") result = event.result;
    else onEvent(event as Progress);
  }
  if (!result)
    throw new Error(
      "The local connection ended before transcription completed.",
    );
  return result;
}
export async function cancel(job: string) {
  if (desktop) await invoke("cancel_transcription", { jobId: job });
}
export async function loadHistory(): Promise<Transcript[]> {
  const entries = validateHistory(desktop ? await invoke("load_history") : JSON.parse(localStorage.getItem("pnola-transcripts") || "[]"));
  if (!desktop) return entries;
  const sessions = await invoke<LiveSnapshot[]>("list_live");
  const ids = new Set(sessions.map(session => session.id));
  return [...sessions.map(liveTranscript), ...entries.filter(entry => !ids.has(entry.id))]
    .sort((a,b) => b.date.localeCompare(a.date)).slice(0,20);

}
let persistence: Promise<unknown> = Promise.resolve();
export function saveHistory(entries: Transcript[]): Promise<unknown> {
  persistence = persistence
    .catch(() => {})
    .then(() =>
      desktop
        ? invoke("save_history", { entries })
        : localStorage.setItem("pnola-transcripts", JSON.stringify(entries)),
    );
  return persistence;
}
export async function copyTranscript(text: string) {
  if (desktop) await writeText(text);
  else await navigator.clipboard.writeText(text);
}
export async function exportTranscript(
  name: string,
  text: string,
): Promise<boolean> {
  const filename =
    (name
      .replace(/\.(wav|m4a|mp3|ogg|webm|flac|aac|mp4)$/i, "")
      .replace(/[^a-z0-9 _-]/gi, "")
      .trim() || "transcript") + ".txt";
  if (desktop) {
    const path = await save({
      defaultPath: filename,
      filters: [{ name: "Text", extensions: ["txt"] }],
    });
    if (!path) return false;
    await invoke("save_transcript", { path, text });
    return true;
  }
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/plain;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}
export function startNativeRecording(): Promise<void> {
  return invoke("start_recording");
}
export async function stopNativeRecording(): Promise<AudioSelection> {
  return nativeSelection(await invoke<NativeAudio>("stop_recording"));
}
export function captureState(): Promise<CaptureState> {
  return invoke("recording_state");
}
export async function onFileDrop(
  callback: (path: string) => void,
): Promise<() => void> {
  return getCurrentWebview().onDragDropEvent((event) => {
    if (event.payload.type === "drop" && event.payload.paths[0])
      callback(event.payload.paths[0]);
  });
}

export function liveTranscript(value: LiveSnapshot): Transcript {
  return { id: value.id, liveSessionId: value.id, name: value.name, date: value.date,
    text: value.text, words: value.words, duration: value.duration, execution: value.execution,
    queued: value.queued, partial: value.phase !== 'complete' };
}
export function startNativeLive(): Promise<LiveSnapshot> {
  return invoke('start_live', { id: crypto.randomUUID(), name: `Live note ${new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' })}`, date: new Date().toISOString() });
}
export function nativeLiveState(): Promise<LiveSnapshot | null> { return invoke('live_state'); }
export function stopNativeLive(): Promise<LiveSnapshot> { return invoke('stop_live'); }
export function resumeNativeLive(id:string): Promise<LiveSnapshot> { return invoke('resume_live', {id}); }
export function copyLive(id:string): Promise<void> { return invoke<string>('live_text', {id}).then(writeText); }
export async function exportLive(entry:Transcript): Promise<boolean> {
  const path = await save({defaultPath:entry.name.replace(/[^a-z0-9 _-]/gi,'')+'.txt',filters:[{name:'Text',extensions:['txt']}]});
  if (!path) return false;
  await invoke('export_live',{id:entry.liveSessionId,path}); return true;
}

export function deleteLive(id:string): Promise<void> { return invoke('delete_live',{id}); }
