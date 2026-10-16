export interface EngineStatus {
  status: "loading" | "ready" | "unavailable";
  detail: string;
  busy: boolean;
  sample: boolean;
  transcription_available?:boolean;
  speakers_available?:boolean;
}
export interface AudioSelection {
  name: string;
  url: string;
  bytes: number;
  duration: number;
  path?: string;
  blob?: Blob;
  sha256?:string;
}
export interface Transcript {
  id: string;
  name: string;
  text: string;
  date: string;
  duration: number;
  execution: number;
  partial?: boolean;
  liveSessionId?: string;
  words?: number;
  queued?: number;
  attribution?:SpeakerAttribution;
  speakerNames?:Record<string,string>;
  audio_path?:string;
  audio_sha256?:string;
  asr_error?:string|null;
  wordsEdited?:boolean;
}
export interface LiveSnapshot {
  id: string; name: string; date: string; phase: 'recording' | 'stopping' | 'complete' | 'error' | 'interrupted';
  text: string; duration: number; execution: number; words: number;
  completed: number; queued: number; seconds: number; level: number; error?: string;
}
export interface Result {
  text: string;
  duration_seconds: number;
  execution_micros: number;
  parts?: number;
  attribution?:SpeakerAttribution;
  asr_error?:string|null;
  audio_path?:string;
  audio_sha256?:string;
}
export interface SpeakerTurn { start_seconds:number;end_seconds:number;speaker:number; }
export type WordAttribution = {kind:'speaker';speaker:number} | {kind:'overlap';speakers:number[]} | {kind:'unknown';reason:string};
export interface AttributedWord {text:string;start_seconds:number;end_seconds:number;attribution:WordAttribution;candidates:{speaker:number;fraction:number}[];}
export type SpeakerAttribution = {status:'unavailable';reason:string} | {
  status:'complete';session_id:string;revision:number;complete:boolean;
  timeline:{turns:SpeakerTurn[];audio_duration_seconds:number;model_revision:string;runtime_version:string;backend:string;geometry:Record<string,number>};
  words:AttributedWord[];words_complete:boolean;timing_method:string;alignment_error:string|null;
};
export type Progress =
  | {
      type: "progress";
      stage: "converting" | "transcribing" | "diarizing";
      accepted_audio_seconds?:number;
      total_audio_seconds?:number;
      index?: number;
      total?: number;
    }
  | { type: "part"; index: number; total: number; result: Result }
  | { type:"timeline"; attribution:SpeakerAttribution };
export type Phase = "idle" | "preparing" | "transcribing" | "cancelling";
export interface CaptureState {
  phase: "idle" | "starting" | "recording" | "stopping" | "stopped" | "error";
  seconds: number;
  level: number;
  error?: string;
}
