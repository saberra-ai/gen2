import type {
  AudioSelection,
  EngineStatus,
  Phase,
  Progress,
  Transcript,
  SpeakerAttribution,
  LiveSpeakers,
} from "./types";
export interface StudioState {
  phase: Phase;
  engine: EngineStatus;
  audio: AudioSelection | null;
  current: Transcript | null;
  history: Transcript[];
  historyReady: boolean;
  job: string | null;
  progress: Progress | null;
  error: string | null;
}
export const initialState: StudioState = {
  phase: "idle",
  engine: {
    status: "loading",
    detail: "Loading local engine…",
    busy: false,
    sample: false,
  },
  audio: null,
  current: null,
  history: [],
  historyReady: false,
  job: null,
  progress: null,
  error: null,
};
export type Action =
  | { type: "engine"; value: EngineStatus }
  | { type: "live"; entry: Transcript }
  | { type: "history"; entries: Transcript[] }
  | { type: "preparing" }
  | { type: "audio"; value: AudioSelection | null }
  | { type: "reset" }
  | { type: "begin"; job: string; name: string }
  | { type: "progress"; job: string; value: Progress }
  | { type: "complete"; job: string; entry: Transcript;save?:boolean }
  | { type:"save-current" }
  | { type:"rename-speaker";entryId:string;sessionId:string;speaker:number;name:string }
  | { type: "cancelling" }
  | { type: "cancelled"; job: string }
  | { type: "error"; message: string; job?: string }
  | { type: "edit"; text: string }
  | { type: "open"; id: string;audio?:AudioSelection|null }
  | { type: "delete"; id: string };
export function reducer(state: StudioState, action: Action): StudioState {
  switch (action.type) {
    case "live": {
      const previous=state.history.find(entry=>entry.id===action.entry.id);
      const sameSession=!!previous?.liveSpeakers?.session_id&&previous.liveSpeakers.session_id===action.entry.liveSpeakers?.session_id;
      if(sameSession&&previous!.liveSpeakers!.revision>action.entry.liveSpeakers!.revision)return state;
      if(sameSession&&previous!.liveSpeakers!.status!=='active'&&action.entry.liveSpeakers!.status==='active')return state;
      const entry=sameSession?{...action.entry,speakerNames:previous?.speakerNames}:action.entry;
      return { ...state, audio: state.current?.id === action.entry.id ? state.audio : null,
        current: entry, error: null,
        history: [entry, ...state.history.filter(value => value.id !== entry.id)].slice(0,20) };
    }
    case "engine":
      return { ...state, engine: action.value };
    case "history":
      return {
        ...state,
        history: action.entries.slice(0, 20),
        historyReady: true,
      };
    case "preparing":
      return { ...state, phase: "preparing", error: null };
    case "audio":
      return { ...state, audio: action.value, phase: "idle", error: null };
    case "reset":
      return {
        ...state,
        audio: null,
        current: null,
        phase: "idle",
        error: null,
        progress: null,
        job: null,
      };
    case "begin":
      return {
        ...state,
        phase: "transcribing",
        job: action.job,
        progress: null,
        error: null,
        current: {
          id: "",
          name: action.name,
          text: "",
          date: "",
          duration: 0,
          execution: 0,
          partial: true,
        },
      };
    case "progress": {
      if (state.job !== action.job) return state;
      if (action.value.type === "progress")
        return { ...state, progress: action.value };
      if(action.value.type==='timeline') return {...state,current:state.current?{...state.current,attribution:action.value.attribution}:null};
      const result = action.value.result,
        current = state.current!;
      return {
        ...state,
        progress: action.value,
        current: {
          ...current,
          text: [current.text, result.text].filter(Boolean).join("\n\n"),
          duration: current.duration + result.duration_seconds,
          execution: current.execution + result.execution_micros,
          partial: true,
        },
      };
    }
    case "complete":
      return state.job !== action.job
        ? state
        : {
            ...state,
            phase: "idle",
            job: null,
            progress: null,
            current: action.entry,
            history: action.save===false?state.history:[action.entry, ...state.history].slice(0, 20),
          };
    case 'save-current':
      return !state.current||state.phase!=='idle'?state:{...state,history:[state.current,...state.history.filter(entry=>entry.id!==state.current!.id)].slice(0,20)};
    case 'rename-speaker': {
      const current=state.current;
      const session=current?.attribution?.status==='complete'?current.attribution:current?.liveSpeakers;
      if(!current||state.phase!=='idle'||current.id!==action.entryId||session?.session_id!==action.sessionId||!session?.timeline?.turns.some(t=>t.speaker===action.speaker)||action.name.length>80) return state;
      const updated={...current,speakerNames:{...current.speakerNames,[String(action.speaker)]:action.name}};
      return {...state,current:updated,history:state.history.map(entry=>entry.id===updated.id?updated:entry)};
    }
    case "cancelling":
      return { ...state, phase: "cancelling" };
    case "cancelled":
      return state.job !== action.job
        ? state
        : { ...state, phase: "idle", job: null, progress: null, error: null };
    case "error":
      return action.job && state.job !== action.job
        ? state
        : {
            ...state,
            phase: "idle",
            job: null,
            error: action.message,
            progress: null,
          };
    case "edit": {
      if (!state.current || state.phase !== "idle" || state.current.liveSessionId) return state;
      const attribution=state.current.attribution;
      const current = { ...state.current, text: action.text,wordsEdited:true,attribution:attribution?.status==='complete'?{...attribution,words:[],words_complete:false}:attribution };
      return {
        ...state,
        current,
        history: state.history.map((entry) =>
          entry.id === current.id ? current : entry,
        ),
      };
    }
    case "open":
      return {
        ...state,
        audio:action.audio||null,
        current:
          state.history.find((entry) => entry.id === action.id) ||
          state.current,
        error: null,
      };
    case "delete":
      return {
        ...state,
        history: state.history.filter((entry) => entry.id !== action.id),
      };
  }
}
export function validateHistory(value: unknown): Transcript[] {
  if (!Array.isArray(value))
    throw new Error(
      "Saved history is not a transcript list. Your existing file has been preserved.",
    );
  return value
    .filter(
      (entry): entry is Transcript =>
        !!entry &&
        typeof entry.id === "string" &&
        typeof entry.name === "string" &&
        typeof entry.text === "string" &&
        typeof entry.date === "string" &&
        Number.isFinite(entry.duration) &&
        Number.isFinite(entry.execution),
    )
    .slice(0, 20)
    .map(entry=>{
      const {attribution,speakerNames,liveSpeakers,...original}=entry;
      const valid=validateAttribution(attribution);
      const live=validateLiveSpeakers(liveSpeakers);
      const names=speakerNames&&typeof speakerNames==='object'?Object.fromEntries(Object.entries(speakerNames).filter(([id,name])=>/^[1-8]$/.test(id)&&typeof name==='string'&&name.length<=80)):undefined;
      return {...original,...(valid?{attribution:valid}:{}),...(live?{liveSpeakers:live}:{}),...(names?{speakerNames:names}:{})};
    });
}

function validateLiveSpeakers(value:unknown):LiveSpeakers|undefined {
  if(!value||typeof value!=='object')return undefined;
  const v=value as LiveSpeakers;
  if(!['unavailable','active','complete','interrupted'].includes(v.status)||
    !(v.session_id===null||typeof v.session_id==='string'&&v.session_id.length<=64)||
    !Number.isSafeInteger(v.revision)||v.revision<0||
    !Number.isFinite(v.accepted_audio_seconds)||v.accepted_audio_seconds<0||v.accepted_audio_seconds>3600+1e-6||
    !Number.isFinite(v.processed_audio_seconds)||v.processed_audio_seconds<0||v.processed_audio_seconds>3600+1e-6||
    !(v.reason===null||typeof v.reason==='string'))return undefined;
  if(v.timeline){
    const t=v.timeline;
    if(!Number.isFinite(t.audio_duration_seconds)||t.audio_duration_seconds<=0||t.audio_duration_seconds>3600||
      typeof t.model_revision!=='string'||typeof t.runtime_version!=='string'||typeof t.backend!=='string'||
      !Array.isArray(t.geometry)||t.geometry.length!==6||!t.geometry.every(Number.isInteger)||
      !Array.isArray(t.turns)||t.turns.length>100000)return undefined;
    let previous=0;
    for(const turn of t.turns){if(!turn||!Number.isFinite(turn.start_seconds)||!Number.isFinite(turn.end_seconds)||turn.start_seconds<previous||turn.end_seconds<turn.start_seconds||turn.end_seconds>t.audio_duration_seconds||!Number.isInteger(turn.speaker)||turn.speaker<1||turn.speaker>8)return undefined;previous=turn.start_seconds;}
  }else if(v.timeline!==null||v.status==='complete')return undefined;
  return v;
}

function validateAttribution(value:unknown):SpeakerAttribution|undefined {
  if(!value||typeof value!=='object')return undefined;
  const v=value as SpeakerAttribution;
  if(v.status==='unavailable')return typeof v.reason==='string'?v:undefined;
  if(v.status!=='complete'||typeof v.session_id!=='string'||!Number.isSafeInteger(v.revision)||v.revision<1||v.complete!==true||!v.timeline||!Number.isFinite(v.timeline.audio_duration_seconds)||v.timeline.audio_duration_seconds<=0||v.timeline.audio_duration_seconds>3600||typeof v.timeline.model_revision!=='string'||typeof v.timeline.runtime_version!=='string'||typeof v.timeline.backend!=='string'||!Array.isArray(v.timeline.turns)||v.timeline.turns.length>100000||!Array.isArray(v.words)||v.words.length>100000||typeof v.words_complete!=='boolean')return undefined;
  let previous=0;
  for(const turn of v.timeline.turns){if(!turn||!Number.isFinite(turn.start_seconds)||!Number.isFinite(turn.end_seconds)||turn.start_seconds<previous||turn.end_seconds<turn.start_seconds||turn.end_seconds>v.timeline.audio_duration_seconds||!Number.isInteger(turn.speaker)||turn.speaker<1||turn.speaker>8)return undefined;previous=turn.start_seconds;}
  for(const word of v.words){
    if(!word||typeof word.text!=='string'||!Number.isFinite(word.start_seconds)||!Number.isFinite(word.end_seconds)||word.start_seconds<0||word.end_seconds<word.start_seconds||word.end_seconds>v.timeline.audio_duration_seconds||!word.attribution)return undefined;
    const a=word.attribution;
    if(a.kind==='speaker'){if(!Number.isInteger(a.speaker)||a.speaker<1||a.speaker>8)return undefined;}
    else if(a.kind==='overlap'){if(!Array.isArray(a.speakers)||a.speakers.length>8||a.speakers.some(id=>!Number.isInteger(id)||id<1||id>8))return undefined;}
    else if(a.kind!=='unknown'||typeof a.reason!=='string')return undefined;
  }
  return v;
}
export function formatTime(value: number) {
  const seconds = Math.max(0, Math.floor(value || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
