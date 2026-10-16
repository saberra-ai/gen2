import { useEffect, useReducer, useRef, useState } from "react";
import * as backend from "../lib/api/backend";
import { initialState, reducer } from "../lib/studio/state";
import type { AudioSelection, LiveSnapshot } from "../lib/studio/types";
export function useStudio() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [toast, setToast] = useState("");
  const active = useRef<{
    id: string;
    abort: AbortController;
    cancelled: boolean;
  } | null>(null);
  const stateRef = useRef(state);
  const historyOpen=useRef(0);
  stateRef.current = state;
  useEffect(() => {
    let disposed = false;
    const poll = async () => {
      try {
        const value = await backend.engineStatus();
        if (!disposed) dispatch({ type: "engine", value });
      } catch {
        if (!disposed)
          dispatch({
            type: "engine",
            value: {
              status: "unavailable",
              detail: "Local engine disconnected. Restart Pnola.",
              busy: false,
              sample: false,
            },
          });
      }
    };
    void poll();
    const timer = setInterval(poll, 2000);
    void backend
      .loadHistory()
      .then((entries) => {
        if (!disposed) dispatch({ type: "history", entries });
      })
      .catch((error) => {
        if (!disposed) dispatch({ type: "error", message: String(error) });
      });
    return () => {
      disposed = true;
      clearInterval(timer);
      active.current?.abort.abort();
      if (active.current) void backend.cancel(active.current.id);
    };
  }, []);
  useEffect(() => {
    if (!state.historyReady) return;
    const timer = setTimeout(() => {
      void backend
        .saveHistory(state.history)
        .catch((error) =>
          dispatch({
            type: "error",
            message: `Could not save history: ${String(error)}`,
          }),
        );
    }, 200);
    return () => clearTimeout(timer);
  }, [state.history, state.historyReady]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 2500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    const url = state.audio?.url;
    return () => {
      if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
    };
  }, [state.audio?.url]);
  const fail = (error: unknown) =>
    dispatch({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  async function prepare(loader: () => Promise<AudioSelection | null>) {
    if (stateRef.current.phase !== "idle") return;
    historyOpen.current++;
    dispatch({ type: "preparing" });
    try {
      const value = await loader();
      if (value) dispatch({ type: "audio", value });
      else dispatch({ type: "audio", value: stateRef.current.audio });
    } catch (error) {
      fail(error);
    }
  }
  useEffect(() => {
    if (!backend.desktop) return;
    let unlisten: (() => void) | undefined,
      disposed = false;
    void backend
      .onFileDrop((path) => {
        if (stateRef.current.phase === "idle")
          void prepare(() => backend.importPath(path));
      })
      .then((stop) => {
        if (disposed) stop();
        else unlisten = stop;
      });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);
  async function run() {
    const audio = stateRef.current.audio;
    if (!audio || active.current) return;
    historyOpen.current++;
    const operation = {
      id: crypto.randomUUID(),
      abort: new AbortController(),
      cancelled: false,
    };
    active.current = operation;
    dispatch({ type: "begin", job: operation.id, name: audio.name });
    try {
      const result = await backend.transcribe(
        audio,
        operation.id,
        (value) => dispatch({ type: "progress", job: operation.id, value }),
        operation.abort.signal,
      );
      if (operation.cancelled) {
        dispatch({ type: "cancelled", job: operation.id });
        return;
      }
      dispatch({
        type: "complete",
        job: operation.id,
        save:!result.asr_error||!!result.text.trim(),
        entry: {
          id: crypto.randomUUID(),
          name: audio.name,
          text: result.text,
          date: new Date().toISOString(),
          duration: result.duration_seconds,
          execution: result.execution_micros,
          partial:!!result.asr_error,
          asr_error:result.asr_error,
          attribution:result.attribution,
          audio_path:result.audio_path||audio.path,
          audio_sha256:result.audio_sha256||audio.sha256,
        },
      });
      if (result.asr_error) setToast('Transcription failed. The speaker timeline remains available.');
      else if (!result.text.trim())
        setToast("Bark returned an empty transcript. Try a clearer recording.");
    } catch (error) {
      if (operation.cancelled || operation.abort.signal.aborted)
        dispatch({ type: "cancelled", job: operation.id });
      else
        dispatch({
          type: "error",
          job: operation.id,
          message: error instanceof Error ? error.message : String(error),
        });
    } finally {
      if (active.current === operation) active.current = null;
    }
  }
  async function stop() {
    const operation = active.current;
    if (!operation) return;
    operation.cancelled = true;
    dispatch({ type: "cancelling" });
    operation.abort.abort();
    try {
      await backend.cancel(operation.id);
    } catch (error) {
      fail(error);
    }
  }
  return {
    state,
    dispatch,
    reset:()=>{historyOpen.current++;dispatch({type:'reset'});},
    toast,
    fail,
    run,
    stop,
    choose: () => prepare(backend.chooseAudio),
    importFile: (file: File) =>
      prepare(() => backend.importBlob(file, file.name)),
    sample: () => prepare(backend.sampleAudio),
    remove: async (id:string) => {
      const entry = stateRef.current.history.find(entry => entry.id === id);
      try { if (entry?.liveSessionId) await backend.deleteLive(entry.liveSessionId); dispatch({type:'delete',id}); }
      catch (error) { fail(error); }
    },
    acceptLive: (value: LiveSnapshot) => dispatch({type: "live", entry: backend.liveTranscript(value)}),
    acceptRecording: (audio: AudioSelection) =>{
      historyOpen.current++;
      dispatch({ type: "audio", value: audio });
    },
    open:async(id:string)=>{
      if(stateRef.current.phase!=='idle')return;
      const request=++historyOpen.current;
      const entry=stateRef.current.history.find(entry=>entry.id===id);
      if(!entry)return;
      let audio:AudioSelection|null=null;
      if(backend.desktop&&entry.audio_path&&entry.audio_sha256){
        try{
          const candidate=await backend.importPath(entry.audio_path);
          if(candidate.sha256===entry.audio_sha256)audio=candidate;
          else setToast('The original audio has changed. Playback is unavailable.');
        }catch{setToast('The original audio is unavailable. Your transcript is preserved.');}
      }
      if(request===historyOpen.current&&stateRef.current.phase==='idle')dispatch({type:'open',id,audio});
    },
    copy: async () => {
      try {
        if (state.current?.liveSessionId) await backend.copyLive(state.current.liveSessionId); else await backend.copyTranscript(state.current?.text || "");
        setToast("Your words, copied.");
      } catch (error) {
        fail(error);
      }
    },
    export: async () => {
      try {
        if (
          state.current &&
          ((state.current.liveSessionId ? await backend.exportLive(state.current) : await backend.exportTranscript(
            state.current.name,
            state.current.text,
          )))
        )
          setToast("Transcript saved.");
      } catch (error) {
        fail(error);
      }
    },
  };
}
