import { useEffect, useRef, useState } from "react";
import * as backend from "../lib/api/backend";
import type { AudioSelection, CaptureState, LiveSnapshot } from "../lib/studio/types";

const idle: CaptureState = { phase: "idle", seconds: 0, level: 0 };
export function useRecorder(
  onAudio: (audio: AudioSelection) => void,
  onError: (error: unknown) => void,
  onLive?: (value: LiveSnapshot) => void,
) {
  const [state, setState] = useState<CaptureState>(idle);
  const [live, setLive] = useState<LiveSnapshot | null>(null);
  const current = useRef(state);
  current.current = state;
  const callbacks = useRef({ onAudio, onError, onLive });
  callbacks.current = { onAudio, onError, onLive };
  const media = useRef<{
    recorder: MediaRecorder;
    stream: MediaStream;
    context: AudioContext;
    analyser: AnalyserNode;
    started: number;
  } | null>(null);
  const mounted = useRef(true);
  useEffect(
    () => () => {
      mounted.current = false;
      media.current?.stream.getTracks().forEach((track) => track.stop());
      void media.current?.context.close();
      if (
        backend.desktop &&
        ["starting", "recording"].includes(current.current.phase)
      )
        void backend.stopNativeLive().catch(() => {});
    },
    [],
  );
  async function stop() {
    if (!["recording", "error", "stopped"].includes(current.current.phase))
      return;
    current.current = { ...current.current, phase: "stopping" };
    setState(current.current);
    try {
      if (backend.desktop) {
        const value = await backend.stopNativeLive();
        if (mounted.current) { setLive(value); callbacks.current.onLive?.(value); setState(idle); if (value.error) callbacks.current.onError(value.error); }

      } else {
        media.current?.recorder.stop();
      }
    } catch (error) {
      if (mounted.current) {
        setState(idle);
        callbacks.current.onError(error);
      }
    }
  }
  useEffect(() => {
    if (state.phase !== "recording" && !(backend.desktop && state.phase === "stopping")) return;
    let disposed = false,
      inFlight = false;
    const tick = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        if (backend.desktop) {
          const value = await backend.nativeLiveState();
          if (!disposed && value) {
            setLive(value); callbacks.current.onLive?.(value);
            if (value.phase === 'error') { callbacks.current.onError(value.error || 'Live transcription failed. Saved audio can be resumed.'); await stop(); }
            else if (value.phase === 'complete') setState(idle);
            else setState({phase: value.phase === 'recording' ? 'recording' : 'stopping', seconds:value.seconds,level:value.level});
          }

        } else if (media.current) {
          const recorder = media.current;
          const data = new Uint8Array(recorder.analyser.fftSize);
          recorder.analyser.getByteTimeDomainData(data);
          const level = Math.sqrt(
            data.reduce((sum, v) => sum + ((v - 128) / 128) ** 2, 0) /
              data.length,
          );
          const seconds = (Date.now() - recorder.started) / 1000;
          if (seconds >= 59) await stop();
          else if (!disposed)
            setState({
              phase: "recording",
              seconds,
              level: Math.sqrt(level),
              error: undefined,
            });
        }
      } catch (error) {
        if (!disposed) callbacks.current.onError(error);
      } finally {
        inFlight = false;
      }
    };
    const timer = setInterval(tick, 100);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [state.phase]);
  async function start() {
    if (current.current.phase !== "idle") return;
    current.current = { ...idle, phase: "starting" };
    setState(current.current);
    try {
      if (backend.desktop) {
        const value = await backend.startNativeLive();
        if (mounted.current) { setLive(value); callbacks.current.onLive?.(value); setState({ phase: "recording", seconds: 0, level: 0 }); }
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
        throw new Error(
          "Microphone recording is unavailable in this browser. Upload an audio file instead.",
        );
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true },
      });
      const mimeType = [
        "audio/webm;codecs=opus",
        "audio/mp4",
        "audio/webm",
      ].find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      const context = new AudioContext();
      await context.resume();
      const analyser = context.createAnalyser();
      analyser.fftSize = 128;
      context.createMediaStreamSource(stream).connect(analyser);
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        await context.close();
        const extension = recorder.mimeType.includes("mp4") ? "m4a" : "webm";
        try {
          const audio = await backend.importBlob(
            new Blob(chunks, { type: recorder.mimeType }),
            `Voice note ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.${extension}`,
          );
          audio.duration =
            (Date.now() - (media.current?.started || Date.now())) / 1000;
          if (mounted.current) {
            callbacks.current.onAudio(audio);
            setState(idle);
          }
        } catch (error) {
          if (mounted.current) {
            setState(idle);
            callbacks.current.onError(error);
          }
        }
        media.current = null;
      };
      recorder.onerror = () => {
        stream.getTracks().forEach((track) => track.stop());
        void context.close();
        setState(idle);
        callbacks.current.onError("Microphone recording failed.");
      };
      media.current = {
        recorder,
        stream,
        context,
        analyser,
        started: Date.now(),
      };
      recorder.start();
      setState({ phase: "recording", seconds: 0, level: 0 });
    } catch (error) {
      setState(idle);
      callbacks.current.onError(
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Microphone access was denied. Allow it in your privacy settings, or upload a file."
          : error,
      );
    }
  }
  return {
    state, live,
    resume: async (id: string) => {
      try { const value = await backend.resumeNativeLive(id); setLive(value); callbacks.current.onLive?.(value); setState({phase:'stopping', seconds:value.duration,level:0}); }
      catch (error) { callbacks.current.onError(error); }
    },
    start,
    stop,
    active: ["starting", "recording", "stopping"].includes(state.phase),
  };
}
