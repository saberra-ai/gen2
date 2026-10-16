import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Icon, IconLibrary } from "@/components/studio/icons";
import { useStudio } from "@/hooks/use-studio";
import { useRecorder } from "@/hooks/use-recorder";
import { formatTime } from "@/lib/studio/state";
import { desktop } from "@/lib/api/backend";
import { LiveSpeakersPanel } from "@/components/studio/live-speakers-panel";
import { SpeakerPanel } from "@/components/studio/speaker-panel";

export function App() {
  const studio = useStudio(),
    { state } = studio;
  const recorder = useRecorder(studio.acceptRecording, studio.fail, studio.acceptLive);
  const [mode, setMode] = useState<"record" | "upload">("record");
  const [dragging, setDragging] = useState(false);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const player=useRef<HTMLAudioElement>(null);
  const playbackMatches=!!state.audio&&!!state.current?.audio_sha256&&state.audio.sha256===state.current.audio_sha256;
  const busy = state.phase !== "idle",
    locked = busy || recorder.active;
  const progress = state.progress;
  const speakerProgress=progress?.type==='progress'&&progress.stage==='diarizing'?progress:null;
  const part =
      progress&&'index' in progress ? progress.index||0:0,
    total = progress&&'total' in progress?progress.total||0:0;
  const text = state.current?.text || "",
    words = state.current?.words ?? (text.trim() ? text.trim().split(/\s+/u).length : 0);
  const preparing = state.phase === "preparing";
  const processing =
    state.phase === "transcribing" || state.phase === "cancelling";
  const engineText = desktop && recorder.active
    ? `Live transcription · ${recorder.live?.queued || 0} sections queued`
    : processing
    ? state.phase === "cancelling"
      ? "Stopping transcription…"
      : speakerProgress?`Finding speakers locally${speakerProgress.total_audio_seconds?` · ${formatTime(speakerProgress.accepted_audio_seconds||0)} / ${formatTime(speakerProgress.total_audio_seconds)}`:'…'}`:total
        ? `Transcribing part ${Math.min(part + 1, total)} of ${total}…`
        : "Preparing audio locally…"
    : state.engine.detail;
  const capture = recorder.state;
  const recording = capture.phase === "recording";
  const title = preparing
    ? "Preparing your audio…"
    : capture.phase === "starting"
      ? "Starting microphone…"
      : capture.phase === "stopping"
        ? "Finishing your recording…"
        : recording
          ? "Keep talking. We’re transcribing."
          : state.audio
            ? "A thought, ready to become words."
            : "Make room for a thought.";
  const browse = () => {
    if (locked) return;
    if (desktop) void studio.choose();
    else input.current?.click();
  };
  function reset() {
    studio.reset();
    setPlaybackDuration(0);
  }
  return (
    <>
      <IconLibrary />
      <div className="app-shell">
        <aside className="sidebar">
          <button
            className="brand"
            aria-label="Pnola home"
            disabled={locked}
            onClick={reset}
          >
            <span className="brand-mark">
              <Icon name="wave" />
            </span>
            pnola<span className="brand-dot">.</span>
          </button>
          <Button
            id="new-session"
            className="new-button"
            onClick={reset}
            disabled={locked}
          >
            <Icon name="plus" />
            New transcription<span>↗</span>
          </Button>
          <div className="nav-item">
            <Icon name="wave" />
            Transcription studio
            <span className="tiny-dot" />
          </div>
          <div className="recent-heading">
            <span>RECENT TRANSCRIPTS</span>
            <span id="history-count">{state.history.length}</span>
          </div>
          <div id="history" className="history-list">
            {!state.history.length && (
              <p className="history-empty">
                Your recent thoughts will live here.
              </p>
            )}
            {state.history.map((entry) => (
              <div
                className={`history-item${state.current?.id === entry.id ? " selected" : ""}`}
                key={entry.id}
              >
                <button
                  disabled={locked}
                  onClick={() =>
                    void studio.open(entry.id)
                  }
                >
                  <strong>{entry.name}</strong>
                  <small>
                    {new Date(entry.date).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                    })}{" "}
                    · {formatTime(entry.duration)}
                  </small>
                </button>
                <button
                  disabled={locked}
                  className="delete-history"
                  aria-label={`Delete ${entry.name}`}
                  onClick={() =>
                    void studio.remove(entry.id)
                  }
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <div className="sidebar-bottom">
            <span className="local-icon">
              <Icon name="lock" />
            </span>
            <div>
              <strong>Just you and your words.</strong>
              <p>Audio stays on this device.</p>
            </div>
          </div>
          <div className="version">
            PNOLA / {desktop ? "DESKTOP" : "LOCAL"} STUDIO <span>v0.2</span>
          </div>
        </aside>
        <main>
          <header className="topbar">
            <div>
              <span className="breadcrumb">Workspace</span>
              <span className="slash">/</span>
              <span>Transcription studio</span>
            </div>
            <Badge className="local-badge">
              <span />
              {desktop ? "On your Mac" : "Local inference"}
            </Badge>
          </header>
          <div className="workspace">
            <div className="page-heading">
              <div>
                <div className="eyebrow">THINK OUT LOUD.</div>
                <h1>Your voice, in writing.</h1>
                <p>
                  A small studio for big thoughts. Record, transcribe, make it
                  yours.
                </p>
              </div>
              <div className="engine-badge">
                <Icon name="wave" />
                <div>
                  <strong>Bark 2.0</strong>
                  <span>Powered by Gen2</span>
                </div>
              </div>
            </div>
            <div className="studio-grid">
              <Card
                role="region"
                className="capture-panel"
                aria-labelledby="capture-title"
              >
                <div className="panel-heading">
                  <span className="step-number">01</span>
                  <h2 id="capture-title">Your audio</h2>
                  <span className="language">ENGLISH</span>
                </div>
                <div
                  className="mode-tabs"
                  role="tablist"
                  aria-label="Audio source"
                >
                  {(["record", "upload"] as const).map((tab, index) => (
                    <button
                      key={tab}
                      id={`${tab}-tab`}
                      role="tab"
                      className={mode === tab ? "active" : ""}
                      aria-selected={mode === tab}
                      aria-controls={`${tab}-view`}
                      disabled={locked}
                      onClick={() => setMode(tab)}
                      onKeyDown={(event) => {
                        if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
                          event.preventDefault();
                          setMode(index ? "record" : "upload");
                        }
                      }}
                    >
                      <Icon name={tab === "record" ? "mic" : "upload"} />
                      {tab === "record" ? (desktop ? "Go live" : "Record audio") : "Upload a file"}
                    </button>
                  ))}
                </div>
                {mode === "record" ? (
                  <div
                    id="record-view"
                    role="tabpanel"
                    aria-labelledby="record-tab"
                    className={`record-view${recording ? " recording" : ""}`}
                  >
                    <div className="record-label">
                      <span />
                      {recording
                        ? (desktop ? "LIVE TRANSCRIPTION" : "LISTENING TO YOUR THOUGHT")
                        : capture.phase === "starting"
                          ? "STARTING MICROPHONE"
                          : "READY WHEN YOU ARE"}
                    </div>
                    <div className="waveform" aria-hidden="true">
                      {Array.from({ length: 43 }, (_, i) => (
                        <i
                          key={i}
                          style={{
                            height: recording
                              ? `${Math.max(5, capture.level * 56 * (0.4 + 0.6 * Math.sin(i * 1.7) ** 2))}px`
                              : `${7 + Math.sin(i * 1.7) ** 2 * 23 * Math.sin(((i + 1) / 44) * Math.PI)}px`,
                          }}
                        />
                      ))}
                    </div>
                    <div className="timer">
                      {formatTime(
                        recording
                          ? capture.seconds
                          : state.current?.liveSessionId ? state.current.duration : state.audio?.duration || playbackDuration,
                      )}
                      <span>{desktop ? " live" : " / 01:00"}</span>
                    </div>
                    <Button
                      id="record-button"
                      className="record-button"
                      aria-label={
                        recording ? "Stop live transcription" : "Start live transcription"
                      }
                      disabled={
                        busy || state.engine.status !== "ready" || state.engine.transcription_available===false ||
                        capture.phase === "starting" ||
                        capture.phase === "stopping"
                      }
                      onClick={() =>
                        void (recording ? recorder.stop() : recorder.start())
                      }
                    >
                      {recording ? (
                        <span className="stop-square" />
                      ) : (
                        <Icon name="mic" />
                      )}
                    </Button>
                    <h3>{title}</h3>
                    <p>
                      {recording ? (
                        "Words arrive as you speak. Stop whenever you’re done."
                      ) : state.audio ? (
                        "Listen back, then transcribe when you’re ready."
                      ) : (
                        <>
                          Tap the microphone and start speaking.
                          <br />
                          {desktop
                            ? "Live transcription. No session time limit."
                            : "One idea, up to 60 seconds."}
                        </>
                      )}
                    </p>
                  </div>
                ) : (
                  <div
                    id="upload-view"
                    role="tabpanel"
                    aria-labelledby="upload-tab"
                    className="upload-view"
                  >
                    <button
                      id="dropzone"
                      className={`dropzone${dragging ? " dragover" : ""}`}
                      aria-label="Browse audio files"
                      disabled={locked}
                      onClick={browse}
                      onDragOver={(event) => {
                        event.preventDefault();
                        if (!locked) setDragging(true);
                      }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={(event) => {
                        event.preventDefault();
                        setDragging(false);
                        const file = event.dataTransfer.files[0];
                        if (file && !locked && !desktop)
                          void studio.importFile(file);
                      }}
                    >
                      <span className="upload-icon">
                        <Icon name="upload" />
                      </span>
                      <strong>
                        {preparing
                          ? "Preparing your audio…"
                          : "Drop your audio here"}
                      </strong>
                      <span>
                        or <u>browse files</u>
                      </span>
                      <small>
                        WAV, MP3, M4A, OGG, WebM · up to 2 hours / 256 MB
                      </small>
                    </button>
                  </div>
                )}
                <input
                  ref={input}
                  id="file-input"
                  type="file"
                  accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac,.aac"
                  hidden
                  disabled={locked}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void studio.importFile(file);
                    event.target.value = "";
                  }}
                />
                {state.audio && (
                  <div className="audio-preview">
                    <div className="file-summary">
                      <Icon name="file" />
                      <div>
                        <strong id="file-name">{state.audio.name}</strong>
                        <span id="file-meta">
                          {formatTime(state.audio.duration || playbackDuration)}{" "}
                          · {(state.audio.bytes / 1024 / 1024).toFixed(1)} MB ·
                          processed in short sections
                        </span>
                      </div>
                      <Button
                        className="icon-button"
                        aria-label="Remove audio"
                        disabled={locked}
                        onClick={() => {
                          studio.dispatch({ type: "audio", value: null });
                          setPlaybackDuration(0);
                        }}
                      >
                        <Icon name="x" />
                      </Button>
                    </div>
                    <audio
                      ref={player}
                      key={state.audio.url}
                      id="player"
                      src={state.audio.url}
                      controls
                      onLoadedMetadata={(event) =>
                        setPlaybackDuration(
                          Number.isFinite(event.currentTarget.duration)
                            ? event.currentTarget.duration
                            : 0,
                        )
                      }
                    />
                  </div>
                )}
                <div className="capture-footer">
                  <div className="input-settings">
                    <span>
                      Model <strong>Bark 2.0</strong>
                    </span>
                    <span>
                      Language <strong>English</strong>
                    </span>
                  </div>
                  <Button
                    id="transcribe"
                    className="primary-button"
                    disabled={
                      !state.audio || locked || state.engine.status !== "ready"
                    }
                    onClick={() => void studio.run()}
                  >
                    <span>
                      {processing
                        ? speakerProgress?'Finding speakers…':total
                          ? `Transcribing ${Math.min(part + 1, total)} / ${total}…`
                          : "Preparing audio…"
                        : "Transcribe audio"}
                    </span>
                    <Icon name={processing ? "wave" : "arrow"} />
                  </Button>
                  {desktop && state.current?.liveSessionId && (state.current.queued || 0) > 0 && !locked && (
                    <Button className="primary-button resume-live" onClick={() => void recorder.resume(state.current!.liveSessionId!)}>Finish saved live audio</Button>
                  )}
                  {processing && (
                    <div className="job-progress">
                      <progress
                        aria-label="Transcription progress"
                        value={speakerProgress?.total_audio_seconds?speakerProgress.accepted_audio_seconds:total ? part : undefined}
                        max={speakerProgress?.total_audio_seconds||total||1}
                      />
                      <Button
                        className="cancel-button"
                        disabled={state.phase === "cancelling"}
                        onClick={() => void studio.stop()}
                      >
                        {state.phase === "cancelling"
                          ? "Stopping…"
                          : "Stop transcription"}
                      </Button>
                    </div>
                  )}
                  <p className="local-note">
                    <Icon name="lock" />
                    Processed locally. No audio uploads to the cloud.
                  </p>
                </div>
              </Card>
              <Card
                role="region"
                className={`transcript-panel${processing ? " processing" : ""}`}
                aria-labelledby="transcript-title"
              >
                <div className="panel-heading">
                  <span className="step-number">02</span>
                  <h2 id="transcript-title">Your words</h2>
                  <span id="transcript-state" className="transcript-state">
                    {desktop && recorder.active ? "LIVE" : processing
                      ? total
                        ? `PART ${part} / ${total}`
                        : "PROCESSING"
                      : state.current
                        ? state.current.partial
                          ? "PARTIAL"
                          : state.current.liveSessionId ? "SAVED LIVE" : "EDITABLE"
                        : "TRANSCRIPT"}
                  </span>
                </div>
                <div className="transcript-toolbar">
                  <span id="transcript-name">
                    {state.current?.name || "Untitled transcription"}
                  </span>
                  <div>
                    <Button
                      className="icon-button"
                      aria-label="Copy transcript"
                      disabled={!text || processing}
                      onClick={() => void studio.copy()}
                    >
                      <Icon name="copy" />
                    </Button>
                    <Button
                      className="icon-button"
                      aria-label="Download transcript"
                      disabled={!text || processing}
                      onClick={() => void studio.export()}
                    >
                      <Icon name="download" />
                    </Button>
                  </div>
                </div>
                <div className="transcript-body">
                  {state.current?.attribution&&<SpeakerPanel key={state.current.id} entry={state.current} disabled={processing} saved={state.history.some(entry=>entry.id===state.current?.id)} canSeek={playbackMatches} onSeek={seconds=>{if(playbackMatches&&player.current){player.current.currentTime=seconds;void player.current.play().catch(()=>{});}}} onRename={(speaker,name)=>{if(state.current?.attribution?.status==='complete')studio.dispatch({type:'rename-speaker',entryId:state.current.id,sessionId:state.current.attribution.session_id,speaker,name});}} onSave={()=>studio.dispatch({type:'save-current'})}/>}
                  {state.current?.liveSpeakers&&<LiveSpeakersPanel key={state.current.id} entry={state.current} onRename={(speaker,name)=>{const sessionId=state.current?.liveSpeakers?.session_id;if(sessionId&&state.current)studio.dispatch({type:'rename-speaker',entryId:state.current.id,sessionId,speaker,name});}}/>}
                  {state.current && (!processing || state.current.text) ? (
                    <Textarea
                      id="transcript"
                      aria-label="Editable transcript"
                      spellCheck
                      value={state.current.text}
                      readOnly={processing || !!state.current.liveSessionId}
                      onChange={(event) =>
                        studio.dispatch({
                          type: "edit",
                          text: event.target.value,
                        })
                      }
                    />
                  ) : (
                    <div className="empty-state">
                      <span className="empty-art">
                        <Icon name="wave" />
                      </span>
                      <h3>
                        {processing
                          ? "Making room for your words…"
                          : "Good thoughts start here."}
                      </h3>
                      <p>
                        {processing ? (
                          "Your transcript appears as each section finishes."
                        ) : (
                          <>
                            Your transcript will appear in this space.
                            <br />
                            Record a thought or upload some audio.
                          </>
                        )}
                      </p>
                      {state.engine.sample && !processing && (
                        <Button
                          className="sample-button"
                          disabled={locked || state.engine.status !== "ready"}
                          onClick={() => {
                            setMode("upload");
                            void studio.sample();
                          }}
                        >
                          Try a sample recording <span>↗</span>
                        </Button>
                      )}
                    </div>
                  )}
                </div>
                <div className="transcript-footer">
                  <span id="word-count">
                    {words} {words === 1 ? "word" : "words"}
                  </span>
                  <span id="transcript-meta">
                    {state.current ? (
                      state.current.liveSessionId ? `${formatTime(state.current.duration)} transcribed · ${state.current.queued || 0} queued · latest sections shown` : `${formatTime(state.current.duration)} audio · ${(state.current.execution / 1_000_000).toFixed(1)}s processing`
                    ) : (
                      <>
                        Ready for your first thought
                        <span className="small-spark">✳</span>
                      </>
                    )}
                  </span>
                </div>
              </Card>
            </div>
            <div className="bottom-row">
              <span
                id="engine-status"
                className={state.engine.status}
                role="status"
              >
                <span className="status-dot" />
                <span id="engine-text">{engineText}</span>
              </span>
              <span>YOUR VOICE, A LITTLE CLEARER.</span>
            </div>
            {state.error && (
              <div id="message" role="alert">
                {state.error}
              </div>
            )}
          </div>
        </main>
      </div>
      {studio.toast && (
        <div id="toast" role="status">
          {studio.toast}
        </div>
      )}
    </>
  );
}
