import { useState } from 'react';
import type { Transcript } from '@/lib/studio/types';
import { formatTime } from '@/lib/studio/state';

export function LiveSpeakersPanel({entry,onRename}:{entry:Transcript;onRename:(speaker:number,name:string)=>void}) {
  const [shown,setShown]=useState(50);
  const snapshot=entry.liveSpeakers;
  if(!snapshot)return null;
  const turns=snapshot.timeline?.turns||[];
  const speakers=[...new Set(turns.map(turn=>turn.speaker))].sort();
  const names=entry.speakerNames||{};
  const label=(id:number)=>names[String(id)]?.trim()||`Speaker ${id}`;
  return <section className="speaker-panel" aria-label="Live speaker timeline">
    <div className="speaker-heading"><strong>Speakers</strong><span>{snapshot.status==='complete'?'Final timeline':snapshot.status==='active'?'Provisional timeline':'Incomplete timeline'} · {speakers.length} detected</span></div>
    <p className="speaker-note">{snapshot.status==='active'?'Live labels can change across the entire timeline. ':''}Timings are approximate. The transcript is processed in completed sections; live turns do not assign words to voices.</p>
    {snapshot.reason&&<p role="status">{snapshot.reason}</p>}
    <div className="speaker-names">{speakers.map(id=><label key={id}><span>Speaker {id}</span><input aria-label={`Name for live speaker ${id}`} maxLength={80} value={names[String(id)]||''} placeholder={`Speaker ${id}`} onChange={event=>onRename(id,event.target.value)}/></label>)}</div>
    <p className="speaker-note">{formatTime(snapshot.processed_audio_seconds)} processed · revision {snapshot.revision}. Live audio is saved in sections for transcription recovery; timeline playback is unavailable.</p>
    <ol className="speaker-segments">{turns.slice(0,shown).map((turn,index)=><li key={index}><span>{formatTime(turn.start_seconds)}</span><div><strong>{label(turn.speaker)}</strong><span>{formatTime(turn.start_seconds)}–{formatTime(turn.end_seconds)}</span></div></li>)}</ol>
    {!turns.length&&snapshot.status==='active'&&<p className="speaker-note">Waiting for the first speaker snapshot.</p>}
    {shown<turns.length&&<button className="speaker-more" onClick={()=>setShown(shown+50)}>Show 50 more ({turns.length-shown} remaining)</button>}
  </section>;
}
