import { useState } from "react";
import type { Transcript, AttributedWord } from "@/lib/studio/types";
import { formatCue } from "@/lib/studio/state";

function wordKey(word:AttributedWord):string {
  const a=word.attribution;
  return a.kind==='speaker'?`speaker:${a.speaker}`:a.kind==='overlap'?`overlap:${a.speakers.join(',')}`:`unknown:${a.reason}`;
}
export function SpeakerPanel({entry,canSeek,disabled,saved,onSeek,onRename,onSave}:{entry:Transcript;canSeek:boolean;disabled:boolean;saved:boolean;onSeek:(seconds:number)=>void;onRename:(speaker:number,name:string)=>void;onSave:()=>void}) {
  const [view,setView]=useState<'turns'|'words'>('turns');
  const [shown,setShown]=useState(50);
  const attribution=entry.attribution;
  if(!attribution)return null;
  if(attribution.status==='unavailable')return <section className="speaker-panel" aria-label="Speaker labels"><p>Speaker labels unavailable. Your transcript is preserved.</p><details><summary>Details</summary><p>{attribution.reason}</p></details></section>;
  const names=entry.speakerNames||{};
  const name=(id:number)=>names[String(id)]?.trim()||`Speaker ${id}`;
  const speakers=[...new Set(attribution.timeline.turns.map(t=>t.speaker))].sort();
  const groups:{key:string;start:number;end:number;text:string;label:string;uncertain:boolean}[]=[];
  for(const word of attribution.words){
    const a=word.attribution,key=wordKey(word);
    const label=a.kind==='speaker'?name(a.speaker):a.kind==='overlap'?`Overlap · ${a.speakers.map(name).join(' + ')}`:'Uncertain speaker';
    const previous=groups.at(-1);
    if(previous&&previous.key===key&&word.start_seconds-previous.end<=0.8){previous.text+=` ${word.text}`;previous.end=Math.max(previous.end,word.end_seconds);}
    else groups.push({key,start:word.start_seconds,end:word.end_seconds,text:word.text,label,uncertain:a.kind!=='speaker'});
  }
  const turns=attribution.timeline.turns;
  const activeView=view==='words'&&groups.length?'words':'turns';
  const total=activeView==='turns'?turns.length:groups.length;
  return <section className="speaker-panel" aria-label="Speaker timeline">
    <div className="speaker-heading"><strong>Speakers</strong><span>Final timeline · {speakers.length} detected</span></div>
    <p className="speaker-note">Labels are anonymous. Timings are approximate; overlap does not separate voices.</p>
    {entry.asr_error&&<p role="status">Transcription is unavailable or incomplete. The speaker timeline is preserved.</p>}
    {entry.asr_error&&!saved&&<button className="speaker-save" onClick={onSave}>Save speaker timeline</button>}
    <div className="speaker-names">{speakers.map(id=><label key={id}><span>Speaker {id}</span><input aria-label={`Name for speaker ${id}`} maxLength={80} disabled={disabled} value={names[String(id)]||''} placeholder={`Speaker ${id}`} onChange={event=>onRename(id,event.target.value)}/></label>)}</div>
    <div className="speaker-tabs" role="tablist" aria-label="Speaker view">
      <button role="tab" aria-selected={activeView==='turns'} onClick={()=>{setView('turns');setShown(50);}}>Turns</button>
      <button role="tab" aria-selected={activeView==='words'} onClick={()=>{setView('words');setShown(50);}} disabled={!groups.length}>Words</button>
    </div>
    {!canSeek&&<p className="speaker-note">Playback needs the unchanged original audio. Saved labels remain available.</p>}
    {entry.wordsEdited&&<p className="speaker-note">Text was edited. Word labels were cleared; speaker turns are unchanged.</p>}
    {attribution.alignment_error&&<p className="speaker-note">Some words could not be aligned. Their speaker labels are unavailable.</p>}
    <ol className="speaker-segments">{activeView==='turns'?turns.slice(0,shown).map((turn,index)=><li key={index}>
      <button aria-label={`Play ${name(turn.speaker)} at ${formatCue(turn.start_seconds)}`} disabled={!canSeek} onClick={()=>onSeek(turn.start_seconds)}>{formatCue(turn.start_seconds)}</button>
      <div><strong>{name(turn.speaker)}</strong><span>{formatCue(turn.start_seconds)}–{formatCue(turn.end_seconds)}</span></div>
    </li>):groups.slice(0,shown).map((group,index)=><li className={group.uncertain?'uncertain':''} key={index}>
      <button aria-label={`Play words at ${formatCue(group.start)}`} disabled={!canSeek} onClick={()=>onSeek(group.start)}>{formatCue(group.start)}</button>
      <div><strong>{group.label}</strong><p>{group.text}</p></div>
    </li>)}</ol>
    {!total&&<p className="speaker-note">No speech turns detected.</p>}
    {shown<total&&<button className="speaker-more" onClick={()=>setShown(shown+50)}>Show 50 more ({total-shown} remaining)</button>}
  </section>;
}
