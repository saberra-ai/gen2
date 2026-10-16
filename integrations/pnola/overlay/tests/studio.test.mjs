import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initialState, reducer, validateHistory } from '../src/lib/studio/state.ts';
const entry={id:'entry-1',name:'A thought',text:'Finished thought',date:'2026-10-14T20:00:00Z',duration:10,execution:100};
test('reactive progress accumulates text and ignores stale job events',()=>{
  let state=reducer(initialState,{type:'begin',job:'current',name:'Audio'});
  const progress={type:'part',index:1,total:2,result:{text:'First thought.',duration_seconds:10,execution_micros:20}};
  assert.equal(reducer(state,{type:'progress',job:'old',value:progress}),state);
  state=reducer(state,{type:'progress',job:'current',value:progress});
  assert.equal(state.current.text,'First thought.');assert.equal(state.current.partial,true);
  state=reducer(state,{type:'progress',job:'current',value:{...progress,index:2,result:{...progress.result,text:'Second thought.'}}});
  assert.equal(state.current.text,'First thought.\n\nSecond thought.');assert.equal(state.current.duration,20);
  assert.equal(reducer(state,{type:'complete',job:'old',entry}),state);
});
test('cancellation keeps completed sections and rejects late results',()=>{
  let state=reducer(initialState,{type:'begin',job:'current',name:'Audio'});
  state=reducer(state,{type:'progress',job:'current',value:{type:'part',index:1,total:2,result:{text:'Keep this.',duration_seconds:10,execution_micros:20}}});
  state=reducer(state,{type:'cancelled',job:'current'});
  assert.equal(state.phase,'idle');assert.equal(state.current.text,'Keep this.');assert.equal(state.history.length,0);
  assert.equal(reducer(state,{type:'complete',job:'current',entry}),state);
});
test('editing an empty transcript remains editable and updates saved history',()=>{
  let state=reducer(initialState,{type:'begin',job:'current',name:'Audio'});
  state=reducer(state,{type:'complete',job:'current',entry});
  state=reducer(state,{type:'edit',text:''});assert.equal(state.current.text,'');assert.equal(state.history[0].text,'');
  state=reducer(state,{type:'edit',text:'Rewritten.'});assert.equal(state.history[0].text,'Rewritten.');
});
test('invalid saved history does not silently become an empty file',()=>{
  assert.throws(()=>validateHistory({unexpected:'data'}),/preserved/);
  assert.deepEqual(validateHistory([entry,{id:'broken'}]),[entry]);
});

test('live updates retain one history entry beyond two hours and protect its bounded preview from edits',()=>{
  const live={...entry,id:'live',liveSessionId:'live',text:'Latest sections',words:10000,queued:2,duration:7208,partial:true};
  let state={...initialState,audio:{name:'old.wav'}};
  state=reducer(state,{type:'live',entry:live});
  assert.equal(state.audio,null);assert.equal(state.current.words,10000);
  for(let index=0;index<100;index++)state=reducer(state,{type:'live',entry:{...live,words:10000+index}});
  assert.equal(state.history.length,1);assert.equal(state.current.duration,7208);
  const edited=reducer(state,{type:'edit',text:'Would truncate full saved transcript'});
  assert.equal(edited,state);assert.equal(edited.current.text,'Latest sections');
  state=reducer(state,{type:'live',entry:{...live,partial:false,queued:0}});
  assert.equal(state.history[0].partial,false);assert.equal(state.current.queued,0);
});

test('speaker names and timelines survive storage, while text edits invalidate word labels',()=>{
  const attribution={status:'complete',session_id:'session-1',revision:4,complete:true,timeline:{turns:[{start_seconds:0,end_seconds:1,speaker:1}],audio_duration_seconds:10,model_revision:'model',runtime_version:'runtime',backend:'cpu',geometry:{}},words:[{text:'Finished',start_seconds:0,end_seconds:0.5,attribution:{kind:'speaker',speaker:1},candidates:[{speaker:1,fraction:1}]}],words_complete:true,timing_method:'bark_token_intervals_v1',alignment_error:null};
  let state=reducer(initialState,{type:'begin',job:'current',name:'Audio'});
  state=reducer(state,{type:'complete',job:'current',entry:{...entry,attribution}});
  state=reducer(state,{type:'rename-speaker',entryId:entry.id,sessionId:'session-1',speaker:1,name:'Alex'});
  assert.equal(state.current.speakerNames['1'],'Alex');
  assert.equal(validateHistory(JSON.parse(JSON.stringify(state.history)))[0].speakerNames['1'],'Alex');
  state=reducer(state,{type:'edit',text:'Changed presentation.'});
  assert.equal(state.current.attribution.words.length,0);
  assert.equal(state.current.attribution.words_complete,false);
  assert.equal(state.current.attribution.timeline.turns.length,1);
  assert.equal(state.current.speakerNames['1'],'Alex');
});

test('a standalone timeline is saved only after explicit selection',()=>{
  let state=reducer(initialState,{type:'begin',job:'current',name:'Audio'});
  state=reducer(state,{type:'complete',job:'current',entry:{...entry,text:'',asr_error:'Speech failed'},save:false});
  assert.equal(state.history.length,0);assert.equal(state.current.asr_error,'Speech failed');
  state=reducer(state,{type:'save-current'});assert.equal(state.history.length,1);
});

test('opening history clears unrelated playback audio',()=>{
  const state={...initialState,history:[entry],audio:{url:'wrong.wav',name:'Different recording'}};
  assert.equal(reducer(state,{type:'open',id:entry.id}).audio,null);
});

test('live speaker updates replace revisions and retain names only within one native session',()=>{
  const speakers={status:'active',session_id:'native-one',revision:3,accepted_audio_seconds:8,processed_audio_seconds:5,timeline:{turns:[{start_seconds:0,end_seconds:4,speaker:1}],audio_duration_seconds:5,model_revision:'model',runtime_version:'runtime',backend:'cpu',geometry:[9,4,0,264,264,222],native_compute_micros:1},reason:null};
  const live={...entry,id:'live',liveSessionId:'live',liveSpeakers:speakers};
  let state=reducer(initialState,{type:'live',entry:live});
  state=reducer(state,{type:'rename-speaker',entryId:'live',sessionId:'native-one',speaker:1,name:'Alex Morgan'});
  assert.equal(state.current.speakerNames['1'],'Alex Morgan');
  state=reducer(state,{type:'live',entry:{...live,liveSpeakers:{...speakers,revision:4}}});
  assert.equal(state.current.speakerNames['1'],'Alex Morgan');
  const newer=state;
  assert.equal(reducer(state,{type:'live',entry:live}),newer);
  assert.equal(validateHistory(JSON.parse(JSON.stringify(state.history)))[0].liveSpeakers.revision,4);
  state=reducer(state,{type:'live',entry:{...live,liveSpeakers:{...speakers,revision:4,status:'interrupted',reason:'Queue full'}}});
  assert.equal(reducer(state,{type:'live',entry:{...live,liveSpeakers:{...speakers,revision:4}}}),state);
  state=reducer(state,{type:'live',entry:{...live,liveSpeakers:{...speakers,session_id:'different-native-session',revision:1}}});
  assert.equal(state.current.speakerNames,undefined);
});

test('malformed live speaker history does not attach invalid labels to valid text',()=>{
  const restored=validateHistory([{...entry,liveSpeakers:{status:'complete',session_id:'s',revision:1,accepted_audio_seconds:1,processed_audio_seconds:1,timeline:{turns:[{start_seconds:0,end_seconds:2,speaker:1}],audio_duration_seconds:1,model_revision:'model',runtime_version:'runtime',backend:'cpu',geometry:[9,4,0,264,264,222]},reason:null}}])[0];
  assert.equal(restored.text,entry.text);
  assert.equal(restored.liveSpeakers,undefined);
});
