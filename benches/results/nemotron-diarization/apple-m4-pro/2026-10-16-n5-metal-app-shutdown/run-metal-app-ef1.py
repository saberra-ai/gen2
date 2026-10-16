from pathlib import Path
import hashlib,http.server,json,os,subprocess,threading
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'release Metal app qualification ef1fc1a';app=root/'Pnola.app'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((repo/'tools/nemotron-diarization/metal-app.protocol.json').read_text());assert h(Path(__file__))==protocol['runner_sha256'];assert h(root/'manifest.json')==protocol['manifest_sha256']
m=json.loads((root/'manifest.json').read_text())
for item in m['files'].values():
 p=Path(item['path'])
 if 'symlink' in item:assert p.is_symlink() and os.readlink(p)==item['symlink']
 else:assert p.stat().st_size==item['bytes'] and h(p)==item['sha256'],p
commands=[];results={};base={'PATH':'/usr/bin:/bin'}
def run(name,args,env=None,sandbox=True,timeout=900):
 command=[str(x) for x in args]
 if sandbox:command=['/usr/bin/sandbox-exec','-f',str(root/'offline.sb'),*command]
 commands.append({'name':name,'argv':command,'environment':env or base,'expected_failure':name.endswith('denied')})
 (root/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print('START '+name,flush=True)
 with (root/(name+'.stdout.log')).open('x') as out,(root/(name+'.stderr.log')).open('x') as err:
  r=subprocess.run(command,env=env or base,cwd=root,stdout=out,stderr=err,timeout=timeout)
 print('END '+name+' '+str(r.returncode),flush=True)
 return r.returncode
assert run('app-signature',['/usr/bin/codesign','--verify','--deep','--strict',app],sandbox=False)==0
# Prove network denial against an immediately verified local listening server.
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'control')
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();url='http://127.0.0.1:'+str(server.server_port)
outside=run('network-allowed',['/usr/bin/curl','--silent','--show-error','--max-time','3',url],sandbox=False)
inside=run('network-denied',['/usr/bin/curl','--silent','--show-error','--max-time','3',url]);server.shutdown()
source=cache/'NeMo-Speech.cpp/LICENSE'
source_out=run('source-allowed',['/bin/cat',source],sandbox=False);source_in=run('source-denied',['/bin/cat',source])
results['sandbox_controls']={'passed':outside==0 and inside!=0 and source_out==0 and source_in!=0,'outside_network_exit':outside,'inside_network_exit':inside,'outside_source_exit':source_out,'inside_source_exit':source_in}
assert results['sandbox_controls']['passed']
for name in ['mixed_real_laya_chat_first','mixed_real_laya_diarization_first']:
 status=run(name,[root/'mixed-native-tests','--ignored','--exact',name,'--nocapture'],env=m['environment'])
 results[name]={'passed':status==0,'exit_code':status,'scope':'Actual chat (Metal-enabled build), full Laya, Bark and Metal diarizer; admission/unload/reload, short text/silence only; no new model quality or latency claim.'}
native=root/'native-ami.json'
results['native_reference']={'passed':run('native-ami',[root/'native-metal',app/'Contents/Resources/nemotron/Nemotron-3-Diarization.q8_0.gguf',root/'ami-60s.wav','1','0',native])==0}
output=root/'recorded.json'
status=run('app-recorded',[app/'Contents/MacOS/pnola-desktop','--verify-bundle',root/'ami-60s.wav',output])
results['app_recorded']={'passed':False,'exit_code':status}
if status==0:
 x=json.loads(output.read_text());timeline=x['attribution']['timeline'];reference=json.loads(native.read_text());actual=timeline['turns'];expected=reference['turns'];maximum=0
 assert len(actual)==len(expected)
 for a,b in zip(actual,expected):
  assert a['speaker']==b['speaker']
  maximum=max(maximum,abs(a['start_seconds']-b['start_seconds']),abs(a['end_seconds']-min(60,b['end_seconds'])))
 assert maximum<=1e-6 and timeline['backend']=='metal' and timeline['device'] and x['asr_error'] is None and x['parts']==x['parts_completed']
 results['app_recorded']={'passed':True,'backend':timeline['backend'],'device':timeline['device'],'turns':len(actual),'maximum_native_turn_error_seconds':maximum,'asr_parts':x['parts_completed'],'words':len(x['attribution']['words']),'source_audio_sha256':x['audio_sha256']}
negative_env={**base,'GEN2_NEMO_EXECUTION':'metal','GEN2_NEMO_LIBRARY':str(root/'control-cpu/lib/libnemo_speech_asr_c.dylib')}
negative=root/'unavailable-metal.json';status=run('metal-unavailable',[app/'Contents/MacOS/pnola-desktop','--verify-bundle',root/'ami-60s.wav',negative],env=negative_env)
results['requested_metal_does_not_fallback']={'passed':False,'exit_code':status}
if status==0:
 x=json.loads(negative.read_text());assert x['attribution']['status']=='unavailable' and x['text'] and x['asr_error'] is None and x['parts']==x['parts_completed'];assert 'Metal' in x['attribution']['reason']
 results['requested_metal_does_not_fallback']={'passed':True,'asr_parts':x['parts_completed'],'speaker_reason':x['attribution']['reason']}
live=root/'live.json';status=run('app-live',[app/'Contents/MacOS/pnola-desktop','--verify-live-speakers',root/'live-source-60s.wav',live])
results['app_live']={'passed':False,'exit_code':status}
if status==0:
 x=json.loads(live.read_text());t=x['speakers']['timeline'];assert x['passed'] and x['sections']==11 and x['queued']==0 and not x['microphone_used'] and x['speakers']['status']=='complete';assert t['backend']=='metal' and t['device']
 results['app_live']={'passed':True,'backend':t['backend'],'device':t['device'],'audio_seconds':x['duration_seconds'],'asr_sections':x['sections'],'words':x['words'],'native_turns':len(t['turns']),'observed_provisional_revisions':len(x['provisional_revisions']),'final_revision':x['speakers']['revision'],'persisted_turn_tolerance_seconds':x['persisted_turn_tolerance_seconds'],'export_bytes':x['export_bytes'],'microphone_used':False}
(root/'verification.json').write_text(json.dumps({'release_qualified':False,'scope':m['scope'],'all_requested_checks_passed':all(r['passed'] for r in results.values()),'results':results},indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
