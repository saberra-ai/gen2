from pathlib import Path
import hashlib,http.server,json,os,subprocess,threading
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'release Metal app qualification 024a88f';app=root/'Pnola.app'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((repo/'tools/nemotron-diarization/metal-app-shutdown.protocol.json').read_text());assert h(Path(__file__))==protocol['runner_sha256'];assert h(root/'manifest.json')==protocol['manifest_sha256']
m=json.loads((root/'manifest.json').read_text())
for v in m['files'].values():
 p=Path(v['path'])
 if 'symlink' in v:assert p.is_symlink() and os.readlink(p)==v['symlink']
 else:assert p.stat().st_size==v['bytes'] and h(p)==v['sha256'],p
commands=[];results={};base={'PATH':'/usr/bin:/bin'}
def save():
 (root/'verification.json').write_text(json.dumps({'release_qualified':False,'scope':m['scope'],'all_requested_checks_passed':len(results)==7 and all(x['passed'] for x in results.values()),'results':results},indent=2)+'\n')
def run(name,args,env=None,sandbox=True):
 command=[str(x) for x in args]
 if sandbox:command=['/usr/bin/sandbox-exec','-f',str(root/'offline.sb'),*command]
 commands.append({'name':name,'argv':command,'environment':env or base});(root/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print('START '+name,flush=True)
 with (root/(name+'.stdout.log')).open('x') as out,(root/(name+'.stderr.log')).open('x') as err:
  r=subprocess.run(command,env=env or base,cwd=root,stdout=out,stderr=err,timeout=900)
 print('END '+name+' '+str(r.returncode),flush=True);return r.returncode
status=run('app-signature',['/usr/bin/codesign','--verify','--deep','--strict',app],sandbox=False);results['signature']={'passed':status==0,'exit_code':status};save()
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'control')
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();url='http://127.0.0.1:'+str(server.server_port)
outside=run('network-allowed',['/usr/bin/curl','--silent','--show-error','--max-time','3',url],sandbox=False);inside=run('network-denied',['/usr/bin/curl','--silent','--show-error','--max-time','3',url]);server.shutdown()
source=cache/'NeMo-Speech.cpp/LICENSE';source_out=run('source-allowed',['/bin/cat',source],sandbox=False);source_in=run('source-denied',['/bin/cat',source]);old_in=run('predecessor-app-denied',['/bin/cat',cache/'release Metal app qualification ef1fc1a/Pnola.app/Contents/Info.plist'])
results['sandbox_controls']={'passed':outside==0 and inside!=0 and source_out==0 and source_in!=0 and old_in!=0,'network_allowed':outside,'network_denied':inside,'source_allowed':source_out,'source_denied':source_in,'predecessor_denied':old_in};save();assert results['sandbox_controls']['passed']
for name,audio in [('shutdown-1s','shutdown-1s.wav'),('app-recorded','ami-60s.wav')]:
 output=root/(name+'-result.json');status=run(name,[app/'Contents/MacOS/pnola-desktop','--verify-bundle',root/audio,output]);r={'passed':False,'exit_code':status};results[name]=r;save()
 if status==0:
  x=json.loads(output.read_text());t=x['attribution']['timeline'];assert t['backend']=='metal' and t['device'] and x['asr_error'] is None and x['parts']==x['parts_completed']
  r.update(passed=True,backend=t['backend'],device=t['device'],asr_parts=x['parts_completed'],words=len(x['attribution']['words']),turns=len(t['turns']))
  if name=='app-recorded':
   expected=json.loads((root/'native-ami.json').read_text())['turns'];actual=t['turns'];assert len(actual)==len(expected)
   maximum=max(max(abs(a['start_seconds']-b['start_seconds']),abs(a['end_seconds']-min(60,b['end_seconds']))) for a,b in zip(actual,expected));assert all(a['speaker']==b['speaker'] for a,b in zip(actual,expected)) and maximum<=1e-6
   r['maximum_native_turn_error_seconds']=maximum
 save()
negative=root/'unavailable-metal.json';status=run('metal-unavailable',[app/'Contents/MacOS/pnola-desktop','--verify-bundle',root/'ami-60s.wav',negative],env={**base,'GEN2_NEMO_EXECUTION':'metal','GEN2_NEMO_LIBRARY':str(root/'control-cpu/lib/libnemo_speech_asr_c.dylib')})
r={'passed':False,'exit_code':status};results['requested_metal_does_not_fallback']=r;save()
if status==0:
 x=json.loads(negative.read_text());assert x['attribution']['status']=='unavailable' and x['text'] and x['asr_error'] is None and x['parts']==x['parts_completed'] and 'Metal' in x['attribution']['reason'];r.update(passed=True,asr_parts=x['parts_completed'],words=len(x['attribution']['words']),speaker_reason=x['attribution']['reason']);save()
live=root/'live.json';status=run('app-live',[app/'Contents/MacOS/pnola-desktop','--verify-live-speakers',root/'live-source-60s.wav',live]);r={'passed':False,'exit_code':status};results['app_live']=r;save()
if status==0:
 x=json.loads(live.read_text());t=x['speakers']['timeline'];assert x['passed'] and x['sections']==11 and x['queued']==0 and not x['microphone_used'] and x['speakers']['status']=='complete' and t['backend']=='metal' and t['device'];r.update(passed=True,backend=t['backend'],device=t['device'],audio_seconds=x['duration_seconds'],asr_sections=x['sections'],words=x['words'],turns=len(t['turns']),observed_provisional_revisions=len(x['provisional_revisions']),final_revision=x['speakers']['revision'],persisted_turn_tolerance_seconds=x['persisted_turn_tolerance_seconds'],export_bytes=x['export_bytes'],microphone_used=False);save()
status=run('missing-input-error',[app/'Contents/MacOS/pnola-desktop','--verify-bundle',root/'missing.wav',root/'missing-result.json']);results['missing_input_error']={'passed':status==1,'exit_code':status,'scope':'Controlled error exit; predecessor also exited1, no original crash claimed.'};save()
print((root/'verification.json').read_text(),flush=True)
assert all(x['passed'] for x in results.values())
