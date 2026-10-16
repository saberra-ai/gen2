from pathlib import Path
import hashlib,http.server,json,os,subprocess,threading
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'release Metal app qualification 024a88f';app=root/'Pnola.app'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((repo/'tools/nemotron-diarization/metal-app-shutdown-resume.protocol.json').read_text());assert h(Path(__file__))==protocol['runner_sha256'];assert h(root/'manifest.json')==protocol['manifest_sha256']
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
commands=json.loads((root/'commands.json').read_text());results=json.loads((root/'verification.json').read_text())['results']
assert results['requested_metal_does_not_fallback']['exit_code']==0
x=json.loads((root/'unavailable-metal.json').read_text())
assert x['attribution']['status']=='unavailable' and x['text'] and x['asr_error'] is None and x['parts']==x['parts_completed'] and 'Metal' in x['attribution']['reason']
results['requested_metal_does_not_fallback']={'passed':True,'exit_code':0,'asr_parts':x['parts_completed'],'speaker_reason':x['attribution']['reason'],'result_reused_without_model_rerun':True}
save()
live=root/'live.json';status=run('app-live',[app/'Contents/MacOS/pnola-desktop','--verify-live-speakers',root/'live-source-60s.wav',live]);r={'passed':False,'exit_code':status};results['app_live']=r;save()
if status==0:
 x=json.loads(live.read_text());t=x['speakers']['timeline'];assert x['passed'] and x['sections']==11 and x['queued']==0 and not x['microphone_used'] and x['speakers']['status']=='complete' and t['backend']=='metal' and t['device'];r.update(passed=True,backend=t['backend'],device=t['device'],audio_seconds=x['duration_seconds'],asr_sections=x['sections'],words=x['words'],turns=len(t['turns']),observed_provisional_revisions=len(x['provisional_revisions']),final_revision=x['speakers']['revision'],persisted_turn_tolerance_seconds=x['persisted_turn_tolerance_seconds'],export_bytes=x['export_bytes'],microphone_used=False);save()
status=run('missing-input-error',[app/'Contents/MacOS/pnola-desktop','--verify-bundle',root/'missing.wav',root/'missing-result.json']);results['missing_input_error']={'passed':status==1,'exit_code':status,'scope':'Controlled error exit; predecessor also exited1, no original crash claimed.'};save()
print((root/'verification.json').read_text(),flush=True)
assert all(x['passed'] for x in results.values())
