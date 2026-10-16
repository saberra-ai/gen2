from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
protocol_path=repo/'tools/nemotron-diarization/attribution-ami.protocol.json'
p=json.loads(protocol_path.read_text());h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert h(Path(__file__))==p['runner_sha256']
root=cache/'n4-frozen-ami-app-v1';assert root.is_dir()
for name,expected in p['retained_first_execution'].items():assert h(root/name)==expected
(root/'resume-protocol.json').write_bytes(protocol_path.read_bytes());shutil.copy2(__file__,root/'resume.py')
app=Path(p['app_path'])
for name,expected in p['app_files'].items():
 path=app/name
 if 'symlink' in expected:assert path.is_symlink() and os.readlink(path)==expected['symlink']
 else:assert path.stat().st_size==expected['bytes'] and h(path)==expected['sha256'],name
assert h(Path(p['sandbox_path']))==p['sandbox_sha256']
assert h(cache/'quality-corpus-v1/manifest.json')==p['corpus_manifest_sha256']
for name,expected in p['tool_sha256'].items():assert h(repo/'tools/nemotron-diarization'/name)==expected
with (root/'signature.log').open('x') as out:subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',app],stdout=out,stderr=subprocess.STDOUT,check=True)
sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from qualify_attribution import score
from qualify_alignment import normalize,evaluate
results=[]
for case in p['cases']:
 identifier=case['id'];wav=cache/'quality-corpus-v1'/case['wav'];transcript=cache/'quality-corpus-v1'/case['transcript']
 assert h(wav)==case['wav_sha256'] and h(transcript)==case['transcript_sha256']
 reference=[{'text':w['text'],'speaker':w['speaker'],'start_seconds':w['start'],'end_seconds':w['end'],'source_id':w['source_id']} for w in json.loads(transcript.read_text())['words']]
 reference.sort(key=lambda w:(w['start_seconds'],w['speaker'],w['source_id']))
 (root/(identifier+'-reference.json')).write_text(json.dumps({'reference_words':reference,'timing':'secondary automatic forced alignment of human words; no manually timed words','source_sha256':h(transcript)},indent=2)+'\n')
 result_path=root/(identifier+'-result.json')
 command=['/usr/bin/sandbox-exec','-f',p['sandbox_path'],str(app/'Contents/MacOS/pnola-desktop'),'--verify-bundle',str(wav),str(result_path)]
 (root/(identifier+'-command.json')).write_text(json.dumps({'argv':command,'environment':{'PATH':'/usr/bin:/bin'}},indent=2)+'\n')
 if result_path.exists():
  assert h(result_path)==p['retained_first_execution'][result_path.name]
  print('REUSE completed '+identifier,flush=True)
 else:
  print('START '+identifier,flush=True)
  with (root/(identifier+'.stdout.log')).open('x') as out,(root/(identifier+'.stderr.log')).open('x') as err:subprocess.run(command,env={'PATH':'/usr/bin:/bin'},stdout=out,stderr=err,timeout=600,check=True)
 result=json.loads(result_path.read_text())
 assert result['audio_sha256']==case['wav_sha256']
 assert result['parts']==result['parts_completed'] and result['parts']>=2 and result['asr_error'] is None
 assert result['duration_seconds']==120 and result['attribution']['complete']
 metrics=score(reference,result['attribution'])
 predicted=[(normalize(w['text']),w['start_seconds'],w['end_seconds']) for w in result['attribution']['words'] if normalize(w['text'])]
 timing=evaluate([(normalize(w['text']),w['start_seconds'],w['end_seconds']) for w in reference if normalize(w['text'])],predicted)
 report={'slice':identifier,'source_audio_sha256':h(wav),'result_sha256':h(result_path),'attribution_metrics':metrics,'word_alignment_diagnostic':timing,'release_qualified':False}
 (root/(identifier+'-metrics.json')).write_text(json.dumps(report,indent=2)+'\n');results.append(report)
 print('PASS '+identifier+' '+json.dumps(metrics),flush=True)
(root/'summary.json').write_text(json.dumps({'release_qualified':False,'scope':p['scope'],'results':results},indent=2)+'\n')
