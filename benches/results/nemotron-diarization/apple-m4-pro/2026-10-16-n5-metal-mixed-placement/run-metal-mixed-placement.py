from pathlib import Path
import hashlib,json,os,subprocess
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'release Metal mixed placement 8572cc8';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((repo/'tools/nemotron-diarization/metal-mixed-placement.protocol.json').read_text());assert h(Path(__file__))==p['runner_sha256'] and h(root/'manifest.json')==p['manifest_sha256'];m=json.loads((root/'manifest.json').read_text())
for v in m['files'].values():
 f=Path(v['path'])
 if 'symlink' in v:assert f.is_symlink() and os.readlink(f)==v['symlink']
 else:assert f.stat().st_size==v['bytes'] and h(f)==v['sha256'],f
commands=[];results=[]
for name in ['mixed_real_laya_chat_first','mixed_real_laya_diarization_first']:
 command=['/usr/bin/sandbox-exec','-f',str(root/'offline.sb'),str(root/'mixed-native-tests'),'--ignored','--exact',name,'--nocapture'];commands.append({'name':name,'argv':command,'environment':m['environment']});(root/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print('START',name,flush=True)
 with (root/(name+'.stdout.log')).open('x') as out,(root/(name+'.stderr.log')).open('x') as err:r=subprocess.run(command,env=m['environment'],cwd=root,stdout=out,stderr=err,timeout=600)
 text=(root/(name+'.stdout.log')).read_text();placement={}
 for line in text.splitlines():
  if line.startswith('chat-placement '):
   key,data=line.removeprefix('chat-placement ').split(': ',1);placement[key]=json.loads(data)
 passed=r.returncode==0 and set(placement)=={'initial','restored'} and all(v['layers']>0 and v['backend']=='MTL' and v['device'] for v in placement.values())
 results.append({'name':name,'exit_code':r.returncode,'passed':passed,'public_reported_chat_placement':placement});(root/'verification.json').write_text(json.dumps({'release_qualified':False,'scope':m['scope'],'all_requested_checks_passed':len(results)==2 and all(x['passed'] for x in results),'results':results},indent=2)+'\n');print('END',name,r.returncode,flush=True)
print((root/'verification.json').read_text(),flush=True);assert all(x['passed'] for x in results)
