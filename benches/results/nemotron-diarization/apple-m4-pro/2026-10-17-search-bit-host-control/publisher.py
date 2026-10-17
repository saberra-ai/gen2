"""Saved-data replay/publication only: no permission, native or VM calls."""
import ast,base64,hashlib,json,os,stat,struct,subprocess
from pathlib import Path
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
root=cache/'macos-search-bit-control-local-v1'
def descriptor(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>1024**2:raise ValueError('bounded regular single-link receipt input required')
    data=path.read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
expected={'protocol.json','runner.py','offline.sb','denied.gguf','qualification.json','macos_vm_search_bit_control.py','macos_vm_permissions_probe.py','offline-controls.stdout.log','offline-controls.stderr.log','host-control.stdout.log','host-control.stderr.log'}
admitted={root/name:descriptor(root/name) for name in expected|{'collection.json'}}
admitted[Path(__file__)]=descriptor(Path(__file__))
manifest=json.loads((root/'collection.json').read_text())['files']
assert set(manifest)==expected
for name,want in manifest.items():assert descriptor(root/name)==want
q=json.loads((root/'qualification.json').read_text());p=json.loads((root/'protocol.json').read_text())
assert q['source_commit']=='d5c4f4e5983404d032c30b8c97a872d77d61e708'
assert q['pins']['protocol']==descriptor(root/'protocol.json') and q['pins']['runner']==descriptor(root/'runner.py')
for name,want in p['tools'].items():
    assert descriptor(root/name)==want
    data=subprocess.check_output(['git','show',q['source_commit']+':tools/nemotron-diarization/'+name],cwd=repo,env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'})
    assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==want
protocol_data=subprocess.check_output(['git','show',q['source_commit']+':tools/nemotron-diarization/macos-search-bit-control.protocol.json'],cwd=repo)
assert protocol_data==(root/'protocol.json').read_bytes()
pre_path=cache/'search-bit-control-pre-execution-v1.json'
admitted[pre_path]=descriptor(pre_path)
pre=json.loads(pre_path.read_text())
assert set(pre['files'])=={'run-macos-search-bit-control-v1.py','test-search-bit-controller-spawn-v1.py','search-bit-controller-spawn-controls-v1.json','search-bit-control-full-suite-v1.stdout.log','search-bit-control-full-suite-v1.stderr.log'}
assert pre['source_commit']==q['source_commit'] and pre['protocol']==descriptor(root/'protocol.json')
for name,want in pre['files'].items():
    admitted[cache/name]=descriptor(cache/name)
    assert admitted[cache/name]==want
assert descriptor(root/'runner.py')==pre['files']['run-macos-search-bit-control-v1.py']
spawn=json.loads((cache/'search-bit-controller-spawn-controls-v1.json').read_text())
assert spawn['passed'] is True and spawn['new_vm_operations']==0 and spawn['host_fixture_invocations']==0
assert len(spawn['controls'])==2 and {x['signal'] for x in spawn['controls']}=={2,15}
assert all(x['returncode']==-9 and x['reaped'] is True for x in spawn['controls'])
assert q['host_control_qualified'] is True and q['guest_access_qualified'] is False and q['new_guest_operations']==0
assert q['host']=={'cpu':'Apple M4 Pro','platform':'macOS-26.3-arm64-arm-64bit','euid':502}
report=json.loads((root/'host-control.stdout.log').read_text());assert q['report']==report
for name,want in p['required'].items():assert report[name] is want
assert report['runtime']=='/Applications/Xcode.app/Contents/Developer/usr/bin/python3' and report['python_version']=='3.9.6'
assert report['denials_before']=={'users':13,'groups':13} and report['ctime_changed'] is True
strip=lambda d:{k:v for k,v in d.items() if k not in {'permissions','full_mode','ctime_ns'}}
before,during,restored=[report[k] for k in ['before','during','restored']]
assert strip(before)==strip(during)==strip(restored)
assert [x['permissions'] for x in [before,during,restored]]==[0o600,0o700,0o600]
assert all(x['file_type']==stat.S_IFDIR and x['flags']==0 and x['observed_uid']==502 for x in [before,during,restored])
assert report['root_before']==report['root_restored']
assert set(report['children'])=={'users','groups'}
assert all(x['file_type']==stat.S_IFDIR and x['permissions']==0o600 and x['device']==before['device'] for x in report['children'].values())
security=[report['security_'+phase] for phase in ['before','during','restored']]
assert security[0]==security[1]==security[2]
raw=base64.b64decode(security[0]['raw_base64'],validate=True)
assert raw==struct.pack('<IiI',12,8,0) and hashlib.sha256(raw).hexdigest()==security[0]['sha256']
assert security[0]['status']=='ok' and security[0]['returncode']==0 and security[0]['acl_policy_qualified'] is False and security[0]['recognized_filesec'] is False
assert [x['name'] for x in q['commands']]==['offline-controls','host-control']
prefix=['/usr/bin/sandbox-exec','-f',str(root/'offline.sb')]
negative=next(ast.literal_eval(n.value) for n in ast.parse((root/'runner.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='negative' for t in n.targets))
assert q['commands'][0]['argv']==prefix+['/usr/bin/python3','-B','-c',negative,str(root/'denied-write.bin'),str(root/'denied.gguf')]
assert q['commands'][1]['argv']==prefix+p['argv']
env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(root),'TMPDIR':str(root)+'/', 'LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
assert all(type(x['returncode']) is int and x['returncode']==0 and x['reaped'] is True and x['pid']>0 and x['cwd']==str(root) and x['environment']==env and 0<x['wall_seconds']<=p['timeout_seconds']+1 and 'error' not in x and not x.get('forced_process_group_termination',False) for x in q['commands'])
assert not (root/'denied-write.bin').exists()
dest=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-search-bit-host-control'
dest.mkdir(mode=0o700)
copies={**{name:root/name for name in expected|{'collection.json'}},**{name:cache/name for name in ['search-bit-control-pre-execution-v1.json',*pre['files']]},'publisher.py':Path(__file__)}
for name,origin in copies.items():
    (dest/name).write_bytes(origin.read_bytes())
    assert descriptor(dest/name)==admitted[origin]
for origin,want in admitted.items():assert descriptor(origin)==want
qualification={'source_commit':q['source_commit'],'host_control_qualified':True,'guest_access_qualified':False,'acl_policy_qualified':False,'fresh_machine_qualified':False,'release_qualified':False,'permission_restoration_confirmed':True,'ctime_changed':True,'new_guest_operations':0,'command_count':2,'ordinary_signal_spawn_controls':2,'full_suite':pre['full_suite'],'scope':'Actual fresh host fixture operation only; saved-data collection invokes no native/permission/VM operation.'}
(dest/'collection-qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
# Exact copied identity and source closure precede the final public manifest.
assert {x.name for x in dest.iterdir()}==set(copies)|{'collection-qualification.json'}
files={name:descriptor(dest/name) for name in set(copies)|{'collection-qualification.json'}}
for name,origin in copies.items():assert files[name]==admitted[origin]
for origin,want in admitted.items():assert descriptor(origin)==want
(dest/'manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
assert {x.name for x in dest.iterdir()}==set(files)|{'manifest.json'}
for name,want in files.items():assert descriptor(dest/name)==want
print(json.dumps({'destination':str(dest),'members':len(files)+1,'bytes':sum(x.stat().st_size for x in dest.iterdir()),'host_control_qualified':True,'new_guest_operations':0}))
