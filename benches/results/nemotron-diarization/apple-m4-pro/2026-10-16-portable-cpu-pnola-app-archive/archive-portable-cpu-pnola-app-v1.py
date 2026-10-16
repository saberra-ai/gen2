"""Snapshot the exact complete app for later receiver delivery; no app/model execution."""
import json,os,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard,atomic_json
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory
source=cache/'portable-cpu-pnola-app-build-v1';build=json.loads((source/'build.json').read_text());app=Path(build['app']);assert inventory(app)==build['inventory']
out=cache/'portable-cpu-pnola-app-delivery-v1';assert not out.exists();out.mkdir()
script='from pathlib import Path; import json,sys; from sdk_archive import write_archive; print(json.dumps(write_archive(Path(sys.argv[1]),Path(sys.argv[2]),"portable-cpu-pnola-app")))'
policy=out/'archive.sb';policy.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write* (subpath '+json.dumps(str(app))+'))\n(deny process-exec (subpath '+json.dumps(str(app))+'))\n')
pins={n:descriptor(tools/n) for n in ['sdk_archive.py','artifacts.py','qualification_storage.py','qualify_valid_mel_pnola.py']}
protocol={'scope':__doc__,'source_app':str(app),'inventory':build['inventory'],'constructor_receipt':descriptor(source/'build.json'),'archive_code':script,'launcher':descriptor(Path(__file__)),'tools':pins,'storage':{'start_minimum':3*1024**3,'running_minimum':1024**3,'reserve_bytes':256*1024**2},'neural_admission_floor_unchanged_bytes':5*1024**3,'release_qualified':False}
atomic_json(out/'protocol.json',protocol);r={'archive_construction_passed':False,'models_or_app_executed':False,'release_qualified':False};guard=None
try:
 guard=StorageGuard(out,**protocol['storage']);guard.before_model()
 env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(tools),'PYTHONDONTWRITEBYTECODE':'1'}
 argv=['/usr/bin/sandbox-exec','-f',str(policy),sys.executable,'-c',script,str(app),str(out)]
 command={'argv':argv,'environment':env,'cwd':str(out)};r['command']=command
 with (out/'archive.stdout.log').open('xb') as stdout,(out/'archive.stderr.log').open('xb') as stderr:
  execute_guarded(argv,cwd=out,environment=env,stdout=stdout,stderr=stderr,guard=guard,record=command,persist=lambda:atomic_json(out/'qualification.json',r),timeout=600)
 assert command['returncode']==0
 files=json.loads((out/'portable-cpu-pnola-app-files.json').read_text());assert files['files']==build['inventory']
 assert inventory(app)==build['inventory'] and descriptor(source/'build.json')==protocol['constructor_receipt']
 for name,value in pins.items():assert descriptor(tools/name)==value
 assert descriptor(Path(__file__))==protocol['launcher']
 r.update(archive_construction_passed=True,archive={k:v for k,v in files.items() if k!='files'},archive_descriptor=descriptor(out/files['archive']),inventory_entries=len(files['files']))
except BaseException as error:r['error']=repr(error);raise
finally:
 if guard is not None:guard.release();r['storage']=guard.record
 atomic_json(out/'qualification.json',r)
print(json.dumps({'archive_construction_passed':r['archive_construction_passed'],'archive':r.get('archive')}))
