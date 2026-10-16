"""Compile current Pnola test and CLI helpers only; no model/helper execution."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');app=Path('/Users/victor/workspace/pnola');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard,atomic_json
out=cache/'valid-mel-pnola-helper-build-v2';assert not out.exists();out.mkdir()
prior=json.loads((tools/'valid-mel-mixed.protocol.json').read_text());pins=prior['repository_files']
names={'src-tauri/Cargo.toml','src-tauri/Cargo.lock','src-tauri/build.rs','src-tauri/Info.plist','src-tauri/tauri.conf.json'}
for name in ['src-tauri/src','src-tauri/icons','src-tauri/capabilities','dist']:
 names.update(str(p.relative_to(app))for p in(app/name).rglob('*')if p.is_file()and not p.is_symlink())
app_pins={name:descriptor(app/name)for name in sorted(names)}
policy=out/'compile-only.sb';policy.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath '+json.dumps(str(cache))+') (subpath '+json.dumps(str(app/'.desktop-assets'))+'))\n')
env={k:os.environ[k]for k in ['HOME','PATH','TMPDIR','RUSTUP_HOME','CARGO_HOME','DEVELOPER_DIR']if k in os.environ};env.update(CARGO_TARGET_DIR='/Users/victor/.cache/pnola-nemotron-build',CARGO_BUILD_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',CARGO_INCREMENTAL='0',CARGO_NET_OFFLINE='true',HOMEBREW_NO_AUTO_UPDATE='1',TAURI_CONFIG=json.dumps({'bundle':{'resources':[],'macOS':{'frameworks':[]}}}))
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
protocol={'scope':__doc__,'gen2_source_commit':source,'pnola_baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=app,text=True).strip(),'repository_files':pins,'pnola_files':app_pins,'environment':env,'launcher':descriptor(Path(__file__)),'models_loaded':False,'test_or_native_query_calls':0}
atomic_json(out/'protocol.json',protocol);commands=[];record={'compiled':False,'models_loaded':False,'gen2_source_commit':source,'commands':commands};guard=None
try:
 guard=StorageGuard(out,3*1024**3,1024**3,256*1024**2)
 for name,args in [('tests',['test','--no-run']),('cli',['build'])]:
  argv=['/usr/bin/sandbox-exec','-f',str(policy),'cargo',*args,'--offline','--locked','--manifest-path',str(app/'src-tauri/Cargo.toml'),'--message-format=json']
  command={'name':name,'argv':argv,'cwd':str(app)};commands.append(command);atomic_json(out/'commands.json',commands)
  started=time.monotonic()
  with(out/(name+'.stdout.jsonl')).open('xb')as stdout,(out/(name+'.stderr.log')).open('xb')as stderr:
   process=subprocess.Popen(argv,cwd=app,env=env,stdout=stdout,stderr=stderr,start_new_session=True);command['pid']=process.pid;atomic_json(out/'commands.json',commands);command['returncode']=guard.wait(process,command,1800)
  command['elapsed_seconds']=time.monotonic()-started;atomic_json(out/'commands.json',commands)
  if command['returncode']!=0:raise RuntimeError('Pnola compile-only command failed: '+name)
  binaries=[]
  for line in(out/(name+'.stdout.jsonl')).read_text().splitlines():
   item=json.loads(line)
   if item.get('reason')=='compiler-artifact'and item['target']['name']=='pnola-desktop'and item.get('executable'):binaries.append(Path(item['executable']))
  assert len(binaries)==1
  helper=out/('pnola-tests'if name=='tests'else'pnola-cli');shutil.copy2(binaries[0],helper);record[name]={'source_path':str(binaries[0]),**descriptor(helper)}
  for root,files in [(repo,pins),(app,app_pins)]:
   for path,expected in files.items():assert descriptor(root/path)==expected,path
 record['compiled']=True
except BaseException as error:
 record['error']=str(error);raise
finally:
 if guard is not None:guard.release();record['storage']=guard.record
 atomic_json(out/'build.json',record)
print(json.dumps({'compiled':record['compiled'],'tests':record.get('tests'),'cli':record.get('cli'),'pnola_input_count':len(app_pins),'gen2_input_count':len(pins)}))
