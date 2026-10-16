"""Clone a qualified complete app and replace only its CPU package/frameworks; no app or model execution."""
import ast,json,os,plistlib,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard,atomic_json
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory
from verify_valid_mel_pnola_app import verify_app,unsigned_descriptor
from package_macos import verify
oldroot=cache/'valid-mel-pnola-instrumented-app-build-v1';old=oldroot/'cpu/Pnola.app';producer=cache/'portable-macos-construction-ci-v2';package=producer/'restored-portable-cpu-runtime'
out=cache/'portable-cpu-pnola-app-build-v1';assert not out.exists();out.mkdir()
prior=json.loads((oldroot/'build.json').read_text());oldfiles=inventory(old);assert oldfiles==prior['apps']['cpu']['files']
packagefiles=inventory(package);assert packagefiles==json.loads((producer/'evidence/portable-cpu-runtime-files.json').read_text())['files']
closure=set();queue=['verify_valid_mel_pnola_app.py','qualification_storage.py','qualify_valid_mel_pnola.py','sdk_archive.py']
while queue:
 name=queue.pop()
 if name in closure:continue
 closure.add(name)
 for node in ast.walk(ast.parse((tools/name).read_text())):
  modules=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
  for module in modules:
   path=tools/((module or '').split('.')[0]+'.py')
   if path.is_file():queue.append(path.name)
closure|={'runtime-valid-mel.lock.json','runtime.lock.json','patches/v3-finish-valid-mel.patch','notices/openmdw-1.1/sources.json'}
pins={name:descriptor(tools/name) for name in sorted(closure)}
references={str(path):descriptor(path) for path in [oldroot/'build.json',oldroot/'protocol.json',producer/'evidence/qualification.json',producer/'evidence/portable-cpu-runtime-files.json',repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-construction-v2/qualification.json']}
protocol={'scope':__doc__,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'launcher':descriptor(Path(__file__)),'tools':pins,'references':references,'original_app':str(old),'original_inventory':oldfiles,'portable_package':str(package),'portable_inventory':packagefiles,'storage':{'start_minimum':3*1024**3,'running_minimum':1024**3,'reserve_bytes':256*1024**2},'cases':['static inspect exact original app/package','read and preserve actual original hardened entitlements','APFS clone complete original app','replace only Resources/nemo and its92 private Frameworks plus public entry copy; preserve ORT','adhoc sign each changed framework and bundle with original hardened entitlements','verify whole bundle, unsigned executable equality and exact permitted inventory changes'],'model_or_app_execution':False,'release_qualified':False}
atomic_json(out/'protocol.json',protocol)
r={'scope':__doc__,'source_commit':protocol['source_commit'],'app_construction_passed':False,'native_version_queries_or_neural_calls':0,'commands':[],'release_qualified':False};guard=None
policy=out/'construction.sb';policy.write_text('(version 1)\n(allow default)\n(deny network*)\n')
env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(Path.home()),'LC_ALL':'C'}
def frozen():
 assert inventory(old)==oldfiles and inventory(package)==packagefiles
 for n,v in pins.items():assert descriptor(tools/n)==v,n
 for n,v in references.items():assert descriptor(Path(n))==v,n
 assert descriptor(Path(__file__))==protocol['launcher']
def run(name,argv):
 frozen();guard.before_model();actual=['/usr/bin/sandbox-exec','-f',str(policy),*map(str,argv)]
 command={'name':name,'argv':actual,'cwd':str(out),'environment':env};r['commands'].append(command);atomic_json(out/'build.json',r)
 with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
  execute_guarded(actual,cwd=out,environment=env,stdout=stdout,stderr=stderr,guard=guard,record=command,persist=lambda:atomic_json(out/'build.json',r),timeout=300)
 assert command['returncode']==0,name
 return (out/(name+'.stdout.log')).read_bytes()
try:
 guard=StorageGuard(out,**protocol['storage']);frozen()
 original_static=verify_app(old);manifest=verify(package);assert original_static['backend']==manifest['backend']=='cpu'
 atomic_json(out/'original-static.json',original_static)
 raw=run('original-entitlements',['/usr/bin/codesign','-d','--entitlements',':-',old])
 entitlements=plistlib.loads(raw)
 assert entitlements=={'com.apple.security.cs.disable-library-validation':True,'com.apple.security.device.audio-input':True}
 (out/'Entitlements.plist').write_bytes(raw)
 details=run('original-signature',['/usr/bin/codesign','-d','--verbose=4',old])
 assert 'runtime' in (out/'original-signature.stderr.log').read_text()
 destination=out/'Pnola.app';run('clone',['/bin/cp','-cR',old,destination])
 resources=destination/'Contents/Resources';frameworks=destination/'Contents/Frameworks'
 shutil.rmtree(resources/'nemo');shutil.copytree(package,resources/'nemo',symlinks=True)
 assert inventory(resources/'nemo')==packagefiles
 names=set(manifest['libraries'])|{'libnemo_speech_asr_c.dylib'}
 assert set(f.name for f in frameworks.iterdir())==names|{'libonnxruntime.dylib'}
 for index,name in enumerate(sorted(names)):
  path=frameworks/name;path.unlink();shutil.copy2(package/'lib'/name,path,follow_symlinks=True)
  run('framework-'+str(index),['/usr/bin/codesign','--force','--sign','-','--options','runtime','--timestamp=none',path])
 run('bundle-sign',['/usr/bin/codesign','--force','--sign','-','--options','runtime','--entitlements',out/'Entitlements.plist','--timestamp=none',destination])
 run('bundle-verify',['/usr/bin/codesign','--verify','--deep','--strict',destination])
 signed=run('new-entitlements',['/usr/bin/codesign','-d','--entitlements',':-',destination]);assert plistlib.loads(signed)==entitlements
 run('new-signature',['/usr/bin/codesign','-d','--verbose=4',destination]);assert 'runtime' in (out/'new-signature.stderr.log').read_text()
 current=verify_app(destination);atomic_json(out/'static-verification.json',current)
 assert current['backend']=='cpu' and current['logical_runtime_library_count']==92
 assert unsigned_descriptor(destination/'Contents/MacOS/pnola-desktop')==unsigned_descriptor(old/'Contents/MacOS/pnola-desktop')
 newfiles=inventory(destination)
 changed={n for n in set(newfiles)|set(oldfiles) if newfiles.get(n)!=oldfiles.get(n)}
 allowed=lambda n:n.startswith(('Contents/Resources/nemo/','Contents/Frameworks/','Contents/_CodeSignature/')) or n=='Contents/MacOS/pnola-desktop'
 assert all(allowed(n) for n in changed),changed
 assert descriptor(frameworks/'libonnxruntime.dylib')==descriptor(old/'Contents/Frameworks/libonnxruntime.dylib')
 frozen();r.update(app_construction_passed=True,app=str(destination),inventory=newfiles,changed_inventory_entries=sorted(changed),unsigned_executable_payload_unchanged=True,original_hardened_entitlements_preserved=True,static_verification=descriptor(out/'static-verification.json'))
except BaseException as error:r['error']=repr(error);raise
finally:
 if guard is not None:guard.release();r['storage']=guard.record
 atomic_json(out/'build.json',r)
print(json.dumps({'app_construction_passed':True,'commands':len(r['commands']),'changed_entries':len(r['changed_inventory_entries']),'neural_calls':0}))
