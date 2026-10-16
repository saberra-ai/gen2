"""Clone/cache-byte verification only; no model, native query or helper execution."""
import json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from package_macos import verify
from qualification_storage import atomic_json
out=cache/'valid-mel-mixed-inputs-v1';assert not out.exists();out.mkdir();work=out/'work';work.mkdir()
build=cache/'valid-mel-mixed-helper-build-v1';b=json.loads((build/'command.json').read_text());assert b['compiled']is True and b['returncode']==0 and b['models_loaded']is False
for name,expected in b['repository_files'].items():assert descriptor(repo/name)==expected,name
app=cache/'release Metal app qualification 024a88f/Pnola.app/Contents'
pairs=[('cpu/nemo',cache/'valid-mel-macos-package-v1/relocated/nemo'),('metal/nemo',cache/'valid-mel-metal-macos-package-v1/relocated/nemo'),('models/nemotron',app/'Resources/nemotron'),('models/bark',app/'Resources/bark'),('models/laya',cache/'relocated mixed models/laya'),('models/chat.gguf',cache/'relocated mixed models/chat.gguf'),('providers/libonnxruntime.dylib',app/'Frameworks/libonnxruntime.dylib'),('bin/mixed-native-tests',build/'mixed-native-tests')]
commands=[];origins={}
for name,source in pairs:
 destination=work/name;destination.parent.mkdir(parents=True,exist_ok=True)
 command=['/bin/cp','-cR',str(source.resolve()),str(destination)];r=subprocess.run(command,capture_output=True,text=True);commands.append({'name':name,'argv':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});atomic_json(out/'commands.json',commands);r.check_returncode()
 files=[source]if source.is_file()else sorted(p for p in source.rglob('*')if p.is_file()and not p.is_symlink())
 for path in files:
  relative=Path('.')if source.is_file()else path.relative_to(source)
  copied=destination if source.is_file()else destination/relative
  expected=descriptor(path);assert descriptor(copied)==expected;origins[str(copied.relative_to(work))]={'source':str(path),**expected}
for backend in ['cpu','metal']:
 manifest=verify(work/backend/'nemo');assert manifest['backend']==backend and manifest['runtime_profile']=='gen2-valid-mel-v1'
files={str(p.relative_to(work)):descriptor(p)for p in sorted(work.rglob('*'))if p.is_file()and not p.is_symlink()}
aliases={str(p.relative_to(work)):str(p.readlink())for p in sorted(work.rglob('*'))if p.is_symlink()}
record={'schema_version':1,'scope':__doc__,'models_loaded':False,'native_queries_or_helper_cases_executed':0,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'helper_build_record':descriptor(build/'command.json'),'repository_files':b['repository_files'],'helper':descriptor(work/'bin/mixed-native-tests'),'origins':origins,'files':files,'aliases':aliases,'logical_regular_file_bytes':sum(v['bytes']for v in files.values()),'private_sdk_libraries':{'cpu':92,'metal':93},'future_scope':'Four fresh residency/admission/sequential-smoke tests: actualLaya+Bark+Metalchat with CPU/Metaldiarization, eachloadorder. Eachcase uses1s+20ms silentdiarization and20ms silentBark, not speechparity/fullquality or concurrentNN proof.'}
atomic_json(out/'build.json',record);print(json.dumps({'prepared':True,'files':len(files),'aliases':len(aliases),'logical_bytes':record['logical_regular_file_bytes'],'models_loaded':False}))
