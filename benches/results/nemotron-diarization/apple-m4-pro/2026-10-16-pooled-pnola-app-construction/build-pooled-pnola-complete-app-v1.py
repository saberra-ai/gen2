"""Build and clone one complete pooled CPU instrumented hardened app; no app, model, version query or GUI execution."""
import json,os,plistlib,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');app=Path('/Users/victor/workspace/pnola');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard,atomic_json
from qualify_valid_mel_pnola import execute_guarded
from package_macos import verify
from verify_pooled_pnola_app import verify_app
from qualify_macos_pooled_mixed import verify_vendor_worktree
from sdk_archive import inventory
out=cache/'pooled-pnola-complete-app-build-v1';assert not out.exists();out.mkdir();work=cache/'valid-mel-pnola-inputs-v2/work';compiled=json.loads((tools/'macos-pooled-mixed-v2.protocol.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
# Historical debug inputs remain in the existing immutable snapshot.
snapshot=cache/'valid-mel-pnola-debug-source-snapshot-v1'
old_apps={name:inventory(path)for name,path in {'cpu':cache/'valid-mel-pnola-app-cpu-snapshot-v1/Pnola.app','metal':cache/'valid-mel-pnola-metal-app-build-v2/Pnola.app'}.items()}
source_names={'package.json','package-lock.json','index.html','tsconfig.json','vite.config.ts'}
for dirname in ['src','scripts','public','src-tauri/src','src-tauri/icons','src-tauri/capabilities']:
 source_names.update(str(p.relative_to(app))for p in(app/dirname).rglob('*')if p.is_file()and not p.is_symlink())
source_names.update(str(p.relative_to(app))for p in(app/'src-tauri').iterdir()if p.is_file())
app_pins={name:descriptor(app/name)for name in sorted(source_names)};gen2_pins={name:descriptor(repo/name)for name in compiled['repository_files']};new_module='integrations/pnola/overlay/src-tauri/src/verification_images.rs';gen2_pins[new_module]=descriptor(repo/new_module)
pooled_package=cache/'pooled-macos-construction-ci-v1/restored-pooled-cpu-runtime'
package_pins={'cpu':verify(pooled_package)}
assert package_pins['cpu']['runtime_profile']=='gen2-valid-mel-pool-v1' and package_pins['cpu']['backend']=='cpu'
extra_tools={name:descriptor(tools/name)for name in sorted(set(compiled['tools'])|{'verify_pooled_pnola_app.py','verify_valid_mel_pnola_app.py'})}
asset_inputs={str(path):inventory(path) for path in [work/'models/bark',work/'models/nemotron',work/'providers']}
asset_files={str(path):descriptor(path) for path in [work/'assets/ami.wav',cache/'release Metal app qualification 024a88f/Pnola.app/Contents/Resources/runtime/LICENSE',app/'node_modules/.package-lock.json']}
launcher_pin=descriptor(Path(__file__))
policy=out/'construction.sb';policy.write_text('(version 1)\n(allow default)\n(deny network*)\n')
env={k:os.environ[k]for k in ['HOME','PATH','TMPDIR','RUSTUP_HOME','CARGO_HOME','DEVELOPER_DIR']if k in os.environ}
env.update(CARGO_TARGET_DIR='/Users/victor/.cache/pnola-nemotron-build',CARGO_BUILD_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',CARGO_INCREMENTAL='0',CARGO_NET_OFFLINE='true',HOMEBREW_NO_AUTO_UPDATE='1',npm_config_offline='true',npm_config_audit='false',PNOLA_SIGNING_IDENTITY='-',GEN2_ROOT=str(repo),GEN2_BARK_BUNDLE=str(work/'models/bark'),GEN2_NEMOTRON_ASSETS=str(work/'models/nemotron'),GEN2_ORT_LIBRARY=str(work/'providers/libonnxruntime.dylib'),GEN2_ORT_LICENSE=str(cache/'release Metal app qualification 024a88f/Pnola.app/Contents/Resources/runtime/LICENSE'),PNOLA_SAMPLE_AUDIO=str(work/'assets/ami.wav'))
commands=[];record={'scope':__doc__,'app_construction_passed':False,'models_loaded_for_inference':False,'native_version_queries_or_neural_calls':0,'source_commit':source,'commands':commands,'apps':{}};guard=None
atomic_json(out/'protocol.json',{'scope':__doc__,'source_commit':source,'gen2_files':gen2_pins,'pnola_source_files':app_pins,'vendor_revision':compiled['vendor_revision'],'vendor_source_files':compiled['vendor_sources'],'extra_tools':extra_tools,'asset_inputs':asset_inputs,'asset_files':asset_files,'historical_debug_input_snapshot_for_provenance_only':str(snapshot),'environment':env,'model_package_manifests':{'cpu':descriptor(pooled_package/'package.json')},'launcher':descriptor(Path(__file__)),'model_payloads_read_for_copy_and_hash_only':True,'construction_storage':{'start_minimum':8*1024**3,'running_minimum':1024**3,'reserve_bytes':256*1024**2},'neural_admission_unchanged':'5GiB after256MiBreserve; no neural calls here','old_app_inventories':old_apps,'app_verifier':descriptor(tools/'verify_pooled_pnola_app.py'),'trace_failure':descriptor(cache/'valid-mel-pnola-complete-app-runtime-v1/qualification.json'),'callback_proof':descriptor(cache/'process-image-callback-v1/result.json')})
def frozen():
 for root,records in [(repo,gen2_pins),(app,app_pins)]:
  for name,expected in records.items():assert descriptor(root/name)==expected,name
 assert verify(pooled_package)==package_pins['cpu']
 assert descriptor(Path(__file__))==launcher_pin
 for name,expected in asset_inputs.items():assert inventory(Path(name))==expected,name
 for name,expected in asset_files.items():assert descriptor(Path(name))==expected,name
 verify_vendor_worktree(repo/'vendor/llama-cpp-sys-2/llama.cpp',compiled['vendor_revision'])
 for name,expected in compiled['vendor_sources'].items():assert descriptor(repo/'vendor/llama-cpp-sys-2/llama.cpp'/name)==expected,name
 for name,expected in extra_tools.items():assert descriptor(tools/name)==expected,name
 for name,path in {'cpu':cache/'valid-mel-pnola-app-cpu-snapshot-v1/Pnola.app','metal':cache/'valid-mel-pnola-metal-app-build-v2/Pnola.app'}.items():assert inventory(path)==old_apps[name]
def run(name,argv,variables):
 command={'name':name,'argv':argv,'environment':variables,'cwd':str(app)};commands.append(command);atomic_json(out/'commands.json',commands);started=time.monotonic()
 try:
  with(out/(name+'.stdout.log')).open('xb')as stdout,(out/(name+'.stderr.log')).open('xb')as stderr:
   execute_guarded(argv,cwd=app,environment=variables,stdout=stdout,stderr=stderr,guard=guard,record=command,persist=lambda:atomic_json(out/'commands.json',commands),timeout=1800)
 finally:command['elapsed_seconds']=time.monotonic()-started;atomic_json(out/'commands.json',commands)
 if command['returncode']!=0:raise RuntimeError('app construction command failed: '+name)
try:
 guard=StorageGuard(out,8*1024**3,1024**3,256*1024**2)
 for backend in ['cpu']:
  frozen();variables={**env,'GEN2_NEMO_PACKAGE':str(pooled_package)}
  guard.before_model();run(backend+'-build',['/usr/bin/sandbox-exec','-f',str(policy),'npm','run','desktop:build'],variables)
  built=Path(env['CARGO_TARGET_DIR'])/'release/bundle/macos/Pnola.app';dest=out/backend/'Pnola.app';dest.parent.mkdir()
  run(backend+'-clone',['/bin/cp','-cR',str(built),str(dest)],env)
  run(backend+'-signature',['/usr/bin/codesign','--verify','--deep','--strict',str(dest)],env)
  package=verify(dest/'Contents/Resources/nemo');assert package==package_pins[backend]
  info=plistlib.loads((dest/'Contents/Info.plist').read_bytes());assert info['LSMinimumSystemVersion']==package['minimum_macos']
  bundle=json.loads((dest/'Contents/Resources/nemotron/bundle.json').read_text());assert bundle['runtime_profile']=='gen2-valid-mel-pool-v1'
  frames=dest/'Contents/Frameworks';expected=set(package['libraries'])|{'libnemo_speech_asr_c.dylib','libonnxruntime.dylib'}
  assert {f.name for f in frames.iterdir()}==expected
  static=verify_app(dest);assert static['backend']==backend;atomic_json(out/(backend+'-static-verification.json'),static)
  record['apps'][backend]={'path':str(dest),'static_verification':descriptor(out/(backend+'-static-verification.json')),'package_manifest':descriptor(dest/'Contents/Resources/nemo/package.json'),'info_plist':descriptor(dest/'Contents/Info.plist'),'executable':descriptor(dest/'Contents/MacOS/pnola-desktop'),'minimum_macos':info['LSMinimumSystemVersion'],'framework_entries':len(expected),'files':inventory(dest)}
  atomic_json(out/'build.json',record);frozen()
 record['app_construction_passed']=True
except BaseException as error:record['error']=repr(error);raise
finally:
 if guard is not None:guard.release();record['storage']=guard.record
 atomic_json(out/'build.json',record)
print(json.dumps({'app_construction_passed':record['app_construction_passed'],'backends':list(record['apps']),'models_loaded_for_inference':False}))
