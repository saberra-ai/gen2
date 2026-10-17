"""Independently collect saved pooled CPU mixed residency evidence; no model calls."""
from pathlib import Path
import json,re,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory,restore_archive
from package_macos import verify
from macos_loaded_libraries import audit_loaded_libraries
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
root=cache/'pooled-mixed-local-v3';read=lambda p:json.loads(p.read_text());p=read(root/'protocol.json');q=read(root/'qualification.json');source='f21c5616d51d6f20600ccee51489ca8178ec13f3'
assert q['source_commit']==source and q['pooled_cpu_mixed_residency_passed']is True and 'error'not in q and q['release_qualified']is False
assert descriptor(root/'protocol.json')==q['protocol'] and (root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-mixed-v3.protocol.json'],cwd=repo)
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files']);verify_git_sources(repo,source,pins);verify_git_sources(repo/'vendor/llama-cpp-sys-2/llama.cpp',p['vendor_revision'],p['vendor_sources'])
expected=['compile-controls','mel-build-input-readable','rustc-version','mixed-helper-build','runtime-controls','premodel-controls','version-only','list',*p['tests']]
assert [c['name']for c in q['commands']]==expected and all(c['returncode']==0 for c in q['commands']) and q['neural_commands_started']==p['tests']
assert q['producer_artifact']['id']==p['producer']['id']==12082417995
inputs=read(root/'build-inputs.json');work=root/'work';package=work/'nemo';manifest=verify(package)
assert manifest['runtime_profile']=='gen2-valid-mel-pool-v1'and manifest['backend']=='cpu'
for name,directory in [('work',work),('producer',root/'producer'),('public_helpers',root/'public-helpers')]:assert inventory(directory)==inputs['protected_inputs'][name]
for n,v in p['model_and_provider_files'].items():assert descriptor(work/n)==v
for n,v in p['compiler_tools'].items():assert descriptor(Path(v['path']))==v['file']
assert inputs['compiler_environment']['CMAKE']==p['compiler_tools']['cmake']['path'] and inputs['compiler_environment']['CMAKE_BUILD_PARALLEL_LEVEL']=='2'
helpers=read(root/'pooled-mixed-helpers-files.json');assert {k:v for k,v in helpers.items()if k!='files'}==q['helper_archive'];assert inventory(work/'bin')==helpers['files'];restore_archive(root/helpers['archive'],helpers,root/'independent-helper-restore');assert inventory(root/'independent-helper-restore')==helpers['files']
assert len(q['cases'])==2
for row in q['cases']:
 name=row['name'];assert row['passed']and row['diarization_backend']=='cpu'
 text=(root/(name+'.stdout.log')).read_text();assert re.search(r'^test '+re.escape(name)+r' \.\.\. ok$',text,re.MULTILINE)and'1 passed; 0 failed; 0 ignored;'in text
 placement={}
 for line in text.splitlines():
  if line.startswith('chat-placement '):
   phase,value=line.removeprefix('chat-placement ').split(': ',1);assert phase not in placement;placement[phase]=json.loads(value)
 assert placement==row['chat_placement']and set(placement)=={'initial','restored'}
 assert all(v['layers']>0 and v['backend']=='MTL'and v['device']=='Apple M4 Pro'for v in placement.values())
 replay=audit_with_auxiliary((root/(name+'.stderr.log')).read_text(),package,manifest,work/'bin/mixed-native-tests',{str(work/'providers/libonnxruntime.dylib'):p['model_and_provider_files']['providers/libonnxruntime.dylib']});assert replay==q['loaded_images'][name]
assert audit_loaded_libraries((root/'version-only.stderr.log').read_text(),package,manifest,work/'bin/load-probe')==q['loaded_images']['version-only']
assert audit_loaded_libraries((root/'list.stderr.log').read_text(),package,manifest,work/'bin/mixed-native-tests',False)==q['loaded_images']['list']
assert q['storage']['released']and q['storage']['minimum_observed_free_bytes']>=1024**3
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-mixed';assert not out.exists();out.mkdir()
for path in root.iterdir():
 if path.is_file()and path.suffix in {'.json','.log','.sb'}:shutil.copy2(path,out/path.name)
shutil.copy2(__file__,out/Path(__file__).name)
qualification={'schema_version':1,'pooled_cpu_mixed_residency_passed':True,'release_qualified':False,'source_commit':source,'parent_source_pins_verified':len(pins),'vendor_source_pins_verified':len(p['vendor_sources']),'actual_commands':len(expected),'neural_load_order_cases':p['tests'],'cases':q['cases'],'all_protected_inputs_unchanged':True,'helper_archive_independently_restored':True,'loaded_images_replayed':list(q['loaded_images']),'new_model_calls_during_collection':0,'limits':['Same named-M4 mixed residency/refusal/eviction/restoration/shutdown smoke with tiny silence only. No simultaneous neural, speech accuracy, app, GUI, fresh-machine, uncontended performance or release claim.','Full weight/provider/helper payloads remain local and are independently rehashed; Git retains descriptors and observed process results.']}
(out/'collection-qualification.json').write_text(json.dumps(qualification,indent=2)+'\n');files={x.relative_to(out).as_posix():descriptor(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members')
