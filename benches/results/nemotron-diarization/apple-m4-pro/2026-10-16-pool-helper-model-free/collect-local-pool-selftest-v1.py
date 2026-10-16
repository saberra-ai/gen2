"""Retain actual model-free prototype attempts; replay bytes only."""
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from package_macos import verify_files,verify_package_provenance
from macos_loaded_libraries import audit_loaded_libraries
from sdk_archive import inventory
v1=cache/'portable-pool-probe-build-v1';v2=cache/'portable-pool-probe-build-v2';v3=cache/'portable-pool-probe-link-v3';control=cache/'portable-pool-selftest-v1'
a=json.loads((v1/'build.json').read_text());b=json.loads((v2/'build.json').read_text());c=json.loads((v3/'link.json').read_text());q=json.loads((control/'qualification.json').read_text())
assert a['returncode']==1 and 'unused variable' in (v1/'compile.stderr.log').read_text() and 'different signs' in (v1/'compile.stderr.log').read_text()
assert [x['returncode'] for x in b['commands']]==[0,0,0,1]
assert 'Undefined symbols' in (v2/'link.stderr.log').read_text() and '_ggml_backend_dev_count' in (v2/'link.stderr.log').read_text()
assert c['returncode']==0 and c['previous_build']==descriptor(v2/'build.json')
assert a['source_pins']==b['source_pins']==q['source_pins'] and len(b['source_pins'])==117
for name,d in b['source_pins'].items():assert descriptor(Path(name))==d
for name,d in c['object_inputs'].items():assert descriptor(Path(name))==d
binary=v3/'work/bin/native-pool-probe';assert descriptor(binary)==c['binary']==q['binary'] and q['link_record']==descriptor(v3/'link.json')
base=cache/'NeMo-Speech-valid-mel-tagged-v1'
materialized=json.loads((cache/'portable-macos-construction-ci-v2/evidence/build/llama-materialized-audit.json').read_text())['files']
headers={}
for name,d in b['source_pins'].items():
 f=Path(name)
 if f.is_relative_to(base/'src'):
  rel=f.relative_to(base).as_posix();raw=subprocess.check_output(['git','show','083861c241061562b8ec29eb06d7f6cdc3af8775:'+rel],cwd=base)
  assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==d
  headers[rel]=d
 if f.is_relative_to(base/'build/cpu-diar/_deps/llama.cpp'):
  rel=f.relative_to(base/'build/cpu-diar/_deps/llama.cpp').as_posix();assert materialized[rel]==d;headers[rel]=d
assert [x['returncode'] for x in q['commands']]==[0,0] and [x['name'] for x in q['commands']]==['sandbox-controls','pool-selftest']
assert q['commands'][1]['argv'][-1]=='--self-test' and q['passed']
assert q['result']=={'model_constructed':False,'frontend_executed':False,'backend_lifetimes':20,'persistent_pool_lifetimes':10,'threads':4}
assert json.loads((control/'pool-selftest.stdout.log').read_text())==q['result']
assert descriptor(control/'model-free.sb')==q['policy']
binding_commit='7db170d0c8f17b3aea670769577d2d3cd2deaf5b'
frozen_producer=json.loads(subprocess.check_output(['git','show',binding_commit+':tools/nemotron-diarization/macos-portable-runtime-v2.protocol.json'],cwd=repo))['producer']
construction=cache/'portable-macos-construction-ci-v2'
original_inventories={}
for kind,count in [('sdk',84),('runtime',123)]:
 name='portable-cpu-'+kind+'-files.json';path=construction/'evidence'/name
 assert descriptor(path)==frozen_producer['files'][name]
 expected=json.loads(path.read_text())['files']
 assert len(expected)==count and inventory(construction/('restored-portable-cpu-'+kind))==expected
 original_inventories[kind]={'manifest':descriptor(path),'entries':count,'all_entries_match':True}
package=cache/'portable-macos-construction-ci-v2/restored-portable-cpu-runtime';manifest=json.loads((package/'package.json').read_text());verify_files(package,manifest['files']);verify_package_provenance(package,manifest)
assert audit_loaded_libraries((control/'pool-selftest.stderr.log').read_text(),package,manifest,binary)==q['loaded_images'] and q['loaded_images']['private_library_count']==92
source=repo/'tools/nemotron-diarization/native_pool_probe.cpp'
assert descriptor(source)==b['source_pins'][str(source)]
record={'schema_version':1,'scope':'Actual M4 model-free private-C++ diagnostic helper preparation/control only; no SDK changes or neural inference, no public API/ABI or release qualification.','helper_source':descriptor(source),'header_and_two_upstream_cpp_inputs_match_pinned_native_or_qualified_materialization':headers,'all_117_compiler_inputs_unchanged':True,'original_construction_binding_commit':binding_commit,'original_sdk_and_runtime_inventories':original_inventories,'failed_initial_warning_and_missing_link_attempts_retained':True,'successful_objects_reused_for_link_continuation':True,'binary':descriptor(binary),'model_free_result':q['result'],'private_libraries':92,'model_or_frontend_execution':False,'neural_abba_experiment_executed':False,'release_qualified':False,'limits':['Private C++ headers and two unmodified helper-local BackendManager/logging translation units require exact version matching; this is not a public API.','The original qualified SDK/package bytes are unchanged. This control proves no model inference or timing parity.','A future same-input experiment must compare every probability row with the qualified C ABI output before interpreting timings.']}
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pool-helper-model-free';assert not out.exists();out.mkdir(parents=True)
for label,root in [('initial-warning-failure',v1),('initial-link-failure',v2),('link-continuation',v3),('model-free-control',control)]:
 target=out/label;target.mkdir()
 for f in root.iterdir():
  if f.is_file() and f.suffix in ['.json','.log','.sb']:shutil.copy2(f,target/f.name)
shutil.copy2(__file__,out/Path(__file__).name)
(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n')
files={str(f.relative_to(out)):descriptor(f) for f in out.rglob('*') if f.is_file()}
(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'files':len(files)+1,'source':str(out)}))
