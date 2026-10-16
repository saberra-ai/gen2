"""Collect saved Metal construction evidence; inspect bytes without native loads."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
src=cache/'valid-mel-tagged-metal-build-v1'
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-construction'
def rec(path):
    data=path.read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def read(path):return json.loads(path.read_text())
p=read(src/'protocol.json');b=read(src/'build/build.json');commit=b['source_commit']
assert commit=='448c632a5a8a2bf4bc60bfec29a1b44c3935de42'
assert subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/valid-mel-metal-build.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
for name,expected in p['tools'].items():
    raw=subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/'+name],cwd=repo)
    assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected==rec(tools/name)
assert rec(cache/'run-valid-mel-metal-build.py')==p['external_launcher']
for name,expected in p['preparation_files'].items():assert rec(Path(p['preparation_path'])/name)==expected
outer=read(src/'commands.json')
assert [c['phase']for c in outer]==[x['name']for x in p['preflights']]+['construction']
assert all(c['returncode']==0 and c['source_commit']==commit for c in outer)
assert b['commands']==read(src/'build/commands.json') and len(b['commands'])==11 and all(c['returncode']==0 for c in b['commands'])
assert [c['phase']for c in b['commands']]==['pristine-llama-archive','configure','build','install','model-free-state-test','version-only','metadata-cmake','metadata-ninja','metadata-clang','metadata-xcode-sdk','metadata-brew-packages']
assert b['runtime_reported_version']==(src/'build/version-only.stdout.log').read_text().strip()==p['acceptance']['runtime_version']
assert '100% tests passed, 0 tests failed out of 1' in (src/'build/model-free-state-test.stdout.log').read_text()
assert b['backend']=='metal' and b['pristine_llama_compiler_tree_verified'] and not b['cpu_ggml_patches_applied']
assert b['model_free_state_test_completed'] and b['version_function_only_executed']
for flag in ['model_instantiated_or_inference_performed','artifact_admitted_for_inference','production_qualified','release_qualified']:assert not b[flag]
source=Path(p['source_path']);sdk=Path(p['prefix_path'])
sys.path.insert(0,str(tools))
from runtime_profiles import validate_provenance
from valid_mel_metal_audit import KINDS,embedded_payloads,pristine_files,shader_text
validate_provenance(b)
audit=read(src/'build/pristine-metal-llama-audit.json')
assert audit['files']==pristine_files(source/'llama.cpp') and len(audit['files'])==3619
assert rec(src/'build/pristine-metal-llama.tar')==audit['archive']
shader=read(src/'build/metal-shader-audit.json')
root=source/'llama.cpp/ggml/src/ggml-metal'
payloads={kind:shader_text(root,kind)for kind in KINDS}
assert shader['actual_installed_payloads']==embedded_payloads(sdk/'lib/libggml-metal.0.25.1.dylib',payloads)
assert rec(sdk/'lib/libggml-metal.0.25.1.dylib')==shader['installed_library']
for name,expected in shader['generated_files'].items():assert rec(src/'build/metal-payloads'/name)==expected
assert len(shader['generated_files'])==40 and shader['shader_text_bundles']==20
assert len(b['libraries'])==7 and len(b['installed_library_paths'])==18
for name,expected in b['installed_library_paths'].items():
    path=sdk/'lib'/name
    assert rec(path)=={k:expected[k]for k in ['bytes','sha256']}
    assert (str(path.readlink())if path.is_symlink()else None)==expected['symlink_target']
archive=b['runtime_archive'];assert rec(Path(archive['path']))=={k:archive[k]for k in ['bytes','sha256']}
assert not out.exists()
shutil.copytree(src,out,ignore=shutil.ignore_patterns('pristine-metal-llama.tar'))
shutil.copytree(Path(p['preparation_path']),out/'preparation')
shutil.copy2(source/'llama.cpp/LICENSE',out/'LICENSE.ggml')
for name in ['run-valid-mel-metal-build.py','collect-valid-mel-metal-build.py']:shutil.copy2(cache/name,out/name)
for name in p['tools']:
    target=out/'construction-tools'/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/'+name],cwd=repo))
q={'schema_version':1,'source_commit':commit,'tagged_metal_sdk_constructed':True,'actual_saved_runtime_version':b['runtime_reported_version'],'outer_commands_passed':len(outer),'construction_phase_commands_passed':11,'model_free_preflight_tests_passed':12,'model_free_state_test_completed':True,'version_function_only_executed':True,'pristine_compiler_files':3619,'cpu_ggml_patches_applied':False,'generated_shader_text_bundles':20,'generated_shader_and_assembly_files':40,'actual_installed_payload_spans_verified':20,'embedded_payloads_are_shader_text_not_precompiled_gpu_code':True,'sdk_library_paths':18,'sdk_real_libraries':7,'runtime_static_archive_bound':True,'neural_calls':0,'model_instantiated_or_inference_performed':False,'artifact_admitted_for_inference':False,'production_qualified':False,'release_qualified':False,'omitted_binary_inputs':{'pristine_llama_archive':audit['archive'],'installed_sdk_libraries':b['installed_library_paths'],'runtime_archive':b['runtime_archive']},'limits':p['limits']+['Collection replays saved hashes, physical source inventory, shader flattening and installed Mach-O payload spans using Python byte reads only. No native library load/version/state test/build/model rerun. Native SDK, source Git archive, weights and audio are omitted; their original hashes and compiler/source manifests remain.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
files={str(f.relative_to(out)):rec(f)for f in sorted(out.rglob('*'))if f.is_file()}
(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'members':len(files),'shader_text_bundles':20,'native_calls_in_collection':0,'neural_calls':0}))
