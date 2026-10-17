"""Actual local source composition only; no compiler/helper/native/model/network."""
import sys,json,hashlib,subprocess
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'bounded-windows-source-admission-v1';source=root/'source';out=root/'composition'
out.mkdir();sys.path.insert(0,str(tools))
from pooled_runtime import BOUNDED_PROFILE,load_profile,compatibility_patch_bytes,profile_provenance
from platform_bounded_sources import verify_completed_source,INFERENCE_FILES
from windows_naming import prepare_private_names
from windows_compat import apply_windows_lf_compat
from build_valid_mel_candidate import descriptor
path,lock=load_profile(BOUNDED_PROFILE);patch=out/'bounded-inference.patch';patch.write_bytes(compatibility_patch_bytes(lock))
verify_completed_source(source,None,lock,patch,True)
hook,naming=prepare_private_names(source,tools,out)
compat=apply_windows_lf_compat(source,out)
build=source/'build/gen2-cpu-diar';build.mkdir(parents=True)
verify_completed_source(source,build,lock,patch,True)
record={'schema_version':1,**profile_provenance(path,lock),'actual_local_source_composition_passed':True,
 'windows_compiler_or_native_behavior_qualified':False,'release_qualified':False,'new_native_or_neural_calls':0,
 'inference_patch':descriptor(patch),'source_files':{n:descriptor(source/n) for n in sorted(INFERENCE_FILES|{'cmake/llama_cpp.cmake'})},
 'private_naming_prepared_before_lf_overlay':True,'naming':naming,'build_compatibility':compat,
 'limits':['Actual Git composition and pinned source hashes only on macOS; no SDK/build/version query/native helper/model/Windows API/workflow run.',
           'MSVC construction, actual compiler tree, negative-load thread cleanup, RTF, lifecycle, packaging, offline chat and fresh desktop remain unqualified.']}
(out/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'actual_local_source_composition_passed':True,'source_files':len(record['source_files']),'new_native_or_neural_calls':0}))
