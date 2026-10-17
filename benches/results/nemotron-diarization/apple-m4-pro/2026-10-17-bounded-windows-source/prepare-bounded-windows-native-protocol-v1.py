"""Metadata-only preparation; no builder/helper/model or workflow invocation."""
import json,sys,subprocess,hashlib
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
sys.path.insert(0,str(tools))
from pooled_runtime import BOUNDED_PROFILE,load_profile,profile_provenance
p=json.loads((tools/'valid-mel-platform-native.protocol.json').read_text())
p['scope']='Fresh WindowsServer2022 x86_64 AVX2/FMA/F16C bounded CPU SDK construction; same-host full native/public parity, seven lifecycle gates and native/public RTF each strictly below one. Not offline packaging, fresh machine, chat coexistence, heldout quality or release.'
lockpath,lock=load_profile(BOUNDED_PROFILE);p['runtime_provenance']=profile_provenance(lockpath,lock)
p['rtf_limit_exclusive']=1.0
p['native_probe_derivation']='Replace exactly one quoted original gen2-valid-mel-v1 version literal with bounded CPU-v2; preserve all other source bytes and retain original/derived descriptors.'
p['source_policy'][0]='Require pristine pinned native root and LLAMA submodule, then apply exact valid-Mel, CPU-thread-cap and persistent-pool inference patches in locked order; check pre/post source hashes and exact four-file diff.'
p['source_policy'].append('Bind committed protocol and source bytes before construction; preserve functional comparison/lifecycle evidence even if either RTF is at least one. No automatic retry.')
p['limits'][1]='Same-host Windows parity/lifecycle/RTF does not establish offline/fresh-machine packaging, desktop latency, negative-load thread cleanup, or supported release.'
p['tools'].update({n:{} for n in ['platform_bounded_windows_native.py','platform_bounded_sources.py','platform_bounded_metrics.py','test_platform_bounded_windows.py','pooled_runtime.py','runtime-valid-mel-pool.lock.json','runtime-valid-mel-pool-cpu.lock.json','patches/cpu-thread-cap.patch','patches/persistent-cpu-pool.patch','frozen_git_sources.py','ctest_output.py','test_ctest_output.py']})
names=subprocess.check_output(['git','ls-files','src','Cargo.toml','Cargo.lock','build.rs','.cargo','examples/nemotron_valid_mel_stream.rs','tests/diarization_native.rs'],cwd=repo,text=True).splitlines()
p['repository_files'].update({n:{} for n in names})
p['repository_files']['.github/workflows/diarization-bounded-windows-native.yml']={}
def desc(path):
 b=path.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
for key,root in [('tools',tools),('repository_files',repo)]:p[key]={n:desc(root/n) for n in sorted(p[key])}
path=tools/'bounded-windows-native.protocol.json';path.write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'protocol':desc(path),'tools':len(p['tools']),'repository_files':len(p['repository_files']),'new_native_model_or_workflow_calls':0}))
