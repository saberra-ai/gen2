import hashlib
import json
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def rec(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
def tree(p):return {str(q.relative_to(p)):{**rec(q),'symlink_target':str(q.readlink())if q.is_symlink()else None}for q in sorted(p.rglob('*'))if q.is_file()}
long=cache/'valid-mel-cpu-long-v1/result.json';assert json.loads(long.read_text())['fixed_cpu_30m_1h_compaction_parity_passed']
assert json.loads(Path(str(long.parent)+'.qualification.command.json').read_text())['returncode']==0
old=json.loads((tools/'valid-mel-macos-package-v2.protocol.json').read_text())
work=cache/'valid-mel-metal-runtime-inputs-v1/work';missing=cache/'valid-mel-metal-load-control-v1/missing/nemo';cpu=cache/'valid-mel-macos-runtime-inputs-v1/work/nemo'
p={k:old[k]for k in ['schema_version','runtime_version','pcm_sha256','lifecycle_tests','command_timeout_seconds']}
p['command_timeout_seconds']=180
p.update(scope='Fixed relocated tagged macOS arm64 Metal model/loader/public-native and lifecycle qualification on the named Apple M4 Pro. No heldout quality or all-release claim.',output_path=str(cache/'valid-mel-metal-package-qualification-v1'),work_path=str(work),load_probe_path=str(cache/'valid-mel-macos-load-probe-v1/load-probe'),public_prior_build_path=str(cache/'tagged-valid-mel-probes-v1/build.json'),prior_cpu_long_result_path=str(long),cpu_package_path=str(cpu),missing_entry_path=str(missing/'lib/libnemo_speech_asr_c.dylib'),missing_library_path=str(missing/'lib/libgen2_nemo_metal_ggml-base.0.25.1.dylib'),expected_metal_device='Apple M4 Pro',combined_metal_test='explicit_metal_matches_native_reference_and_cpu_sdk_cannot_silently_fallback',rust_sources=json.loads((work.parent/'build.json').read_text())['rust_source_files'])
names=['qualify_valid_mel_metal_package.py','qualify_valid_mel_macos_package.py','macos_loaded_libraries.py','macos_multiple_loaded_libraries.py','test_macos_multiple_loaded_libraries.py','macos_runtime_load_probe.cpp','test_macos_loaded_libraries.py','compare_stream_valid_mel.py','test_stream_comparison_valid_mel.py','test_macos_package_comparison.py','package_macos.py','test_package_macos.py','runtime_profiles.py','test_runtime_profiles.py','build_valid_mel_candidate.py','artifacts.py','runtime.lock.json','runtime-valid-mel.lock.json','patches/v3-finish-valid-mel.patch']
p['tools']={n:rec(tools/n)for n in names}
p['tree_files']={str(root):tree(root)for root in [work,missing,cpu]}
inputs={}
for root in [work.parent,cache/'valid-mel-macos-load-probe-v1',cache/'tagged-valid-mel-probes-v1']:
 for f in root.iterdir():
  if f.is_file():inputs[str(f)]=rec(f)
for name in ['build-valid-mel-metal-runtime-inputs.py','run-valid-mel-metal-package.py','freeze-valid-mel-metal-package.py']:
 inputs[str(cache/name)]=rec(cache/name)
for f in [long,Path(str(long.parent)+'.qualification.command.json'),cache/'valid-mel-macos-runtime-inputs-v1/build.json',cache/'valid-mel-tagged-metal-build-v1/build/build.json']:
 inputs[str(f)]=rec(f)
b=json.loads((work.parent/'build.json').read_text());inputs.update(b['helper_source_files'])
prior=json.loads((cache/'tagged-valid-mel-probes-v1/build.json').read_text())
for n in prior['source_files']:inputs[str(repo/n)]=rec(repo/n)
for receipt in ['2026-10-16-valid-mel-metal-construction','2026-10-16-valid-mel-metal-macos-package','2026-10-16-valid-mel-macos-runtime','2026-10-16-valid-mel-cpu-long']:
 root=repo/'benches/results/nemotron-diarization/apple-m4-pro'/receipt
 assert root.is_dir()
 for f in root.rglob('*'):
  if f.is_file():inputs[str(f)]=rec(f)
p['inputs']=inputs;p['external_launcher']=rec(cache/'run-valid-mel-metal-package.py')
p['preflights']=old['preflights']+[{'name':'multiple-loader-preflight','pattern':'test_macos_multiple_loaded_libraries.py','tests':2}]
p['denied_paths']=[str(cache/name)for name in ['sdk','sdk-metal','sdk-valid-mel-cpu-v1','sdk-valid-mel-tagged-cpu-v1','sdk-valid-mel-tagged-metal-v1','NeMo-Speech.cpp','NeMo-Speech-valid-mel-v1','NeMo-Speech-valid-mel-tagged-v1','NeMo-Speech-valid-mel-metal-v1','valid-mel-macos-package-v1','valid-mel-metal-macos-package-v1','assets','native-f32-conversion-v1','tail-framing-v1']]+['/opt/homebrew','/Users/victor/.cache/gen2-bark-onnx/build']
p['readonly_paths']=[str(root)for root in [work,missing,cpu,cache/'valid-mel-macos-load-probe-v1',tools,repo/'src',repo/'tests']]
p['execution_order']=['31 model-free preflight tests; require completed CPU long-session PASS and frozen compiled helper/provenance/asset/package bytes.','Version-only relocated Metal closure, then missing-private dependency control before model creation.','One independent C API 60s GPU0 stream and one public Rust --metal stream on identical PCM; compare every6000row/48000probabilities and allturns.','Seven exact existing lifecycle tests, one child each, explicit Metal execution.','Isolated CPU-only SDK explicit Metal admission failure with no probability output, then one exact combined CPU-rejection/Metal-parity test in another child.','Recheck all frozen inputs/tool/Rust bytes. Preserve failures and available raw arrays, do not retry.']
p['acceptance']={'public_native_probability_tolerance':1e-6,'public_native_turn_tolerance_seconds':1e-6,'public_native_frames':6000,'metal_device':'Apple M4 Pro','native_gpu_ordinal':0,'metal_private_libraries':93,'cpu_negative_private_libraries':92,'combined_trace_private_libraries':185,'combined_test_probability_maximum_exclusive':1e-5,'combined_test_turn_maximum_exclusive_seconds':1e-6,'lifecycle_tests':7,'actual_reload_cycles_in_existing_test':100,'warm_cache_native_images':0,'missing_private_library_expected_exit':1,'cpu_metal_unavailable_expected_exit':1}
p['limits']=['Fixed13child commands, version/missing controls, two60s streams, seven existing one-test lifecycle calls, isolatedCPUunavailablecontrol and one existingcombinedtest. Nativeforwardcount notinstrumented. No addedHF/originalNeMo/DER/qualitytargets or extraNNcases. Eachchildboundedby180seconds, timeoutkillsitsprocessgroup andfailsattempt.','CPU long-sessionattemptmustcompletePASSfirst. PurePythonarray/admissioncontrollerunsandboxed; eachnativechildgetsstrictnetwork/source/Homebrew/originalpackage/read-onlyinputprofile exactlyonce. All13commands keepstdout/stderr/actualdyldtraces andprocessgroup/status. No automaticretry.','Everyfinalvalidrow andallturnsmustmatchprimarypublic/native at1e-6 withactualnormalizedPCMidentity. No extraMelrowallowance. Actualpublicbackend/device andCAPIrequestedordinalmustmatchMetal/AppleM4Pro/0. UnsupportednativegraphpartitionsmayuseCPU; thisdoesnotclaimall-GPU execution.','All93privateimagesmustmatchqualifiedMetalpackagebytes ineachsuccessfulMetalruntimeprocess; warmcachemustloadnone; isolatedCPUnegativeusesall92CPUimages. Thecombinedtestrequiresbothcompleteprivateclosures intherawprocesshistory andexistingprobability<1e-5/turn<1e-6 checks. CPUrejectedloadisdroppedbeforeMetalmodelload, so thisprovessequentialCPUrejectionthenMetalparity inoneprocess, notsimultaneousCPU/Metalresidency orchat/Bark/Layacoexistence.','ThisisoneM4Pro/ordinarybackground/warmfilecache qualification. ItdoesnotqualifyMetal30m/1hstreams, heldoutquality, physicalmic/GUI/desktoplatency, mixedmodels, otherOS/hardware, signing/notarization orrelease. OriginalAMI<=20%andallotherroadmapgatesremainunchanged.']
path=tools/'valid-mel-metal-package.protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2)+'\n');print(rec(path))
