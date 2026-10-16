import hashlib
import json
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def rec(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
def tree(root):return {str(p.relative_to(root)):{**rec(p),'symlink_target':str(p.readlink())if p.is_symlink()else None}for p in sorted(root.rglob('*'))if p.is_file()}
old=json.loads((tools/'valid-mel-cpu-long.protocol.json').read_text())
work=cache/'valid-mel-metal-runtime-inputs-v1/work'
prior=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-runtime/qualification.json'
assert json.loads(prior.read_text())['fixed_packaged_metal_parity_and_lifecycle_passed']
p={k:old[k]for k in ['schema_version','runtime_version','cases','command_timeout_seconds','acceptance','source_identity_policy']}
p.update(scope='Fixed macOS arm64 tagged Metal 30-minute and one-hour repeated smoke streams. Native/public retained-tail parity, compaction, provisional revisions and measured process RSS only.',output_path=str(cache/'valid-mel-metal-long-v1'),work_path=str(work),build_metadata_path=str(work.parent/'build.json'),prior_package_qualification_path=str(prior),expected_metal_device='Apple M4 Pro')
p['acceptance']={**p['acceptance'],'full_runtime_private_images_per_process':93,'backend':'metal','device':'Apple M4 Pro','native_gpu_ordinal':0}
names=['qualify_valid_mel_metal_long.py','valid_mel_long_metrics.py','test_valid_mel_long_metrics.py','qualify_valid_mel_macos_package.py','macos_loaded_libraries.py','test_macos_loaded_libraries.py','compare_stream_valid_mel.py','test_stream_comparison_valid_mel.py','package_macos.py','test_package_macos.py','runtime_profiles.py','test_runtime_profiles.py','build_valid_mel_candidate.py','artifacts.py','runtime.lock.json','runtime-valid-mel.lock.json','patches/v3-finish-valid-mel.patch','native_stream_probe_valid_mel.cpp']
p['tools']={n:rec(tools/n)for n in names}
b=json.loads((work.parent/'build.json').read_text());p['rust_sources']=b['rust_source_files'];p['work_files']=tree(work)
inputs={}
for root in [work.parent,prior.parent]:
 for f in root.rglob('*'):
  if f.is_file()and not f.is_relative_to(work):inputs[str(f)]=rec(f)
inputs.update(b['helper_source_files'])
for name in ['build-valid-mel-metal-runtime-inputs.py','run-valid-mel-metal-long.py','freeze-valid-mel-metal-long.py']:
 inputs[str(cache/name)]=rec(cache/name)
p['inputs']=inputs;p['external_launcher']=rec(cache/'run-valid-mel-metal-long.py')
p['denied_paths']=[str(cache/name)for name in ['sdk','sdk-metal','sdk-valid-mel-cpu-v1','sdk-valid-mel-tagged-cpu-v1','sdk-valid-mel-tagged-metal-v1','NeMo-Speech.cpp','NeMo-Speech-valid-mel-v1','NeMo-Speech-valid-mel-tagged-v1','NeMo-Speech-valid-mel-metal-v1','valid-mel-macos-package-v1','valid-mel-metal-macos-package-v1','valid-mel-macos-runtime-inputs-v1','assets','native-f32-conversion-v1','tail-framing-v1']]+['/opt/homebrew','/Users/victor/.cache/gen2-bark-onnx/build']
p['readonly_paths']=[str(root)for root in [work,tools,repo/'src',repo/'tests']]
p['preflights']=[{**case,'tests':5}if case['name']=='long-metrics-preflight'else case for case in old['preflights']]
p['execution_order']=['31 model-free preflight tests and frozen source/input/prior fixed Metal qualification admission.','30-minute independent C API GPU0 stream, then public Rust --metal stream; saved-array/loader/RSS/revision comparison.','60-minute independent C API GPU0 stream, then public Rust --metal stream; same comparison.','Recheck all frozen bytes and staged tree; preserve failures without retry.']
p['limits']=['Exactly four model instantiations/stream runs in the fixed order on repeated named smoke PCM. No additional helper/reload/HF/originalNeMo/DER/heldout targets or extra NN cases. Internal forward counts are not instrumented. Each command is bounded by1800seconds; timeout kills the whole measured process group and fails the attempt. No automatic retries.','All31preflights are network-sandboxed. PurePythonadmission/arraycontrollerhasnooutersandbox; eachnativechildgetsstrictnetwork/source/Homebrew/originalpackage denials andread-onlystagedinputs exactlyonce. Actualdyldtracesmustmatchall93privateMetalpackageimages withqualifiedbytes.','Everyretainedprobabilityrow includingthelast andallturnsmustmatchnative/public at1e-6; totals180000/360000frames, actualnormalizedrepeatedPCM/repeatcount exact, positivecompactedbase andprescribed60000-rowten-minuteminimumtailretained. No historicalextraMelrowallowance. Backend/device/ordinalmustbeMetal/AppleM4Pro/0; unsupportedgraphpartitionsmaystilluseCPU.','MeasuredmacOSprocesspeakRSSmustnotexceed1024MiB andcomputeRTFmustremain<1. Boundedcoalescedrevisionsmustcoverthesessionwithstrictpositiveincreasingrevision/audio/walltimes andactualbackpressure. OrdinaryOS/background/warmfilecache; non-neuralcompilationmayoverlap. Notuncontendedperformanceorrealcapture/UIlatency. ProcessRSSisnotanaggregateGPUallocationorwhole-systemmemorymeasurement.','Thisrepeatedfixturevalidatesretention/compaction/RSS/revisions/native-publiccorrectnessonlyonthenamedM4Pro. Itdoesnotqualifydiverselong-audioquality, simultaneousCPU/Metal/chat/Bark/Layacoexistence, otherhardware/OS, physicalmic/desktop, signing/notarizationorallreleasegates. OriginalAMI<=20%andallroadmapcriteriaremainunchanged.']
path=tools/'valid-mel-metal-long.protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2)+'\n');print(rec(path))
