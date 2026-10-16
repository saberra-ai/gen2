"""Collect complete conversion/source accounting without admitting inference."""
import hashlib,json,shutil,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');base=cache/'native-f32-conversion-v1';native=cache/'NeMo-Speech.cpp'
sys.path.insert(0,str(native/'llama.cpp/gguf-py'))
from gguf import GGUFReader
read=lambda p:json.loads(p.read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();record=lambda p:{'bytes':p.stat().st_size,'sha256':digest(p)}
source='23d2db780a697dc3bf3c6f20a8948e82fd907b4e';protocol=read(base/'protocol.json');command=read(base/'command.json');account=read(base/'source-tensor-accounting.json');construction=read(base/'construction.json')
assert command['source_commit']==source and command['returncode']==0
assert command['protocol_sha256']==account['protocol_sha256']==digest(base/'protocol.json')
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/native-f32-conversion.protocol.json'],cwd=repo)==(base/'protocol.json').read_bytes()
assert command['environment']==protocol['environment'] and command['argv'][3:]==protocol['argv']
assert digest(base/'offline.sb')==command['sandbox_sha256']and'(deny network*)'in(base/'offline.sb').read_text()
for group,root in [('converter_files',native),('gguf_files',native/'llama.cpp/gguf-py')]:
 for name,expected in protocol[group].items():assert record(root/name)==expected
artifact=base/construction['name'];assert record(artifact)==construction['artifact']==account['artifact']
assert account['exact_source_mapping_cast_accounting_passed']and not account['errors']
assert account['source_tensor_count']==363 and account['stored_tensor_count']==360
assert account['source_dtype_counts']=={'torch.bfloat16':363}and account['stored_dtype_counts']=={'F32':359,'F16':1}
assert all(account[key]is False for key in ['neural_model_inference_performed','release_qualified','artifact_admitted_for_inference'])
assert account['checkpoint_deserialization_performed']is True and account['audit_deserialization_weights_only']is True
reader=GGUFReader(str(artifact));by_name={t.name:t for t in reader.tensors};assert len(by_name)==360
mapped=[row for row in account['source_tensors']if row['destination']is not None];skipped=[row for row in account['source_tensors']if row['destination']is None]
assert len(mapped)==360 and len(skipped)==3 and {row['destination']for row in mapped}==set(by_name)
assert {row['source']for row in skipped}=={'preprocessor.featurizer.window','sortformer_modules.hidden_to_spks.weight','sortformer_modules.hidden_to_spks.bias'}
for row in mapped:
 tensor=by_name[row['destination']];actual=np.ascontiguousarray(tensor.data)
 assert row['exact_source_mapping_cast_match']is True and hashlib.sha256(actual.tobytes()).hexdigest()==row['output_sha256']==row['expected_cast_sha256']
 assert actual.nbytes==row['output_bytes']and list(actual.shape)==row['output_shape']and tensor.tensor_type.name==row['output_type']
 assert np.isfinite(actual).all()
fp16,=[row for row in mapped if row['output_type']=='F16'];assert fp16['destination']=='subpixel_upsample.weight'
old_native_bank=cache/'frontend-diagnostic-ec88564/native-bank.f32'
bank=by_name['preprocessor.fb'].data;assert hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest()==digest(old_native_bank)
inputs=cache/'research/native-f32-control'
assert record(inputs/protocol['checkpoint']['name'])=={key:protocol['checkpoint'][key]for key in ['bytes','sha256']}
assert record(inputs/'model_config.yaml')==protocol['source_config']
# Retain the benign writer warning verbatim; the parsed file has one final
# architecture key and no duplicate tensor names.
stderr=(base/'converter.stderr.log').read_text()
assert stderr.strip()=="Duplicated key name 'general.architecture', overwriting it with new value 'sortformer' of type STRING"
assert account['metadata']['general.architecture']=='sortformer'and account['metadata']['sortformer.version']=='v3'
output=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-construction';output.mkdir(exist_ok=False)
for name in ['protocol.json','command.json','construction.json','source-tensor-accounting.json','offline.sb','converter.stdout.log','converter.stderr.log']:
 shutil.copy2(base/name,output/name)
for name in ['acquisition.lock.json','repository-inventory.json','nemo-members.json']:
 shutil.copy2(inputs/name,output/name)
for name in ['run-native-f32-conversion.py','audit-native-f32-conversion.py','collect-native-f32-construction.py']:
 shutil.copy2(cache/name,output/name)
qualification={'schema_version':1,'source_commit':source,'artifact':construction['artifact'],'artifact_name':construction['name'],
 'checkpoint_deserialization_performed':True,'neural_model_inference_performed':False,'original_gate_status_changed':False,'release_qualified':False,'artifact_admitted_for_inference':False,
 'source_tensor_count':363,'all_source_tensors_bfloat16':True,'stored_tensor_count':360,'stored_f32':359,'stored_f16':1,'f16_tensor':'subpixel_upsample.weight',
 'all_mapped_names_shapes_dtypes_bytes_equal_independent_source_casts':True,'all_stored_values_finite':True,'all_source_keys_accounted_including_three_intentional_exclusions':True,
 'native_filterbank_exactly_matches_previous_q8_frontend_control':True,
 'converter_exit_code':0,'converter_elapsed_seconds':command['elapsed_seconds'],
 'writer_warning_retained':'Upstream writer inserts general.architecture then the converter explicitly writes it again; warning retained. Parsed final value is sortformer, no duplicate output tensor names.',
 'limits':protocol['limitations']+['Source BF16 weights are upcast for F32 arithmetic; this does not recover precision absent from the original checkpoint.','Artifact construction and exact cast accounting only. This metadata does not authorize or claim completed native F32 inference; numerical-execution protocol/admission remains separate.','Original checkpoint, converted GGUF and source tensor payloads remain outside Git.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
files={p.relative_to(output).as_posix():record(p)for p in output.rglob('*')if p.is_file()}
(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'evidence_files':len(files),'source_tensors':363,'stored_f32':359,'stored_f16':1,'all_stored_finite':True,'native_bank_same_as_prior_q8':True,'inference_admitted':False}))
