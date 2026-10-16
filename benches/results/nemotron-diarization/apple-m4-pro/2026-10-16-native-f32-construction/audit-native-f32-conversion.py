"""Independently account original source weights against every control GGUF tensor."""
import hashlib,json,re,sys,tarfile
from pathlib import Path
import numpy as np
import torch
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');output=cache/'native-f32-conversion-v1'
protocol=json.loads((output/'protocol.json').read_text());construction=json.loads((output/'construction.json').read_text())
sys.path.insert(0,str(cache/'NeMo-Speech.cpp/llama.cpp/gguf-py'))
from gguf import GGUFReader
hash_bytes=lambda data:hashlib.sha256(data).hexdigest()
record=lambda p:{'bytes':p.stat().st_size,'sha256':hash_bytes(p.read_bytes())}
artifact=output/construction['name'];assert record(artifact)==construction['artifact']
checkpoint=cache/'research/native-f32-control'/protocol['checkpoint']['name']
assert record(checkpoint)=={key:protocol['checkpoint'][key]for key in ['bytes','sha256']}
with tarfile.open(checkpoint)as bundle:
 with bundle.extractfile('model_weights.ckpt')as stream:sd=torch.load(stream,map_location='cpu',weights_only=True)
if 'state_dict'in sd:sd=sd['state_dict']
assert isinstance(sd,dict)and all(isinstance(value,torch.Tensor)for value in sd.values())
reader=GGUFReader(str(artifact));tensors={t.name:t for t in reader.tensors};assert len(tensors)==len(reader.tensors)
rows=[];stored=set();errors=[];counts={};source_counts={}
conv=re.compile(r'^(encoder\.layers\.\d+\.conv\.depthwise_conv|encoder\.pre_encode\.conv\.\d+|subpixel_upsample)\.weight$')
pointwise=re.compile(r'^encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight$')
for name,value in sd.items():
 value=value.detach().cpu().contiguous();raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
 row={'source':name,'source_shape':list(value.shape),'source_dtype':str(value.dtype),'source_bytes':len(raw),'source_sha256':hash_bytes(raw)}
 source_counts[str(value.dtype)]=source_counts.get(str(value.dtype),0)+1
 destination=None
 if name=='preprocessor.featurizer.fb':destination='preprocessor.fb';reason='original trained bank, leading singleton removed'
 elif name in {'encoder.pos_enc.pe','preprocessor.featurizer.window','sortformer_modules.hidden_to_spks.weight','sortformer_modules.hidden_to_spks.bias'}:
  reason={'encoder.pos_enc.pe':'analytical positional encoding; unused for V3 RoPE','preprocessor.featurizer.window':'native frontend generates symmetric Hann from config','sortformer_modules.hidden_to_spks.weight':'unused auxiliary speaker head, explicitly excluded upstream','sortformer_modules.hidden_to_spks.bias':'unused auxiliary speaker head, explicitly excluded upstream'}[name]
 elif name.startswith(('spec_augmentation.','loss.'))or name.endswith('.num_batches_tracked'):reason='explicit training-only metadata/buffer exclusion'
 elif name.startswith(('sortformer_modules.encoder_proj.','sortformer_modules.subpixel_upsample.')):destination=name[len('sortformer_modules.'):];reason='strip module owner prefix'
 elif name=='sortformer_modules.learnable_sil_emb':destination='learnable_sil_emb';reason='learned silence embedding'
 elif name.startswith('sortformer_modules.'):destination='head.'+name[len('sortformer_modules.'):];reason='inference speaker head namespace'
 elif name.startswith('transformer_encoder.'):destination='transformer.'+name[len('transformer_encoder.'):];reason='transformer namespace'
 elif name.startswith('encoder.'):destination=name;reason='encoder identity mapping'
 else:reason='unrecognized source key';errors.append(name+':unrecognized skip')
 row.update(destination=destination,reason=reason)
 if destination is not None:
  if destination in stored:errors.append(name+':duplicate destination')
  stored.add(destination)
  array=value.float().numpy()
  if name=='preprocessor.featurizer.fb':array=np.squeeze(array,axis=0)
  if pointwise.match(destination)and array.ndim==3 and array.shape[-1]==1:array=array.squeeze(axis=-1)
  dtype=np.dtype('<f2')if conv.match(destination)else np.dtype('<f4')
  expected=np.ascontiguousarray(array,dtype=dtype);expected_type='F16'if dtype.itemsize==2 else'F32'
  tensor=tensors.get(destination)
  if tensor is None:errors.append(name+':missing output');rows.append(row);continue
  actual=np.ascontiguousarray(tensor.data)
  matched=(tuple(actual.shape)==tuple(expected.shape)and tensor.tensor_type.name==expected_type and actual.tobytes()==expected.tobytes())
  if not matched:errors.append(name+':stored shape/type/bytes mismatch')
  row.update(output_shape=list(actual.shape),ggml_shape=[int(x)for x in tensor.shape],output_type=tensor.tensor_type.name,
             output_bytes=actual.nbytes,output_sha256=hash_bytes(actual.tobytes()),expected_cast_sha256=hash_bytes(expected.tobytes()),exact_source_mapping_cast_match=matched)
  counts[tensor.tensor_type.name]=counts.get(tensor.tensor_type.name,0)+1
 rows.append(row)
if stored!=set(tensors):errors.append('unaccounted output names:'+str(sorted(set(tensors)-stored)))
fields={name:field.contents()for name,field in reader.fields.items()if not name.startswith('GGUF.')}
assert fields['general.architecture']=='sortformer'and fields['sortformer.version']=='v3'and fields['general.file_type']==0
assert counts=={'F32':359,'F16':1}and len(rows)==363 and len(stored)==360
result={'schema_version':1,'protocol_sha256':record(output/'protocol.json')['sha256'],'artifact':record(artifact),
 'source_tensor_count':len(sd),'source_dtype_counts':source_counts,'stored_tensor_count':len(tensors),'stored_dtype_counts':counts,
 'exact_source_mapping_cast_accounting_passed':not errors,'errors':errors,'source_tensors':rows,'metadata':fields,
 'checkpoint_deserialization_performed':True,'audit_deserialization_weights_only':True,'neural_model_inference_performed':False,
 'release_qualified':False,'artifact_admitted_for_inference':False,
 'limits':['Manual independent name mapping and dtype/shape/byte accounting against source state dict; no reuse of upstream remap or dtype functions. This does not establish HF/original-NeMo graph/cache/frontend equivalence.','F16 subpixel convolution is retained despite nominal file_type0; not an all-FP32 artifact.','No native model run, output scoring, tuning or release model replacement occurred. Review is still required before inference admission.']}
(output/'source-tensor-accounting.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key]for key in ['source_tensor_count','source_dtype_counts','stored_tensor_count','stored_dtype_counts','exact_source_mapping_cast_accounting_passed','errors']}))
if errors:raise SystemExit(1)
