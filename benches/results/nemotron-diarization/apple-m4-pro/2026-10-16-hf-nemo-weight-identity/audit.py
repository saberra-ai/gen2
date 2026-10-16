"""Fixed read-only tensor identity audit; no neural model objects or inference."""
import ast,collections,datetime,hashlib,importlib.metadata,json,pathlib,re,struct,sys,tarfile
import numpy as np
import torch
from safetensors import safe_open
ROOT=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0')
OUT=ROOT/'research/hf-nemo-weight-identity'
LOCK=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/runtime.lock.json')
EXPECTED_LOCK='e66e5651608636bbb04528fed7944ab8972a778d1141b36fef3fb391eefd0dcc'
MAP=ROOT/'python-reference/lib/python3.12/site-packages/transformers/models/nemotron3_diarization/convert_nemotron3_diarization_to_hf.py'
MAP_SHA='6a4de669f624642eb723d5361900f0e6c6c07d1d7153edf645e2fae0e0c71da1'
NEMO=ROOT/'research/native-f32-control/Nemotron-3-Diarization.nemo'
HF=ROOT/'assets/model.safetensors'
SOURCE_BYTES=198666820
SOURCE_SHA='9930e1719c85b35da083c557cdbd6503072c311ffe8bc9cbd3371d2609e21d0f'
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(4*1024*1024):h.update(chunk)
 return h.hexdigest()
def fingerprint(t):
 t=t.detach().cpu().contiguous()
 if t.dtype==torch.bfloat16:raw=t.view(torch.uint16).numpy().astype('<u2',copy=False).tobytes()
 elif t.dtype==torch.float32:raw=t.numpy().astype('<f4',copy=False).tobytes()
 else:raise AssertionError(('unexpected dtype',t.dtype))
 return {'dtype':str(t.dtype),'shape':list(t.shape),'elements':t.numel(),'bytes':len(raw),'sha256':sha(raw)},raw
assert sha(LOCK.read_bytes())==EXPECTED_LOCK
lock=json.loads(LOCK.read_text());artifact=next(x for x in lock['artifacts'] if x['name']=='model.safetensors')
assert HF.stat().st_size==artifact['size'] and file_sha(HF)==artifact['sha256']
assert NEMO.stat().st_size==198676480 and file_sha(NEMO)=='867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d'
rawmap=MAP.read_bytes();assert sha(rawmap)==MAP_SHA
assert (ROOT/'research/quality-next/hf-convert.py').read_bytes()==rawmap
(OUT/'pinned-hf-converter.py').write_bytes(rawmap)
tree=ast.parse(rawmap);want_consts={'STATE_DICT_MAPPING','KEYS_TO_DROP'};want_funcs={'split_fused_qkv','convert_state_dict'};nodes=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in want_consts for x in node.targets):nodes.append(node)
 elif isinstance(node,ast.FunctionDef) and node.name in want_funcs:nodes.append(node)
assert len(nodes)==4
selected=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(selected);ns={'re':re};exec(compile(selected,str(MAP)+'[mapping AST only]','exec'),ns)
(OUT/'extracted-mapping.ast.txt').write_text(ast.dump(selected,indent=2)+'\n')
with HF.open('rb') as f:header_size=struct.unpack('<Q',f.read(8))[0];header_bytes=f.read(header_size)
header=json.loads(header_bytes);hf_header={k:v for k,v in header.items() if k!='__metadata__'}
assert header_size==48568 and len(hf_header)==417 and all(x['dtype']=='F32' for x in hf_header.values())
with tarfile.open(NEMO,'r:*') as tar:
 members=[x for x in tar.getmembers() if x.name=='model_weights.ckpt'];assert len(members)==1 and members[0].isfile() and members[0].size==SOURCE_BYTES
 with tar.extractfile(members[0]) as f:
  h=hashlib.sha256()
  while b:=f.read(4*1024*1024):h.update(b)
  assert h.hexdigest()==SOURCE_SHA
 with tar.extractfile(members[0]) as f:source=torch.load(f,map_location='cpu',weights_only=True)
assert isinstance(source,dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in source.items())
assert len(source)==363 and {str(v.dtype) for v in source.values()}=={'torch.bfloat16'}
torch.set_num_threads(4)
mapped=ns['convert_state_dict'](source);assert len(mapped)==417 and set(mapped)==set(hf_header)
source_inventory=[];excluded=[];trace={};qkv=[]
for key,t in source.items():
 info,_=fingerprint(t);source_inventory.append({'name':key,**info})
 drop=sorted(x for x in ns['KEYS_TO_DROP'] if re.match(x,key))
 if drop:
  excluded.append({'name':key,'matched_drop_regex':drop,**info});continue
 match=re.match(r'^encoder\.layers\.(\d+)\.attn\.w_qkv\.weight$',key)
 if match:
  assert t.shape[0]%3==0
  width=t.shape[0]//3;destinations=[]
  for index,part in enumerate(['q','k','v']):
   name=f'model.audio_tower.layers.{match.group(1)}.self_attn.{part}_proj.weight';assert name not in trace
   trace[name]={'source':key,'source_row_interval':[index*width,(index+1)*width],'operation':'split dim0 then exact BF16->F32 cast'};destinations.append({'target':name,'rows':[index*width,(index+1)*width]})
  qkv.append({'source':key,'source_shape':list(t.shape),'source_sha256':info['sha256'],'destinations':destinations})
 else:
  candidates=[re.sub(a,b,key) for a,b in ns['STATE_DICT_MAPPING'].items() if re.match(a,key)];assert len(candidates)==1
  name=candidates[0];assert name not in trace;trace[name]={'source':key,'operation':'rename then exact BF16->F32 cast'}
assert set(trace)==set(mapped) and len(excluded)==8 and len(qkv)==31
comparisons=[]
with safe_open(str(HF),framework='pt',device='cpu') as target:
 assert set(target.keys())==set(mapped)
 for name in sorted(mapped):
  expected=mapped[name].to(dtype=torch.float32);actual=target.get_tensor(name)
  assert actual.dtype==torch.float32 and actual.shape==expected.shape and list(actual.shape)==hf_header[name]['shape']
  ei,eb=fingerprint(expected);ai,ab=fingerprint(actual);equal=eb==ab
  comparisons.append({'name':name,**trace[name],'shape':list(actual.shape),'dtype':'F32','bytes':len(ab),'expected_bf16_cast_sha256':ei['sha256'],'actual_hf_sha256':ai['sha256'],'exact_bytes_match':equal})
assert all(x['exact_bytes_match'] for x in comparisons)
versions={x:importlib.metadata.version(x) for x in ['torch','numpy','safetensors','transformers']}
transformer_dist=importlib.metadata.distribution('transformers');direct=json.loads(transformer_dist.read_text('direct_url.json'));assert direct['vcs_info']['commit_id']=='683b721971417b3b2c64fba9001f525607547e5a'
report={'schema_version':1,'audit_passed':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All mapped parameter bytes agree after the pinned pure HF mapping and exact BF16->F32 cast. No neural model instantiation, inference, GGUF conversion, frontend equivalence, cache-state equivalence, quality or release qualification.','runtime_lock_sha256':EXPECTED_LOCK,'checkpoint':{'path':str(NEMO),'bytes':198676480,'sha256':file_sha(NEMO),'member':'model_weights.ckpt','member_bytes':SOURCE_BYTES,'member_sha256':SOURCE_SHA},'hf':{'path':str(HF),'bytes':HF.stat().st_size,'sha256':artifact['sha256'],'header_bytes':header_size,'header_sha256':sha(header_bytes),'tensors':len(hf_header),'dtype':'F32'},'mapping':{'installed_path':str(MAP),'sha256':MAP_SHA,'primary_url':'https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/convert_nemotron3_diarization_to_hf.py','ast_only_selected':sorted(want_consts|want_funcs),'module_imported':False,'extracted_ast_sha256':file_sha(OUT/'extracted-mapping.ast.txt')},'dependencies':versions,'transformers_direct_url':direct,'source_count':len(source),'source_dtype_counts':dict(collections.Counter(str(x.dtype) for x in source.values())),'qkv_source_count':len(qkv),'qkv_output_count':len(qkv)*3,'excluded_source_count':len(excluded),'mapped_target_count':len(mapped),'exact_matched_count':sum(x['exact_bytes_match'] for x in comparisons),'all_source_inventory':source_inventory,'excluded_source_tensors':excluded,'qkv_split_trace':qkv,'mapped_tensor_comparisons':comparisons,'script_sha256':file_sha(pathlib.Path(__file__)),'torch_load_weights_only':True,'transformers_imported':any(x=='transformers' or x.startswith('transformers.') for x in sys.modules)}
assert not report['transformers_imported']
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['audit_passed','source_count','source_dtype_counts','qkv_source_count','excluded_source_count','mapped_target_count','exact_matched_count','dependencies']}));print('report SHA256',file_sha(OUT/'report.json'))
for x in excluded:print('EXCLUDED',x['name'],x['shape'],x['sha256'])
