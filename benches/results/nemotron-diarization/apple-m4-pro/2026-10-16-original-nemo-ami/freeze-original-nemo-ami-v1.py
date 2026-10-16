import ast,hashlib,json,pathlib,subprocess
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');p=json.loads((tools/'original-nemo-calibration.protocol.json').read_text());env=pathlib.Path(p['environment_root']);stage=cache/'original-nemo-ami-inputs-v1';stage.mkdir(exist_ok=False);commands=[]
def desc(f):return {'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
corpus=cache/'quality-corpus-v1';quality=cache/'quality qualification 125f907';cases=[];inputs={k:v for k,v in p['inputs'].items()if k not in [p['pcm_path'],p['reference_path'],p['reference_metadata_path'],p['hf_first_mel_path'],p['hf_final_mel_path']]}
for id,suffix,expected_der in [('ami-close','ami-ES2004a-Mix-Headset',.414447315),('ami-distant','ami-ES2004a-Array1-01',.500222677)]:
 original=cache/'tail-framing-v1'/id/'audio.f32';pcm=stage/(id+'.f32');argv=['/bin/cp','-c',str(original),str(pcm)];r=subprocess.run(argv,capture_output=True,text=True);commands.append({'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0 and desc(original)==desc(pcm)
 meta=quality/(suffix+'-fp32.json');ref=quality/(suffix+'-fp32.npy');labels=corpus/(suffix+'.reference.json');metadata=json.loads(meta.read_text());assert metadata['input_pcm_f32le_sha256']==desc(pcm)['sha256']and metadata['audio_samples']==1920000 and pcm.stat().st_size==7680000
 cases.append({'id':id,'samples':1920000,'pcm_path':str(pcm),'pcm':desc(pcm),'reference_path':str(ref),'reference_metadata_path':str(meta),'labels_path':str(labels),'original_cached_hf_shared_der':expected_der,'uem_seconds':[0,120]})
 for f in [original,pcm,meta,ref,labels,cache/'tail-framing-v1'/id/'framing.json']:inputs[str(f)]=desc(f)
inputs[str(corpus/'manifest.json')]=desc(corpus/'manifest.json')
receipt=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-original-nemo-calibration'
for name in ['collection.json','receiver-qualification.json','qualification.json']:inputs[str(receipt/name)]=desc(receipt/name)
prerequisite=json.loads((receipt/'receiver-qualification.json').read_text());assert prerequisite['strict_original_cpu_restore_and_one_calibration_forward_passed']is True and prerequisite['all_111_valid_probability_rows_compared']is True and prerequisite['values_above_1e_3']==0
modules=set();pending=['original_nemo_calibration.py','qualify_original_nemo_ami.py']
while pending:
 name=pending.pop()
 if name in modules:continue
 modules.add(name)
 for node in ast.walk(ast.parse((tools/name).read_text())):
  names=[node.module]if isinstance(node,ast.ImportFrom)else[a.name for a in node.names]if isinstance(node,ast.Import)else[]
  for module in names:
   local=(module or '').split('.')[0]+'.py'
   if (tools/local).is_file()and local not in modules:pending.append(local)
for key in ['pcm_path','pcm','reference_path','reference_metadata_path','hf_first_mel_path','hf_final_mel_path']:p.pop(key)
p.update(scope='Exactly2fresh original-NeMo strict CPU FP32 restore/whole-waveform forward calls: unchanged120sAMIclose then distant, fixed9/4/0/264/264/222. Prerequisite excluded calibration passed ataf4cfdb. No new HF/native inference, profile/threshold search, target replacement or quality waiver.',cases=cases,inputs=inputs,tools={n:desc(tools/n)for n in sorted(modules)},output_path=str(cache/'original-nemo-ami-runtime-v1'),comparison_policy='All12000valid/eight learned channels without permutation/row removal. Abs1e-3 descriptive differences and exact unchanged10ms/zero-collar/overlap-inclusive full120sUEM/shared>.5DER against original references; preserve failures. The original20%gate is reported diagnostically and never replaces currentQ8/HF failures.',limits=['Two previously observed failed crops; not new held-out evaluation or corpus qualification.','Later official3.1 source and independently resolved environment, not identified3.0 training source.','Original whole-wave frontend and cachedHFchunked frontend are different; do not assume bitwise or full-corpus model equivalence.','Both fixed calls run even if descriptive disagreement or quality failures occur. Stop on execution/geometry/shape/finite-range/source/storage failure. No relaxed-key retries.','No original-NeMo forward on eitherAMIcrop has run at preparation.','OriginalQ8/HF gates, label policy and every historical result remain unchanged. No public signing,GUI/microphone,clean performance,fresh-machine or release claim.'])
p['denied_paths']+= [str(cache/'original-nemo-calibration-runtime-v1'),str(cache/'valid-mel-pnola-inputs-v2'),str(cache/'release CPU qualification d01ef80'),str(cache/'release Metal qualification d01ef80')]
p['readonly_paths']+=[str(stage)];p['storage_policy']['timeout_seconds']=600
(stage/'preparation.json').write_text(json.dumps({'model_executed':False,'commands':commands,'files':{f.name:desc(f)for f in stage.iterdir()if f.is_file()}},indent=2)+'\n');p['inputs'][str(stage/'preparation.json')]=desc(stage/'preparation.json')
path=tools/'original-nemo-ami.protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'protocol':desc(path),'tools':len(modules),'cases':len(cases)}))
