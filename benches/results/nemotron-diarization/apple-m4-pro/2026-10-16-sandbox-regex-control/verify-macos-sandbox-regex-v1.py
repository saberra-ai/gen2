"""Replay historical wildcard weakness and corrected denial; no model import/work."""
import hashlib,json,pathlib,shutil,subprocess,sys,time
cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');out=cache/'macos-sandbox-regex-control-v1';out.mkdir(exist_ok=False)
old=cache/'original-nemo-env-v1/import.sb';oldbytes=old.read_bytes();oldtext=oldbytes.decode();needle='.*'+chr(92)*2+'.';fixed='.*'+chr(92)+'.';assert oldtext.count(needle)==1;corrected=oldtext.replace(needle,fixed)
(out/'historical-import.sb').write_bytes(oldbytes);(out/'corrected-import.sb').write_text(corrected)
suffixes=['nemo','gguf','safetensors','onnx','npy','f32'];probes=[]
for suffix in suffixes:
 p=out/('dummy.'+suffix);p.write_bytes(b'not a model');assert p.read_bytes()==b'not a model';probes.append(p)
allow='import pathlib,sys; assert pathlib.Path(sys.argv[1]).read_bytes()==b"not a model"; print("historical wildcard permitted dummy GGUF read")'
denied='''import pathlib,socket,sys
for name in sys.argv[1:]:
 try:
  with pathlib.Path(name).open("rb")as stream: stream.read(1)
 except PermissionError: pass
 else: raise RuntimeError("expected denied read: "+name)
try: socket.create_connection(("127.0.0.1",9),timeout=1)
except PermissionError: pass
else: raise RuntimeError("network was not denied")
print("all requested reads and loopback denied")
'''
(out/'denied-control.py').write_text(denied);(out/'allowed-control.py').write_text(allow)
python=cache/'evidence-tools-env/bin/python';home=out/'home';home.mkdir();env={'HOME':str(home),'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'};commands=[]
for name,policy,script,arguments in [('historical-wildcard',out/'historical-import.sb',out/'allowed-control.py',[probes[1]]),('historical-explicit-directory-and-network',out/'historical-import.sb',out/'denied-control.py',[cache/'research/native-f32-control/model_config.yaml']),('corrected-six-formats-and-network',out/'corrected-import.sb',out/'denied-control.py',probes)]:
 argv=['/usr/bin/sandbox-exec','-f',str(policy),str(python),str(script),*map(str,arguments)];row={'name':name,'argv':argv,'environment':env};commands.append(row);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');start=time.monotonic()
 with(out/(name+'.stdout.log')).open('xb')as so,(out/(name+'.stderr.log')).open('xb')as se:r=subprocess.run(argv,cwd=out,env=env,stdout=so,stderr=se,timeout=10)
 row.update(returncode=r.returncode,seconds=time.monotonic()-start);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert r.returncode==0,name
record=lambda p:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
q={'schema_version':1,'old_policy_source':str(old),'old_policy':record(old),'historical_wildcard_denied_weight_formats':False,'historical_explicit_model_directory_and_network_denial_verified':True,'corrected_policy_denied_all_six_formats_and_network':True,'format_suffixes':suffixes,'model_or_frontend_or_native_execution':False,'import_receipt_correction':'The original37member import-only receipt overstated wildcard weight-format denial. Its explicit model-directory/network denial and class-import evidence remain separate; the original policy/receipt are unchanged.','limits':['All format probes are11-byte dummy files, never parsed as model/tensor data. The actual known-directory control reads at most1byte of model_config.yaml if denied protection were absent.','This corrects only a sandbox wildcard claim, not any model/quality outcome. Neural original-NeMo calibration/AMI qualifiers use explicit oracle/label path denials rather than this wildcard.','Source/environment writes are still denied by both saved policies. No source, model, SDK, prior receipt or user app was modified.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');shutil.copy2(__file__,out/pathlib.Path(__file__).name)
target=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-sandbox-regex-control';assert not target.exists();target.mkdir()
for p in out.iterdir():
 if p.is_file():shutil.copy2(p,target/p.name)
files={p.name:record(p)for p in target.iterdir()if p.is_file()};(target/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files)+1,'historical_wildcard_denied':False,'corrected_six_formats_and_network_denied':True,'model_calls':0}))
