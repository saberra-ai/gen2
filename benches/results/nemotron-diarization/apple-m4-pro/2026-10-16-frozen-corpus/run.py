"""Sequential offline calibration and all frozen CPU/Metal corpus cases."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,wave
import numpy as np
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
root=cache/'quality qualification 125f907'
corpus=cache/'quality-corpus-v1'
marker=root/'build-manifest.json'
started=time.monotonic()
while not marker.exists():
 if time.monotonic()-started>7200:raise RuntimeError('timed qualification/build did not finish within2hours; no corpus inference started')
 time.sleep(5)
sys.path.insert(0,str(tools))
import qualify_corpus as q
from compare_stream import compare
from corpus import pcm_header
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q.validate_protocol(tools/'corpus.protocol.json',corpus)
manifest=q.validate_corpus(corpus)
build=json.loads(marker.read_text())
for name,digest in build['binaries'].items():assert sha(root/name)==digest
assert sha(tools/'corpus.protocol.json')==build['protocol_sha256']
shutil.copy2(__file__,root/'run.py')
shutil.copy2(tools/'corpus.protocol.json',root/'frozen-protocol.json')
python=cache/'python-reference/bin/python'
base_env={'PATH':'/usr/bin:/bin'}
ref_env={**base_env,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}

def run(name,args,*,reference=False,stdout_json=False):
 policy=root/('reference-offline.sb' if reference else 'offline.sb')
 command=['/usr/bin/time','-l','/usr/bin/sandbox-exec','-f',str(policy),*map(str,args)]
 env=ref_env if reference else base_env
 (root/(name+'.command.json')).write_text(json.dumps({'command':command,'environment':env},indent=2)+'\n')
 print('START '+name,flush=True)
 output=root/(name+('.json' if stdout_json else '.stdout.log'))
 with output.open('x') as out,(root/(name+'.time.log')).open('x') as log:
  subprocess.run(command,env=env,stdout=out,stderr=log,check=True)
 print('PASS '+name,flush=True)

def reference(name,wav):
 run(name,[python,tools/'reference.py','--assets',cache/'assets','--wav',wav,'--output',root/name],reference=True)
 meta=json.loads((root/(name+'.json')).read_text())
 q.validate_oracle_environment(meta)
 assert meta['script_sha256']==q.digest(tools/'reference.py')
 assert meta['wav_sha256']==sha(wav)
 return meta

def paths(backend):
 return cache/('release CPU qualification d01ef80' if backend=='cpu' else 'release Metal qualification d01ef80')

def native_pair(name,wav,backend):
 package_root=paths(backend)
 run(name+'-gen2',[root/'nemotron_stream',*(['--metal'] if backend=='metal' else []),package_root/'assets',package_root/'nemo/lib/libnemo_speech_asr_c.dylib',wav,'1',root/(name+'-gen2.f32')],stdout_json=True)
 run(name+'-native',[root/('native-'+backend),package_root/'assets/Nemotron-3-Diarization.q8_0.gguf',wav,'1','0' if backend=='metal' else '-1',root/(name+'-native.json')])
 gen2=json.loads((root/(name+'-gen2.json')).read_text())
 native=json.loads((root/(name+'-native.json')).read_text())
 result=compare(gen2,native,(root/(name+'-native.json.f32')).read_bytes(),(root/(name+'-gen2.f32')).read_bytes())
 raw=wav.read_bytes();offset,samples=pcm_header(raw[:65536],len(raw))
 expected=hashlib.sha256((np.frombuffer(raw,dtype='<i2',count=samples,offset=offset).astype(np.float32)/np.float32(32768)).astype('<f4').tobytes()).hexdigest()
 for record in [gen2,native]:
  assert record['input_pcm_f32le_sha256']==expected
  assert record['input_repeats']==1
 return result,expected

# This is an excluded EN2002d smoke crop, not a frozen target recording.
calibration=root/'partial-1.117s-smoke.wav'
with wave.open(str(cache/'assets/ami_en2002d_2132.wav'),'rb') as source:
 data=source.readframes(17872)
with wave.open(str(calibration),'wb') as out:
 out.setparams((1,2,16000,0,'NONE','not compressed'));out.writeframes(data)
meta=reference('calibration-reference',calibration)
calibrations=[]
for backend in ['cpu','metal']:
 result,expected=native_pair('calibration-'+backend,calibration,backend)
 assert meta['input_pcm_f32le_sha256']==expected
 assert meta['shape']==[111,8]
 assert result['native_frame_count']==112
 calibrations.append(result)
(root/'calibration.json').write_text(json.dumps({'passed':True,'scope':'excluded smoke1.117s partial-frame and actual-input-digest calibration before corpus inference','reference_shape':meta['shape'],'results':calibrations},indent=2)+'\n')
print('CALIBRATION PASSED; beginning all8frozen cases',flush=True)
results=[]
for item in manifest['slices']:
 identifier=item['id'];wav=corpus/item['wav']
 assert sha(wav)==manifest['files'][item['wav']]['sha256']
 reference(identifier+'-fp32',wav)
 for backend in ['cpu','metal']:
  name=identifier+'-'+backend
  native_pair(name,wav,backend)
  command=[str(python),str(tools/'qualify_corpus.py'),'--corpus',str(corpus),'--slice',identifier,'--gen2',str(root/(name+'-gen2.json')),'--native',str(root/(name+'-native.json')),'--reference',str(root/(identifier+'-fp32')),'--output',str(root/(name+'-score.json'))]
  # Scoring has no model loading/network; retain exact command and output.
  (root/(name+'-score.command.json')).write_text(json.dumps(command,indent=2)+'\n')
  subprocess.run(command,env=ref_env,check=True)
  results.append(json.loads((root/(name+'-score.json')).read_text()))
(root/'summary.json').write_text(json.dumps({'release_qualified':False,'scope':'frozen local tuning-heldout CPU/Metal per-slice quality; no general unseen-data or complete release claim','source_commit':build['source_commit'],'protocol_sha256':build['protocol_sha256'],'all_slice_gates_passed':all(r['all_slice_gates_passed'] for r in results),'results':results},indent=2)+'\n')
print('ALL CORPUS CASES RECORDED '+str(root/'summary.json'),flush=True)
