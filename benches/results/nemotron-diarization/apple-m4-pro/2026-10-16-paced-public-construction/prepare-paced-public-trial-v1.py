"""Collect model-free construction, then freeze one independent paced trial."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import wave

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
build=cache/'paced-public-probe-construction-v1'
read=lambda path:json.loads(path.read_text())
def desc(path):
    data=Path(path).read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
p=read(build/'protocol.json');q=read(build/'construction.json');source=q['source_commit']
assert source=='e658b0cd763001842f54b848c66f26c515b1b9ae'
for name,identity in p['tools'].items():assert desc(tools/name)==identity
assert (tools/'frozen_git_sources.py').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/frozen_git_sources.py'],cwd=repo)
sys.path.insert(0,str(tools))
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory
verify_git_sources(repo,source,p['repository_files'])
verify_git_sources(repo,source,{'tools/nemotron-diarization/'+name:identity for name,identity in p['tools'].items()})
assert (build/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/paced-stream-construction.protocol.json'],cwd=repo)
for name,identity in p['repository_files'].items():assert desc(repo/name)==identity
for path,identity in p['compiler_files'].items():assert desc(path)==identity
for name,identity in read(build/'files.json')['files'].items():assert desc(build/name)==identity
assert q['constructed'] is True and q['model_calls']==0 and q['release_qualified'] is False
assert [row['name'] for row in q['commands']]==['sandbox-controls','rustc-version','rustfmt-check','cargo-build','error-evidence-regression']
assert all(row['returncode']==0 for row in q['commands'])
assert q['storage']['released'] is True
test=(build/'error-evidence-regression.stdout.log').read_text()
assert 'test tests::partial_evidence_survives_replay_error ... ok' in test and '1 passed; 0 failed' in test
assert desc(build/'bin/paced-public')==q['binary']
for name,identity in q['used_gen2_source_files'].items():assert p['repository_files'][name]==identity==desc(repo/name)
receipt=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-paced-public-construction'
assert not receipt.exists();receipt.mkdir()
for path in [build/'protocol.json',build/'construction.json',build/'files.json',build/'model-free.sb',build/'paced-public.d',
    *build.glob('*.stdout.log'),*build.glob('*.stderr.log'),Path(__file__),cache/'paced-public-probe-construction-v1.controller.log',
    cache/'platform-ci/model-free-247-paced-local.log',cache/'paced-stream-metrics-targeted.log']:
    shutil.copy2(path,receipt/path.name)
summary={'schema_version':1,'source_commit':source,'constructed':True,'model_calls':0,
    'gen2_source_pins':len(p['repository_files']),'tool_pins':len(p['tools']),'compiler_config_pins':len(p['compiler_files']),
    'used_gen2_source_dependencies':len(q['used_gen2_source_files']),'rust_error_evidence_regression_passed':True,
    'binary':q['binary'],'release_qualified':False,'limits':p['limits']}
(receipt/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
(receipt/'collection.json').write_text(json.dumps({'files':{path.name:desc(path) for path in receipt.iterdir() if path.is_file()}},indent=2)+'\n')
old=cache/'pooled-pnola-complete-app-inputs-v2'
old_result=cache/'pooled-pnola-complete-app-runtime-v2'
work=cache/'paced-public-inputs-v1/work'
assert not work.parent.exists()
for name in ['bin','assets','baseline']:(work/name).mkdir(parents=True,exist_ok=True)
shutil.copytree(old/'nemo',work/'nemo',symlinks=True)
shutil.copy2(build/'bin/paced-public',work/'bin/paced-public')
shutil.copy2(old/'bin/load-probe',work/'bin/load-probe')
shutil.copy2(old/'assets/model.gguf',work/'assets/Nemotron-3-Diarization.q8_0.gguf')
shutil.copy2(old/'assets/live-80s.wav',work/'assets/live-80s.wav')
assert desc(work/'assets/Nemotron-3-Diarization.q8_0.gguf')=={'bytes':107012128,'sha256':'08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1'}
for root,name,target in [(build,'protocol.json','build-protocol.json'),(build,'construction.json','construction.json'),
                        (old_result/'results','native-80s.json','native-80s.json'),
                        (old_result/'results','native-80s.json.f32','native-80s.json.f32')]:
    shutil.copy2(root/name,work/'baseline'/target)
with wave.open(str(work/'assets/live-80s.wav'),'rb') as wav:
    assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())==(1,2,16000,1280640)
    data=wav.readframes(1280640)
h=hashlib.sha256()
for sample, in struct.iter_unpack('<h',data):h.update(struct.pack('<f',sample/32768.))
actual_pcm=h.hexdigest()
assert actual_pcm==read(work/'baseline/native-80s.json')['input_pcm_f32le_sha256']=='dc94f3f490576f50d11e23728f8fa8bb129ff78b4af41eff3d4961a776b45858'
assert inventory(old/'nemo')==inventory(work/'nemo')
names=set();pending=['qualify_paced_public_stream','test_paced_stream_metrics']
while pending:
    name=pending.pop()
    if name+'.py' in names:continue
    path=tools/(name+'.py');names.add(path.name)
    for node in ast.walk(ast.parse(path.read_text())):
        imports=[node.module] if isinstance(node,ast.ImportFrom) else [entry.name for entry in node.names] if isinstance(node,ast.Import) else []
        pending.extend(n for n in imports if n and (tools/(n+'.py')).is_file())
trial={'schema_version':1,'work_path':str(work),'output_path':str(cache/'paced-public-runtime-v1'),
    'source_cache_root':str(cache),'pcm_f32le_sha256':actual_pcm,
    'tools':{name:desc(tools/name) for name in sorted(names)},'work_inventory':inventory(work),
    'prerequisite_records':{str(path):desc(path) for path in [build/'construction.json',build/'protocol.json',build/'files.json',
        receipt/'collection.json',old_result/'qualification.json',old_result/'protocol.json',
        repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-app-runtime/collection.json']},
    'command_order':['runtime-controls','premodel-controls','version-only','error-before-model','paced-public'],
    'storage':{'start_minimum':5*2**30,'running_minimum':2**30,'reserve_bytes':256*2**20},'timeout_seconds':600,
    'limits':['One new endpoint-paced80.04s publicGen2 CPU process on existing licensed AMI cyclicfixture; no physicalmic, GUI, newaudio, qualitytest or defaultpolicychange. ActualnormalizedPCMhash mustmatch savedindependentCABIreference; compareall8004rows/eightchannels andallturns withinestablished1e-6 bounds, reportbitexactness separately.',
      'Model/runtime byteidentical to priorqualifiedpooledCPU package. Savedindependentnative80.04s reference is preregistered, never supplied to the helper. This reuses priorneural evidence, not a new nativeprocess or clean performancecomparison.',
      'Runtime/premodeldenial and libc-canonicalization controls; nativeversion processrequiresall92privateimages. Pre-modelhelperfailure mustpersistcompletefalseemptyrows/checkpoints,return1 andload0privateimages. One paced neuralprocessrequiresall92images. NativeSDK/source/reference/provenance/prioroutput data denied; protectedworkwrites andnetwork denied.',
      'Keep4002 endpoint/readiness/admission/Busyrows,15or16 consecutiveprovisionalturnobservations andfinalrevision17. Lastprovisional16maycoalescebeforefinishreturn; never inventitsclock. Measureactualpublicfinishdrain/flus h/delivery andcallerobservation; keepallreportedcomponentsandfailures, none are claimedas isolatedqueue/lookahead/publication or acousticcorrectlabeldelay.',
      'Pacedpublicelapsed includes deliberatecapture-clockwaits and is not computeRTF; nativecomputewallratio is reportedseparately. Scheduling/admission/nativecompute overlap andmustnotbesummed. No stableprefix orfinalityinference, earlyfuturesamplefeeding orposthocchannelselection.',
      'Original5GiBstart/1GiBrunning/256MiBphysicalreserve/600sdeadline/processgroupkill andfailurestop. Helper180ssoftdeadline; partialreturned-errorJSONretained. Noautomaticretry. All originalquality/wordgold/GUI/mic/freshmachine/publicsigning/releasegates remain open; no mainpush/completion claim.']}
out=tools/'paced-public-runtime.protocol.json';assert not out.exists();out.write_text(json.dumps(trial,indent=2)+'\n')
print('construction receipt',len(list(receipt.iterdir())),'members; trial',desc(out),'tools',len(names),'work',len(trial['work_inventory']))
