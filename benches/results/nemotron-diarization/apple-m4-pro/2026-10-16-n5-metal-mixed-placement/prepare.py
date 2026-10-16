from pathlib import Path
import hashlib,json,os,shutil,subprocess
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');root=cache/'release Metal mixed placement 8572cc8';root.mkdir();old=cache/'release Metal app qualification ef1fc1a';app=cache/'release Metal app qualification 024a88f/Pnola.app';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
artifacts=[json.loads(line) for line in (cache/'n5-metal-chat-placement-artifacts.jsonl').read_text().splitlines()]
a=next(x for x in artifacts if x.get('reason')=='compiler-artifact' and x['target']['name']=='diarization_mixed_native');assert {'metal','backend-llamacpp','backend-diar-nemo','asr-dynamic','laya-dynamic'} <= set(a['features']);shutil.copy2(a['executable'],root/'mixed-native-tests');(root/'compiler-artifact.json').write_text(json.dumps(a,indent=2)+'\n')
assert h(repo/'tests/diarization_mixed_native.rs')==hashlib.sha256(subprocess.check_output(['git','show','8572cc8:tests/diarization_mixed_native.rs'],cwd=repo)).hexdigest()
files={}
oldmanifest=json.loads((old/'manifest.json').read_text());newmanifest=json.loads((app.parent/'manifest.json').read_text())
for k,v in oldmanifest['files'].items():
 if k.startswith('laya/') or k=='chat.gguf':files[k]=v
for k,v in newmanifest['files'].items():
 if k.startswith('app/'):files[k]=v
shutil.copy2(app.parent/'offline.sb',root/'offline.sb')
for n in ['mixed-native-tests','offline.sb','compiler-artifact.json']:
 p=root/n;files[n]={'path':str(p),'bytes':p.stat().st_size,'sha256':h(p)}
env={**oldmanifest['environment'],'GEN2_NEMOTRON_ASSETS':str(app/'Contents/Resources/nemotron'),'GEN2_NEMO_LIBRARY':str(app/'Contents/Frameworks/libnemo_speech_asr_c.dylib'),'GEN2_BARK_BUNDLE':str(app/'Contents/Resources/bark'),'GEN2_ORT_LIBRARY':str(app/'Contents/Frameworks/libonnxruntime.dylib'),'GEN2_CHAT_EXPECT_BACKEND':'MTL'}
m={'source_commit':'8572cc8','source_sha256':h(repo/'tests/diarization_mixed_native.rs'),'release_qualified':False,'files':files,'environment':env,'scope':'Samehost offline full Laya/Bark/chat/Metal Nemotron coexistence in both loadorders, sharedbudget refusal andeviction/restore. Public chat placement must report nonzeroGPUlayers/backendMTL/device atinitial/restored generation. Metadata mirrors native loader parameters/registry; no pertensor orkerneltelemetry, quality, latency or freshmachine claim.'}
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');shutil.copy2(__file__,root/'prepare.py');print(root)
