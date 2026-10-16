from pathlib import Path
import shutil,json,hashlib,subprocess,re,os,wave
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'release Metal app qualification ef1fc1a';app=root/'Pnola.app';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'ami-80.04s.wav').rename(root/'live-source-60s.wav')
for name in ['ami-60s.wav','live-source-60s.wav']:
 with wave.open(str(root/name)) as f:assert f.getnframes()==960000 and f.getframerate()==16000
shutil.copy2(cache/'quality qualification 125f907/native-metal',root/'native-metal')
probe=root/'native-metal';out=subprocess.check_output(['otool','-l',str(probe)],text=True)
for old in re.findall(r'cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+) \(offset \d+\)',out):subprocess.run(['install_name_tool','-delete_rpath',old,str(probe)],check=True)
subprocess.run(['install_name_tool','-add_rpath',str(app/'Contents/Frameworks'),str(probe)],check=True)
subprocess.run(['codesign','--force','--sign','-',str(probe)],check=True)
denied=['/opt/homebrew','/Users/victor/workspace/gen2-nemotron3-n0','/Users/victor/workspace/gen2-bark-onnx','/Users/victor/workspace/pnola','/Users/victor/models','/Users/victor/.cache/pnola-nemotron-build','/Users/victor/.cache/gen2-bark-onnx/build',str(cache/'NeMo-Speech.cpp'),str(cache/'sdk-cpu'),str(cache/'sdk-metal'),str(cache/'assets'),str(cache/'package-cpu-n5-final'),str(cache/'package-metal-n5'),str(cache/'relocated Pnola frameworks candidate'),str(cache/'release Metal qualification d01ef80'),str(cache/'quality qualification 125f907')]
(root/'offline.sb').write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+ ' '.join('(subpath '+json.dumps(p)+')' for p in denied)+')\n')
models=cache/'relocated mixed models';files={}
for base,label in [(app,'app'),(root/'control-cpu','control-cpu'),(models/'laya','laya')]:
 for p in sorted(base.rglob('*')):
  key=label+'/'+str(p.relative_to(base))
  if p.is_symlink():files[key]={'path':str(p),'symlink':os.readlink(p)}
  elif p.is_file():files[key]={'path':str(p),'bytes':p.stat().st_size,'sha256':h(p)}
for p in [root/'mixed-native-tests',root/'native-metal',root/'ami-60s.wav',root/'live-source-60s.wav',root/'offline.sb',models/'chat.gguf']:
 files[p.name]={'path':str(p),'bytes':p.stat().st_size,'sha256':h(p)}
manifest={'release_qualified':False,'source_commit':'ef1fc1a','source_files':{n:h(repo/n) for n in ['tests/diarization_mixed_native.rs','integrations/pnola/manifest.json']},'app_package':json.loads((app/'Contents/Resources/nemo/package.json').read_text())['backend'],'host':subprocess.check_output(['sw_vers'],text=True),'scope':'same-host M4 Pro/macOS26.3 Metal app/backend selection, saved device, offline relocation and actual mixed chat/Bark/Laya admission; no accuracy/default/fresh-machine or physical GUI/mic claim','files':files,'environment':{'PATH':'/usr/bin:/bin','GEN2_NEMO_EXECUTION':'metal','GEN2_NEMOTRON_ASSETS':str(app/'Contents/Resources/nemotron'),'GEN2_NEMO_LIBRARY':str(app/'Contents/Frameworks/libnemo_speech_asr_c.dylib'),'GEN2_BARK_BUNDLE':str(app/'Contents/Resources/bark'),'GEN2_ORT_LIBRARY':str(app/'Contents/Frameworks/libonnxruntime.dylib'),'GEN2_CHAT_GGUF':str(models/'chat.gguf'),'GEN2_LAYA_BUNDLE':str(models/'laya')},'denied_source_roots':denied,'live_fixture':'Existing verifier consumes60s source and builds80.04s capture sequence internally; externalinputisnot80.04s.'}
assert manifest['app_package']=='metal'
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copy2(__file__,root/'prepare.py')
print('Prepared',len(files),'frozen files in',root)
