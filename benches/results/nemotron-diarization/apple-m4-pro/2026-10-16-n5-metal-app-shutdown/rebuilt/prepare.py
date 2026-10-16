from pathlib import Path
import hashlib,json,os,shutil,subprocess
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
old=cache/'release Metal app qualification ef1fc1a'; root=cache/'release Metal app qualification 024a88f';root.mkdir()
app=root/'Pnola.app';subprocess.run(['/usr/bin/ditto','/Users/victor/.cache/pnola-nemotron-build/release/bundle/macos/Pnola.app',str(app)],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def tree(base):
 return {str(p.relative_to(base)):({'symlink':os.readlink(p)} if p.is_symlink() else {'bytes':p.stat().st_size,'sha256':h(p)}) for p in sorted(base.rglob('*')) if p.is_file() or p.is_symlink()}
unchanged={}
for rel in ['Contents/Frameworks','Contents/Resources/nemotron','Contents/Resources/bark','Contents/Resources/nemo']:
 before=tree(old/'Pnola.app'/rel);after=tree(app/rel);assert before==after,rel
 unchanged[rel]={'identical':True,'files':len(after),'tree_sha256':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest()}
for n in ['ami-60s.wav','live-source-60s.wav','shutdown-1s.wav','native-ami.json','native-ami.json.f32']:
 shutil.copy2(old/n,root/n)
shutil.copytree(old/'control-cpu',root/'control-cpu',symlinks=True)
shutil.copy2(cache/'n5-pnola-metal-shutdown-build.log',root/'build.log')
denied=json.loads((old/'manifest.json').read_text())['denied_source_roots']+[str(old)]
(root/'offline.sb').write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(p)+')' for p in denied)+')\n')
files={}
for base,label in [(app,'app'),(root/'control-cpu','control-cpu')]:
 for name,v in tree(base).items():files[label+'/'+name]={'path':str(base/name),**v}
for n in ['ami-60s.wav','live-source-60s.wav','shutdown-1s.wav','native-ami.json','native-ami.json.f32','offline.sb','build.log']:
 p=root/n;files[n]={'path':str(p),'bytes':p.stat().st_size,'sha256':h(p)}
source={}
for n in ['src-tauri/src/engine.rs','src-tauri/src/live.rs','src-tauri/src/transcription.rs']:
 actual=Path('/Users/victor/workspace/pnola')/n;committed=subprocess.check_output(['git','show','024a88f:integrations/pnola/overlay/'+n],cwd=repo);assert actual.read_bytes()==committed
 source[n]=h(actual)
manifest={'schema_version':1,'release_qualified':False,'app_source_commit':'024a88f','predecessor_source_commit':'ef1fc1a','predecessor_manifest_sha256':h(old/'manifest.json'),'predecessor_verification_sha256':h(old/'verification.json'),'unchanged_native_and_model_trees':unchanged,'app_source_files':source,'scope':'Same-host M4 Pro/macOS26.3 signed relocated Metal app CLI shutdown regression; clean recorded/live exit, unchanged native output, unavailable-Metal ASR preservation; no GUI, microphone, fresh-machine, quality or throughput qualification. Earlier mixed-model evidence is retained with exact native/model tree identity, not rerun.','files':files,'denied_source_roots':denied}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(__file__,root/'prepare.py');print(json.dumps(unchanged,indent=2));print(root)
