"""Clone bytes for current-profile Pnola adapter qualification; no neural execution."""
import json,subprocess,sys,wave
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from qualification_storage import atomic_json
from package_macos import verify
out=cache/'valid-mel-pnola-inputs-v1';assert not out.exists();out.mkdir();work=out/'work';work.mkdir();compiled=cache/'valid-mel-pnola-helper-build-v2';b=json.loads((compiled/'build.json').read_text());assert b['compiled']and not b['models_loaded'];mixed=cache/'valid-mel-mixed-inputs-v1/work'
pairs=[('cpu/nemo',mixed/'cpu/nemo'),('metal/nemo',mixed/'metal/nemo'),('models/nemotron',mixed/'models/nemotron'),('models/bark',mixed/'models/bark'),('providers/libonnxruntime.dylib',mixed/'providers/libonnxruntime.dylib'),('bin/pnola-tests',compiled/'pnola-tests'),('cli/Contents/MacOS/pnola-cli',compiled/'pnola-cli'),('cli/Contents/Frameworks/libonnxruntime.dylib',mixed/'providers/libonnxruntime.dylib'),('assets/ami.wav',cache/'valid-mel-macos-runtime-inputs-v1/work/assets/ami_en2002d_2132.wav'),('assets/bark-reference.json',cache/'n4-bark-ami.json'),('assets/live.wav',cache/'relocated Pnola frameworks candidate/ami.wav'),('references/cpu.json',cache/'valid-mel-macos-package-qualification-v2/public.json'),('references/metal.json',cache/'valid-mel-metal-package-qualification-v1/public.json')]
commands=[];origins={}
for name,source in pairs:
 dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);argv=['/bin/cp','-cR',str(source.resolve()),str(dest)];result=subprocess.run(argv,capture_output=True,text=True);commands.append({'name':name,'argv':argv,'returncode':result.returncode,'stderr':result.stderr});atomic_json(out/'commands.json',commands);result.check_returncode()
 files=[source]if source.is_file()else[p for p in source.rglob('*')if p.is_file()and not p.is_symlink()]
 for f in files:
  target=dest if source.is_file()else dest/f.relative_to(source);assert descriptor(f)==descriptor(target);origins[str(target.relative_to(work))]={'source':str(f),**descriptor(f)}
for backend in ['cpu','metal']:assert verify(work/backend/'nemo')['backend']==backend
for name,expected in [('ami.wav',60.),('live.wav',80.04)]:
 with wave.open(str(work/'assets'/name))as f:assert f.getnchannels()==1 and f.getframerate()==16000 and f.getsampwidth()==2 and f.getnframes()/16000==expected
atomic_json(out/'build.json',{'scope':__doc__,'models_loaded':False,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'helper_build_record':descriptor(compiled/'build.json'),'helper_build_protocol':descriptor(compiled/'protocol.json'),'origins':origins,'files':{str(f.relative_to(work)):descriptor(f)for f in sorted(work.rglob('*'))if f.is_file()and not f.is_symlink()},'aliases':{str(f.relative_to(work)):str(f.readlink())for f in work.rglob('*')if f.is_symlink()}})
print('prepared Pnola byte clones; no native/model/query execution')
