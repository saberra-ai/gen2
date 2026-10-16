"""Frozen source/binary setup, delayed until sequential CPU timing completes."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,wave
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
root=cache/'quality qualification 125f907'
marker=cache/'release CPU qualification d01ef80/comparison-60m.json'
while not marker.exists():
 time.sleep(5)
assert json.loads(marker.read_text())['passed']
print('CPU timed qualification complete; preparing quality binaries',flush=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip().startswith('125f907')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
sys.path.insert(0,str(tools))
import qualify_corpus as q
corpus=cache/'quality-corpus-v1'
q.validate_protocol(tools/'corpus.protocol.json',corpus)
q.validate_corpus(corpus)
root.mkdir(exist_ok=False)
(root/'reference-offline.sb').write_text('(version 1)\n(allow default)\n(deny network*)\n')
shutil.copy2(cache/'release CPU qualification d01ef80/offline.sb',root/'offline.sb')
shutil.copy2(__file__,root/'prepare.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env=dict(os.environ,CARGO_TARGET_DIR='/Users/victor/.cache/gen2-bark-onnx/build')
cmd=['cargo','build','--locked','--release','--no-default-features','--features','backend-diar-nemo','--example','nemotron_stream']
with (root/'rust-build.log').open('w') as log:subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/examples/nemotron_stream',root/'nemotron_stream')
llama=cache/'NeMo-Speech.cpp/llama.cpp'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=llama,text=True).strip()=='bd4f514db14d87fded667787a7a963bfbaa98e89'
headers={}
for name in ['sha256/sha256.c','sha256/sha256.h','rotate-bits/rotate-bits.h']:
 path=llama/'vendor/hash'/name
 pinned=subprocess.check_output(['git','show','HEAD:vendor/hash/'+name],cwd=llama)
 assert path.read_bytes()==pinned
 headers[name]=sha(path)
commands=[cmd]
for backend in ['cpu','metal']:
 package=cache/('release CPU qualification d01ef80' if backend=='cpu' else 'release Metal qualification d01ef80')/'nemo'
 sdk=cache/f'sdk-{backend}'
 prior=json.loads((package.parent/'manifest.json').read_text())
 for name,digest in prior['header_hashes'].items():assert sha(sdk/name)==digest
 assert sha(cache/'NeMo-Speech.cpp/examples/wav_reader.h')==prior['wav_reader_sha256']
 cmd=['clang++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-x','c++',str(tools/'native_stream_probe.cpp'),str(llama/'vendor/hash/sha256/sha256.c'),f'-I{llama}/vendor/hash',f'-I{sdk}/include',f'-I{cache}/NeMo-Speech.cpp/examples',f'-L{package}/lib','-lnemo_speech_asr_c',f'-Wl,-rpath,{package}/lib','-o',str(root/f'native-{backend}')]
 with (root/f'cpp-{backend}-build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 commands.append(cmd)
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'protocol_sha256':sha(tools/'corpus.protocol.json'),'corpus_manifest_sha256':sha(corpus/'manifest.json'),'source_sha256':{n:sha(tools/n) for n in q.PROTOCOL_TOOLS},'compile_commands':commands,'pinned_llama_sha256_sources':headers,'binaries':{n:sha(root/n) for n in ['nemotron_stream','native-cpu','native-metal']},'package_manifests':{b:sha(cache/f'release {b.upper() if b=="cpu" else "Metal"} qualification d01ef80/nemo/package.json') for b in ['cpu','metal']}}
(root/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Quality binaries compiled and source/inputs pinned: '+str(root),flush=True)
