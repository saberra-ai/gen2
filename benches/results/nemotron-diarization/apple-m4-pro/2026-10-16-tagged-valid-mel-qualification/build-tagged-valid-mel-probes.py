"""Compile tagged qualification helpers and separate Rust artifact; execute no probe/model."""
import hashlib,json,pathlib,subprocess,time,wave
import numpy as np
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';source=cache/'NeMo-Speech-valid-mel-tagged-v1';sdk=cache/'sdk-valid-mel-tagged-cpu-v1';out=cache/'tagged-valid-mel-probes-v1';metadata=cache/'valid-mel-tagged-metadata-v1/recovered-build.json';build=json.loads(metadata.read_text())
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not out.exists();out.mkdir()
for name,expected in build['installed_library_paths'].items():
 p=sdk/'lib'/name;assert rec(p)=={k:expected[k]for k in ['bytes','sha256']}and(str(p.readlink())if p.is_symlink()else None)==expected['symlink_target']
archive=pathlib.Path(build['runtime_archive']['path']);assert rec(archive)=={k:build['runtime_archive'][k]for k in ['bytes','sha256']}
profile=out/'compile-only.sb';denied=[cache/x for x in ['assets','native-f32-conversion-v1','quality-corpus-v1','tail-framing-v1','research/native-f32-control']];profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(str(p))+')'for p in denied)+')\n')
env={'PATH':'/Users/victor/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','LC_ALL':'C','HOME':str(pathlib.Path.home()),'CARGO_TARGET_DIR':'/Users/victor/.cache/gen2-bark-onnx/build','CARGO_NET_OFFLINE':'true','PYTHONDONTWRITEBYTECODE':'1'}
commands=[]
def run(name,argv):
 actual=['/usr/bin/sandbox-exec','-f',str(profile),*map(str,argv)];record={'name':name,'argv':actual,'environment':env,'cwd':str(repo)};commands.append(record);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');start=time.monotonic()
 with (out/(name+'.stdout.log')).open('xb')as stdout,(out/(name+'.stderr.log')).open('xb')as stderr:r=subprocess.run(actual,cwd=repo,env=env,stdout=stdout,stderr=stderr)
 record.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-start);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(json.dumps({'name':name,'returncode':r.returncode}),flush=True);r.check_returncode()
common=['/usr/bin/c++','-std=c++17','-O2','-fvisibility=hidden','-DNEMO_SPEECH_GGML_PATCHED=1',str(tools/'valid_mel_finish_probe.cpp')]
for inc in ['src/asr','src/asr/features','src/asr/encoder','src/asr/decoders','src/asr/vad','src/asr/diar','src/asr/postproc','src/asr/pnc','src/common','src/runtime/ggml','build/cpu-diar/_deps/llama.cpp/ggml/include']:common.append('-I'+str(source/inc))
run('finish-probe',common+[str(archive),'-L'+str(sdk/'lib'),'-lnemo_speech_asr','-lggml','-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(sdk/'lib'),'-o',out/'valid-mel-finish-probe'])
hashing=source/'llama.cpp/vendor/hash'
run('c-api-probe',['/usr/bin/c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-x','c++',tools/'native_stream_probe_valid_mel.cpp',hashing/'sha256/sha256.c','-I'+str(sdk/'include'),'-I'+str(source/'examples'),'-I'+str(hashing),'-L'+str(sdk/'lib'),'-lnemo_speech_asr_c','-Wl,-rpath,'+str(sdk/'lib'),'-o',out/'native-stream-probe-valid-mel'])
run('rust-release',['/Users/victor/.cargo/bin/cargo','build','--offline','--locked','--release','--no-default-features','--features','backend-diar-nemo','--example','nemotron_valid_mel_stream','--message-format=json'])
messages=[json.loads(line)for line in(out/'rust-release.stdout.log').read_text().splitlines()if line.startswith('{')]
artifacts=[r for r in messages if r.get('reason')=='compiler-artifact'and r['target']['name']=='nemotron_valid_mel_stream'];assert len(artifacts)==1 and artifacts[0]['executable'];executable=pathlib.Path(artifacts[0]['executable']);import shutil;shutil.copy2(executable,out/'nemotron-valid-mel-stream')
# Construct a lossless PCM16 container of the already retained calibration.
audio=cache/'tail-framing-v1/calibration/audio.f32';a=np.fromfile(audio,dtype='<f4');pcm=np.rint(a*32768).astype('<i2');assert len(a)==17872 and np.array_equal(a,pcm.astype('<f4')/32768)
with wave.open(str(out/'calibration.wav'),'wb')as writer:writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000);writer.writeframes(pcm.tobytes())
with wave.open(str(out/'calibration.wav'),'rb')as reader:assert(np.frombuffer(reader.readframes(17872),dtype='<i2').astype('<f4')/32768).tobytes()==audio.read_bytes()
for name in ['valid-mel-finish-probe','native-stream-probe-valid-mel','nemotron-valid-mel-stream']:(out/(name+'.load-commands.txt')).write_bytes(subprocess.check_output(['/usr/bin/otool','-l',str(out/name)]))
record={'source_commit_at_build':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'sdk_construction_metadata':rec(metadata),'runtime_archive':build['runtime_archive'],'commands':commands,'input_pcm':rec(audio),'lossless_pcm16_wav':rec(out/'calibration.wav'),'rust_artifact':artifacts[0],'binaries':{name:rec(out/name)for name in ['valid-mel-finish-probe','native-stream-probe-valid-mel','nemotron-valid-mel-stream']},'source_files':{name:rec(repo/name)for name in ['Cargo.toml','Cargo.lock','examples/nemotron_stream.rs','tools/nemotron-diarization/valid_mel_finish_probe.cpp','tools/nemotron-diarization/native_stream_probe_valid_mel.cpp',*['src/diarization/'+p.name for p in (repo/'src/diarization').glob('*.rs')]]},'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'rustc':subprocess.check_output(['/Users/victor/.cargo/bin/rustc','--version','--verbose'],text=True),'neural_calls':0,'probes_executed':False,'pcm_container_conversion_lossless':True}
(out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'compiled':True,'binary_count':3,'neural_calls':0}))
