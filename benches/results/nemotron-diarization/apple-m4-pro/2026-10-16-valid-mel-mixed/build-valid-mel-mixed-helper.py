"""Compile only: future Mac mixed-runtime helper; all model cache reads denied."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard,atomic_json
out=cache/'valid-mel-mixed-helper-build-v1';assert not out.exists();out.mkdir()
prior=json.loads((tools/'valid-mel-linux-runtime.protocol.json').read_text());names=set(prior['repository_files'])|{'tests/diarization_mixed_native.rs','assets/bark/nemo128.f32'}
pins={n:descriptor(repo/n)for n in sorted(names)};source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
profile=out/'compile-only.sb';profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath '+json.dumps(str(cache))+'))\n')
env={k:os.environ[k]for k in ['HOME','PATH','TMPDIR','RUSTUP_HOME','CARGO_HOME','DEVELOPER_DIR']if k in os.environ}
env.update(CARGO_TARGET_DIR='/Users/victor/.cache/gen2-bark-onnx/build',CARGO_BUILD_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',CARGO_INCREMENTAL='0',CARGO_NET_OFFLINE='true',HOMEBREW_NO_AUTO_UPDATE='1')
argv=['/usr/bin/sandbox-exec','-f',str(profile),'cargo','test','--offline','--locked','--release','--no-default-features','--features','backend-llamacpp,metal,asr-dynamic,laya-dynamic,backend-diar-nemo','--test','diarization_mixed_native','--no-run','--message-format=json']
protocol={'scope':__doc__,'source_commit':source,'repository_files':pins,'gitlinks':prior['gitlinks'],'argv':argv,'environment':env,'models_loaded':False,'native_or_neural_test_cases_executed':0,'storage':{'start_bytes':2*1024**3,'running_floor_bytes':1024**3,'diagnostic_reserve_bytes':256*1024**2},'launcher':descriptor(Path(__file__)),'compiler_models_read_policy':'All gen2-nemotron3-n0 cache reads denied to compiler descendants; Cargo offline and network denied. This is helper compilation, not SDK construction/NN qualification.'}
atomic_json(out/'protocol.json',protocol);record={'source_commit':source,'models_loaded':False,'native_or_neural_test_cases_executed':0,'argv':argv,'compiled':False};guard=None;process=None
try:
 guard=StorageGuard(out,2*1024**3,1024**3,256*1024**2)
 with(out/'cargo.stdout.jsonl').open('xb')as stdout,(out/'cargo.stderr.log').open('xb')as stderr:
  started=time.monotonic();process=subprocess.Popen(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  record['pid']=process.pid;atomic_json(out/'command.json',record)
  record['returncode']=guard.wait(process,record,1800);record['elapsed_seconds']=time.monotonic()-started
 if record['returncode']!=0:raise RuntimeError('compile-only Cargo command failed')
 binaries=[]
 for line in(out/'cargo.stdout.jsonl').read_text().splitlines():
  item=json.loads(line)
  if item.get('reason')=='compiler-artifact'and item['target']['name']=='diarization_mixed_native'and item.get('executable'):binaries.append(Path(item['executable']))
 assert len(binaries)==1
 helper=out/'mixed-native-tests';shutil.copy2(binaries[0],helper)
 for name,expected in pins.items():assert descriptor(repo/name)==expected,name
 for name,expected in prior['gitlinks'].items():
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo/name,text=True).strip()==expected
  assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo/name)
 record.update(compiled=True,helper=descriptor(helper),helper_source_path=str(binaries[0]),repository_files=pins)
except BaseException as error:
 record['error']=str(error)
 if process is not None and process.poll()is None:
  import signal
  os.killpg(process.pid,signal.SIGKILL);process.wait();record['returncode']=process.returncode
 raise
finally:
 if guard is not None:guard.release();record['storage']=guard.record
 atomic_json(out/'command.json',record)
print(json.dumps({'compiled':True,'helper':record['helper'],'models_loaded':False,'elapsed_seconds':record['elapsed_seconds']}))
