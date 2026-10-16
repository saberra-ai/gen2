import hashlib,json,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');native=cache/'NeMo-Speech.cpp';sdk=cache/'sdk-cpu';out=cache/'calibration-graph-probe-v1';out.mkdir(exist_ok=False)
source=repo/'tools/nemotron-diarization/calibration_graph_probe.cpp'
includes=['src/asr','src/asr/features','src/asr/encoder','src/asr/decoders','src/asr/vad','src/asr/diar','src/asr/postproc','src/asr/pnc','src/common','src/runtime/ggml','llama.cpp/ggml/include']
argv=['/usr/bin/c++','-std=c++17','-O2',str(source)]+['-I'+str(native/x)for x in includes]+['-L'+str(sdk/'lib'),'-lnemo_speech_asr','-lnemo_speech_runtime_ggml','-Wl,-rpath,'+str(sdk/'lib'),'-o',str(out/'calibration-graph-probe')]
start=time.monotonic();r=subprocess.run(argv,capture_output=True);(out/'build.stdout.log').write_bytes(r.stdout);(out/'build.stderr.log').write_bytes(r.stderr)
def record(p):return{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
row={'argv':argv,'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'source':record(source),'compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'binary':record(out/'calibration-graph-probe')if r.returncode==0 else None};(out/'build.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row));print(r.stderr.decode());r.check_returncode()
