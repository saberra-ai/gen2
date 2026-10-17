"""Model-free libc canonicalization reproduction, then a data-only denial control."""
from pathlib import Path
import json
import subprocess

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
trial = cache / 'context-development-diagnostic-v1'
original = trial / 'runtime.sb'
fixed = cache / 'context-cache-metadata-control.sb'
text = original.read_text()
before = '(deny file-read* (require-all (subpath "' + str(cache) + '") '
assert text.count(before) == 1
fixed.write_text(text.replace(before, before.replace('file-read*', 'file-read-data')))
code = '''import ctypes,json
from pathlib import Path
base=Path("/Users/victor/.cache/gen2-nemotron3-n0/context-development-execution-inputs-v1/work/assets")
lib=ctypes.CDLL(None,use_errno=True)
lib.realpath.argtypes=[ctypes.c_char_p,ctypes.c_void_p];lib.realpath.restype=ctypes.c_void_p
result={}
for name,path in [('directory',base),('model_path',base/'Nemotron-3-Diarization.q8_0.gguf')]:
 buffer=ctypes.create_string_buffer(4096);ctypes.set_errno(0)
 value=lib.realpath(str(path).encode(),buffer)
 result[name]={'success':bool(value),'errno':ctypes.get_errno(),'resolved':buffer.value.decode()}
for name,path in [('outside_model',Path("/Users/victor/.cache/gen2-nemotron3-n0/ami-prefix-inputs-v1/work-v2/assets/Nemotron-3-Diarization.q8_0.gguf")),('staged_reference',base.parent/'references/transcriber.json')]:
 try:
  with path.open('rb') as stream:stream.read(1)
 except PermissionError:
  result[name]='denied'
 else:raise RuntimeError('protected data became readable')
print(json.dumps(result))
'''
env = {'PATH':'/usr/bin:/bin','HOME':str(trial/'home'),'TMPDIR':str(trial/'tmp')+'/', 'LC_ALL':'C'}
records = {}
for name,profile in [('original',original),('data_only',fixed)]:
    argv=['/usr/bin/sandbox-exec','-f',str(profile),'/usr/bin/python3','-B','-c',code]
    process=subprocess.run(argv,cwd=trial,env=env,capture_output=True,text=True,timeout=30)
    assert process.returncode == 0
    value=json.loads(process.stdout)
    assert value['outside_model'] == value['staged_reference'] == 'denied'
    assert value['directory']['success'] is (name == 'data_only')
    assert value['model_path']['success'] is (name == 'data_only')
    records[name]={'argv':argv,'environment':env,'returncode':process.returncode,'stdout':value,'stderr':process.stderr}
out=cache/'context-cache-policy-diagnosis-v1.json'
assert not out.exists()
out.write_text(json.dumps({'model_calls':0,'records':records},indent=2)+'\n')
print(out)
