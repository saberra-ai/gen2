"""Decode the admitted M1 library only; do not execute the native runtime."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'pooled-cpu-profile-ci-v1'
producer = cache / 'pooled-macos-construction-ci-v1'
package = producer / 'restored-pooled-cpu-runtime'
read = lambda path:json.loads(Path(path).read_text())
def desc(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
library = package / 'lib/libgen2_nemo_ggml-cpu.0.25.1.dylib'
expected = read(package / 'package.json')['files']['lib/'+library.name]
assert desc(library) == {'bytes':expected['size'],'sha256':expected['sha256']}
source = cache / 'pooled-cpu-construction-local-v1/source/build/cpu-diar/_deps/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c'
audit_path = producer / 'evidence/build/llama-materialized-audit.json'
assert desc(source) == read(audit_path)['files']['ggml/src/ggml-cpu/ggml-cpu.c']
tool = Path(subprocess.check_output(['/usr/bin/xcrun','--find','llvm-objdump'],text=True).strip())
output,errors = root / 'qualified-graph-thread.txt',root / 'qualified-graph-thread.stderr.log'
argv = [str(tool),'--disassemble-symbols=_ggml_graph_compute_thread',str(library)]
before = desc(library)
with output.open('xb') as stdout,errors.open('xb') as stderr:
    result = subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=60)
assert result.returncode == 0 and desc(library) == before
text = output.read_text()
base = int(re.search(r'^([0-9a-f]+) <_ggml_graph_compute_thread>:',text,re.M)[1],16)
instructions = {int(match[1],16):match[2] for match in re.finditer(r'^\s*([0-9a-f]+):\s+[0-9a-f]+\s+([^\r\n]+)',text,re.M)}
selected = {str(offset):instructions[base+offset] for offset in [320,324,328,332,336]}
assert selected['320'].startswith('ldr') and selected['324'].startswith('cmp')
assert selected['328'].startswith('b.ne') and selected['332'].strip() == 'yield'
assert selected['336'].startswith('b\t')
record = {'schema_version':1,'native_runtime_executed':False,'library':{'path':str(library),**before},
    'tool':{'path':str(tool),**desc(tool)},'argv':argv,'returncode':0,'output':desc(output),'stderr':desc(errors),
    'source':{'path':str(source),**desc(source),'matched_m1_compiler_audit':desc(audit_path)},
    'function_base_hex':hex(base),'reported_sample_offsets_decoded':selected,
    'limits':'Read-only instruction/source inspection. The coalesced sample node includes undisplayed offsets; do not assign its entire count to this loop or interpret it as whole-process CPU utilization.'}
(root / 'instruction-inspection.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
