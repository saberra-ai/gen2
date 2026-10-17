"""Retain actual loop evidence and terminal feature CI; no native execution."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
root = cache / 'windows-q8-scale-ci-v1'
target = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-scale-loop-inspection'
source = '4c72846674d373915fec72fbb0fccbc5a8a1e068'


def desc(path):
    h = hashlib.sha256(); size = 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2): size += len(block); h.update(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


bound = {}


def bind(path, expected=None):
    raw = path.read_bytes()
    value = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    key = str(path)
    if key in bound:
        assert value == bound[key]
    if expected is not None:
        assert value == expected
    bound.setdefault(key, value)
    return raw


def read(path): return json.loads(bind(path))


bind(Path(__file__))


inspection = read(root / 'loop-inspection-v1.json')
assert inspection['source_commit'] == source and inspection['run_id'] == 39072824849
assert inspection['new_native_calls'] == 0 and inspection['production_qualified'] is False
assert set(inspection['functions']) == {'control', 'scale'}
assert all(row['fma_instructions'] == 8 for row in inspection['functions'].values())
for path, expected in inspection['inputs'].items(): bind(Path(path), expected)
preexecution_path = cache / 'q8-scale-loop-inspector-pre-execution-v1.json'
preexecution = read(preexecution_path)
for path, expected in preexecution['files'].items(): bind(Path(path), expected)
assert preexecution['parent_algorithm_ast_equal'] is True
functions = []
for name in ('inspect-q8-scale-loops-v1.py', 'inspect-q8-pair-loops-v1.py'):
    tree = ast.parse(bind(cache / name))
    function, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'inspect_loop']
    functions.append(ast.dump(function, include_attributes=False))
assert functions[0] == functions[1]
run_path = cache / 'q8-scale-feature-run-terminal-v1.json'
jobs_path = cache / 'q8-scale-feature-terminal-v1.json'
run, jobs = read(run_path), read(jobs_path)
assert run['id'] == 39072824905 and run['head_sha'] == source
assert run['status'] == 'completed' and run['conclusion'] == 'success'
assert run['repository']['full_name'] == 'saberra-ai/gen2'
assert jobs['total_count'] == len(jobs['jobs']) == 13
assert all(row['run_id'] == run['id'] and row['head_sha'] == source and row['status'] == 'completed'
    and row['conclusion'] == 'success' for row in jobs['jobs'])
files = {'inspection.json': root / 'loop-inspection-v1.json',
    'inspection-pre-execution.json': preexecution_path,
    'publisher.py': Path(__file__), 'feature-ci/run.json': run_path, 'feature-ci/jobs.json': jobs_path}
for path in inspection['inputs']:
    p = Path(path)
    if p.suffix.lower() == '.obj': name = 'objects/' + p.name
    elif p.name.startswith('disasm-'): name = 'disassembly/' + p.name
    elif p.name == 'qualification.json': name = 'replay/delivery-admission.json'
    else: name = 'replay/' + p.name
    assert name not in files
    files[name] = p
assert len({name.casefold() for name in files}) == len(files)
assert all(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1 for path in files.values())
inputs = {str(path): bound[str(path)] for path in files.values()}
assert all(value['bytes'] <= 256 * 1024**2 for value in inputs.values())
assert sum(value['bytes'] for value in inputs.values()) <= 1024**3
assert not target.exists() and not target.is_symlink()
target.mkdir()
for name, path in files.items():
    dest = target / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == inputs[str(path)]
    with dest.open('xb') as stream: stream.write(raw)
    assert desc(dest) == inputs[str(path)]
result = {'source_commit': source, 'run_id': inspection['run_id'], 'feature_run_id': run['id'],
    'feature_jobs_success': 13, 'publisher': bound[str(Path(__file__))], 'receiver_inputs': inputs,
    'new_native_calls': 0, 'production_qualified': False,
    'scope': 'Static exact-loop counts and terminal feature CI only. Original failed speed/observer gates remain failed; no measured traffic or bottleneck cause.'}
with (target / 'qualification.json').open('x') as stream: stream.write(json.dumps(result, indent=2) + '\n')
for path, expected in inputs.items(): assert desc(Path(path)) == expected
observed = {}
folded = set()
entries = 0
for parent, directories, names in os.walk(target, followlinks=False):
    for name in [*directories, *names]:
        p = Path(parent) / name; relative = p.relative_to(target).as_posix(); entries += 1
        assert entries <= 10000 and relative.casefold() not in folded
        folded.add(relative.casefold()); mode = p.lstat().st_mode
        assert not stat.S_ISLNK(mode)
        if stat.S_ISDIR(mode): continue
        assert stat.S_ISREG(mode) and p.stat().st_nlink == 1
        observed[relative] = desc(p)
assert set(observed) == set(files) | {'qualification.json'}
assert all(observed[name] == inputs[str(path)] for name, path in files.items())
assert sum(value['bytes'] for value in observed.values()) <= 1024**3
with (target / 'collection.json').open('x') as stream:
    stream.write(json.dumps({'schema_version': 1, 'files': observed}, indent=2) + '\n')
print(json.dumps({'receipt': str(target), 'members': len(observed) + 1,
    'bytes_without_manifest': sum(value['bytes'] for value in observed.values()),
    'feature_jobs_success': 13, 'new_native_calls': 0}))
