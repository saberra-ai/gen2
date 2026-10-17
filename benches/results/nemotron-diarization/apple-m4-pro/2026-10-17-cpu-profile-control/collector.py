"""Publish only saved model-free development-test data; no compiler/native calls."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import math
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'ggml-cpu-profile-control-local-v2'

def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def read(path):
    return json.loads(path.read_text())

identity = descriptor(Path(__file__))
inputs, execution, result = (read(root / name) for name in ('inputs.json', 'execution-inputs-v1.json', 'control-results-v1.json'))
assert result['completed'] is True and result['production_qualified'] is False and 'error' not in result
assert descriptor(root / 'execution-inputs-v1.json') == result['execution_inputs']
assert descriptor(root / 'inputs.json') == execution['compilation_inputs']
runner = cache / 'run-cpu-profile-control-local-v2.py'
assert descriptor(runner) == result['runner'] == execution['runner']
for name, expected in execution['subject_inputs'].items():
    assert descriptor(tools / name) == descriptor(root / 'execution-subject-inputs-v1' / name) == expected
for name, expected in inputs['subject_inputs'].items():
    assert descriptor(root / 'subject-inputs' / name) == expected
sys.path.insert(0, str(tools))
def source_inventory(root):
    values = {}
    if root.is_symlink() or not root.is_dir():
        raise ValueError('regular source directory required')
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('source inventory symlink')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('source inventory nonregular entry')
        values[path.relative_to(root).as_posix()] = descriptor(path)
    return values

from ggml_cpu_profile import CANONICAL_GGML_C_SHA256, OPERATION_NAMES, summarize
from ggml_cpu_profile_source import PATH, transform
expected = inputs['canonical_ggml_files'].copy()
expected[PATH] = inputs['instrumented_cpu_source']
assert source_inventory(root / 'source') == expected
assert descriptor(root / 'control.cpp') == inputs['control_source'] == execution['subject_inputs']['ggml_cpu_profile_control.cpp']
original_cpu = cache / 'windows-compiler-fp-ci-v1/compiler-fp-replay-v1/ggml' / PATH
assert transform(original_cpu.read_bytes()) == (root / 'source' / PATH).read_bytes()
canonical_c = root / 'source/ggml/src/ggml.c'
assert descriptor(canonical_c)['sha256'] == CANONICAL_GGML_C_SHA256
block = re.search(r'static const char \* GGML_OP_NAME\[GGML_OP_COUNT\] = \{(.*?)\n\};', canonical_c.read_text(), re.S)[1]
assert tuple(re.findall(r'"([A-Z0-9_]+)"', block)) == OPERATION_NAMES
assert len(result['compiler_sources']) == 27
assert all(descriptor(Path(name)) == value for name, value in result['compiler_sources'].items())
assert descriptor(root / 'build/profile-control') == result['binary']
assert [(c['threads'], c['index'], c['enabled']) for c in result['cases']] == [(t,i,e) for t in (1,2,4) for i,e in enumerate((False,True,True,False))]
files = {'compilation-inputs.json': root / 'inputs.json', 'execution-inputs.json': root / 'execution-inputs-v1.json',
    'control-results.json': root / 'control-results-v1.json', 'runner.py': runner,
    'cpu-profile.patch.log': root / 'cpu-profile.patch.log', 'control.cpp': root / 'control.cpp',
    'CMakeLists.txt': root / 'CMakeLists.txt', 'CMakeCache.txt': root / 'build/CMakeCache.txt',
    'compile_commands.json': root / 'build/compile_commands.json'}
for name in ('configure.stdout.log','configure.stderr.log','export-commands.stdout.log','export-commands.stderr.log',
             'build.stdout.log','build.stderr.log','python-suite.stdout.log','python-suite.stderr.log'):
    files[name] = root / name
for prefix, folder in [('compiled-subjects', 'subject-inputs'), ('execution-subjects', 'execution-subject-inputs-v1')]:
    for name in execution['subject_inputs']:
        files[prefix + '/' + name] = root / folder / name
vectors = []
for case in result['cases']:
    assert type(case['returncode']) is int and case['returncode'] == 0
    phase = f"workers-{case['threads']}-case-{case['index']}"
    log = root / (phase + '.stderr.log')
    assert summarize(log.read_text(), case['enabled'], expected_graphs=3, expected_threads=case['threads']) == case['profile']
    for suffix in ('.stdout.log', '.stderr.log'):
        files[phase + suffix] = root / (phase + suffix)
    assert len(case['outputs']) == 3
    for name, expected_output in case['outputs'].items():
        path = root / name
        assert descriptor(path) == expected_output and expected_output['bytes'] == 8192
        vector = struct.unpack('<2048f', path.read_bytes())
        assert all(math.isfinite(value) for value in vector) and any(value != 0 for value in vector)
        files[name] = path
        vectors.append(path.read_bytes())
assert len(vectors) == 36 and all(raw == vectors[0] for raw in vectors)
assert result['all_36_outputs_bit_exact_across_worker_counts'] is True
assert 'Ran 407 tests' in (root / 'python-suite.stderr.log').read_text()
assert 'OK (skipped=7)' in (root / 'python-suite.stderr.log').read_text()
prior = root / 'failed-collection-admission-v1'
prior_failure = root / 'collection-admission-v1.json'
failure = read(prior_failure)
assert source_inventory(prior) == failure['original_files']
assert descriptor(prior / 'collection.json') == failure['original_manifest']
assert descriptor(prior / 'collector.py') == failure['original_collector']
files.update({'collector-v1.py': prior / 'collector.py', 'collection-v1.json': prior / 'collection.json',
              'collection-qualification-v1.json': prior / 'collection-qualification.json',
              'collection-admission-v1.json': prior_failure})
bindings = {name: descriptor(path) for name, path in files.items()}
dest = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-cpu-profile-control'
assert not dest.exists()
dest.mkdir()
for name, path in files.items():
    target = dest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert descriptor(target) == bindings[name]
shutil.copyfile(Path(__file__), dest / 'collector.py')
assert descriptor(dest / 'collector.py') == descriptor(Path(__file__)) == identity
assert all(descriptor(path) == descriptor(dest / name) == bindings[name] for name, path in files.items())
summary = {'scope': result['scope'], 'collection_completed': True, 'collector': identity,
    'native_processes': 12, 'graph_evaluations': 36, 'canonical_source_files': 3425,
    'compiler_sources': 27, 'probability_or_model_calls': 0, 'raw_output_bytes_each': 8192,
    'all_outputs_bit_exact': True, 'full_python_tests': 407, 'skipped_windows_tests': 7,
    'canonical_operation_names': len(OPERATION_NAMES), 'canonical_ggml_c_sha256': CANONICAL_GGML_C_SHA256,
    'profiled_cpu_source': inputs['instrumented_cpu_source'], 'production_qualified': False,
    'prior_publication_review': descriptor(prior_failure),
    'windows_or_model_qualified': False,
    'limits': ['Working-tree development test; compilation subjects and corrected execution decoder are separately recorded.',
               'CPU wall timings include internal waits, OS scheduling and observer effects; they do not establish uninstrumented kernel cost.',
               'No actual Windows graph profile, neural inference, SDK construction or release performance result is established.',
               'Native binaries remain in the local development cache; Git retains hashes, compile inputs, logs and all synthetic output arrays.']}
assert source_inventory(prior) == failure['original_files']
(dest / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
(dest / 'collection.json').write_text(json.dumps({'files':source_inventory(dest)}, indent=2) + '\n')
print(json.dumps({'receipt':str(dest),'files':len(source_inventory(dest)),'all_outputs_bit_exact':True}))
