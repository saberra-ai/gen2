"""Model-free development control, not a Windows/model/SDK qualification."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/ggml-cpu-profile-control-local-v2')

def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

identity = descriptor(Path(__file__))
inputs = json.loads((root / 'inputs.json').read_text())
assert descriptor(root / 'source/ggml/src/ggml-cpu/ggml-cpu.c') == inputs['instrumented_cpu_source']
assert descriptor(root / 'control.cpp') == inputs['control_source']
assert descriptor(tools / 'ggml_cpu_profile_source.py') == inputs['subject_inputs']['ggml_cpu_profile_source.py']
assert descriptor(tools / 'ggml_cpu_profile_control.cpp') == inputs['subject_inputs']['ggml_cpu_profile_control.cpp']
# Decoder correction happened after compilation, before the first native control.
# Keep original compilation inputs intact and bind the execution decoder separately.
snapshot = root / 'execution-subject-inputs-v1'
snapshot.mkdir()
for name in inputs['subject_inputs']:
    shutil.copyfile(tools / name, snapshot / name)
subjects = {name: descriptor(snapshot / name) for name in inputs['subject_inputs']}
(root / 'execution-inputs-v1.json').write_text(json.dumps({'compilation_inputs': descriptor(root / 'inputs.json'),
    'subject_inputs': subjects, 'runner': identity}, indent=2) + '\n')
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import source_inventory
from ggml_cpu_profile import CANONICAL_GGML_C_SHA256, OPERATION_NAMES, environment, summarize
from ggml_cpu_profile_source import PATH
expected = inputs['canonical_ggml_files'].copy()
expected[PATH] = inputs['instrumented_cpu_source']
assert source_inventory(root / 'source') == expected
canonical_c = root / 'source/ggml/src/ggml.c'
assert descriptor(canonical_c)['sha256'] == CANONICAL_GGML_C_SHA256
block = re.search(r'static const char \* GGML_OP_NAME\[GGML_OP_COUNT\] = \{(.*?)\n\};', canonical_c.read_text(), re.S)[1]
assert tuple(re.findall(r'"([A-Z0-9_]+)"', block)) == OPERATION_NAMES
commands = json.loads((root / 'build/compile_commands.json').read_text())
assert len(commands) == 27 and len({row['file'] for row in commands}) == 27
compiler_sources = {}
for row in commands:
    path = Path(row['file'])
    assert path == root / 'control.cpp' or path.is_relative_to(root / 'source')
    compiler_sources[str(path)] = descriptor(path)
binary = root / 'build/profile-control'
binary_identity = descriptor(binary)
result = {'scope': 'Model-free development test on macOS arm64 only; no SDK/model/Windows/release qualification',
          'completed': False, 'runner': identity, 'binary': binary_identity,
          'execution_inputs': descriptor(root / 'execution-inputs-v1.json'),
          'compiler_sources': compiler_sources, 'cases': [], 'production_qualified': False}
target = root / 'control-results-v1.json'
with target.open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
try:
    all_vectors = []
    for threads in (1, 2, 4):
        reference = None
        for index, enabled in enumerate((False, True, True, False)):
            phase = f'workers-{threads}-case-{index}'
            assert descriptor(binary) == binary_identity
            argv = [str(binary), str(threads), str(root / phase)]
            started = time.monotonic_ns()
            with (root / (phase + '.stdout.log')).open('xb') as out, (root / (phase + '.stderr.log')).open('xb') as err:
                child = subprocess.run(argv, env=environment(os.environ, enabled), stdout=out, stderr=err, timeout=30)
            row = {'threads': threads, 'index': index, 'enabled': enabled, 'argv': argv,
                   'returncode': child.returncode, 'wall_ns': time.monotonic_ns() - started}
            result['cases'].append(row)
            child.check_returncode()
            assert (root / (phase + '.stdout.log')).read_text().splitlines() == [f'graph-control threads={threads} repeat={repeat} bytes=8192' for repeat in range(3)]
            row['profile'] = summarize((root / (phase + '.stderr.log')).read_text(), enabled,
                                       expected_graphs=3, expected_threads=threads)
            if enabled:
                assert all(worker['fused'] == 1 for graph in row['profile']['graphs'] for worker in graph['workers'])
            row['outputs'] = {}
            for repeat in range(3):
                path = root / (phase + f'.{repeat}.f32')
                raw = path.read_bytes()
                assert len(raw) == 8192
                values = struct.unpack('<2048f', raw)
                assert all(math.isfinite(value) for value in values) and any(value != 0 for value in values)
                if reference is None:
                    reference = raw
                assert raw == reference
                row['outputs'][path.name] = descriptor(path)
                all_vectors.append(raw)
            assert descriptor(binary) == binary_identity
            target.write_text(json.dumps(result, indent=2) + '\n')
    assert source_inventory(root / 'source') == expected
    assert all(descriptor(Path(name)) == value for name, value in compiler_sources.items())
    assert all(descriptor(tools / name) == descriptor(snapshot / name) == value for name, value in subjects.items())
    assert descriptor(Path(__file__)) == identity and descriptor(binary) == binary_identity
    result['all_36_outputs_bit_exact_across_worker_counts'] = all(raw == all_vectors[0] for raw in all_vectors)
    result['completed'] = True
except BaseException as error:
    result['error'] = repr(error)
    raise
finally:
    target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'cases': len(result['cases']), 'completed': result['completed'],
                  'all_36_outputs_bit_exact_across_worker_counts': result.get('all_36_outputs_bit_exact_across_worker_counts')}))
