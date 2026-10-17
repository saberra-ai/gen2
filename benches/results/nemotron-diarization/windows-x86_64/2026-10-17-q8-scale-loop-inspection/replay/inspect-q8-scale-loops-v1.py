"""Inspect admitted A-scale disassemblies only; no compiler/native calls."""
import hashlib
import json
from pathlib import Path
import re


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def inspect_loop(text, method):
    signature = '::tinyBLAS_Q0_AVX<struct block_q8_0,struct block_q8_0,float>::' + method + '('
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line and not line[0].isspace()
              and line.endswith(':') and signature in line]
    if len(starts) != 1:
        raise ValueError('one exact Q8/Q8/F32 function header required')
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()
                and lines[i].endswith(':')), len(lines))
    instructions = []
    for line in lines[start + 1:end]:
        match = re.match(r'^\s{2}([0-9A-F]{16}):\s+(?:[0-9A-F]{2}\s+)+([a-z][a-z0-9]+)\s*(.*?)\s*$', line)
        if match:
            instructions.append({'offset': int(match[1], 16), 'op': match[2], 'args': match[3]})
    offsets = {row['offset'] for row in instructions}
    if len(offsets) != len(instructions) or not instructions:
        raise ValueError('unique addressed function instructions required')
    edges = []
    for row in instructions:
        if row['op'].startswith('j') and row['op'] != 'jmp' and re.fullmatch('[0-9A-F]{16}', row['args']):
            target = int(row['args'], 16)
            if target < row['offset'] and target in offsets:
                edges.append((target, row['offset']))
    candidates = []
    for first, last in edges:
        body = [row for row in instructions if first <= row['offset'] <= last]
        nested = any((lo, hi) != (first, last) and first <= lo <= hi <= last for lo, hi in edges)
        if not nested and sum(row['op'].startswith('vfmadd') for row in body) == 8:
            candidates.append((first, last, body))
    if len(candidates) != 1:
        raise ValueError('one innermost backward conditional loop with eight FMA instructions required; inspect manually')
    first, last, body = candidates[0]
    accesses = []
    for row in body:
        operands = row['args'].split(',')
        for index, operand in enumerate(operands):
            match = re.search(r'\b(xmmword|ymmword) ptr \[rsp(?:[+-][^\]]*)?\]', operand)
            if match:
                accesses.append({'offset': hex(row['offset']), 'instruction': row['op'] + ' ' + row['args'],
                    'bytes': 16 if match[1] == 'xmmword' else 32, 'access': 'store' if index == 0 else 'load'})
    normalized = '\n'.join(row['op'] + ' ' + row['args'] for row in body).encode()
    return {'function_header': lines[start], 'function_instructions': len(instructions), 'loop_start': hex(first),
        'backedge': hex(last), 'loop_instruction_count': len(body), 'fma_instructions': 8,
        'loop_instructions_sha256': hashlib.sha256(normalized).hexdigest(), 'vector_stack_accesses': accesses,
        'static_vector_stack_operand_bytes': sum(row['bytes'] for row in accesses)}


def main():
    cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
    root = cache / 'windows-q8-scale-ci-v1'
    parent = cache / 'inspect-q8-pair-loops-v1.py'
    assert descriptor(parent) == {'bytes':5650,'sha256':'54fe10286fe4fc683484fda772049089c3eeb2bd548c82a754f884a38c9ebe86'}
    replay = root / 'collection-replay-v1.json'
    q = json.loads(replay.read_text())
    assert q['collector'] == descriptor(cache / 'collect-windows-q8-scale-v3.py')
    preexecution = cache / 'windows-q8-scale-receivers-pre-execution-v3.json'
    pins = json.loads(preexecution.read_text())['files']
    assert q['collector'] == pins['collect-windows-q8-scale-v3.py']
    assert q['source_commit'] == '4c72846674d373915fec72fbb0fccbc5a8a1e068' and q['run_id'] == 39072824849
    assert q['diagnostic_completed'] is True and q['all_1056_outputs_bit_exact'] is True
    assert q['all_12_scale_half_broadcasts_verified'] is True and q['commands_verified'] == 95
    member_path = root / 'delivery-admission-v1/artifact-members.json'
    admission_path = root / 'delivery-admission-v1/qualification.json'
    admission = json.loads(admission_path.read_text())
    assert q['delivery_admission'] == descriptor(admission_path)
    assert admission['artifact_members'] == descriptor(member_path)
    members = json.loads(member_path.read_text())
    delivered = root / 'delivery-admission-v1/delivered'
    inputs = {str(path): descriptor(path) for path in (Path(__file__), parent, preexecution, replay, member_path, admission_path)}
    functions = {}
    for role, method in [('control', 'gemmMx4<2>'), ('scale', 'gemmScaleMx4<2>')]:
        for name in ('kernel-' + role + '.obj', 'disasm-' + role + '.stdout.log'):
            path = delivered / 'evidence' / name
            assert descriptor(path) == members['evidence/' + name]
            inputs[str(path)] = descriptor(path)
        path = delivered / 'evidence' / ('disasm-' + role + '.stdout.log')
        functions[role] = inspect_loop(path.read_text(), method)
    assert all(descriptor(Path(name)) == value for name, value in inputs.items())
    result = {'source_commit': q['source_commit'], 'run_id': q['run_id'], 'inputs': inputs,
        'method': 'Unchanged parent unique Q8/Q8/F32 signature and unique innermost backward conditional loop with eight FMA instructions; explicit RSP vector operands only.',
        'functions': functions, 'new_native_calls': 0, 'production_qualified': False,
        'limits': 'Static instruction operand widths only. No measured traffic, CPU time, spill bottleneck or speedup cause. Original numerical, timing and observer decisions remain unchanged.'}
    with (root / 'loop-inspection-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({role: {'instructions':value['loop_instruction_count'],'static_stack_operand_bytes':value['static_vector_stack_operand_bytes']} for role,value in functions.items()}))


if __name__ == '__main__': main()
