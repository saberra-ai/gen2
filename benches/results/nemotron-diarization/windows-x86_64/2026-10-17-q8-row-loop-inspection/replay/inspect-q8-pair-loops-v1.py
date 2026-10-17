"""Inspect admitted saved disassembly; static operand widths are not traffic."""
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
    root = cache / 'windows-q8-pair-ci-v1'
    replay = root / 'collection-replay-v2.json'
    q = json.loads(replay.read_text())
    assert q['collector'] == descriptor(cache / 'collect-windows-q8-pair-v2.py') == {
        'bytes': 15654, 'sha256': '7c7058112e347d5cc9fc4e4554536d823b385e17191b2beda9925350f80dd0ef'}
    assert q['source_commit'] == '259966ceca113f50d7d172fa63269b0bcb17b37a' and q['run_id'] == 39066832336
    assert q['diagnostic_completed'] is True and q['all_1056_outputs_bit_exact'] is True
    assert q['commands_verified'] == 95
    member_path = root / 'delivery-admission-v1/artifact-members.json'
    admission_path = root / 'delivery-admission-v1/qualification.json'
    admission = json.loads(admission_path.read_text())
    assert q['delivery_admission'] == descriptor(admission_path)
    assert admission['artifact_members'] == descriptor(member_path)
    members = json.loads(member_path.read_text())
    delivered = root / 'delivery-admission-v1/delivered'
    inputs = {str(path): descriptor(path) for path in (Path(__file__), replay, member_path)}
    functions = {}
    for role, method in [('control', 'gemmMx4<2>'), ('pair', 'gemmPair2x4')]:
        for name in ('kernel-' + role + '.obj', 'disasm-' + role + '.stdout.log'):
            path = delivered / 'evidence' / name
            assert descriptor(path) == members['evidence/' + name]
            inputs[str(path)] = descriptor(path)
        path = delivered / 'evidence' / ('disasm-' + role + '.stdout.log')
        functions[role] = inspect_loop(path.read_text(), method)
    assert all(descriptor(Path(name)) == value for name, value in inputs.items())
    result = {'source_commit': q['source_commit'], 'run_id': q['run_id'], 'inputs': inputs,
        'method': 'Unique Q8/Q8/F32 header; unique innermost backward conditional loop with eight FMA instructions; explicit RSP vector operands only.',
        'functions': functions, 'new_native_calls': 0, 'production_qualified': False,
        'limits': 'Static instruction operand widths only. No measured traffic, CPU time, spill bottleneck or speedup cause. Original independent timing and observer decisions remain unchanged.'}
    with (root / 'loop-inspection-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({role: value['static_vector_stack_operand_bytes'] for role, value in functions.items()}))


if __name__ == '__main__': main()
