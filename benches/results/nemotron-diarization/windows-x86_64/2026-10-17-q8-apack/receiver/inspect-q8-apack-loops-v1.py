"""Inspect admitted packed-B-scale objects; original parser, no native calls."""
import hashlib
import json
from pathlib import Path
import re
bound = {}

def read_bytes(path, expected=None):
    raw = path.read_bytes()
    value = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    key = str(path)
    if key in bound:
        assert value == bound[key]
    if expected is not None:
        assert value == expected
    bound.setdefault(key, value)
    return raw

def read(path, expected=None):
    return json.loads(read_bytes(path, expected))

def inspect_loop(text, method, fma_count):
    signature = '::tinyBLAS_Q0_AVX<struct block_q8_0,struct block_q8_0,float>::' + method + '('
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line and (not line[0].isspace()) and line.endswith(':') and (signature in line)]
    if len(starts) != 1:
        raise ValueError('one exact Q8/Q8/F32 function header required')
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and (not lines[i][0].isspace()) and lines[i].endswith(':')), len(lines))
    instructions = []
    for line in lines[start + 1:end]:
        match = re.match('^\\s{2}([0-9A-F]{16}):\\s+(?:[0-9A-F]{2}\\s+)+([a-z][a-z0-9]+)\\s*(.*?)\\s*$', line)
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
        nested = any(((lo, hi) != (first, last) and first <= lo <= hi <= last for lo, hi in edges))
        if not nested and sum((row['op'].startswith('vfmadd') for row in body)) == fma_count:
            candidates.append((first, last, body))
    if len(candidates) != 1:
        raise ValueError('one innermost backward conditional loop with requested FMA count required; inspect manually')
    first, last, body = candidates[0]
    accesses = []
    for row in body:
        operands = row['args'].split(',')
        for index, operand in enumerate(operands):
            match = re.search('\\b(xmmword|ymmword) ptr \\[rsp(?:[+-][^\\]]*)?\\]', operand)
            if match:
                accesses.append({'offset': hex(row['offset']), 'instruction': row['op'] + ' ' + row['args'], 'bytes': 16 if match[1] == 'xmmword' else 32, 'access': 'store' if index == 0 else 'load'})
    normalized = '\n'.join((row['op'] + ' ' + row['args'] for row in body)).encode()
    return {'function_header': lines[start], 'function_instructions': len(instructions), 'loop_start': hex(first), 'backedge': hex(last), 'loop_instruction_count': len(body), 'fma_instructions': fma_count, 'loop_instructions_sha256': hashlib.sha256(normalized).hexdigest(), 'vector_stack_accesses': accesses, 'static_vector_stack_operand_bytes': sum((row['bytes'] for row in accesses))}

def main():
    cache = Path('/home/runner/work/_temp/q8-apack-receiver-v1')
    repo = Path('/home/runner/work/gen2/gen2')
    root = cache / 'windows-q8-apack-ci-v1'
    read_bytes(Path(__file__))
    parent_receipt = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row-loop-inspection'
    parent_manifest = read(parent_receipt / 'collection.json', {'bytes': 2409, 'sha256': '51dcd2e621769519d5d2d2767125621debfb74f64dc96e845828ffb7f5443d66'})['files']
    read_bytes(cache / 'inspect-q8-row-loops-v1.py', parent_manifest['replay/inspect-q8-row-loops-v1.py'])
    pre_path = cache / 'windows-q8-apack-receivers-pre-execution-v1.json'
    pre = read(pre_path)
    collector = cache / 'collect-windows-q8-apack-v1.py'
    read_bytes(collector, pre['files'][collector.name])
    replay_path = root / 'collection-replay-v1.json'
    q = read(replay_path)
    assert q['source_commit'] == '3fd41caa862c34134d008d18b0b6b830b736c864' and q['run_id'] == 39077373013 and (q['job_id'] == 117527216794)
    assert q['collector'] == bound[str(collector)]
    assert q['diagnostic_completed'] is True and q['commands_verified'] == 95
    assert all((q[name] is True for name in ['all_1056_outputs_bit_exact', 'all_12_half_tables_bit_exact', 'all_36_support_rejections_verified', 'all_12_packing_controls_verified']))
    admission_path = root / 'delivery-admission-v1/qualification.json'
    admission = read(admission_path, q['delivery_admission'])
    members_path = root / 'delivery-admission-v1/artifact-members.json'
    members = read(members_path, admission['artifact_members'])
    functions = {}
    negative_controls = 0
    for role, method, fmas in [('control', 'gemmMx4<2>', 8), ('apack', 'gemmPackA<2>', 8)]:
        evidence = root / 'delivery-admission-v1/delivered/evidence'
        for name in ['kernel-' + role + '.obj', 'disasm-' + role + '.stdout.log']:
            read_bytes(evidence / name, members['evidence/' + name])
        text = read_bytes(evidence / ('disasm-' + role + '.stdout.log')).decode()
        functions[role] = inspect_loop(text, method, fmas)
        for wrong_method, wrong_fmas in [(method, 4 if fmas == 8 else 8), (method + 'missing', fmas)]:
            try:
                inspect_loop(text, wrong_method, wrong_fmas)
            except ValueError:
                negative_controls += 1
            else:
                raise AssertionError('wrong helper/FMA count accepted')
    assert negative_controls == 4
    for name, expected in bound.copy().items():
        read_bytes(Path(name), expected)
    result = {'source_commit': q['source_commit'], 'run_id': q['run_id'], 'job_id': q['job_id'], 'inputs': bound, 'method': 'Unchanged admitted row parser: unique Q8/Q8/F32 function and unique innermost backward conditional loop with8control/8candidate FMAs. Explicit RSP vector operand widths only.', 'functions': functions, 'negative_selection_controls': negative_controls, 'normalized_candidate_instructions_per8outputs': functions['apack']['loop_instruction_count'], 'normalized_candidate_stack_operand_bytes_per8outputs': functions['apack']['static_vector_stack_operand_bytes'], 'new_native_calls': 0, 'production_qualified': False, 'limits': 'Both helpers produce8outputs per2x4iteration. Static counts exclude tile setup/reduction/scheduling,are not measured traffic/CPUtime/cause,and cannot replace numerical/timing/observer gates. All original flags remain unchanged.'}
    with (root / 'loop-inspection-v1.json').open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({role: {'instructions': row['loop_instruction_count'], 'fmas': row['fma_instructions'], 'static_stack_operand_bytes': row['static_vector_stack_operand_bytes']} for role, row in functions.items()}))
if __name__ == '__main__':
    main()
