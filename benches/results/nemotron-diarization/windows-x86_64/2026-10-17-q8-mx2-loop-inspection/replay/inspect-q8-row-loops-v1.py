"""Inspect admitted failed row disassembly; static widths are not traffic."""
import hashlib
import json
from pathlib import Path
import re

bound = {}

def read_bytes(path, expected=None):
    raw = path.read_bytes()
    value = {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    key = str(path)
    if key in bound: assert value == bound[key]
    if expected is not None: assert value == expected
    bound.setdefault(key,value)
    return raw

def read(path, expected=None): return json.loads(read_bytes(path,expected))

def inspect_loop(text, method, fma_count):
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
        if not nested and sum(row['op'].startswith('vfmadd') for row in body) == fma_count:
            candidates.append((first, last, body))
    if len(candidates) != 1:
        raise ValueError('one innermost backward conditional loop with requested FMA count required; inspect manually')
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
        'backedge': hex(last), 'loop_instruction_count': len(body), 'fma_instructions': fma_count,
        'loop_instructions_sha256': hashlib.sha256(normalized).hexdigest(), 'vector_stack_accesses': accesses,
        'static_vector_stack_operand_bytes': sum(row['bytes'] for row in accesses)}


def main():
    cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
    repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
    root = cache / 'windows-q8-row-ci-v1'
    receipt = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row'
    read_bytes(Path(__file__))
    manifest = read(receipt/'collection.json', {'bytes':213899,'sha256':'213cd26a9737f0c7ac55a44f76114ed6a4b910a143222d0772e65a2c42b31a2a'})['files']
    replay_path = root/'collection-replay-v1.json'
    q = read(replay_path,manifest['receiver/collection-replay-v1.json'])
    assert q['source_commit']=='2dbf3a008bde6ad405dccd33bb7f015da1901b46' and q['run_id']==39070438530
    assert q['diagnostic_completed'] is True and q['all_1056_outputs_bit_exact'] is True and q['commands_verified']==95
    admission_path = root/'delivery-admission-v1/qualification.json'
    admission = read(admission_path,manifest['receiver/delivery-qualification.json'])
    assert q['delivery_admission']==bound[str(admission_path)]
    members_path = root/'delivery-admission-v1/artifact-members.json'
    members = read(members_path,manifest['receiver/delivery-artifact-members.json'])
    assert admission['artifact_members']==bound[str(members_path)]
    parent = cache/'inspect-q8-pair-loops-v1.py'
    read_bytes(parent,{'bytes':5650,'sha256':'54fe10286fe4fc683484fda772049089c3eeb2bd548c82a754f884a38c9ebe86'})
    functions={};negative_controls=0
    for role,method,fmas in [('control','gemmMx4<2>',8),('row1','gemmMx4<1>',4)]:
        evidence=root/'delivery-admission-v1/delivered/evidence'
        for name in ['kernel-'+role+'.obj','disasm-'+role+'.stdout.log']:
            read_bytes(evidence/name,members['evidence/'+name])
        text=read_bytes(evidence/('disasm-'+role+'.stdout.log')).decode()
        functions[role]=inspect_loop(text,method,fmas)
        for wrong_method,wrong_fmas in [(method,4 if fmas==8 else 8),(method+'missing',fmas)]:
            try: inspect_loop(text,wrong_method,wrong_fmas)
            except ValueError: negative_controls+=1
            else: raise AssertionError('wrong function/FMA-count control accepted')
    assert negative_controls==4
    for name,expected in bound.copy().items(): read_bytes(Path(name),expected)
    result={'source_commit':q['source_commit'],'run_id':q['run_id'],'inputs':bound,
        'method':'Same original unique Q8/Q8/F32 signature and unique innermost backward conditional loop; only expected FMA count parameterized8or4. Explicit RSP vector operand widths only.',
        'functions':functions,'negative_selection_controls':negative_controls,'new_native_calls':0,'production_qualified':False,
        'limits':'Different output counts per tile: control2x4 eight accumulators; row1x4 four. Normalize static counts by two for the same eight outputs when comparing. Static widths/instructions are not measured traffic/CPUtime/cause; original failed timing and observer flags remain false.'}
    with (root/'loop-inspection-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({role:{'instructions':value['loop_instruction_count'],'fmas':value['fma_instructions'],'static_vector_stack_operand_bytes':value['static_vector_stack_operand_bytes']}for role,value in functions.items()}))

if __name__=='__main__':main()
