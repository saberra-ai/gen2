"""Execute only pure receiver functions/conditions; never consume a CI artifact."""
import ast
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import tempfile
import unittest
import sys
import struct

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = Path('/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization')
sys.path.insert(0, str(tools))
from q8_apack_source import control_source
from q8_apack_metrics import admit_packing

raw = (cache / 'collect-windows-q8-apack-v1.py').read_bytes()
tree = ast.parse(raw)
names = {'desc', 'matches_control_source', 'admit_kernel_unit', 'admit_kernel_command', 'admit_packing_case'}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in nodes} == names
scope = {'hashlib': hashlib, 'PureWindowsPath': PureWindowsPath, 'control_source': control_source, 're': re, 'admit_packing': admit_packing}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pure receiver functions>', 'exec'), scope)
assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
              and ast.unparse(node.test) == 'seen == set(compiled) and {control, candidate} <= seen']
assert len(assertions) == 1
membership = compile(ast.Module(body=assertions, type_ignores=[]), '<actual database admission>', 'exec')


class ReceiverControlTests(unittest.TestCase):
    def packing(self, path):
        phase='case-4-0'
        payload=bytes((i+17*j)&255 for i in range(65536) for j in range(32))
        scales=b''.join(struct.pack('<f',struct.unpack('<e',struct.pack('<H',i))[0]) for i in range(65536))
        (path/(phase+'.apack-payload.i8')).write_bytes(payload)
        (path/(phase+'.apack-scales.f32')).write_bytes(scales)
        return phase,payload,scales,{'packing_control':admit_packing(payload,scales)}

    def test_complete_packing_outputs_and_original_lookup_identity_admit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);phase,payload,scales,case=self.packing(path)
            self.assertEqual(scope['admit_packing_case'](case,path,phase,scales),case['packing_control'])

    def test_packing_mutation_missing_output_metadata_and_lookup_mismatch_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);phase,payload,scales,case=self.packing(path)
            target=path/(phase+'.apack-payload.i8')
            target.write_bytes(payload[:-1])
            with self.assertRaises(ValueError):scope['admit_packing_case'](case,path,phase,scales)
            target.write_bytes(payload)
            target.unlink()
            with self.assertRaises(FileNotFoundError):scope['admit_packing_case'](case,path,phase,scales)
            target.write_bytes(payload)
            changed=copy.deepcopy(case);changed['packing_control']['blocks']=65535
            with self.assertRaises(AssertionError):scope['admit_packing_case'](changed,path,phase,scales)
            with self.assertRaises(AssertionError):scope['admit_packing_case'](case,path,phase,scales[:-4])
            (path/(phase+'.apack-scales.f32')).write_bytes(b'\0\0\x80\x3f'+scales[4:])
            with self.assertRaises(ValueError):scope['admit_packing_case'](case,path,phase,scales)

    def test_exact_guard_and_case_arguments_admit(self):
        evidence, repo = PureWindowsPath('D:/trial/evidence'), PureWindowsPath('D:/repo')
        for args in (['--support-control', 'barrier'], ['off', '4', str(evidence / 'case-4-0')]):
            row = {'command': [str(evidence / 'q8_tile_benchmark.exe'), *args], 'cwd': str(repo)}
            scope['admit_kernel_command'](row, evidence, repo, args)

    def test_changed_executable_argument_extra_prefix_and_cwd_reject(self):
        evidence, repo = PureWindowsPath('D:/trial/evidence'), PureWindowsPath('D:/repo')
        args = ['off', '4', str(evidence / 'case-4-0')]
        original = {'command': [str(evidence / 'q8_tile_benchmark.exe'), *args], 'cwd': str(repo)}
        variants = []
        for index, value in ((0, 'D:/wrong.exe'), (1, 'on'), (2, '2'), (3, 'D:/other-output')):
            changed = copy.deepcopy(original); changed['command'][index] = value; variants.append(changed)
        changed = copy.deepcopy(original); changed['command'].append('extra'); variants.append(changed)
        changed = copy.deepcopy(original); changed['command'].pop(); variants.append(changed)
        changed = copy.deepcopy(original); changed['cwd'] = 'D:/other-repo'; variants.append(changed)
        for row in variants:
            with self.subTest(row=row), self.assertRaises(AssertionError):
                scope['admit_kernel_command'](row, evidence, repo, args)

    def unit(self, role):
        name = 'gen2_q8_tile_' + ('control' if role == 'control' else 'candidate') + '_sgemm'
        guard = 'GEN2_Q8_' + ('MX4' if role == 'control' else 'APACK') + '_EXPECTED=1'
        row = {'command': 'clang-cl.exe -D' + guard + ' -Dllamafile_sgemm=' + name + ' /arch:AVX2 -c source.cpp'}
        return {'entry': name, 'command': row['command']}, row

    def test_fixed_role_alias_and_compilation_guard_admit(self):
        for role in ('control', 'apack'):
            unit, row = self.unit(role)
            scope['admit_kernel_unit'](unit, row, role)

    def test_consistent_alias_swap_duplicate_definition_or_wrong_guard_reject(self):
        for role in ('control', 'apack'):
            unit, row = self.unit(role)
            other, _ = self.unit('apack' if role == 'control' else 'control')
            for mode in ('swapped', 'duplicate', 'guard', 'record'):
                value, command = copy.deepcopy(unit), copy.deepcopy(row)
                if mode == 'swapped':
                    command['command'] = command['command'].replace(value['entry'], other['entry']); value['entry'] = other['entry']
                elif mode == 'duplicate': command['command'] += ' /Dllamafile_sgemm=' + value['entry']
                elif mode == 'guard': command['command'] = command['command'].replace('GEN2_Q8_', 'GEN2_Q8_OTHER_')
                else: value['entry'] = other['entry']
                value['command'] = command['command']
                with self.subTest(role=role, mode=mode), self.assertRaises(AssertionError):
                    scope['admit_kernel_unit'](value, command, role)

    def test_missing_control_file_is_false_and_complete_original_control_matches(self):
        original = (tools / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'control.cpp'
            expected_raw = control_source(original)
            expected = {'bytes': len(expected_raw), 'sha256': hashlib.sha256(expected_raw).hexdigest()}
            self.assertFalse(scope['matches_control_source'](path, original, expected))
            path.write_bytes(expected_raw)
            self.assertTrue(scope['matches_control_source'](path, original, expected))
            path.write_bytes(expected_raw + b'changed')
            self.assertFalse(scope['matches_control_source'](path, original, expected))

    def test_both_kernel_units_must_appear_in_unique_database(self):
        control, candidate = PureWindowsPath('D:/control.cpp'), PureWindowsPath('D:/pair.cpp')
        for seen in ({control, candidate}, {control}, {candidate}, set()):
            env = {'seen': seen, 'compiled': {name: {} for name in seen}, 'control': control, 'candidate': candidate}
            if seen == {control, candidate}:
                exec(membership, env)
            else:
                with self.subTest(seen=seen), self.assertRaises(AssertionError): exec(membership, env)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReceiverControlTests))
assert (cache / 'collect-windows-q8-apack-v1.py').read_bytes() == raw
report = {'tests_run': result.testsRun, 'passed': result.wasSuccessful(), 'collector': {
    'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, 'new_compiler_sdk_model_calls': 0,
    'scope': 'Pure receiver function and exact compiler-database condition controls only; actual CI artifact not opened.'}
with (cache / 'q8-apack-receiver-controls-local-v1.json').open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
if not result.wasSuccessful(): raise SystemExit(1)
print(json.dumps(report))
