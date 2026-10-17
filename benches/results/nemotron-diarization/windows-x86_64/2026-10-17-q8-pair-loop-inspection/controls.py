"""Static parsing controls only; no new object/compiler/native execution."""
import importlib.util
import json
from pathlib import Path
import unittest

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
path = cache / 'inspect-q8-pair-loops-v1.py'
spec = importlib.util.spec_from_file_location('inspection', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
before = module.descriptor(path)


class LoopInspectionTests(unittest.TestCase):
    def text(self, fmas=8):
        header = "synthetic (private: void tinyBLAS::tinyBLAS_Q0_AVX<struct block_q8_0,struct block_q8_0,float>::gemmPair2x4(__int64)):"
        rows = [header]
        for index, instruction in enumerate(['vmovups ymmword ptr [rsp+20h],ymm0',
                'vmovaps xmm1,xmmword ptr [rsp+10h]', 'mov rax,qword ptr [rsp+8h]',
                *['vfmadd231ps ymm1,ymm2,ymm3'] * fmas]):
            rows.append(f'  {index:016X}: C5 FC 00           {instruction}')
        rows.append(f'  {fmas+3:016X}: 0F 85 00           jne 0000000000000000')
        return '\n'.join(rows)

    def test_saved_admitted_old_disassembly_reproduces_both_known_static_counts(self):
        root = Path('/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx4')
        manifest = json.loads((root / 'collection.json').read_text())['files']
        source = root / 'evidence/disasm-mx4.stdout.log'
        self.assertEqual(module.descriptor(source), manifest['evidence/disasm-mx4.stdout.log'])
        self.assertEqual(module.inspect_loop(source.read_text(), 'gemm4xN<2>')['static_vector_stack_operand_bytes'], 352)
        self.assertEqual(module.inspect_loop(source.read_text(), 'gemmMx4<2>')['static_vector_stack_operand_bytes'], 192)

    def test_explicit_vector_widths_and_operand_direction_exclude_scalar_stack_operands(self):
        result = module.inspect_loop(self.text(), 'gemmPair2x4')
        self.assertEqual(result['static_vector_stack_operand_bytes'], 48)
        self.assertEqual([(x['bytes'], x['access']) for x in result['vector_stack_accesses']], [(32, 'store'), (16, 'load')])
        self.assertEqual(result['fma_instructions'], 8)

    def test_missing_duplicate_or_wrong_type_function_headers_reject(self):
        original = self.text()
        for text in ('no header', original + '\n' + original,
                     original.replace('block_q8_0,struct block_q8_0', 'block_q4_0,struct block_q8_0')):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError): module.inspect_loop(text, 'gemmPair2x4')

    def test_missing_backward_conditional_or_changed_fma_count_requires_manual_inspection(self):
        for text in (self.text().replace('jne ', 'jmp '), self.text(7), self.text(16)):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError): module.inspect_loop(text, 'gemmPair2x4')

    def test_memory_outside_rsp_is_not_mislabeled_as_stack(self):
        result = module.inspect_loop(self.text().replace('[rsp+20h]', '[rdi+20h]'), 'gemmPair2x4')
        self.assertEqual(result['static_vector_stack_operand_bytes'], 16)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LoopInspectionTests))
assert module.descriptor(path) == before
report = {'tests_run': result.testsRun, 'passed': result.wasSuccessful(), 'inspector': before,
    'new_compiler_native_model_calls': 0, 'scope': 'Static parser controls and earlier admitted disassembly only; current pair artifact unopened.'}
with (cache / 'q8-pair-loop-controls-v1.json').open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
if not result.wasSuccessful(): raise SystemExit(1)
print(json.dumps(report))
