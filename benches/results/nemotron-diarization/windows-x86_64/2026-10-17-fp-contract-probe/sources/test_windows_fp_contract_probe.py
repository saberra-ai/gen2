"""Model-free controls for the scalar observation's inputs and interpretation."""
import json
from pathlib import Path
import unittest

from windows_fp_contract_probe import (COMMON_FLAGS, FUSED, INPUTS, VARIANTS,
                                     compile_command, environment, observation)


class ArithmeticObservationTest(unittest.TestCase):
    def test_compiler_command_changes_only_explicit_contraction_flag(self):
        source, directory = Path('/source/probe.cpp'), Path('/output')
        base = compile_command('/compiler', source, directory, 'clang-default')
        changed = compile_command('/compiler', source, directory, 'clang-no-contract')
        self.assertEqual(base, ['/compiler', *COMMON_FLAGS, str(source),
                                '/Fo' + str(directory / 'probe.obj'), '/Fe' + str(directory / 'probe.exe')])
        self.assertEqual(changed, base[:1 + len(COMMON_FLAGS)] + ['/clang:-ffp-contract=off'] + base[1 + len(COMMON_FLAGS):])
        self.assertEqual(base, compile_command('/compiler', source, directory, 'msvc-default'))
        with self.assertRaises(ValueError):
            compile_command('/compiler', source, directory, 'unknown')

    def test_child_mapping_removes_ambient_flags_without_mutating_parent(self):
        parent = {'Path': 'tools', 'SystemRoot': 'windows', 'Cl': '/fp:fast', '_cl_': '/GL',
                  'gh_token': 'private', 'CXXFLAGS': '-ffast-math', 'NEMO_SPEECH_TIMING': '1'}
        original = parent.copy()
        self.assertEqual(environment(parent), {'Path': 'tools', 'SystemRoot': 'windows'})
        self.assertEqual(parent, original)

    def test_explicit_controls_accept_supported_or_unsupported_prediction(self):
        for variant in VARIANTS:
            for ordinary in (0, FUSED):
                value = {'inputs': INPUTS, 'ordinary': ordinary, 'separate': 0, 'intrinsic': FUSED,
                         'mxcsr': 0x1f80, 'msc_ver': 1944, 'clang_major': 0 if variant == 'msvc-default' else 19,
                         'fma_macro': 0 if variant == 'msvc-default' else 1, 'avx2_macro': 1}
                if variant == 'clang-no-contract' and ordinary == FUSED:
                    with self.assertRaises(ValueError):
                        observation(json.dumps(value), variant)
                else:
                    self.assertEqual(observation(json.dumps(value), variant), value)

    def test_failed_arithmetic_controls_cannot_be_observations(self):
        valid = {'inputs': INPUTS, 'ordinary': FUSED, 'separate': 0, 'intrinsic': FUSED,
                 'mxcsr': 0x1f80, 'msc_ver': 1944, 'clang_major': 19, 'fma_macro': 1, 'avx2_macro': 1}
        for key, value in [('separate', FUSED), ('intrinsic', 0), ('inputs', [1, 2, 3]),
                           ('ordinary', 1), ('ordinary', True), ('mxcsr', -1), ('avx2_macro', 0), ('clang_major', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                observation(json.dumps({**valid, key: value}), 'clang-default')


if __name__ == '__main__':
    unittest.main()
