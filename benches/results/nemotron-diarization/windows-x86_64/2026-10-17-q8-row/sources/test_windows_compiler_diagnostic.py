"""Model-free compiler selection, ABI/ISA provenance and alias controls."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from platform_native import powershell_literal
from windows_compiler_metrics import (ORDER, admit_compiler_paths, build_command, command_compiler,
                                      compiler_environment, compiler_evidence, timing_evidence)
from windows_compiler_diagnostic import Diagnostic


MSVC = r'C:\Program Files\MSVC\cl.exe'
CLANG = r'C:\Program Files\LLVM\clang-cl.exe'
FIELDS = {'CMAKE_GENERATOR': 'Ninja', 'CMAKE_BUILD_TYPE': 'Release', 'BUILD_SHARED_LIBS': 'ON',
    'CMAKE_LINKER': r'C:\Program Files\MSVC\link.exe',
    'GGML_NATIVE': 'OFF', 'GGML_AVX': 'ON', 'GGML_AVX2': 'ON', 'GGML_FMA': 'ON', 'GGML_F16C': 'ON',
    'GGML_AVX512': 'OFF', 'GGML_AVX_VNNI': 'OFF', 'GGML_OPENMP': 'OFF', 'GGML_BACKEND_DL': 'OFF',
    'GGML_METAL': 'OFF', 'GGML_CUDA': 'OFF', 'GGML_VULKAN': 'OFF',
    'NEMO_SPEECH_BUILD_ASR': 'OFF', 'NEMO_SPEECH_BUILD_TTS': 'OFF'}


def fixture(kind):
    compiler = MSVC if kind == 'msvc' else CLANG
    cache = '\n'.join(key + ':STRING=' + value for key, value in FIELDS.items())
    cache += '\nCMAKE_C_COMPILER:FILEPATH=' + compiler + '\nCMAKE_CXX_COMPILER:FILEPATH=' + compiler
    files = {language: '\n'.join('set(CMAKE_' + language + '_' + key + ' "' + value + '")' for key, value in
        [('COMPILER', compiler), ('COMPILER_ID', 'MSVC' if kind == 'msvc' else 'Clang'),
         ('COMPILER_VERSION', '19.44' if kind == 'msvc' else '20.1.2'), ('SIMULATE_ID', 'MSVC')]) for language in ('C', 'CXX')}
    database = [{'file': r'D:\build\ggml-cpu\ggml-cpu.c',
                 'command': '"' + compiler + '" -DGGML_AVX2 -DGGML_FMA -DGGML_F16C /arch:AVX2 /O2 /c source.c'}]
    return cache, database, files, compiler


class CompilerDiagnosticTest(unittest.TestCase):
    def test_command_failure_preserves_actual_cleanup_observation(self):
        reaped = subprocess.CompletedProcess(['fixture-only'], 9)
        reaped.process_tree_reaped = True
        timeout = subprocess.TimeoutExpired(['fixture-only'], 2)
        timeout.process_tree_reaped = True
        for value, expected, error_type in [(reaped, True, subprocess.CalledProcessError),
                (timeout, True, subprocess.TimeoutExpired), (RuntimeError('unverified cleanup'), False, RuntimeError)]:
            with self.subTest(failure=type(value).__name__), tempfile.TemporaryDirectory() as temporary:
                runner = object.__new__(Diagnostic)
                runner.output = runner.repo = Path(temporary)
                runner.kind, runner.commands, runner.environment = 'msvc', [], {}
                runner.frozen = lambda: None
                arguments = {'side_effect': value} if isinstance(value, BaseException) else {'return_value': value}
                with patch('windows_compiler_diagnostic.run_in_job', **arguments), self.assertRaises(error_type):
                    runner.run('fixture-command', ['fixture-only'])
                self.assertEqual(runner.commands[0]['process_tree_reaped'], expected)
                self.assertIn('error', runner.commands[0])
                if isinstance(value, subprocess.CompletedProcess):
                    self.assertEqual(runner.commands[0]['returncode'], 9)

    def test_prediction_requires_both_candidates_and_stable_controls(self):
        def cases(values):
            return [{'compiler': kind, 'result': {'elapsed_micros': value}} for kind, value in zip(ORDER, values)]
        result = timing_evidence(cases([100000000, 85000000, 86000000, 105000000]))
        self.assertTrue(result['compiler_speedup_prediction_supported'])
        self.assertTrue(all(value > 1 for value in result['native_rtfs']))
        self.assertFalse(timing_evidence(cases([100, 80, 80, 140]))['compiler_speedup_prediction_supported'])
        self.assertFalse(timing_evidence(cases([100, 80, 95, 100]))['compiler_speedup_prediction_supported'])
        self.assertTrue(timing_evidence(cases([100, 90, 90, 110]))['compiler_speedup_prediction_supported'])
        for invalid in [cases([100, 80, 80]), cases([100, 80, 0, 100]), cases([100, 80, True, 100])]:
            with self.assertRaises(ValueError):
                timing_evidence(invalid)

    def test_only_existing_compiler_selections_change(self):
        source = "& 'native/build.ps1' -Backend cpu -Profile asr -Jobs 2 -CMakeArgs @(" + ','.join([
            powershell_literal('-DCMAKE_C_COMPILER=' + MSVC), powershell_literal('-DCMAKE_CXX_COMPILER=' + MSVC),
            "'-DGGML_AVX2=ON'", "'-DGGML_OPENMP=OFF'"]) + ')'
        command = ['pwsh', '-NoProfile', '-NonInteractive', '-Command', source]
        result = build_command(command, MSVC, CLANG, 'clang-cl')
        expected = source.replace(' -Backend cpu -Profile', ' -Backend cpu -Compiler clang-cl -Profile').replace(MSVC, CLANG)
        self.assertEqual(result, command[:4] + [expected])
        self.assertEqual(command[4], source)
        self.assertEqual(ORDER, ['msvc', 'clang-cl', 'clang-cl', 'msvc'])
        self.assertIn('-Compiler msvc', build_command(command, MSVC, MSVC, 'msvc')[4])
        for invalid in [source + source, source.replace('CMAKE_C_COMPILER=', 'MISSING='),
                        source.replace(' -Backend cpu ', ' -Backend cuda '), source + ' -Compiler msvc']:
            with self.subTest(command=invalid), self.assertRaises(ValueError):
                build_command(command[:4] + [invalid], MSVC, CLANG, 'clang-cl')
        with self.assertRaises(ValueError):
            build_command(['cmd', '/c', source], MSVC, CLANG, 'clang-cl')

    def test_actual_compiler_abi_and_portable_commands(self):
        for kind in ('msvc', 'clang-cl'):
            args = fixture(kind)
            result = compiler_evidence(*args, kind)
            self.assertEqual(result['languages']['C']['id'], 'MSVC' if kind == 'msvc' else 'Clang')
            self.assertEqual(result['ggml_cpu_entries'], 1)
            cache, database, files, compiler = args
            with self.assertRaises(ValueError):
                compiler_evidence(cache, database, {**files, 'C': files['C'].replace('COMPILER_ID "', 'OTHER_ID "')}, compiler, kind)
            with self.assertRaises(ValueError):
                compiler_evidence(cache, database, files, compiler, 'gcc')
        cache, database, files, compiler = fixture('clang-cl')
        with self.assertRaisesRegex(ValueError, 'ABI'):
            compiler_evidence(cache, database, {key: value.replace('SIMULATE_ID "MSVC"', 'SIMULATE_ID "GNU"') for key, value in files.items()}, compiler, 'clang-cl')

    def test_cache_and_command_changes_fail_closed(self):
        cache, database, files, compiler = fixture('clang-cl')
        for key in ['GGML_NATIVE', 'GGML_AVX512', 'GGML_OPENMP', 'GGML_CUDA']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                compiler_evidence(cache.replace(key + ':STRING=OFF', key + ':STRING=ON'), database, files, compiler, 'clang-cl')
        for addition in [' /arch:AVX512', ' /arch:AVX10.2', ' -march=native', ' -mavx-vnni', ' -mavx512f', ' /clang:-mcpu=native']:
            with self.subTest(flag=addition), self.assertRaises(ValueError):
                compiler_evidence(cache, [{**database[0], 'command': database[0]['command'] + addition}], files, compiler, 'clang-cl')
        for change in [database[0]['command'].replace('-DGGML_FMA ', ''), database[0]['command'].replace(CLANG, MSVC)]:
            with self.assertRaises(ValueError):
                compiler_evidence(cache, [{**database[0], 'command': change}], files, compiler, 'clang-cl')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compiler_evidence(cache + '\nGGML_AVX2:BOOL=ON', database, files, compiler, 'clang-cl')
        with self.assertRaises(ValueError):
            compiler_evidence(cache, [], files, compiler, 'clang-cl')

    def test_short_path_requires_explicit_file_identity_admission(self):
        cache, database, files, compiler = fixture('msvc')
        short = r'C:\PROGRA~1\MSVC\cl.exe'
        changed = [{**database[0], 'command': database[0]['command'].replace(MSVC, short)}]
        with self.assertRaises(ValueError):
            compiler_evidence(cache, changed, files, compiler, 'msvc')
        self.assertEqual(compiler_evidence(cache, changed, files, compiler, 'msvc', [short])['compiler_entries'], 1)
        self.assertEqual(command_compiler('"' + MSVC + '" /O2'), MSVC)
        self.assertEqual(command_compiler(short + ' /O2'), short)

    def test_alias_binding_rejects_equal_bytes_in_different_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, alias, other = root / 'compiler bytes', root / 'alias', root / 'different'
            binary.write_bytes(b'fixture-only binary')
            other.write_bytes(binary.read_bytes())
            os.link(binary, alias)
            result = admit_compiler_paths(binary, [binary, alias])
            self.assertEqual(result[str(binary)], result[str(alias)])
            with self.assertRaises(ValueError):
                admit_compiler_paths(binary, [other])

    def test_compiler_flags_are_child_only_and_case_insensitive(self):
        parent = {'PATH': 'fixture', 'Cl': '/arch:AVX512', '_CL_': '/Od', 'CXXFLAGS': '-march=native', 'CC': 'other'}
        saved = copy.deepcopy(parent)
        self.assertEqual(compiler_environment(parent), {'PATH': 'fixture'})
        self.assertEqual(parent, saved)


if __name__ == '__main__':
    unittest.main()
