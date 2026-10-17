"""Admit the exact saved 97-command experiment and reject confounded candidates."""
import copy
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from build_valid_mel_candidate import descriptor as real_descriptor
from platform_native import powershell_literal
import windows_math_errno_metrics as metrics
from windows_compiler_fp_metrics import build_command as contraction_build_command
from windows_math_errno_sources import derive_runner
from windows_worker_metrics import compare_native as worker_compare


class MathErrnoTest(unittest.TestCase):
    def fixture(self, variant):
        cache, database, compiler = metrics.baseline('clang-cl')
        root = Path(__file__).resolve().parents[2] / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-parity-failure/evidence-clang-cl'
        files = {language: (root / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')}
        flags = ' '.join(metrics.FLAGS[variant])
        for language in ('C', 'CXX'):
            field = 'CMAKE_' + language + '_FLAGS'
            cache = cache.replace(field + ':STRING=', field + ':STRING=' + flags + ' ')
            cache += '\n' + field + '_INIT:STRING=' + flags + '\n'
        database = [{**row, 'command': row['command'].replace(' /DWIN32 ', ' ' + flags + ' /DWIN32 ')} for row in database]
        return cache, database, files, compiler

    def admit(self, variant, cache, database, files, compiler):
        def descriptor(path):
            if str(path) == compiler['compiler']:
                return compiler['compiler_binary']
            if str(path) == compiler['linker']:
                return compiler['linker_binary']
            return real_descriptor(path)
        with patch.object(metrics, 'descriptor', side_effect=descriptor):
            return metrics.compiler_evidence(cache, database, files, compiler['compiler'], 'clang-cl',
                                             compiler['compiler_path_aliases'], variant)

    def test_exact_controller_derivation_and_missing_or_duplicate_sites_reject(self):
        tools = Path(__file__).resolve().parent
        raw = (tools / 'windows_compiler_fp_diagnostic.py').read_bytes()
        self.assertEqual(derive_runner(raw), (tools / 'windows_math_errno_diagnostic.py').read_bytes())
        for invalid in [raw + raw, raw.replace(b'compiler_admission=admission,', b'unknown')]:
            with self.assertRaises(ValueError):
                derive_runner(invalid)

    def test_build_preserves_both_compilers_and_adds_only_candidate_initializer(self):
        compilers = {kind: metrics.baseline(kind)[2] for kind in ('msvc', 'clang-cl')}
        msvc, clang = (compilers[kind]['compiler'] for kind in ('msvc', 'clang-cl'))
        text = "& 'native/build.ps1' -Backend cpu -Profile asr -Jobs 2 -CMakeArgs @(" + ','.join(
            powershell_literal('-DCMAKE_' + language + '_COMPILER=' + msvc) for language in ('C', 'CXX')) + ')'
        command = ['pwsh', '-NoProfile', '-NonInteractive', '-Command', text]
        identities = {row['compiler']: row['compiler_binary'] for row in compilers.values()}
        def descriptor(path):
            return identities[str(path)] if str(path) in identities else real_descriptor(path)
        with patch('windows_compiler_fp_metrics.descriptor', side_effect=descriptor):
            base = contraction_build_command(command, msvc, clang, 'clang-cl')
            self.assertEqual(metrics.build_command(command, msvc, clang, 'clang-cl', 'control'), base)
            actual = metrics.build_command(command, msvc, clang, 'clang-cl', 'no-errno')
            expected = base[4]
            for language in ('C', 'CXX'):
                expected = expected.replace(powershell_literal('-DCMAKE_' + language + '_FLAGS_INIT:STRING=' + metrics.FLAG),
                    powershell_literal('-DCMAKE_' + language + '_FLAGS_INIT:STRING=' + ' '.join(metrics.FLAGS['no-errno'])))
            self.assertEqual(actual, base[:4] + [expected])
            with self.assertRaises(ValueError):
                metrics.build_command(command, msvc, clang, 'msvc', 'control')
            with self.assertRaises(ValueError):
                metrics.build_command(command, msvc, clang, 'clang-cl', 'unknown')

    def test_both_variants_admit_all_97_actual_commands(self):
        for variant in metrics.FLAGS:
            fixture = self.fixture(variant)
            before = copy.deepcopy(fixture[1])
            value = self.admit(variant, *fixture)
            self.assertEqual(value['math_errno_admission']['flags'], metrics.FLAGS[variant])
            self.assertEqual(value['math_errno_admission']['compiler_commands_matching_original'], 97)
            self.assertEqual(fixture[1], before)

    def test_missing_duplicate_or_additional_math_and_optimization_flags_reject(self):
        for variant in metrics.FLAGS:
            cache, database, files, compiler = self.fixture(variant)
            first = database[0]['command']
            changes = [first + ' /fp:fast', first + ' /Od', first + ' ' + metrics.ERRNO_FLAG,
                       first.replace(metrics.FLAG + ' ', '')]
            if variant == 'no-errno':
                changes.append(first.replace(metrics.ERRNO_FLAG + ' ', ''))
            for text in changes:
                changed = [{**database[0], 'command': text}, *database[1:]]
                with self.subTest(variant=variant, text=text), self.assertRaises(ValueError):
                    self.admit(variant, cache, changed, files, compiler)

    def test_cache_initializer_release_and_platform_changes_reject(self):
        for variant in metrics.FLAGS:
            cache, database, files, compiler = self.fixture(variant)
            changed = [cache.replace('_FLAGS_RELEASE:STRING=/O2', '_FLAGS_RELEASE:STRING=/Od'),
                cache.replace('/DWIN32 /D_WINDOWS', '/D_WINDOWS'),
                cache.replace('_FLAGS_INIT:STRING=', '_FLAGS_INIT:STRING=/fp:fast ')]
            for text in changed:
                with self.assertRaises(ValueError):
                    self.admit(variant, text, database, files, compiler)

    def test_actual_saved_driver_and_only_math_errno_state_change(self):
        root = Path(__file__).resolve().parents[2] / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-fp-contract-probe'
        control = (root / 'clang-no-contract-driver.stderr.log').read_text()
        self.assertEqual(metrics.admit_math_driver(control, 'control')['effective_optimization'], '-O3')
        candidate = control.replace('"-fmath-errno" ', '')
        self.assertFalse(metrics.admit_math_driver(candidate, 'no-errno')['math_errno'])
        for text, variant in [(candidate, 'control'), (control, 'no-errno'), (control + control, 'control'),
                (control.replace('"-O3"', '"-O2"'), 'control'),
                (control.replace('"-O3"', '"-O3" "-O2"'), 'control'),
                (candidate.replace('"-O3"', '"-O3" "-menable-unsafe-fp-math"'), 'no-errno'),
                (candidate.replace('"haswell"', '"skylake-avx512"'), 'no-errno'),
                (candidate.replace('"haswell"', '"haswell" "-target-cpu" "skylake-avx512"'), 'no-errno'),
                ('"-cc1" "-target-cpu"', 'no-errno'),
                (candidate.replace('"-ffp-contract=off"', '"-ffp-contract=on"'), 'no-errno')]:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                metrics.admit_math_driver(text, variant)

    def test_full_vector_comparison_requires_byte_identity_beyond_original_tolerance(self):
        path = Path(__file__).resolve().parents[2] / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'
        reference, raw = json.loads(path.read_text()), Path(str(path) + '.f32').read_bytes()
        self.assertTrue(metrics.compare_native(reference, reference, raw, raw, 4, 4)['probabilities_bit_exact'])
        bits, = struct.unpack_from('<I', raw)
        changed = struct.pack('<I', bits + 1) + raw[4:]
        self.assertFalse(worker_compare(reference, reference, raw, changed, 4, 4)['probabilities_bit_exact'])
        with self.assertRaisesRegex(ValueError, 'byte-exact'):
            metrics.compare_native(reference, reference, raw, changed, 4, 4)

    def test_integer_timing_boundaries_and_strict_real_time(self):
        def cases(times):
            return [{'compiler': name, 'result': {'elapsed_micros': value}} for name, value in zip(metrics.ORDER, times)]
        exact = metrics.timing_evidence(cases([60000000, 57000000, 57000000, 66000000]))
        self.assertTrue(exact['speedup_prediction_supported'])
        self.assertTrue(exact['both_candidates_strictly_real_time'])
        self.assertFalse(metrics.timing_evidence(cases([60000000,57000001,57000000,60000000]))['speedup_prediction_supported'])
        self.assertFalse(metrics.timing_evidence(cases([60000000,57000000,57000000,66000001]))['control_drift_at_most_10_percent'])
        self.assertFalse(metrics.timing_evidence(cases([70000000,60000000,59999999,70000000]))['both_candidates_strictly_real_time'])
        for times in [[True,1,1,1],[1.0,1,1,1],[0,1,1,1],[-1,1,1,1]]:
            with self.assertRaises(ValueError):
                metrics.timing_evidence(cases(times))
        with self.assertRaises(ValueError):
            metrics.timing_evidence(list(reversed(cases([1,1,1,1])))[1:])


if __name__ == '__main__':
    unittest.main()
