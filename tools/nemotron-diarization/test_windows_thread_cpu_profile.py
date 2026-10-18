"""Actual saved compiler/source baselines plus diagnostic corruption controls."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_valid_mel_candidate import descriptor as real_descriptor
import windows_thread_cpu_profile_metrics as metrics
from windows_thread_cpu_profile_sources import (CPU_SOURCE, PATCH_TOOL, PROFILE_PATCH,
    derive_compiler, derive_verifier, derive_worker, install_profile, verify_profile_series, profile_source_hashes)


TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
RESULT = REPO / metrics.REFERENCE


class WindowsMatrixBatchProfileTest(unittest.TestCase):
    def test_derived_build_code_is_exact(self):
        for target, origin, derive, site in [
                ('windows_thread_cpu_profile_worker.py', 'windows_cpu_profile_worker.py', derive_worker, b'from windows_cpu_profile_metrics import '),
                ('windows_thread_cpu_profile_verify.py', 'windows_cpu_profile_verify.py', derive_verifier, b'from windows_cpu_profile_sources import '),
                ('windows_thread_cpu_profile_compiler.py', 'windows_cpu_profile_compiler.py', derive_compiler, b'from windows_cpu_profile_metrics import (')]:
            original = (TOOLS / origin).read_bytes()
            self.assertEqual(derive(original), (TOOLS / target).read_bytes())
            with self.assertRaises(ValueError):
                derive(original.replace(site, b'unknown source site ', 1))

    def fixture(self):
        root = RESULT / 'evidence-clang-cl'
        cache = (root / 'CMakeCache.txt').read_text().replace(
            'gen2-diagnostic-cpu-threads-4', metrics.metadata(4))
        database = json.loads((root / 'compile_commands.json').read_text())
        database = [{**row, 'command': row['command'].replace('gen2-diagnostic-cpu-threads-4', metrics.metadata(4))}
                    for row in database]
        files = {language: (root / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')}
        compiler = json.loads((root / 'compiler-admission.json').read_text())
        return cache, database, files, compiler

    def admit(self, cache, database, files, compiler):
        def descriptor(path):
            if str(path) == compiler['compiler']:
                return compiler['compiler_binary']
            if str(path) == compiler['linker']:
                return compiler['linker_binary']
            return real_descriptor(path)
        with patch.object(metrics, 'descriptor', side_effect=descriptor):
            return metrics.compiler_evidence(cache, database, files, compiler['compiler'], 'clang-cl',
                                             compiler['compiler_path_aliases'])

    def test_all_97_actual_compiler_commands_admit_only_exact_version_delta(self):
        result = self.admit(*self.fixture())
        self.assertEqual(result['matrix_profile_admission']['version_tokens_changed'], 97)

    def test_extra_flags_wrong_version_or_missing_initializer_reject(self):
        cache, database, files, compiler = self.fixture()
        for text in [database[0]['command'] + ' /O1',
                     database[0]['command'].replace(metrics.metadata(4), 'unknown-version'),
                     database[0]['command'].replace(metrics.FLAG, '')]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.admit(cache, [{**database[0], 'command': text}, *database[1:]], files, compiler)
        with self.assertRaises(ValueError):
            self.admit(cache.replace('_FLAGS_INIT:STRING=' + metrics.FLAG, '_FLAGS_INIT:STRING='), database, files, compiler)

    def test_source_audit_changes_only_the_previously_compiled_cpu_patch(self):
        from ggml_cpu_profile_source import PATH
        audit = copy.deepcopy(metrics.canonical_audit())
        raw = (TOOLS / PATCH_TOOL).read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);target = root / PATH;target.parent.mkdir(parents=True)
            target.write_bytes((TOOLS / 'fixtures/matrix-paths-ggml-cpu.c').read_bytes())
            subprocess.run(['git','apply','--check',TOOLS / PATCH_TOOL],cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            subprocess.run(['git','apply',TOOLS / PATCH_TOOL],cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(real_descriptor(target),CPU_SOURCE)
        audit['files'][PATH] = CPU_SOURCE
        audit['patch_series'].append({'name': PROFILE_PATCH, **real_descriptor(TOOLS / PATCH_TOOL)})
        self.assertEqual(metrics.admit_profile_audit(audit)['changed_files'], [PATH])
        name = next(name for name in audit['files'] if name != PATH)
        audit['files'][name] = {'bytes': 0, 'sha256': '0' * 64}
        with self.assertRaises(ValueError):
            metrics.admit_profile_audit(audit)

    def test_distinct_native_identity_retains_complete_original_comparison(self):
        root = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4'
        path = root / 'case-0-cap-4.json'
        reference = json.loads(path.read_text())
        raw = Path(str(path) + '.f32').read_bytes()
        actual = {**reference, 'runtime_version': metrics.version(4)}
        self.assertTrue(metrics.compare_native(reference, actual, raw, raw)['probabilities_bit_exact'])
        with self.assertRaises(ValueError):
            metrics.compare_native(reference, reference, raw, raw)
        with self.assertRaises(ValueError):
            metrics.metadata(2)

    def test_integer_observer_decisions_keep_original_release_scope(self):
        def cases(times):
            return [{'enabled': enabled, 'result': {'elapsed_micros': elapsed}}
                    for enabled, elapsed in zip(metrics.ORDER, times)]
        self.assertTrue(metrics.timing_evidence(cases([100, 110, 110, 110]))['bounded_observer_interpretation_supported'])
        self.assertFalse(metrics.timing_evidence(cases([100, 111, 110, 110]))['bounded_observer_interpretation_supported'])
        self.assertFalse(metrics.timing_evidence(cases([100, 110, 110, 111]))['bounded_observer_interpretation_supported'])
        invalid = cases([100, 110, 110, 110]); invalid[0]['enabled'] = 0
        with self.assertRaises(ValueError):
            metrics.timing_evidence(invalid)

    def test_exact_series_extension_rejects_corruption_or_reinstallation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.mkdir()
            output = Path(folder) / 'evidence'; output.mkdir()
            (source / 'patches').mkdir()
            names = [row['name'] for row in metrics.canonical_audit()['patch_series']]
            for name in names:
                (source / 'patches' / name).write_text('original ' + name + '\n')
            (source / 'patches/series').write_text('\n'.join(names) + '\n')
            for args in [['init', '-q'], ['add', '.'], ['-c', 'user.name=control', '-c', 'user.email=control@example.invalid', 'commit', '-q', '-m', 'fixture']]:
                subprocess.run(['git', *args], cwd=source, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            install_profile(source, output)
            verify_profile_series(source)
            with self.assertRaises(ValueError):
                install_profile(source, output)
            (source / 'patches' / names[0]).write_text('damaged\n')
            with self.assertRaises(ValueError):
                verify_profile_series(source)

    def test_profile_signature_keeps_matrix_metadata_but_not_worker_local_timings(self):
        from ggml_matrix_profile import summarize
        from test_ggml_matrix_profile import fixture
        profile=summarize('\n'.join(fixture()),True,expected_graphs=1,expected_threads=4,expected_matrices=1)
        expected=metrics.profile_signature(profile)
        profile['graphs'][0]['workers'][0]['matrices'][0]['kernel_us']+=1
        self.assertEqual(metrics.profile_signature(profile),expected)
        profile['graphs'][0]['workers'][0]['matrices'][0]['a'][1]+=1
        self.assertNotEqual(metrics.profile_signature(profile),expected)

    def test_completed_control_rechecks_binary_sources_libraries_and_outputs(self):
        from windows_thread_cpu_profile_diagnostic import verify_control_evidence
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ('control.exe','control.cpp','ggml.lib','output.f32','trace.log')]
            for path in paths:
                path.write_bytes(path.name.encode())
            binary, source, library, output, log = paths
            control = {'binary':real_descriptor(binary),
                'inputs':{str(path):real_descriptor(path) for path in (source,library)},
                'outputs_and_logs':{str(path):real_descriptor(path) for path in (output,log)}}
            verify_control_evidence(binary,control)
            for path in paths:
                original = path.read_bytes()
                path.write_bytes(original+b'changed')
                with self.subTest(path=path.name),self.assertRaises(ValueError):
                    verify_control_evidence(binary,control)
                path.write_bytes(original)
            verify_control_evidence(binary,control)

    def test_hash_table_changes_only_the_exact_extended_series(self):
        parent = json.loads((TOOLS / 'windows-compiler-fp.protocol.json').read_text())['variants']['4']['source_sha256']
        changed = profile_source_hashes(parent)
        self.assertEqual({name for name in changed if changed[name] != parent[name]}, {'patches/series'})
        self.assertEqual(profile_source_hashes(parent), changed)
        with self.assertRaises(ValueError):
            profile_source_hashes({**parent, 'patches/series': '0' * 64})


if __name__ == '__main__':
    unittest.main()
