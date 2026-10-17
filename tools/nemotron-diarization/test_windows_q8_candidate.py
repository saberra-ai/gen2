"""Production-candidate source/compiler semantics and retained graph input controls."""
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from build_valid_mel_candidate import descriptor as real_descriptor
from pooled_runtime import BOUNDED_PROFILE, load_profile
import windows_q8_candidate_metrics as metrics
from windows_q8_candidate_sources import (admit_recipe, candidate_source_hashes, derive_verifier,
    expected_recipe, load_recipe, RECIPE_ID)
from windows_q8_candidate_controls import LIBRARY_NAMES, retain_import_libraries, model_free_controls
from windows_q8_model_controls import verify_bindings
from windows_q8_model_sources import candidate_source, PATCH_TOOL, Q8_PATCH, PATH
from windows_q8_model_sources import install_q8_patch
from windows_q8_candidate_verify import verify_completed_source
import test_platform_bounded_windows as bounded_fixtures
from windows_compat import rewrite_materializer

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
FP = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp/evidence-clang-cl'


class Q8CandidateTest(unittest.TestCase):
    def test_composed_candidate_source_admits_only_bounded_inference_and_appended_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, cmake, original_cmake, patch_file, payload, lock = bounded_fixtures.BoundedWindowsTests().source_fixture(root)
            (source / 'patches').mkdir()
            (source / 'patches/series').write_bytes((TOOLS / 'fixtures/q8-model-original-series.txt').read_bytes())
            for item in metrics.bounded_audit()['patch_series']:
                (source / 'patches' / item['name']).write_text('controlled ' + item['name'] + '\n')
            bounded_fixtures.git(source, 'add', 'patches')
            bounded_fixtures.git(source, '-c', 'user.name=QualificationFixture', '-c',
                'user.email=fixture@example.invalid', 'commit', '-qm', 'controlled original patch series')
            lock['runtime']['revision'] = bounded_fixtures.git(source, 'rev-parse', 'HEAD').decode().strip()
            lock['runtime']['source_sha256']['patches/series'] = real_descriptor(source / 'patches/series')['sha256']
            output = root / 'evidence'; output.mkdir()
            install_q8_patch(source, output, 'mx4')
            with patch('windows_q8_candidate_verify.compatibility_patch_bytes', return_value=payload), patch(
                    'windows_q8_candidate_verify.ORIGINAL_SHA256', hashlib.sha256(original_cmake).hexdigest()):
                verify_completed_source(source, None, lock, patch_file, True)
                build = source / 'build/gen2-cpu-diar'; build.mkdir(parents=True)
                cmake.write_bytes(rewrite_materializer(original_cmake))
                verify_completed_source(source, build, lock, patch_file, True)
                for relative in ['patches/' + Q8_PATCH, 'src/asr/diar/diarizer.cpp', 'patches/series']:
                    target = source / relative; raw = target.read_bytes(); target.write_bytes(raw + b'unbound\n')
                    with self.subTest(relative=relative), self.assertRaises(ValueError):
                        verify_completed_source(source, build, lock, patch_file, True)
                    target.write_bytes(raw)
                extra = source / 'unrecorded.h'; extra.write_text('unrecorded compiler input\n')
                with self.assertRaises(ValueError): verify_completed_source(source, build, lock, patch_file, True)
                extra.unlink()
                verify_completed_source(source, build, lock, patch_file, True)

    def test_recipe_preserves_semantic_tag_and_pins_distinct_build_identity(self):
        _, value = load_recipe()
        self.assertEqual(value, expected_recipe())
        self.assertEqual(value['recipe_id'], RECIPE_ID)
        self.assertEqual(value['runtime_profile'], BOUNDED_PROFILE)
        self.assertEqual(value['runtime_reported_version'], metrics.VERSION)
        for field, replacement in [('runtime_profile', 'unknown'), ('recipe_id', 'unversioned'),
                ('source_sha256_after', {}), ('compiler', {}), ('platform', {'os': 'Linux'})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                admit_recipe({**value, field: replacement})
        with self.assertRaises(ValueError): admit_recipe({**value, 'extra_policy': True})

    def test_source_verifier_is_exact_derived_bounded_policy(self):
        raw = (TOOLS / 'platform_bounded_sources.py').read_bytes()
        derived = derive_verifier(raw)
        self.assertEqual(derived, (TOOLS / 'windows_q8_candidate_verify.py').read_bytes())
        with self.assertRaises(ValueError):
            derive_verifier(raw.replace(b'    expected = INFERENCE_FILES | ', b'unknown'))
        _, lock = load_profile(BOUNDED_PROFILE)
        before = lock['runtime']['source_sha256']
        after = candidate_source_hashes(before)
        self.assertEqual({name for name in before if before[name] != after[name]}, {'patches/series'})
        # These native inference identities include no diagnostic cap/printf delta.
        self.assertEqual(after['src/asr/diar/diarizer.cpp'], before['src/asr/diar/diarizer.cpp'])
        self.assertEqual(after['src/runtime/ggml/backend.cpp'], before['src/runtime/ggml/backend.cpp'])

    def compiler_fixture(self):
        old = 'gen2-diagnostic-cpu-threads-4'
        cache = (FP / 'CMakeCache.txt').read_text().replace(old, BOUNDED_PROFILE)
        database = json.loads((FP / 'compile_commands.json').read_text())
        database = [{**row, 'command': row['command'].replace(old, BOUNDED_PROFILE)} for row in database]
        files = {name: (FP / ('compiler-' + name + '.cmake')).read_text() for name in ('C', 'CXX')}
        compiler = json.loads((FP / 'compiler-admission.json').read_text())
        return cache, database, files, compiler

    def admit_compiler(self, cache, database, files, compiler):
        def descriptor(path):
            if str(path) == compiler['compiler']: return compiler['compiler_binary']
            if str(path) == compiler['linker']: return compiler['linker_binary']
            return real_descriptor(path)
        with patch.object(metrics, 'descriptor', side_effect=descriptor):
            return metrics.compiler_evidence(cache, database, files, compiler['compiler'], 'clang-cl', compiler['compiler_path_aliases'])

    def test_all_97_actual_compiler_commands_allow_only_declared_deltas(self):
        fixture = self.compiler_fixture()
        before = copy.deepcopy(fixture)
        result = self.admit_compiler(*fixture)
        self.assertEqual(result['candidate_compiler_admission']['version_tokens_changed'], 97)
        self.assertEqual(fixture, before)

    def test_unknown_flags_isa_semantics_and_initializer_changes_reject(self):
        cache, database, files, compiler = self.compiler_fixture()
        for command in [database[0]['command'] + ' /fp:fast', database[0]['command'] + ' /arch:AVX512',
                database[0]['command'].replace(metrics.FLAG, ''),
                database[0]['command'].replace(BOUNDED_PROFILE, 'unqualified')]:
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.admit_compiler(cache, [{**database[0], 'command': command}, *database[1:]], files, compiler)
        for text in [cache.replace('_FLAGS_INIT:STRING=' + metrics.FLAG, '_FLAGS_INIT:STRING='),
                cache.replace('_FLAGS_RELEASE:STRING=/O2', '_FLAGS_RELEASE:STRING=/Od'),
                cache.replace('GGML_NATIVE:BOOL=OFF', 'GGML_NATIVE:BOOL=ON')]:
            with self.assertRaises(ValueError): self.admit_compiler(text, database, files, compiler)

    def candidate_audit(self):
        audit = copy.deepcopy(metrics.bounded_audit())
        raw = candidate_source(); identity = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        audit['files'][PATH] = audit['actual_compiler_sources'][PATH] = identity
        for name in audit['all_compiler_sources']:
            if PureWindowsPath(name).as_posix().endswith('/' + PATH): audit['all_compiler_sources'][name] = identity
        audit['patch_series'].append({'name': Q8_PATCH, **real_descriptor(TOOLS / PATCH_TOOL)})
        return audit

    def test_source_audit_requires_original_bounded_native_source_and_exact_sgemm(self):
        audit = self.candidate_audit()
        result = metrics.admit_candidate_audit(audit)
        self.assertFalse(result['diagnostic_worker_telemetry'])
        self.assertEqual(result['compiler_inputs'], 97)
        for field in ('files', 'actual_compiler_sources', 'all_compiler_sources'):
            changed = copy.deepcopy(audit)
            changed[field][next(iter(changed[field]))] = {'bytes': 0, 'sha256': '0' * 64}
            with self.assertRaises(ValueError): metrics.admit_candidate_audit(changed)
        with self.assertRaises(ValueError): metrics.admit_candidate_audit(metrics.bounded_audit())

    def test_actual_bounded_pair_replays_exact_output_but_keeps_original_rtf_failure(self):
        root = REPO / metrics.BOUNDED_RESULT
        public, native = [json.loads((root / name).read_text()) for name in ('public.json', 'native-ami.json')]
        raw_native, raw_public = [(root / name).read_bytes() for name in ('native-ami.json.f32', 'public.f32')]
        result = metrics.compare_candidate_pair(public, native, raw_native, raw_public)
        self.assertTrue(result['probabilities_bit_exact'])
        self.assertFalse(metrics.both_pairs_realtime([result, result]))
        with self.assertRaises(ValueError): metrics.compare_candidate_pair(public, native, raw_native, raw_public[:-4])
        with self.assertRaises(ValueError): metrics.compare_candidate_pair(public, {**native, 'runtime_version': 'unqualified'}, raw_native, raw_public)

    def test_all_four_rtfs_are_strict_and_no_missing_pair_is_accepted(self):
        good = {'passed': True, 'probabilities_bit_exact': True, 'native_rtf': .99, 'gen2_rtf': .99}
        self.assertTrue(metrics.both_pairs_realtime([good, good]))
        for field in ('native_rtf', 'gen2_rtf'):
            for value in (1, 1.01, 0, True, float('nan')):
                with self.subTest(field=field, value=value):
                    self.assertFalse(metrics.both_pairs_realtime([good, {**good, field: value}]))
        for pairs in ([], [good], [good, good, good]):
            with self.assertRaises(ValueError): metrics.both_pairs_realtime(pairs)
        self.assertFalse(metrics.both_pairs_realtime([good, {**good, 'probabilities_bit_exact': False}]))

    def library_fixture(self, root):
        build, output = root / 'build', root / 'output'
        build.mkdir(); output.mkdir()
        for name in LIBRARY_NAMES:
            path = build / 'ggml/src' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('controlled ' + name)
        return SimpleNamespace(build_directory=build, output=output)

    def test_three_graph_libraries_are_copied_and_both_sides_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.library_fixture(Path(folder))
            records, bindings = retain_import_libraries(runner)
            self.assertEqual(len(records), 3); self.assertEqual(len(bindings), 6)
            verify_bindings(bindings)
            for source, record in records.items():
                for path in (Path(source), runner.output / record['retained_path']):
                    raw = path.read_bytes(); path.write_bytes(raw + b'changed')
                    with self.assertRaises(ValueError): verify_bindings(bindings)
                    path.write_bytes(raw)
            with self.assertRaises(ValueError): retain_import_libraries(runner)

    def test_missing_duplicate_empty_or_symlink_graph_libraries_reject(self):
        for mode in ('missing', 'duplicate', 'empty', 'symlink'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                runner = self.library_fixture(Path(folder))
                name = sorted(LIBRARY_NAMES)[0]; source = runner.build_directory / 'ggml/src' / name
                if mode == 'missing': source.unlink()
                elif mode == 'duplicate': (runner.build_directory / name).write_bytes(source.read_bytes())
                elif mode == 'empty': source.write_bytes(b'')
                else:
                    outside = Path(folder) / 'outside.lib'; outside.write_bytes(source.read_bytes()); source.unlink()
                    try: source.symlink_to(outside)
                    except OSError: self.skipTest('symlink creation unavailable')
                with self.assertRaises(ValueError): retain_import_libraries(runner)

    def test_graph_controls_run_only_after_retention_and_recheck_retained_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.library_fixture(Path(folder)); calls = []
            def parent(actual_runner, unchanged):
                self.assertIs(actual_runner, runner)
                self.assertEqual(len(list((runner.output / 'graph-inputs').glob('*.lib'))), 3)
                unchanged()
                original = {str(path.resolve()): real_descriptor(path) for path in runner.build_directory.rglob('*.lib')}
                return {'bindings': original}, {'controlled': True}
            with patch('windows_q8_candidate_controls.parent_controls', side_effect=parent):
                controls, _ = model_free_controls(runner, lambda: calls.append('unchanged'))
            self.assertTrue(controls['graph_input_payloads_fully_retained']); self.assertTrue(calls)
            self.assertEqual(len(controls['retained_import_libraries']), 3)
            verify_bindings(controls['bindings'])


if __name__ == '__main__': unittest.main()
