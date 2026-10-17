"""Saved full compiler/source admission, exact patch replay and failure controls."""
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from build_valid_mel_candidate import descriptor as real_descriptor
import windows_q8_model_metrics as metrics
from windows_q8_model_sources import (GUARD, PATCH_TOOL, Q8_PATCH, candidate_source, q8_patch,
    derive_worker, derive_verifier, derive_compiler, install_q8_patch, verify_q8_series, source_hashes)
from q8_mx4_source import transform
from windows_worker_metrics import compare_native as worker_compare
from windows_q8_model_controls import compiler_objects, verify_bindings
from freeze_windows_q8_model import consumed_fp_files, FP_RESULT, FP_PAYLOADS

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
RESULT = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp/evidence-clang-cl'


class Q8ModelTest(unittest.TestCase):
    def test_every_consumed_full_sdk_receipt_file_is_frozen(self):
        names = consumed_fp_files(REPO)
        self.assertIn(FP_RESULT + '/collection.json', names)
        for name in FP_PAYLOADS:
            self.assertIn(FP_RESULT + '/evidence-clang-cl/' + name, names)

    def test_all_97_actual_object_bindings_detect_mutation_missing_and_duplicate_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, build, out = root / 'source', root / 'build', root / 'evidence'
            source.mkdir(); build.mkdir(); out.mkdir()
            database = []
            for index in range(97):
                path = source / (str(index) + '.cpp'); path.write_text('source ' + str(index))
                obj = build / (str(index) + '.obj'); obj.write_text('object ' + str(index))
                database.append({'directory': str(build), 'file': str(path), 'output': obj.name,
                                 'command': 'unit-test compiler control'})
            runner = SimpleNamespace(build_directory=build, output=out)
            units, bindings = compiler_objects(runner, database)
            self.assertEqual(len(units), 97)
            self.assertEqual(len(bindings), 291)
            verify_bindings(bindings)
            for path in [build / '96.obj', source / '0.cpp', out / units[31]['retained_object']]:
                original = path.read_bytes(); path.write_bytes(original + b'drift')
                with self.assertRaises(ValueError): verify_bindings(bindings)
                path.write_bytes(original)
            (build / '96.obj').unlink()
            with self.assertRaises(FileNotFoundError): verify_bindings(bindings)
            (build / '96.obj').write_text('object 96')
            for name, rows in [('missing', database[:-1]),
                    ('duplicate', [*database[:-1], {**database[-1], 'output': database[0]['output']}]),
                    ('outside', [{**database[0], 'output': str(source / '0.cpp')}, *database[1:]])]:
                destination = root / name; destination.mkdir()
                with self.subTest(name=name), self.assertRaises(ValueError):
                    compiler_objects(SimpleNamespace(build_directory=build, output=destination), rows)

    def test_exact_worker_verifier_and_compiler_derivations(self):
        for target, origin, derive, site in [
                ('windows_q8_model_worker.py', 'windows_worker_diagnostic_v2.py', derive_worker, b'from windows_worker_metrics import '),
                ('windows_q8_model_verify.py', 'windows_worker_sources_v2.py', derive_verifier, b'from windows_compat import '),
                ('windows_q8_model_compiler.py', 'windows_compiler_fp_diagnostic.py', derive_compiler, b'from windows_compiler_fp_metrics import (')]:
            raw = (TOOLS / origin).read_bytes()
            self.assertEqual(derive(raw), (TOOLS / target).read_bytes())
            with self.assertRaises(ValueError):
                derive(raw.replace(site, b'unknown site ', 1))

    def test_candidate_body_equals_existing_standalone_candidate(self):
        fixture = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        self.assertEqual(candidate_source(), GUARD + transform(fixture))
        self.assertEqual(q8_patch(), (TOOLS / PATCH_TOOL).read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / metrics.PATH
            path.parent.mkdir(parents=True)
            path.write_bytes(fixture)
            # Actual Git patch application checks context and the entire file.
            subprocess.run(['git', 'apply', '--check', str(TOOLS / PATCH_TOOL)], cwd=root, check=True)
            subprocess.run(['git', 'apply', str(TOOLS / PATCH_TOOL)], cwd=root, check=True)
            self.assertEqual(path.read_bytes(), candidate_source())
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(['git', 'apply', '--check', str(TOOLS / PATCH_TOOL)], cwd=root,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def fixture(self, variant):
        old, new = 'gen2-diagnostic-cpu-threads-4', metrics.metadata(4, variant)
        cache = (RESULT / 'CMakeCache.txt').read_text().replace(old, new)
        database = json.loads((RESULT / 'compile_commands.json').read_text())
        database = [{**row, 'command': row['command'].replace(old, new)} for row in database]
        files = {lang: (RESULT / ('compiler-' + lang + '.cmake')).read_text() for lang in ('C', 'CXX')}
        compiler = json.loads((RESULT / 'compiler-admission.json').read_text())
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

    def test_both_sdk_versions_admit_all_97_saved_commands(self):
        for variant in ('control', 'mx4'):
            fixture = self.fixture(variant)
            before = copy.deepcopy(fixture)
            value = self.admit(variant, *fixture)
            self.assertEqual(value['q8_model_admission']['version_tokens_changed'], 97)
            self.assertEqual(fixture, before)

    def test_flags_versions_release_compiler_and_initializer_corruption_reject(self):
        for variant in ('control', 'mx4'):
            cache, database, files, compiler = self.fixture(variant)
            for command in [database[0]['command'] + ' /O1', database[0]['command'] + ' /fp:fast',
                    database[0]['command'] + ' -DGEN2_Q8_MX4_EXPECTED=1',
                    database[0]['command'].replace(metrics.metadata(4, variant), 'unknown-version'),
                    database[0]['command'].replace(metrics.FLAG, '')]:
                with self.subTest(variant=variant, command=command), self.assertRaises(ValueError):
                    self.admit(variant, cache, [{**database[0], 'command': command}, *database[1:]], files, compiler)
            for text in [cache.replace('_FLAGS_INIT:STRING=' + metrics.FLAG, '_FLAGS_INIT:STRING='),
                    cache.replace('_FLAGS_RELEASE:STRING=/O2', '_FLAGS_RELEASE:STRING=/Od'),
                    cache.replace('GGML_NATIVE:BOOL=OFF', 'GGML_NATIVE:BOOL=ON')]:
                with self.assertRaises(ValueError):
                    self.admit(variant, text, database, files, compiler)
            with self.assertRaises(ValueError):
                self.admit(variant, cache, database, files, {**compiler, 'compiler_binary': {'bytes': 1, 'sha256': '0' * 64}})

    def audit(self, variant):
        audit = copy.deepcopy(metrics.canonical_audit())
        if variant == 'mx4':
            raw = candidate_source()
            desc = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            audit['files'][metrics.PATH] = desc
            audit['actual_compiler_sources'][metrics.PATH] = desc
            for name in audit['all_compiler_sources']:
                if PureWindowsPath(name).as_posix().endswith('/' + metrics.PATH):
                    audit['all_compiler_sources'][name] = desc
            audit['patch_series'].append({'name': Q8_PATCH, **real_descriptor(TOOLS / PATCH_TOOL)})
        return audit

    def test_complete_source_maps_and_actual_compiler_input_bytes(self):
        for variant in ('control', 'mx4'):
            audit = self.audit(variant)
            value = metrics.admit_q8_audit(audit, variant)
            self.assertEqual(value['patches'], 19 if variant == 'control' else 20)
            self.assertEqual(value['canonical_files'], 3425)
            corruptions = []
            for field in ('files', 'actual_compiler_sources', 'all_compiler_sources'):
                damaged = copy.deepcopy(audit)
                damaged[field][next(iter(damaged[field]))] = {'bytes': 0, 'sha256': '0' * 64}
                corruptions.append(damaged)
            damaged = copy.deepcopy(audit); damaged['patch_series'].reverse(); corruptions.append(damaged)
            for damaged in corruptions:
                with self.assertRaises(ValueError):
                    metrics.admit_q8_audit(damaged, variant)
            with self.assertRaises(ValueError):
                metrics.admit_q8_audit(audit, 'mx4' if variant == 'control' else 'control')

    def test_actual_series_installation_preserves_original_patches(self):
        series = (TOOLS / 'fixtures/q8-model-original-series.txt').read_bytes()
        original = json.loads((TOOLS / 'windows-compiler-fp.protocol.json').read_text())['variants']['4']['source_sha256']
        for variant in ('control', 'mx4'):
            hashes = source_hashes(original, series, variant)
            with tempfile.TemporaryDirectory() as folder:
                source, output = Path(folder) / 'source', Path(folder) / 'evidence'
                (source / 'patches').mkdir(parents=True); output.mkdir()
                (source / 'patches/series').write_bytes(series)
                names = [row['name'] for row in metrics.canonical_audit()['patch_series']]
                for name in names:
                    (source / 'patches' / name).write_text('original ' + name + '\n')
                for args in [['init', '-q'], ['add', '.'], ['-c', 'user.name=control', '-c', 'user.email=control@example.invalid', 'commit', '-q', '-m', 'fixture']]:
                    subprocess.run(['git', *args], cwd=source, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                install_q8_patch(source, output, variant)
                verify_q8_series(source, variant)
                self.assertEqual(real_descriptor(source / 'patches/series')['sha256'], hashes['patches/series'])
                if variant == 'mx4':
                    with self.assertRaises(ValueError): install_q8_patch(source, output, variant)
                (source / 'patches' / names[0]).write_text('corrupted\n')
                with self.assertRaises(ValueError): verify_q8_series(source, variant)
        with self.assertRaises(ValueError): source_hashes(original, series + b'\n', 'mx4')

    def test_versions_and_full_model_probability_identity(self):
        path = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'
        reference, raw = json.loads(path.read_text()), Path(str(path) + '.f32').read_bytes()
        for variant in ('control', 'mx4'):
            actual = {**reference, 'runtime_version': metrics.version(4, variant)}
            self.assertTrue(metrics.compare_native(reference, actual, raw, raw, variant)['probabilities_bit_exact'])
            bits, = struct.unpack_from('<I', raw)
            changed = struct.pack('<I', bits + 1) + raw[4:]
            self.assertFalse(worker_compare(reference, reference, raw, changed, 4, 4)['probabilities_bit_exact'])
            with self.assertRaisesRegex(ValueError, 'byte-exact'):
                metrics.compare_native(reference, actual, raw, changed, variant)
            for wrong in [reference, {**actual, 'push_count': 57}, {**actual, 'input_pcm_f32le_sha256': '0' * 64}]:
                with self.assertRaises(ValueError): metrics.compare_native(reference, wrong, raw, raw, variant)
        for cap, variant in [(True, 'control'), (2, 'mx4'), (4, 'unknown')]:
            with self.assertRaises(ValueError): metrics.metadata(cap, variant)

    def test_graph_vectors_require_all_shapes_repeats_workers_and_finite_exact_bytes(self):
        def stdout(threads):
            return '\n'.join(f'matrix-control threads={threads} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'
                for shape in range(4) for repeat in range(3)) + '\n'
        outputs = [struct.pack('<f', float(shape + 1)) * (2048 if shape < 2 else 64) for shape in range(4) for _ in range(3)]
        reference = metrics.graph_vectors(stdout(1), 1, outputs)
        for threads in (2, 4): self.assertEqual(metrics.graph_vectors(stdout(threads), threads, outputs, reference), reference)
        for raw in [b'', b'\0' * 8192, struct.pack('<f', float('nan')) * 2048, struct.pack('<f', 2.0) * 2048]:
            with self.assertRaises(ValueError): metrics.graph_vectors(stdout(4), 4, [raw, *outputs[1:]], reference)
        with self.assertRaises(ValueError): metrics.graph_vectors(stdout(1), 4, outputs)
        with self.assertRaises(ValueError): metrics.graph_vectors(stdout(4), 4, outputs[:-1])

    def test_timing_predicates_are_independent_and_integer_exact(self):
        def cases(times):
            return [{'index': index, 'variant': name, 'result': {'elapsed_micros': value}}
                for index, (name, value) in enumerate(zip(metrics.ORDER, times))]
        good = metrics.timing_evidence(cases([60_000_000, 57_000_000, 57_000_000, 66_000_000]))
        self.assertTrue(good['full_model_speedup_prediction_supported'])
        self.assertTrue(good['both_candidates_strictly_real_time'])
        value = metrics.timing_evidence(cases([60_000_000, 59_999_999, 59_999_999, 60_000_000]))
        self.assertTrue(value['both_candidates_strictly_real_time']); self.assertFalse(value['full_model_speedup_prediction_supported'])
        self.assertFalse(metrics.timing_evidence(cases([70_000_000,60_000_000,59_999_999,70_000_000]))['both_candidates_strictly_real_time'])
        self.assertFalse(metrics.timing_evidence(cases([60_000_000,57_000_001,57_000_000,60_000_000]))['full_model_speedup_prediction_supported'])
        self.assertFalse(metrics.timing_evidence(cases([60_000_000,57_000_000,57_000_000,66_000_001]))['control_drift_at_most_10_percent'])
        for times in [[True,1,1,1], [1.0,1,1,1], [0,1,1,1], [-1,1,1,1]]:
            with self.assertRaises(ValueError): metrics.timing_evidence(cases(times))
        with self.assertRaises(ValueError): metrics.timing_evidence(cases([1,1,1,1])[::-1])


if __name__ == '__main__':
    unittest.main()
