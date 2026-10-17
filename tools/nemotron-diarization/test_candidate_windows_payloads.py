"""Controlled payload mutations; synthetic timings are not SDK runtime evidence."""
import copy
import json
from pathlib import Path, PureWindowsPath
import struct
import tempfile
import unittest

from build_valid_mel_candidate import descriptor
from candidate_windows_compiler_receipt import verify_compiler_records
from candidate_windows_payloads import (verify_graph_payloads, verify_histories, replay_histories, verify_lifecycle_logs,
                                        verify_objects, verify_output_bindings, verify_probe_and_graph_commands)
from windows_q8_candidate import ORDER, LIFECYCLE
from windows_q8_candidate_metrics import BOUNDED_RESULT
from windows_q8_candidate_sources import load_recipe
from windows_q8_model_sources import PATH
from windows_compiler_fp_metrics import baseline
from windows_session_timing import file_inventory
import test_windows_q8_candidate as compiler_fixtures

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CandidateWindowsPayloadTests(unittest.TestCase):
    def compiler_fixture(self):
        fixture = compiler_fixtures.Q8CandidateTest()
        cache, database, languages, original = fixture.compiler_fixture()
        record = fixture.admit_compiler(cache, database, languages, original)
        record.update(compiler_binary=original['compiler_binary'],
                      linker_binary=original['linker_binary'], compiler_path_aliases=original['compiler_path_aliases'])
        return cache, database, languages, record, load_recipe()[1]

    def test_full_saved_compiler_baseline_replays_without_windows_binaries(self):
        cache, database, languages, record, recipe = self.compiler_fixture()
        self.assertEqual(verify_compiler_records(cache, database, languages, record, recipe), record)

    def test_changed_flags_source_commands_aliases_and_linker_records_reject(self):
        cache, database, languages, record, recipe = self.compiler_fixture()
        variants = []
        changed = copy.deepcopy(database); changed[0]['command'] += ' /O1'; variants.append((cache, changed, record))
        variants.append((cache, database[:-1], record))
        changed = copy.deepcopy(record); changed['linker_binary']['sha256'] = '0' * 64; variants.append((cache, database, changed))
        changed = copy.deepcopy(record); changed['compiler_path_aliases'] = {}; variants.append((cache, database, changed))
        variants.append((cache.replace('/clang:-ffp-contract=off', '/clang:-ffp-contract=fast'), database, record))
        for text, commands, metadata in variants:
            with self.subTest(text=text[:20], commands=len(commands)), self.assertRaises(ValueError):
                verify_compiler_records(text, commands, languages, metadata, recipe)

    def history_fixture(self, root):
        old = REPO / BOUNDED_RESULT
        native, public = [json.loads((old / name).read_text()) for name in ('native-ami.json', 'public.json')]
        raw = (old / 'native-ami.json.f32').read_bytes()
        cases = []
        for index, kind in enumerate(ORDER):
            phase = f'case-{index}-{kind}'
            result = copy.deepcopy(native if kind == 'native' else public)
            result['elapsed_micros'] = 59_000_000  # Admission control, never measured performance.
            target, probabilities = root / (phase + '.json'), root / (phase + '.json.f32')
            write_json(target, result); probabilities.write_bytes(raw)
            (root / (phase + '.stdout.log')).write_text(json.dumps(result) if kind == 'public' else '')
            (root / (phase + '.log')).write_text('ordinary production log\n')
            cases.append({'index': index, 'kind': kind, 'result': result, 'result_path': target.name,
                'probability_path': probabilities.name, 'result_file': descriptor(target), 'probabilities': descriptor(probabilities)})
        from windows_q8_candidate_metrics import compare_candidate_pair
        comparisons = [compare_candidate_pair(cases[1]['result'], cases[0]['result'], raw, raw),
                       compare_candidate_pair(cases[2]['result'], cases[3]['result'], raw, raw)]
        return {'cases': cases, 'comparisons': comparisons}

    def test_complete_saved_histories_replay_both_pairs_and_all_probabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); record = self.history_fixture(root)
            self.assertEqual(verify_histories(root, record), record['comparisons'])

    def test_changed_truncated_or_telemetry_history_rejects(self):
        for mode in ('bytes', 'truncated', 'telemetry', 'comparison'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); record = self.history_fixture(root)
                if mode in ('bytes', 'truncated'):
                    path = root / record['cases'][2]['probability_path']
                    raw = path.read_bytes(); path.write_bytes(raw[:-4] if mode == 'truncated' else b'\0\0\0\0' + raw[4:])
                elif mode == 'telemetry': (root / 'case-1-public.log').write_text('[DEBUG-worker-v1] forbidden\n')
                else: record['comparisons'][0]['native_rtf'] = .8
                with self.assertRaises(ValueError): verify_histories(root, record)

    def test_actual_slow_history_cannot_hide_behind_stored_pass_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); record = self.history_fixture(root)
            case = record['cases'][3]; case['result']['elapsed_micros'] = 60_000_000
            path = root / case['result_path']; write_json(path, case['result']); case['result_file'] = descriptor(path)
            with self.assertRaises(ValueError): verify_histories(root, record)

    def test_failed_timing_replay_preserves_actual_result_without_package_admission(self):
        from windows_q8_candidate_metrics import compare_candidate_pair, both_pairs_realtime
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); record = self.history_fixture(root)
            case = record['cases'][3]; case['result']['elapsed_micros'] = 63_000_000
            path = root / case['result_path']; write_json(path, case['result']); case['result_file'] = descriptor(path)
            native, public = record['cases'][3], record['cases'][2]
            record['comparisons'][1] = compare_candidate_pair(public['result'], native['result'],
                (root / native['probability_path']).read_bytes(), (root / public['probability_path']).read_bytes())
            self.assertFalse(both_pairs_realtime(replay_histories(root, record)))
            with self.assertRaises(ValueError): verify_histories(root, record)

    def test_closed_output_binding_rejects_extra_missing_and_changed_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ci = PureWindowsPath('D:/trial/evidence')
            (root / 'saved.f32').write_bytes(b'controlled')
            commands = [{'name': 'controlled'}]; write_json(root / 'commands.json', commands)
            record = {'commands': commands, 'retained_output_bindings': {str(ci / 'saved.f32'): descriptor(root / 'saved.f32')}}
            verify_output_bindings(root, record, ci)
            (root / 'extra').write_text('extra')
            with self.assertRaises(ValueError): verify_output_bindings(root, record, ci)
            (root / 'extra').unlink(); (root / 'saved.f32').write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_output_bindings(root, record, ci)
            (root / 'saved.f32').unlink()
            with self.assertRaises(ValueError): verify_output_bindings(root, record, ci)

    def objects_fixture(self, root):
        native, ci = PureWindowsPath('D:/trial/source'), PureWindowsPath('D:/trial/evidence')
        build = native / 'build/gen2-cpu-diar'; database, units, inputs = [], [], {}
        for index in range(97):
            source = build / '_deps/llama.cpp' / PATH if index == 0 else native / f'src/unit-{index}.cpp'
            output = PureWindowsPath(f'objects/unit-{index}.obj')
            row = {'file': str(source), 'directory': str(build), 'output': str(output), 'command': 'controlled compiler command'}
            retained = 'compiler-objects/' + output.as_posix()
            path = root / retained; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f'object {index}'.encode())
            source_identity = {'bytes': index + 1, 'sha256': f'{index:064x}'}
            database.append(row); inputs[str(source)] = source_identity
            units.append({'command': row, 'source': source_identity, 'object': descriptor(path), 'retained_object': retained})
        (root / 'kernel.obj').write_bytes((root / units[0]['retained_object']).read_bytes())
        record = {'kernel': {'translation_units': units, 'compiler_command': database[0]}}
        return record, database, {'all_compiler_sources': inputs}, native, ci

    def test_all_97_retained_objects_bind_source_command_and_actual_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); build, database, audit, native, ci = self.objects_fixture(root)
            self.assertEqual(len(verify_objects(root, build, database, audit, native, ci)), 291)
            (root / build['kernel']['translation_units'][5]['retained_object']).write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_objects(root, build, database, audit, native, ci)

    def test_missing_duplicate_or_reassigned_object_association_rejects(self):
        for mode in ('missing', 'duplicate', 'source'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); build, database, audit, native, ci = self.objects_fixture(root)
                units = build['kernel']['translation_units']
                if mode == 'missing': units.pop()
                elif mode == 'duplicate': database[1] = database[0]; units[1] = units[0]
                else: units[4]['source'] = units[5]['source']
                with self.assertRaises(ValueError): verify_objects(root, build, database, audit, native, ci)

    def graph_fixture(self, root):
        native, ci, repo = PureWindowsPath('D:/trial/source'), PureWindowsPath('D:/trial/evidence'), PureWindowsPath('D:/repo')
        build = native / 'build/gen2-cpu-diar'; bindings, libraries, cases = {}, {}, []
        for stem in ('ggml', 'ggml-base', 'ggml-cpu'):
            path = build / ('gen2_nemo_windows_' + stem + '.lib')
            retained = 'graph-inputs/' + path.name; target = root / retained
            target.parent.mkdir(exist_ok=True); target.write_bytes(('controlled import library ' + stem).encode())
            identity = descriptor(target); libraries[str(path)] = {'retained_path': retained, **identity}
            bindings.update({str(path): identity, str(ci / retained): identity})
        source = 'tools/nemotron-diarization/ggml_matrix_profile_control.cpp'
        bindings[str(repo / source)] = descriptor(REPO / source)
        binary = root / 'matrix-profile-control.exe'; binary.write_bytes(b'controlled graph executable')
        bindings[str(ci / binary.name)] = descriptor(binary)
        for threads in (1, 2, 4):
            phase = 'matrix-control-' + str(threads); paths, rows = [], []
            for shape in range(4):
                for repeat in range(3):
                    path = root / (phase + f'.{shape}.{repeat}.f32')
                    path.write_bytes(struct.pack('<f', float(shape + 1)) * (2048 if shape < 2 else 64)); paths.append(path)
                    rows.append(f'matrix-control threads={threads} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}')
            stdout, log = root / (phase + '.stdout.log'), root / (phase + '.log')
            stdout.write_text('\n'.join(rows) + '\n'); log.write_text('')
            outputs = {str(ci / p.name): descriptor(p) for p in [*paths, stdout, log]}
            bindings.update(outputs); cases.append({'threads': threads, 'outputs_and_logs': outputs})
        return {'retained_import_libraries': libraries, 'graph_input_payloads_fully_retained': True,
                'bindings': bindings, 'vectors': 36, 'cases': cases}, native, build, ci, repo

    def test_all_three_library_payloads_and_36_full_graph_vectors_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); controls, _, build, ci, repo = self.graph_fixture(root)
            verify_graph_payloads(root, controls, build, ci, repo)
            self.assertEqual(len(controls['bindings']), 50)

    def test_missing_changed_library_or_inconsistent_complete_graph_rejects(self):
        for mode in ('missing', 'library', 'vector'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); controls, _, build, ci, repo = self.graph_fixture(root)
                if mode == 'missing': (root / 'graph-inputs/gen2_nemo_windows_ggml.lib').unlink()
                elif mode == 'library': (root / 'graph-inputs/gen2_nemo_windows_ggml.lib').write_bytes(b'changed')
                else:
                    path = root / 'matrix-control-2.0.0.f32'; path.write_bytes(struct.pack('<f', 2.0) * 2048)
                    identity = descriptor(path); name = str(ci / path.name)
                    controls['bindings'][name] = controls['cases'][1]['outputs_and_logs'][name] = identity
                with self.assertRaises((OSError, ValueError)): verify_graph_payloads(root, controls, build, ci, repo)

    def test_actual_graph_compiler_command_must_link_retained_original_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); controls, native, build, ci, repo = self.graph_fixture(root)
            trial, hashing, compiler = native.parent, native / 'llama.cpp/vendor/hash', baseline('msvc')[2]['compiler']
            probe = [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/TP', str(ci / 'native_stream_probe_bounded.cpp'),
                str(hashing / 'sha256/sha256.c'), '/I' + str(trial / 'sdk/include'), '/I' + str(native / 'examples'),
                '/I' + str(hashing), '/link', str(trial / 'sdk/lib/gen2_nemo_windows_nemo_speech_asr_c.lib'),
                '/OUT:' + str(trial / 'sdk/bin/gen2_native_stream_probe_valid_mel.exe')]
            libs = [str(build / ('gen2_nemo_windows_' + stem + '.lib')) for stem in ('ggml', 'ggml-base', 'ggml-cpu')]
            matrix = [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/TP',
                str(repo / 'tools/nemotron-diarization/ggml_matrix_profile_control.cpp'),
                '/I' + str(build / '_deps/llama.cpp/ggml/include'), '/link', libs[0],
                '/OUT:' + str(ci / 'matrix-profile-control.exe'), *libs[1:]]
            commands = [{'name': 'native-probe-build', 'command': probe, 'cwd': str(ci)},
                        {'name': 'matrix-control-build', 'command': matrix, 'cwd': str(ci)}]
            commands += [{'name': 'matrix-control-' + str(t), 'command': [str(ci / 'matrix-profile-control.exe'), str(t), str(ci / ('matrix-control-' + str(t)))],
                          'cwd': str(repo)} for t in (1, 2, 4)]
            verify_probe_and_graph_commands(commands, controls, native, ci, repo)
            changed = copy.deepcopy(commands)
            changed[0]['command'][0] = changed[1]['command'][0] = 'D:\\unrelated\\cl.exe'
            with self.assertRaises(ValueError): verify_probe_and_graph_commands(changed, controls, native, ci, repo)
            commands[1]['command'][-1] = 'D:\\unretained.lib'
            with self.assertRaises(ValueError): verify_probe_and_graph_commands(commands, controls, native, ci, repo)

    def lifecycle_fixture(self, root):
        record = self.history_fixture(root); native = record['cases'][0]
        write_json(root / 'reference/ami.json', {'segments': [
            {'start': t['start_seconds'], 'end': t['end_seconds'], 'speaker': t['speaker']} for t in native['result']['turns']]})
        (root / 'reference/ami.json.f32').write_bytes((root / native['probability_path']).read_bytes())
        (root / 'native-lifecycle-list.stdout.log').write_text('\n'.join(test + ': test' for test in LIFECYCLE))
        for index, test in enumerate(LIFECYCLE):
            (root / f'native-lifecycle-{index}.stdout.log').write_text('test ' + test + ' ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n')
            (root / f'native-lifecycle-{index}.log').write_text('')
        return record

    def test_all_seven_lifecycle_logs_and_full_native_reference_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); record = self.lifecycle_fixture(root)
            verify_lifecycle_logs(root, record)

    def test_ignored_multiple_passes_or_changed_reference_rejects(self):
        for mode in ('ignored', 'multiple', 'reference'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); record = self.lifecycle_fixture(root)
                if mode == 'reference': (root / 'reference/ami.json.f32').write_bytes(b'wrong reference')
                else:
                    path = root / 'native-lifecycle-2.stdout.log'; raw = path.read_text()
                    path.write_text(raw.replace('1 passed', '11 passed' if mode == 'multiple' else '0 passed').replace('0 ignored', '1 ignored' if mode == 'ignored' else '0 ignored'))
                with self.assertRaises(ValueError): verify_lifecycle_logs(root, record)


if __name__ == '__main__': unittest.main()
