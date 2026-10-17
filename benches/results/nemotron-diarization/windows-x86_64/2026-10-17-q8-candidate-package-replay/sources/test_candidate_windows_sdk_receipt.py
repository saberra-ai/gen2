"""Synthetic package prerequisite controls; no SDK payload is qualified here."""
import copy
import json
from pathlib import Path, PureWindowsPath
import unittest

from pooled_runtime import BOUNDED_PROFILE, load_profile, profile_provenance
from windows_q8_candidate import ORDER, LIFECYCLE, PROTOCOL
from windows_q8_candidate_sources import load_recipe, RECIPE_ID
from candidate_windows_sdk_receipt import (PRODUCER, PROTOCOL_IDENTITY, SUPERVISION,
    REALTIME_FAILURE, validate_observation_metadata, validate_producer_metadata, admit_command_sequence, admit_supervision)

TOOLS = Path(__file__).resolve().parent


class Q8CandidateSdkReceiptTest(unittest.TestCase):
    def fixture(self):
        lock_path, lock = load_profile(BOUNDED_PROFILE)
        _, recipe = load_recipe()
        p = json.loads((TOOLS / PROTOCOL).read_text())
        phases = ['clone-native', 'checkout-native', 'clone-llama', 'checkout-llama']
        phases += [phase for i in range(3) for phase in ('inference-patch-check-' + str(i), 'inference-patch-' + str(i))]
        phases += ['compiler-version', 'cmake-version', 'git-line-ending-config', 'sdk-build', 'sdk-install', 'compiler-llama-archive']
        phases += ['compiler-llama-patch-' + str(i) for i in range(20)]
        phases += ['model-free-state-test', 'native-probe-build', 'windows-provenance', 'gen2-build',
                   'loader-dependencies', 'version-only', 'receiver-host', 'kernel-disassembly',
                   'matrix-control-build', 'matrix-control-1', 'matrix-control-2', 'matrix-control-4',
                   'windows-default-loader-control']
        phases += ['case-' + str(i) + '-' + kind for i, kind in enumerate(ORDER)]
        phases += ['native-lifecycle-list', *['native-lifecycle-' + str(i) for i in range(7)]]
        root, repo = PureWindowsPath('D:/fixture/trial'), PureWindowsPath('D:/fixture/repo')
        commands = [{'name': name, 'command': ['controlled'], 'cwd': str(repo),
            'process_supervision': SUPERVISION, 'returncode': 0, 'process_tree_reaped': True,
            'job_observation': {'worker_assigned': True, 'normal_worker_exit_observed': True,
                'process_tree_reaped': True, 'job_handle_closed': True, 'worker_pid': i + 1,
                'observed_worker_exit_code': 0, 'reaped_worker_exit_code': 0, 'cleanup_errors': [],
                'remaining_before_cleanup': {'assigned_count': 0, 'processes': []}}}
            for i, name in enumerate(phases)]
        by_name = {row['name']: row for row in commands}
        by_name['clone-native']['command'] = ['git', 'clone', 'controlled', str(root / 'source')]
        example = PureWindowsPath('D:/fixture/target/release/examples/nemotron_valid_mel_stream.exe')
        for i, kind in enumerate(ORDER):
            name = f'case-{i}-{kind}'; target = root / 'evidence' / (name + '.json')
            if kind == 'native':
                argv = [root / 'sdk/bin/gen2_native_stream_probe_valid_mel.exe',
                    root / 'assets/Nemotron-3-Diarization.q8_0.gguf', root / 'assets/ami_en2002d_2132.wav', '1', '-1', target]
            else:
                argv = [example, root / 'assets', root / 'sdk/bin/gen2_nemo_windows_nemo_speech_asr_c.dll',
                        root / 'assets/ami_en2002d_2132.wav', '1', PureWindowsPath(str(target) + '.f32')]
            by_name[name]['command'] = list(map(str, argv))
        cargo = ['cargo', 'test', '--locked', '--release', '--no-default-features', '--features',
                 'backend-diar-nemo', '--test', 'diarization_native', '--']
        by_name['native-lifecycle-list']['command'] = cargo + ['--ignored', '--list']
        for i, test in enumerate(LIFECYCLE):
            by_name['native-lifecycle-' + str(i)]['command'] = cargo + ['--ignored', '--exact', test, '--test-threads=1']
        return {**profile_provenance(lock_path, lock), 'schema_version': 1, 'source_commit': PRODUCER,
            'protocol': PROTOCOL_IDENTITY, 'build_recipe_id': RECIPE_ID, 'build_recipe': recipe,
            'order': ORDER, 'completed_lifecycle_tests': LIFECYCLE, 'limits': p['limits'],
            'qualification_completed': True, 'development_native_gates_passed': True,
            'functional_lifecycle_gates_passed': True, 'realtime_gate_passed': True,
            'graph_input_payloads_fully_retained': True, 'release_qualified': False, 'package_qualified': False,
            'build': {'reported_version': lock['runtime']['reported_version'], 'public_executables': {str(example): {}}},
            'comparisons': [{'passed': True, 'probabilities_bit_exact': True, 'gen2_rtf': .9, 'native_rtf': .9}] * 2,
            'cases': [{'index': i, 'kind': kind, 'command_started': True,
                       'result_path': f'case-{i}-{kind}.json', 'probability_path': f'case-{i}-{kind}.json.f32'}
                      for i, kind in enumerate(ORDER)], 'commands': commands}

    def test_current_producer_metadata_requires_separate_payload_replay(self):
        lock_path, lock = validate_producer_metadata(self.fixture())
        self.assertEqual(lock['profile'], BOUNDED_PROFILE)
        self.assertTrue(lock_path.is_file())

    def test_complete_failed_timing_can_be_replayed_but_cannot_admit_a_package(self):
        record = self.fixture()
        record['comparisons'] = [dict(row) for row in record['comparisons']]
        record['comparisons'][1]['native_rtf'] = 1.05
        record.update(realtime_gate_passed=False, development_native_gates_passed=False, error=REALTIME_FAILURE)
        self.assertEqual(validate_observation_metadata(record)[1]['profile'], BOUNDED_PROFILE)
        with self.assertRaises(ValueError): validate_producer_metadata(record)
        for delta in ({'error': 'unrelated failure'}, {'functional_lifecycle_gates_passed': False},
                      {'realtime_gate_passed': True}, {'development_native_gates_passed': True}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                validate_observation_metadata({**record, **delta})
        for bad in (0, True, float('nan'), float('inf')):
            changed = copy.deepcopy(record)
            changed['comparisons'][1]['native_rtf'] = bad
            with self.subTest(timing=bad), self.assertRaises(ValueError): validate_observation_metadata(changed)

    def test_failed_or_relabelled_producer_does_not_admit(self):
        fixture = self.fixture()
        for delta in ({'source_commit': '0' * 40}, {'protocol': {}}, {'schema_version': 2}, {'schema_version': True},
                {'build_recipe_id': 'different'}, {'build_recipe': {}}, {'runtime_profile': 'different'},
                {'qualification_completed': False}, {'development_native_gates_passed': False},
                {'functional_lifecycle_gates_passed': False}, {'realtime_gate_passed': False},
                {'graph_input_payloads_fully_retained': False}, {'completed_lifecycle_tests': []},
                {'release_qualified': True}, {'package_qualified': True}, {'error': 'failed'}):
            with self.subTest(delta=delta), self.assertRaises(ValueError): validate_producer_metadata({**fixture, **delta})

    def test_any_slow_missing_or_nonfinite_rtf_rejects_even_with_pass_flags(self):
        for pair in (0, 1):
            for field in ('native_rtf', 'gen2_rtf'):
                for bad in (1, 1.001, 0, True, float('nan'), float('inf')):
                    value = copy.deepcopy(self.fixture())
                    value['comparisons'] = [dict(row) for row in value['comparisons']]
                    value['comparisons'][pair][field] = bad
                    with self.subTest(pair=pair, field=field, bad=bad), self.assertRaises(ValueError):
                        validate_producer_metadata(value)
        with self.assertRaises(ValueError): validate_producer_metadata({**self.fixture(), 'comparisons': []})

    def test_command_list_cannot_drop_duplicate_reorder_or_fail_a_gate(self):
        commands = self.fixture()['commands']
        variants = [commands[1:], commands + [commands[0]], [commands[1], commands[0], *commands[2:]]]
        for key, bad in (('returncode', 1), ('returncode', False), ('process_tree_reaped', False)):
            changed = copy.deepcopy(commands); changed[13][key] = bad; variants.append(changed)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError): admit_command_sequence(value)

    def test_cleanup_requires_observed_integer_exit_closed_handle_and_owned_members(self):
        good = self.fixture()['commands'][0]
        for field, bad in (('observed_worker_exit_code', False), ('reaped_worker_exit_code', 1),
                ('worker_assigned', False), ('normal_worker_exit_observed', False), ('job_handle_closed', False),
                ('cleanup_errors', ['failed']), ('worker_pid', True)):
            row = copy.deepcopy(good); row['job_observation'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError): admit_supervision(row)
        owned = copy.deepcopy(good)
        snapshot = {'assigned_count': 1, 'processes': [
            {'pid': 42, 'membership_confirmed': True, 'image': 'C:\\controlled\\vctip.exe'}]}
        owned['job_observation']['remaining_before_cleanup'] = snapshot
        admit_supervision(owned)
        snapshot['processes'][0]['membership_confirmed'] = False
        with self.assertRaises(ValueError): admit_supervision(owned)

    def test_exact_execution_arguments_bind_sdk_public_example_and_working_directory(self):
        for phase, index, bad in [('case-0-native', 0, 'D:\\other-probe.exe'),
                ('case-1-public', 0, 'D:\\other-public.exe'), ('case-1-public', 2, 'D:\\other-library.dll'),
                ('case-2-public', 3, 'D:\\other.wav'), ('case-3-native', 3, '2'),
                ('native-lifecycle-0', 6, 'different-features')]:
            value = self.fixture(); row, = [r for r in value['commands'] if r['name'] == phase]
            row['command'][index] = bad
            with self.subTest(phase=phase, index=index), self.assertRaises(ValueError): validate_producer_metadata(value)
        value = self.fixture(); row, = [r for r in value['commands'] if r['name'] == 'case-0-native']
        row['cwd'] = 'D:\\other-repository'
        with self.assertRaises(ValueError): validate_producer_metadata(value)


if __name__ == '__main__': unittest.main()
