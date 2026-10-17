"""Package prerequisites for the exact Windows Q8 candidate; no runtime execution."""
import json
from pathlib import Path, PureWindowsPath

from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from pooled_runtime import BOUNDED_PROFILE, load_profile, profile_provenance
from windows_q8_candidate import LIFECYCLE, ORDER, PROTOCOL
from windows_q8_candidate_metrics import both_pairs_realtime
from windows_q8_candidate_sources import load_recipe, RECIPE_ID

PRODUCER = 'fee2177adcd7b75b488c67bea5a593ea17df65de'
PROTOCOL_IDENTITY = {'bytes': 190779, 'sha256': '8d22329d34a58422ffde3fc2067ea3ebfa2e9928d3845e16fb07f91e58ec18ff'}
SUPERVISION = 'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup'


def admit_supervision(command):
    """Require the actual zero exit and complete owned-Job termination evidence."""
    value = command.get('job_observation') or {}
    snapshot = value.get('remaining_before_cleanup') or {}
    records = snapshot.get('processes')
    if (command.get('process_supervision') != SUPERVISION or 'error' in command
            or type(command.get('returncode')) is not int or command['returncode'] != 0
            or command.get('process_tree_reaped') is not True
            or any(value.get(key) is not True for key in ('worker_assigned', 'normal_worker_exit_observed',
                    'process_tree_reaped', 'job_handle_closed'))
            or type(value.get('worker_pid')) is not int or not 0 < value['worker_pid'] <= 0xffffffff
            or type(value.get('observed_worker_exit_code')) is not int or value['observed_worker_exit_code'] != 0
            or type(value.get('reaped_worker_exit_code')) is not int or value['reaped_worker_exit_code'] != 0
            or value.get('cleanup_errors') != [] or 'snapshot_error' in value
            or not isinstance(records, list) or type(snapshot.get('assigned_count')) is not int
            or snapshot['assigned_count'] != len(records) or not 0 <= len(records) <= 4096):
        raise ValueError('zero-exit, observed and reaped Windows Job command required')
    pids = set()
    for row in records:
        pid = row.get('pid')
        if (type(pid) is not int or not 0 < pid <= 0xffffffff or pid in pids
                or type(row.get('membership_confirmed')) is not bool
                or ('image' in row and (row['membership_confirmed'] is not True
                    or not PureWindowsPath(row['image']).is_absolute()))):
            raise ValueError('scoped owned-background process identities required')
        pids.add(pid)


def admit_command_sequence(commands):
    """Keep source/graph/model/lifecycle prerequisite ordering explicit."""
    if not isinstance(commands, list) or any(not isinstance(row, dict) for row in commands):
        raise ValueError('recorded Windows command list required')
    names = [row.get('name') for row in commands]
    required = ['clone-native', 'checkout-native', 'clone-llama', 'checkout-llama']
    required += [name for index in range(3)
                 for name in ('inference-patch-check-' + str(index), 'inference-patch-' + str(index))]
    required += ['compiler-version', 'cmake-version', 'git-line-ending-config', 'sdk-build', 'sdk-install',
                 'compiler-llama-archive']
    required += ['compiler-llama-patch-' + str(index) for index in range(20)]
    required += ['model-free-state-test', 'native-probe-build', 'windows-provenance', 'gen2-build',
                 'loader-dependencies', 'version-only', 'receiver-host', 'kernel-disassembly',
                 'matrix-control-build', 'matrix-control-1', 'matrix-control-2', 'matrix-control-4',
                 'windows-default-loader-control']
    required += ['case-' + str(index) + '-' + kind for index, kind in enumerate(ORDER)]
    required += ['native-lifecycle-list', *['native-lifecycle-' + str(index) for index in range(7)]]
    if names != required:
        raise ValueError('complete original source/build/graph/history/lifecycle command sequence required')
    for row in commands:
        admit_supervision(row)


def admit_execution_commands(commands, public_bindings):
    native = PureWindowsPath(commands[0]['command'][-1])
    if not native.is_absolute() or native.name != 'source':
        raise ValueError('absolute isolated candidate source required')
    root = native.parent
    repo = PureWindowsPath(commands[0]['cwd'])
    candidates = [PureWindowsPath(name) for name in public_bindings
                  if PureWindowsPath(name).name == 'nemotron_valid_mel_stream.exe'
                  and not PureWindowsPath(name).is_relative_to(root / 'evidence')]
    if len(candidates) != 1:
        raise ValueError('one retained original public example required')
    public, = candidates
    assets, evidence = root / 'assets', root / 'evidence'
    expected = []
    for index, kind in enumerate(ORDER):
        name = f'case-{index}-{kind}'
        target = evidence / (name + '.json')
        if kind == 'native':
            argv = [root / 'sdk/bin/gen2_native_stream_probe_valid_mel.exe',
                    assets / 'Nemotron-3-Diarization.q8_0.gguf', assets / 'ami_en2002d_2132.wav', '1', '-1', target]
        else:
            argv = [public, assets, root / 'sdk/bin/gen2_nemo_windows_nemo_speech_asr_c.dll',
                    assets / 'ami_en2002d_2132.wav', '1', PureWindowsPath(str(target) + '.f32')]
        expected.append((name, list(map(str, argv))))
    cargo = ['cargo', 'test', '--locked', '--release', '--no-default-features', '--features',
             'backend-diar-nemo', '--test', 'diarization_native', '--']
    expected.append(('native-lifecycle-list', cargo + ['--ignored', '--list']))
    expected += [('native-lifecycle-' + str(index), cargo + ['--ignored', '--exact', test, '--test-threads=1'])
                 for index, test in enumerate(LIFECYCLE)]
    observed = [row for row in commands if row['name'].startswith(('case-', 'native-lifecycle'))]
    if ([row['name'] for row in observed] != [name for name, _ in expected]
            or any(row['command'] != argv or PureWindowsPath(row['cwd']) != repo
                   for row, (_, argv) in zip(observed, expected))):
        raise ValueError('original ordered history/lifecycle commands and exact SDK/assets required')


def validate_producer_metadata(record):
    """Admit producer identity and mandatory observations; payload replay is separate."""
    tools = Path(__file__).resolve().parent
    lock_path, lock = load_profile(BOUNDED_PROFILE)
    _, recipe = load_recipe()
    protocol_path = tools / PROTOCOL
    if descriptor(protocol_path) != PROTOCOL_IDENTITY:
        raise ValueError('exact admitted Windows Q8 producer protocol required')
    protocol = json.loads(protocol_path.read_text())
    for key, value in profile_provenance(lock_path, lock).items():
        if record.get(key) != value:
            raise ValueError('candidate runtime provenance differs: ' + key)
    flags = ('qualification_completed', 'development_native_gates_passed', 'functional_lifecycle_gates_passed',
             'realtime_gate_passed', 'graph_input_payloads_fully_retained')
    if (any(record.get(key) is not True for key in flags)
            or type(record.get('schema_version')) is not int or record['schema_version'] != 1
            or record.get('release_qualified') is not False
            or record.get('package_qualified') is not False or 'error' in record
            or record.get('source_commit') != PRODUCER or record.get('protocol') != PROTOCOL_IDENTITY
            or record.get('build_recipe_id') != RECIPE_ID or record.get('build_recipe') != recipe
            or record.get('order') != ORDER or record.get('completed_lifecycle_tests') != LIFECYCLE
            or record.get('limits') != protocol['limits']
            or record.get('build', {}).get('reported_version') != lock['runtime']['reported_version']):
        raise ValueError('completed exact Windows Q8 SDK/public/lifecycle qualification required')
    comparisons = record.get('comparisons', [])
    if not both_pairs_realtime(comparisons):
        raise ValueError('all four original below-one native/public RTFs required')
    cases = record.get('cases', [])
    if len(cases) != 4 or [case.get('kind') for case in cases] != ORDER:
        raise ValueError('four ordered candidate observations required')
    for index, case in enumerate(cases):
        if (type(case.get('index')) is not int or case['index'] != index
                or case.get('command_started') is not True
                or case.get('result_path') != f'case-{index}-{ORDER[index]}.json'
                or case.get('probability_path') != f'case-{index}-{ORDER[index]}.json.f32'):
            raise ValueError('complete retained candidate case identities required')
    admit_command_sequence(record.get('commands', []))
    admit_execution_commands(record['commands'], record['build'].get('public_executables', {}))
    pins = {'tools/nemotron-diarization/' + name: value for name, value in protocol['tools'].items()}
    pins.update(protocol['repository_files'])
    pins['tools/nemotron-diarization/' + PROTOCOL] = PROTOCOL_IDENTITY
    verify_git_sources(tools.parents[1], PRODUCER, pins)
    return lock_path, lock
