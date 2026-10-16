"""Guarded Pnola adapter qualification using explicit relocated artifact overrides.

These debug CLI/test helpers do not qualify a signed bundle, GUI or microphone.
"""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from build_valid_mel_candidate import descriptor
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json


def timeline_matches(actual, expected, duration, backend):
    for result in [actual, expected]:
        if (result['runtime_version'] != 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
                or result['model_revision'] != 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
                or result['backend'] != backend
                or result['geometry'] != {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222}
                or not math.isfinite(result['audio_duration_seconds'])
                or abs(result['audio_duration_seconds'] - duration) > 1e-6):
            raise ValueError('Pnola timeline identity/geometry/duration differs')
    if backend == 'metal' and actual.get('device') != 'Apple M4 Pro':
        raise ValueError('Pnola Metal device differs')
    if backend == 'cpu' and actual.get('device') is not None:
        raise ValueError('Pnola CPU device differs')
    if actual['native_frame_count'] != int(duration * 100) or len(actual['turns']) != len(expected['turns']):
        raise ValueError('Pnola native frame/turn count differs')
    maximum = 0.
    for left, right in zip(actual['turns'], expected['turns']):
        if left['speaker'] != right['speaker']:
            raise ValueError('Pnola turn speaker differs')
        for key in ['start_seconds', 'end_seconds']:
            if not all(math.isfinite(value[key]) for value in [left, right]):
                raise ValueError('nonfinite Pnola turn')
            maximum = max(maximum, abs(left[key] - right[key]))
    if maximum > 1e-6:
        raise ValueError('Pnola turn boundary differs')
    return {'turns': len(actual['turns']), 'frames': actual['native_frame_count'], 'maximum_turn_error_seconds': maximum}


def execute_guarded(argv, *, cwd, environment, stdout, stderr, guard, record, persist, timeout):
    process = None
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        record['pid'] = process.pid
        persist()
        record['returncode'] = guard.wait(process, record, timeout)
    except BaseException as error:
        if process is not None and process.poll() is None:
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
                record['forced_process_group_termination'] = True
            except ProcessLookupError:
                pass
            record['returncode'] = process.wait()
        elif process is not None:
            record['returncode'] = process.returncode
        else:
            record['startup_error'] = str(error)
        guard.release()
        raise


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Pnola qualification requires Apple Silicon macOS')
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    protocol_path = tools / 'valid-mel-pnola.protocol.json'; protocol_bytes = protocol_path.read_bytes(); p = json.loads(protocol_bytes)
    work = Path(p['work_path']); output = Path(p['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError('fresh Pnola qualification output required')
    def frozen():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError('Pnola protocol changed')
        for records, root in [(p['tools'], tools), (p['repository_files'], repo), (p['pnola_files'], Path(p['pnola_root'])), (p['files'], work)]:
            for name, expected in records.items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen Pnola input differs: ' + name)
        if ({str(f.relative_to(work)) for f in work.rglob('*') if f.is_file() and not f.is_symlink()} != set(p['files'])
                or {str(f.relative_to(work)): str(f.readlink()) for f in work.rglob('*') if f.is_symlink()} != p['aliases']):
            raise ValueError('Pnola staged inventory differs')
        for name, expected in p['preparation_records'].items():
            if descriptor(Path(name)) != expected:
                raise ValueError('Pnola preparation receipt differs')
    frozen(); manifests = {backend: verify(work / backend / 'nemo') for backend in ['cpu', 'metal']}
    output.mkdir(parents=True); (output / 'protocol.json').write_bytes(protocol_bytes)
    base = '(version 1)\n(allow default)\n(deny network*)\n(deny file-write* (subpath ' + json.dumps(str(work)) + '))\n'
    base += '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n'
    policy = output / 'runtime.sb'; policy.write_text(base)
    list_policy = output / 'list.sb'; list_policy.write_text(base + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(str(work / name)) + ')' for name in ['models', 'cpu', 'metal', 'providers', 'assets', 'references', 'cli']) + ')\n')
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ['HOME'], 'TMPDIR': os.environ.get('TMPDIR', '/tmp'), 'LC_ALL': 'C'}
    commands = []; cases = []; guard = None
    receipt = {'schema_version': 1, 'adapter_gates_passed': False, 'release_qualified': False,
               'scope': p['scope'], 'microphone_used': False, 'protocol': descriptor(protocol_path), 'cases': cases}
    def run(name, executable, arguments, variables, chosen_policy):
        guard.before_model()
        argv = ['/usr/bin/sandbox-exec', '-f', str(chosen_policy), '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', str(executable), *map(str, arguments)]
        record = {'name': name, 'argv': argv, 'environment': variables}; commands.append(record); atomic_json(output / 'commands.json', commands)
        started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                execute_guarded(argv, cwd=output, environment=variables, stdout=stdout, stderr=stderr,
                                guard=guard, record=record, timeout=p['command_timeout_seconds'],
                                persist=lambda: atomic_json(output / 'commands.json', commands))
        finally:
            record['elapsed_seconds'] = time.monotonic() - started; atomic_json(output / 'commands.json', commands)
        if record['returncode'] != 0:
            raise RuntimeError('Pnola command failed: ' + name)
        return (output / (name + '.stdout.log')).read_text(), (output / (name + '.stderr.log')).read_text()
    try:
        receipt['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        guard = StorageGuard(output, **p['storage'])
        helper = work / 'bin/pnola-tests'
        stdout, stderr = run('list', helper, ['--ignored', '--list'], env, list_policy)
        if {line.removesuffix(': test') for line in stdout.splitlines() if line.endswith(': test')} != set(p['ignored_test_names']):
            raise ValueError('Pnola ignored-test inventory differs')
        atomic_json(output / 'list-images.json', audit_loaded_libraries(stderr, work / 'cpu/nemo', manifests['cpu'], helper, require_runtime=False))
        for case in p['cases']:
            frozen(); backend = case['backend']; name = case['name']; result_path = output / (name + '.result.json')
            variables = {**env, 'GEN2_NEMO_EXECUTION': backend, 'GEN2_NEMOTRON_ASSETS': str(work / 'models/nemotron'),
                         'GEN2_NEMO_LIBRARY': str(work / backend / 'nemo/lib/libnemo_speech_asr_c.dylib'),
                         'GEN2_BARK_BUNDLE': str(work / 'models/bark'), 'GEN2_ORT_LIBRARY': str(work / 'providers/libonnxruntime.dylib'),
                         'PNOLA_VERIFY_AUDIO': str(work / 'assets/ami.wav'), 'PNOLA_BARK_REFERENCE': str(work / 'assets/bark-reference.json'),
                         'PNOLA_VERIFY_OUTPUT': str(result_path)}
            if case.get('missing'):
                variables[case['missing']] = str(work / 'missing-unavailable-control')
            if case['kind'] == 'test':
                executable = helper; arguments = ['--ignored', '--exact', case['test'], '--nocapture']
            else:
                executable = work / 'cli/Contents/MacOS/pnola-cli'
                variables['GEN2_ORT_LIBRARY'] = str(work / 'cli/Contents/Frameworks/libonnxruntime.dylib')
                arguments = ['--verify-bundle' if case['kind'] == 'recorded' else '--verify-live-speakers',
                             work / ('assets/ami.wav' if case['kind'] == 'recorded' else 'assets/live.wav'), result_path]
            stdout, stderr = run(name, executable, arguments, variables, policy)
            if case['kind'] == 'test' and ('test ' + case['test'] + ' ... ok' not in stdout or '1 passed; 0 failed; 0 ignored;' not in stdout):
                raise ValueError('exact Pnola test did not pass')
            provider = 'providers/libonnxruntime.dylib' if case['kind'] == 'test' else 'cli/Contents/Frameworks/libonnxruntime.dylib'
            auxiliary = {} if case.get('missing') == 'GEN2_BARK_BUNDLE' else {str(work / provider): p['files'][provider]}
            manifest = dict(manifests[backend], libraries={}) if case.get('missing') == 'GEN2_NEMO_LIBRARY' else manifests[backend]
            images = audit_with_auxiliary(stderr, work / backend / 'nemo', manifest, executable, auxiliary)
            atomic_json(output / (name + '.images.json'), images)
            detail = {'name': name, 'passed': True, 'loaded_images': descriptor(output / (name + '.images.json'))}
            if result_path.exists():
                value = json.loads(result_path.read_text()); expected = json.loads((work / 'references' / (backend + '.json')).read_text())
                if case.get('missing') == 'GEN2_NEMO_LIBRARY':
                    if (value['attribution']['status'] != 'unavailable' or value['asr_error'] is not None
                            or value['text'] != json.loads((work / 'assets/bark-reference.json').read_text())['fulltext']):
                        raise ValueError('Pnola unavailable-diarization recovery differs')
                elif case['kind'] == 'live':
                    if (value['passed'] is not True or value['microphone_used'] is not False or value['speaker_qualification'] is not True
                            or value['sections'] != 11 or value['queued'] != 0 or abs(value['duration_seconds'] - 80.04) > 1e-6
                            or not value['provisional_revisions'] or value['speakers']['status'] != 'complete'
                            or value['speakers']['timeline']['runtime_version'] != expected['runtime_version']
                            or value['speakers']['timeline']['backend'] != backend):
                        raise ValueError('Pnola paced live verification differs')
                elif case.get('capture'):
                    if not value['passed'] or value['microphone_used'] or not value['saved_pcm_matches_input'] or not value['final_matches_recording'] or not value['provisional_revisions']:
                        raise ValueError('Pnola paced capture-output verification differs')
                    detail['turn_comparison'] = timeline_matches(value['final']['timeline'], expected, 60., backend)
                else:
                    if value['attribution']['status'] != 'complete' or value['attribution']['complete'] is not True or value['audio_sha256'] != p['input_wav_sha256']:
                        raise ValueError('Pnola recorded attribution/input differs')
                    detail['turn_comparison'] = timeline_matches(value['attribution']['timeline'], expected, 60., backend)
                    if case.get('missing') == 'GEN2_BARK_BUNDLE' and (value['text'] != '' or not isinstance(value['asr_error'], str) or value['attribution']['words_complete'] is not False):
                        raise ValueError('Pnola unavailable-ASR recovery differs')
                    if not case.get('missing') and (value['text'] != json.loads((work / 'assets/bark-reference.json').read_text())['fulltext'] or value['asr_error'] is not None or not value['attribution']['words_complete'] or len(value['attribution']['words']) != 117):
                        raise ValueError('Pnola recorded Bark text/word preservation differs')
                detail['result'] = descriptor(result_path)
            elif not case.get('shutdown'):
                raise ValueError('Pnola case did not retain its required result')
            cases.append(detail); atomic_json(output / 'cases.json', cases)
        frozen()
        if len(cases) != len(p['cases']):
            raise ValueError('all Pnola cases required')
        receipt['adapter_gates_passed'] = True
    except BaseException as error:
        receipt['error'] = str(error); raise
    finally:
        if guard is not None:
            guard.release(); receipt['storage'] = guard.record
        atomic_json(output / 'qualification.json', receipt)


if __name__ == '__main__':
    main()
