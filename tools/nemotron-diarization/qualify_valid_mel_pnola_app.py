"""Qualify saved complete apps with automatic packaged artifact resolution.

CLI recording and paced live paths only; no GUI or physical microphone.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard, atomic_json
from qualify_valid_mel_pnola import execute_guarded, timeline_matches
from sdk_archive import inventory
from verify_valid_mel_pnola_app import audit_app_images, verify_app


def live_matches(actual, expected, backend):
    for value in [actual, expected]:
        if (value['passed'] is not True or value['microphone_used'] is not False
                or value['speaker_qualification'] is not True or value['sections'] != 11
                or value['queued'] != 0 or value['words'] != 152 or value['export_bytes'] != 803
                or not math.isfinite(value['duration_seconds']) or abs(value['duration_seconds'] - 80.04) > 1e-6
                or not value['provisional_revisions'] or value['speakers']['status'] != 'complete'
                or not math.isfinite(value['speakers']['processed_audio_seconds'])
                or abs(value['speakers']['processed_audio_seconds'] - 80.04) > 1e-6):
            raise ValueError('complete app live session/recovery/export differs')
        revisions = value['provisional_revisions']
        if len(revisions) > 17:
            raise ValueError('unbounded complete app live revisions')
        previous = 0
        for revision in revisions:
            if (type(revision['revision']) is not int or revision['revision'] <= previous
                    or not math.isfinite(revision['elapsed_seconds']) or revision['elapsed_seconds'] < 0
                    or not math.isfinite(revision['processed_audio_seconds'])
                    or not 0 < revision['processed_audio_seconds'] <= 80.04):
                raise ValueError('invalid complete app live revision')
            previous = revision['revision']
        timeline = value['speakers']['timeline']
        if (timeline['runtime_version'] != 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
                or timeline['model_revision'] != 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
                or timeline['backend'] != backend or timeline['geometry'] != [9, 4, 0, 264, 264, 222]
                or timeline['device'] != ('Apple M4 Pro' if backend == 'metal' else None)
                or not math.isfinite(timeline['audio_duration_seconds'])
                or abs(timeline['audio_duration_seconds'] - 80.04) > 1e-6):
            raise ValueError('complete app live timeline identity differs')
    left = actual['speakers']['timeline']['turns']; right = expected['speakers']['timeline']['turns']
    if len(left) != len(right):
        raise ValueError('complete app live turn count differs')
    maximum = 0.
    for a, b in zip(left, right):
        if a['speaker'] != b['speaker']:
            raise ValueError('complete app live speaker differs')
        for key in ['start_seconds', 'end_seconds']:
            if not all(math.isfinite(turn[key]) for turn in [a, b]):
                raise ValueError('nonfinite complete app live turn')
            maximum = max(maximum, abs(a[key] - b[key]))
    if maximum > 1e-6:
        raise ValueError('complete app live turn boundary differs')
    return {'turns': len(left), 'maximum_turn_error_seconds': maximum,
            'native_frame_count_observed': False, 'reference': 'hash-locked prior debug paced-live output; independent recording parity is asserted internally by both helpers'}


def recorded_matches(value, reference, bark_text, backend, wav_sha256, *, unavailable=False):
    if value['audio_sha256'] != wav_sha256 or value['text'] != bark_text or value['asr_error'] is not None:
        raise ValueError('complete app recorded input/Bark text differs')
    if unavailable:
        if (value['attribution']['status'] != 'unavailable'
                or value['attribution'].get('reason') != 'diarization backend unavailable: requested Metal device is unavailable in the native SDK'):
            raise ValueError('CPU app silently accepted unsupported Metal')
        return {'unsupported_metal_preserves_exact_bark_text': True}
    if (value['attribution']['status'] != 'complete' or value['attribution']['complete'] is not True
            or value['attribution']['words_complete'] is not True or len(value['attribution']['words']) != 117):
        raise ValueError('complete app recorded attribution differs')
    return timeline_matches(value['attribution']['timeline'], reference, 60., backend)


def main():
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=tools / 'valid-mel-pnola-app.protocol.json')
    protocol_path = parser.parse_args().protocol
    protocol_bytes = protocol_path.read_bytes(); p = json.loads(protocol_bytes)
    output = Path(p['output_path']); work = Path(p['work_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError('fresh complete-app qualification output required')
    def frozen():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError('complete-app protocol changed')
        for records, root in [(p['tools'], tools), (p['repository_files'], repo), (p['files'], work)]:
            for name, expected in records.items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen complete-app input differs: ' + name)
        if (any(path.is_symlink() for path in work.rglob('*'))
                or {str(path.relative_to(work)) for path in work.rglob('*') if path.is_file()} != set(p['files'])):
            raise ValueError('complete-app staged inventory differs')
        for backend, app in p['apps'].items():
            if inventory(Path(app['path'])) != app['inventory']:
                raise ValueError('complete app inventory changed: ' + backend)
        for name, expected in p['preparation_records'].items():
            if descriptor(Path(name)) != expected:
                raise ValueError('complete app preparation receipt changed')
    frozen()
    verified = {backend: verify_app(Path(app['path'])) for backend, app in p['apps'].items()}
    output.mkdir(parents=True); (output / 'protocol.json').write_bytes(protocol_bytes)
    policy = output / 'runtime.sb'
    protected = [str(work), *[app['path'] for app in p['apps'].values()]]
    policy.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write* '
                      + ' '.join('(subpath ' + json.dumps(path) + ')' for path in protected) + ')\n'
                      + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n')
    temporary = output / 'tmp'; temporary.mkdir(); home = output / 'home'; home.mkdir()
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}
    commands = []; cases = []; guard = None
    r = {'schema_version': 1, 'automatic_packaged_resolution_gates_passed': False, 'release_qualified': False,
         'scope': p['scope'], 'microphone_used': False, 'artifact_environment_overrides_used': False,
         'protocol': descriptor(protocol_path), 'cases': cases}
    try:
        r['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        guard = StorageGuard(output, **p['storage'])
        for case in p['cases']:
            frozen(); guard.before_model()
            name = case['name']; backend = case['backend']; app = Path(p['apps'][backend]['path'])
            result = output / (name + '.result.json'); executable = app / 'Contents/MacOS/pnola-desktop'
            arguments = ['--verify-live-speakers' if case['kind'] == 'live' else '--verify-bundle', str(work / 'ami.wav'), str(result)]
            variables = {**env, **({'GEN2_NEMO_EXECUTION': 'metal'} if case.get('unsupported_metal') else {})}
            argv = ['/usr/bin/sandbox-exec', '-f', str(policy), str(executable), *arguments]
            command = {'name': name, 'argv': argv, 'cwd': str(output), 'environment': variables}
            commands.append(command); atomic_json(output / 'commands.json', commands); started = time.monotonic()
            try:
                with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                    execute_guarded(argv, cwd=output, environment=variables, stdout=stdout, stderr=stderr, guard=guard,
                                    record=command, persist=lambda: atomic_json(output / 'commands.json', commands), timeout=p['command_timeout_seconds'])
            finally:
                command['elapsed_seconds'] = time.monotonic() - started; atomic_json(output / 'commands.json', commands)
            if command['returncode'] != 0:
                raise RuntimeError('complete-app command failed: ' + name)
            images = audit_app_images((output / (name + '.stderr.log')).read_text(), app, verified[backend])
            if not images['process_local_add_image_history_observed']:
                raise ValueError('complete app process-local image history is required')
            atomic_json(output / (name + '.images.json'), images)
            value = json.loads(result.read_text())
            if case['kind'] == 'live':
                comparison = live_matches(value, json.loads((work / (backend + '-live.json')).read_text()), backend)
            else:
                comparison = recorded_matches(value, json.loads((work / (backend + '.json')).read_text()),
                                               p['bark_text'], backend, p['wav_sha256'], unavailable=case.get('unsupported_metal', False))
            cases.append({'name': name, 'passed': True, 'result': descriptor(result), 'images': descriptor(output / (name + '.images.json')), 'comparison': comparison})
            atomic_json(output / 'cases.json', cases)
        frozen(); r['automatic_packaged_resolution_gates_passed'] = len(cases) == len(p['cases'])
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release(); r['storage'] = guard.record
        atomic_json(output / 'qualification.json', r)


if __name__ == '__main__':
    main()
