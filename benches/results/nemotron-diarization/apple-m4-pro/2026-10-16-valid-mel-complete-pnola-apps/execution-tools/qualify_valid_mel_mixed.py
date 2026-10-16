"""Guarded Mac CPU/Metal diarization residency smoke with actual Bark/Laya/chat."""
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import time

from build_valid_mel_candidate import descriptor
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('mixed qualification requires Apple Silicon macOS')
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    protocol_path = tools / 'valid-mel-mixed.protocol.json'
    protocol_bytes = protocol_path.read_bytes(); p = json.loads(protocol_bytes)
    work = Path(p['work_path']); output = Path(p['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError('a fresh mixed qualification output is required')
    def frozen():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError('mixed qualification protocol changed')
        for records, root in [(p['tools'], tools), (p['repository_files'], repo), (p['files'], work)]:
            for name, expected in records.items():
                if descriptor(root / name) != expected:
                    raise ValueError('mixed qualification input differs: ' + name)
        actual_files = {str(path.relative_to(work)) for path in work.rglob('*') if path.is_file() and not path.is_symlink()}
        actual_aliases = {str(path.relative_to(work)): str(path.readlink()) for path in work.rglob('*') if path.is_symlink()}
        if actual_files != set(p['files']) or actual_aliases != p['aliases']:
            raise ValueError('mixed input filesystem differs from frozen inventory')
        for name, target in p['aliases'].items():
            path = work / name
            if not path.is_symlink() or str(path.readlink()) != target:
                raise ValueError('mixed qualification alias differs: ' + name)
        if descriptor(Path(p['build_record_path'])) != p['build_record']:
            raise ValueError('mixed helper preparation receipt differs')
    frozen()
    manifests = {backend: verify(work / backend / 'nemo') for backend in ['cpu', 'metal']}
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    base_policy = '(version 1)\n(allow default)\n(deny network*)\n(deny file-write* (subpath ' + json.dumps(str(work)) + '))\n'
    base_policy += '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n'
    regular_policy = output / 'runtime.sb'; regular_policy.write_text(base_policy)
    listing_policy = output / 'list.sb'
    listing_policy.write_text(base_policy + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(str(work / path)) + ')' for path in ['models', 'cpu', 'metal', 'providers']) + ')\n')
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ['HOME'], 'TMPDIR': os.environ.get('TMPDIR', '/tmp'),
           'LC_ALL': 'C'}
    helper = work / 'bin/mixed-native-tests'; commands = []; cases = []; guard = None; process = None
    result = {'schema_version': 1, 'release_qualified': False, 'mixed_residency_smoke_passed': False,
              'scope': p['scope'], 'protocol': descriptor(protocol_path), 'cases': cases}
    def run(name, argv, environment):
        nonlocal process
        guard.before_model()
        record = {'name': name, 'argv': argv, 'environment': environment}; commands.append(record)
        atomic_json(output / 'commands.json', commands)
        started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=output, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
                record['pid'] = process.pid; atomic_json(output / 'commands.json', commands)
                record['returncode'] = guard.wait(process, record, p['command_timeout_seconds'])
        except OSError as error:
            record['startup_error'] = str(error)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(output / 'commands.json', commands)
        if record['returncode'] != 0:
            raise RuntimeError('mixed qualification command failed: ' + name)
        return (output / (name + '.stdout.log')).read_text(), (output / (name + '.stderr.log')).read_text()
    try:
        result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        guard = StorageGuard(output, **p['storage'])
        stdout, stderr = run('list', ['/usr/bin/sandbox-exec', '-f', str(listing_policy), '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', str(helper), '--ignored', '--list'], env)
        listed = {line.removesuffix(': test') for line in stdout.splitlines() if line.endswith(': test')}
        if listed != set(p['compiled_test_names']):
            raise ValueError('mixed helper did not compile all expected real/synthetic fixture selectors')
        listing = audit_loaded_libraries(stderr, work / 'cpu/nemo', manifests['cpu'], helper, require_runtime=False)
        atomic_json(output / 'list-images.json', listing)
        for case in p['cases']:
            frozen()
            backend = case['backend']; name = backend + '-' + case['test']
            case_env = {**env, 'GEN2_NEMO_EXECUTION': backend,
                        'GEN2_NEMOTRON_ASSETS': str(work / 'models/nemotron'),
                        'GEN2_NEMO_LIBRARY': str(work / backend / 'nemo/lib/libnemo_speech_asr_c.dylib'),
                        'GEN2_BARK_BUNDLE': str(work / 'models/bark'),
                        'GEN2_ORT_LIBRARY': str(work / 'providers/libonnxruntime.dylib'),
                        'GEN2_CHAT_GGUF': str(work / 'models/chat.gguf'),
                        'GEN2_LAYA_BUNDLE': str(work / 'models/laya'), 'GEN2_CHAT_EXPECT_BACKEND': 'MTL'}
            stdout, stderr = run(name, ['/usr/bin/sandbox-exec', '-f', str(regular_policy), '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', str(helper), '--ignored', '--exact', case['test'], '--nocapture'], case_env)
            placement = {}
            for line in stdout.splitlines():
                if line.startswith('chat-placement '):
                    phase, value = line.removeprefix('chat-placement ').split(': ', 1); placement[phase] = json.loads(value)
            if ('1 passed; 0 failed; 0 ignored;' not in stdout or set(placement) != {'initial', 'restored'}
                    or any(value['layers'] <= 0 or value['backend'] != 'MTL' or value['device'] != p['expected_chat_device'] for value in placement.values())):
                raise ValueError('actual mixed test or public chat placement did not pass')
            images = audit_with_auxiliary(stderr, work / backend / 'nemo', manifests[backend], helper,
                         {str(work / 'providers/libonnxruntime.dylib'): p['files']['providers/libonnxruntime.dylib']})
            if images['runtime']['private_library_count'] != p['expected_runtime_images'][backend]:
                raise ValueError('mixed runtime closure count differs')
            atomic_json(output / (name + '.images.json'), images)
            cases.append({'name': name, 'backend': backend, 'passed': True, 'chat_placement': placement,
                          'loaded_images': descriptor(output / (name + '.images.json'))})
            atomic_json(output / 'cases.json', cases)
        frozen()
        if len(cases) != 4:
            raise ValueError('all four mixed residency tests required')
        result['mixed_residency_smoke_passed'] = True
    except BaseException as error:
        result['error'] = str(error)
        if process is not None and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGKILL); process.wait()
            if commands:
                commands[-1]['returncode'] = process.returncode
                atomic_json(output / 'commands.json', commands)
        raise
    finally:
        if guard is not None:
            guard.release(); result['storage'] = guard.record
        atomic_json(output / 'qualification.json', result)


if __name__ == '__main__':
    main()
