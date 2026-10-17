"""Offline complete pooled CPU app recording/live checks with fresh C ABI references.

Automatic packaged artifact resolution; CLI recording and paced live only.
No physical microphone, GUI, fresh machine, accuracy or public-signing claim.
"""
import argparse
import array
import json
import math
import platform
from pathlib import Path
import subprocess

from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from pooled_pnola_metrics import live_matches, native_reference, recorded_matches
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_pooled_mixed import mixed_policy, mixed_runtime_policy, sandbox_path
from qualify_pooled_native import NEW_VERSION
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory
from verify_pooled_pnola_app import verify_app
from verify_valid_mel_pnola_app import audit_app_images
from qualify_macos_portable_runtime import CONTROL

CASES = ['native-60s', 'native-80s', 'cpu-recorded', 'cpu-live', 'cpu-rejects-metal']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)
    output, work, app = [Path(p[name]).resolve(strict=True) for name in ['output_parent', 'work_path', 'app_path']]
    if Path(p['output_name']).name != p['output_name']:
        raise ValueError('single fresh output directory name required')
    output = output / p['output_name']
    if output.exists() or output.is_symlink() or p['cases'] != CASES:
        raise ValueError('fresh output and exact five neural cases required')
    if output.is_relative_to(work) or work.is_relative_to(output) or output.is_relative_to(app):
        raise ValueError('app/work/output must be disjoint')

    def frozen():
        if args.protocol.read_bytes() != protocol_bytes:
            raise ValueError('pooled app protocol changed')
        for records, root in [(p['tools'], tools), (p['repository_files'], repo)]:
            for name, expected in records.items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen pooled app source changed: ' + name)
        for root, expected in [(work, p['work_inventory']), (app, p['app_inventory'])]:
            if inventory(root) != expected:
                raise ValueError('pooled app or staged input inventory changed')
        for name, expected in p['preparation_records'].items():
            if descriptor(Path(name)) != expected:
                raise ValueError('pooled app preparation receipt changed')

    frozen()
    verified = verify_app(app)
    package = work / 'nemo'
    manifest = verify(package)
    if (manifest['runtime_profile'] != 'gen2-valid-mel-pool-v1' or manifest['backend'] != 'cpu'
            or manifest != verify(app / 'Contents/Resources/nemo')):
        raise ValueError('native and complete app package differ')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_bytes)
    results = output / 'results'
    results.mkdir()
    home, temporary = output / 'home', output / 'tmp'
    home.mkdir()
    temporary.mkdir()
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}
    runtime = output / 'runtime.sb'
    denied = [Path(name) for name in p['denied_paths']]
    runtime.write_text(mixed_runtime_policy(denied, [work, app, *denied], results))
    premodel = output / 'premodel.sb'
    premodel.write_text(model_free_policy(runtime.read_text()) + '(deny file-read* '
        + ' '.join('(subpath ' + sandbox_path(path) + ')' for path in
                   [work / 'assets', app / 'Contents/Resources/bark', app / 'Contents/Resources/nemotron']) + ')\n')
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'automatic_packaged_resolution_gates_passed': False, 'release_qualified': False,
         'scope': __doc__, 'protocol': descriptor(args.protocol), 'commands': [], 'cases': [],
         'neural_commands_started': [], 'loaded_images': {}, 'microphone_used': False,
         'artifact_environment_overrides_used': False}
    guard = None
    persist = lambda: atomic_json(output / 'qualification.json', r)

    def run(name, argv, *, sandbox=runtime, inference=False, trace=False, variables=env):
        frozen()
        guard.before_model()
        if inference:
            r['neural_commands_started'].append(name)
        child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)] if trace else list(map(str, argv))
        actual, environment = sandboxed_command(child, sandbox, variables)
        row = {'name': name, 'argv': actual, 'cwd': str(output), 'environment': environment, 'inference_admitted': inference}
        r['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=output, environment=environment, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
        if row['returncode'] != 0:
            raise ValueError('pooled complete app command failed: ' + name)
        return (output / (name + '.stdout.log')).read_text()

    try:
        guard = StorageGuard(output, **p['storage'])
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        prior = results / 'prior-output-control.txt'
        for path in [*probes, prior]:
            path.write_bytes(b'not a model')
            path.read_bytes()
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps({
            'reads': [str(repo / 'Cargo.toml'), str(prior)], 'writes': [str(work / 'assets/ami.wav')]})])
        run('premodel-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps({
            'reads': [*map(str, probes), str(work / 'assets/model.gguf')], 'writes': []})], sandbox=premodel)
        version = run('version-only', [work / 'bin/load-probe', package / 'lib/libnemo_speech_asr_c.dylib', NEW_VERSION], sandbox=premodel, trace=True)
        if version.strip() != NEW_VERSION:
            raise ValueError('pooled complete app native version differs')
        r['loaded_images']['version-only'] = audit_loaded_libraries((output / 'version-only.stderr.log').read_text(), package, manifest, work / 'bin/load-probe')
        natives = {}
        for name, wav, duration in [('native-60s', 'ami.wav', 60.), ('native-80s', 'live-80s.wav', 80.04)]:
            destination = results / (name + '.json')
            run(name, [work / 'bin/native-stream', work / 'assets/model.gguf', work / 'assets' / wav, '1', '-1', destination], inference=True, trace=True)
            value = json.loads(destination.read_text())
            native_reference(value, duration)
            if value['input_pcm_f32le_sha256'] != p['pcm_sha256'][wav]:
                raise ValueError('independent native PCM identity differs')
            probabilities = array.array('f')
            probabilities.frombytes(Path(str(destination) + '.f32').read_bytes())
            if (len(probabilities) != round(duration * 100) * 8
                    or not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)):
                raise ValueError('independent native probability coverage differs')
            r['loaded_images'][name] = audit_loaded_libraries((output / (name + '.stderr.log')).read_text(), package, manifest, work / 'bin/native-stream')
            natives[name] = value
            r['cases'].append({'name': name, 'passed': True, 'result': descriptor(destination), 'probabilities': descriptor(Path(str(destination) + '.f32'))})
            persist()
        executable = app / 'Contents/MacOS/pnola-desktop'
        for name in CASES[2:]:
            result = results / (name + '.json')
            live = name == 'cpu-live'
            unsupported = name == 'cpu-rejects-metal'
            variables = {**env, **({'GEN2_NEMO_EXECUTION': 'metal'} if unsupported else {})}
            run(name, [executable, '--verify-live-speakers' if live else '--verify-bundle', work / 'assets/ami.wav', result], inference=True, variables=variables)
            images = audit_app_images((output / (name + '.stderr.log')).read_text(), app, verified)
            if not images['process_local_add_image_history_observed']:
                raise ValueError('complete app process-local image history required')
            r['loaded_images'][name] = images
            value = json.loads(result.read_text())
            comparison = (live_matches(value, natives['native-80s']) if live else
                recorded_matches(value, natives['native-60s'], p['bark_text'], p['wav_sha256'], unavailable=unsupported))
            r['cases'].append({'name': name, 'passed': True, 'result': descriptor(result), 'comparison': comparison})
            persist()
        frozen()
        r['automatic_packaged_resolution_gates_passed'] = [case['name'] for case in r['cases']] == CASES
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__':
    main()
