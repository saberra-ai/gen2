"""Fresh Mac CPU receiver: exact delivery, offline native/public parity and lifecycle.

This is a fixed smoke/runtime qualification, not held-out quality, an app or a
public release. No source SDK or Homebrew dependency is visible to inference.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import wave

from artifacts import acquire, matches
from build_valid_mel_candidate import descriptor
from compare_stream_valid_mel import compare
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json
from qualify_valid_mel_pnola import execute_guarded
from runtime_profiles import load_profile, profile_provenance
from sdk_archive import inventory, restore_archive, write_archive


CONTROL = '''import json, pathlib, socket, sys
def denied(operation):
    try:
        operation()
    except PermissionError:
        return
    raise RuntimeError("sandbox operation unexpectedly permitted")
cases = json.loads(sys.argv[1])
for name in cases["reads"]:
    denied(lambda name=name: pathlib.Path(name).read_bytes())
for name in cases["writes"]:
    denied(lambda name=name: pathlib.Path(name).write_bytes(b"changed"))
denied(lambda: socket.create_connection(("127.0.0.1", 9), timeout=1))
'''


def policy(*, denied_reads, readonly):
    return ('(version 1)\n(allow default)\n(deny network*)\n'
            + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(str(p)) + ')' for p in denied_reads) + ')\n'
            + '(deny file-write* ' + ' '.join('(subpath ' + json.dumps(str(p)) + ')' for p in readonly) + ')\n')


def cargo_executable(log, name):
    records = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
    values = [r['executable'] for r in records if r.get('reason') == 'compiler-artifact'
              and r['target']['name'] == name and r.get('executable')]
    if len(values) != 1:
        raise ValueError('unique compiled Cargo executable missing: ' + name)
    return Path(values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['producer-zip', 'producer-metadata', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac receiver required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh receiver output required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    output = args.output.resolve()
    inputs = [args.producer_zip.resolve(), args.producer_metadata.resolve()]
    if output.is_relative_to(repo) or any(p.is_relative_to(output) for p in inputs):
        parser.error('receiver output must be separate from checkout and delivery inputs')
    protocol_path = tools / 'macos-portable-runtime.protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    p = json.loads(protocol_bytes)
    lock_path, lock = load_profile()

    def frozen_source():
        if protocol_path.read_bytes() != protocol_bytes or p['runtime_provenance'] != profile_provenance(lock_path, lock):
            raise ValueError('receiver protocol/runtime profile changed')
        for key, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[key].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen receiver source changed: ' + name)

    frozen_source()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    shutil.copy2(args.producer_metadata, output / 'producer-artifact.json')
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'protocol': descriptor(protocol_path),
        'fixed_receiver_runtime_passed': False, 'model_execution_admitted': False,
        'inference_commands_started': [], 'release_qualified': False, 'commands': [], 'loaded_images': {},
        'limits': p['limits']}
    guard = None
    persist = lambda: atomic_json(output / 'qualification.json', r)

    def run(name, argv, profile, environment, *, expected=0, runtime=None, inference=False):
        if guard is None:
            raise ValueError('storage guard missing')
        frozen_source()
        guard.before_model()
        if inference:
            r['inference_commands_started'].append(name)
        actual = ['/usr/bin/sandbox-exec', '-f', str(profile), *map(str, argv)]
        command = {'name': name, 'argv': actual, 'cwd': str(output), 'environment': environment,
                   'expected_returncode': expected, 'inference_admitted': inference}
        r['commands'].append(command)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=output, environment=environment, stdout=stdout, stderr=stderr,
                            guard=guard, record=command, persist=persist, timeout=p['command_timeout_seconds'])
        persist()
        if command['returncode'] != expected:
            raise ValueError('receiver command failed: ' + name)
        text = (output / (name + '.stdout.log')).read_text()
        if runtime is not None:
            r['loaded_images'][name] = audit_loaded_libraries(
                (output / (name + '.stderr.log')).read_text(), package, manifest, argv[0], runtime)
            persist()
        return text

    try:
        guard = StorageGuard(output, **p['storage'])
        producer = output / 'producer'
        r['producer_artifact'] = admit_delivery(args.producer_zip, args.producer_metadata, p['producer'], producer)
        q = json.loads((producer / 'qualification.json').read_text())
        if (not q['construction_and_package_passed'] or q['model_calls'] != 0
                or q['source_commit'] != p['producer']['source_commit']):
            raise ValueError('qualified model-free producer receipt required')
        work = output / 'work'
        work.mkdir()
        sdk, package = output / 'sdk', work / 'nemo'
        for stem, summary, destination in [('portable-cpu-sdk', 'sdk_archive', sdk),
                                            ('portable-cpu-runtime', 'package_archive', package)]:
            files = json.loads((producer / (stem + '-files.json')).read_text())
            if {k: v for k, v in files.items() if k != 'files'} != q[summary]:
                raise ValueError('producer filesystem archive summary differs')
            restore_archive(producer / files['archive'], files, destination)
        manifest = verify(package)  # Static otool/codesign inspection; no library execution.
        atomic_json(output / 'package.json', manifest)
        host = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
                'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
                'xcode': subprocess.check_output(['/usr/bin/xcodebuild', '-version'], text=True)}
        atomic_json(output / 'host.json', host)
        if host['cpu'] != p['expected_receiver_cpu']:
            raise ValueError('receiver CPU differs from frozen M1 virtual baseline')
        assets, helper_source, binary = work / 'assets', output / 'helper-source', work / 'bin'
        for folder in [assets, helper_source, binary]:
            folder.mkdir()
        chosen = [a for a in lock['artifacts'] if a['name'] in p['asset_names']]
        if {a['name'] for a in chosen} != set(p['asset_names']):
            raise ValueError('receiver assets differ from runtime profile')
        acquisitions = {}
        r['acquisition_attempts'] = []
        for entry in chosen:
            guard.before_model()
            r['acquisition_attempts'].append({'name': entry['name'], 'expected': entry})
            persist()
            acquire(entry, assets)
            acquisitions[entry['name']] = descriptor(assets / entry['name'])
            r['acquisition_attempts'][-1]['verified'] = acquisitions[entry['name']]
            persist()
        for relative, entry in p['helper_inputs'].items():
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('unsafe helper source name')
            destination = helper_source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            r['acquisition_attempts'].append({'name': 'helper/' + relative, 'expected': entry})
            persist()
            acquire({**entry, 'name': destination.name}, destination.parent)
            acquisitions['helper/' + relative] = descriptor(destination)
            r['acquisition_attempts'][-1]['verified'] = acquisitions['helper/' + relative]
            persist()
        atomic_json(output / 'acquisitions.json', acquisitions)
        wav_path = assets / 'ami_en2002d_2132.wav'
        with wave.open(str(wav_path), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, 960000):
                raise ValueError('receiver PCM geometry differs')
            import hashlib
            pcm = b''.join(struct.pack('<f', n[0] / 32768) for n in struct.iter_unpack('<h', wav.readframes(960000)))
            if hashlib.sha256(pcm).hexdigest() != p['pcm_sha256']:
                raise ValueError('receiver PCM bytes differ')
        target = output / 'cargo-target'
        cargo = shutil.which('cargo')
        if not cargo:
            raise ValueError('Cargo bootstrap missing')
        cargo_home = Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo'))).resolve()
        rustup_home = Path(os.environ.get('RUSTUP_HOME', str(Path.home() / '.rustup'))).resolve()
        compile_env = {'PATH': str(Path(cargo).parent) + ':/usr/bin:/bin:/usr/sbin:/sbin',
                       'HOME': str(Path.home()), 'CARGO_HOME': str(cargo_home), 'RUSTUP_HOME': str(rustup_home),
                       'CARGO_TARGET_DIR': str(target), 'LC_ALL': 'C',
                       'RUSTFLAGS': '-C target-cpu=generic -C link-arg=-mmacosx-version-min=26.0',
                       'MACOSX_DEPLOYMENT_TARGET': '26.0', 'CC': '/usr/bin/clang', 'CXX': '/usr/bin/clang++',
                       'CFLAGS': '-march=armv8.2-a+dotprod+fp16', 'CXXFLAGS': '-march=armv8.2-a+dotprod+fp16'}
        compile_policy = output / 'compile.sb'
        compile_policy.write_text(policy(denied_reads=[assets], readonly=[sdk, package, producer, repo, '/opt/homebrew']))
        run('compile-controls', [sys.executable, '-c', CONTROL,
            json.dumps({'reads': [str(assets / a['name']) for a in chosen], 'writes': []})], compile_policy, compile_env)
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-arch', 'arm64',
                 '-march=armv8.2-a+dotprod+fp16', '-mmacosx-version-min=26.0']
        run('native-helper-build', ['/usr/bin/c++', *flags, '-x', 'c++', tools / 'native_stream_probe_valid_mel.cpp',
            helper_source / 'sha256/sha256.c', '-I' + str(sdk / 'include'), '-I' + str(helper_source),
            '-L' + str(package / 'lib'), '-lnemo_speech_asr_c', '-Wl,-rpath,@loader_path/../nemo/lib',
            '-o', binary / 'native-stream'], compile_policy, compile_env)
        run('load-helper-build', ['/usr/bin/c++', *flags, tools / 'macos_runtime_load_probe.cpp',
                                 '-o', binary / 'load-probe'], compile_policy, compile_env)
        common = [cargo, '--offline', '--locked', '--release', '--no-default-features',
                  '--features', 'backend-diar-nemo', '--jobs', '2', '--message-format=json']
        # Cargo runs from output; the explicit manifest still resolves the frozen checkout.
        command = [common[0], 'test', *common[1:], '--manifest-path', repo / 'Cargo.toml',
                   '--test', 'diarization_native', '--no-run']
        log = run('lifecycle-build', command, compile_policy, compile_env)
        shutil.copy2(cargo_executable(log, 'diarization_native'), binary / 'native-lifecycle')
        log = run('public-helper-build', [common[0], 'build', *common[1:], '--manifest-path', repo / 'Cargo.toml',
                                         '--example', 'nemotron_valid_mel_stream'], compile_policy, compile_env)
        shutil.copy2(cargo_executable(log, 'nemotron_valid_mel_stream'), binary / 'public-stream')
        load_commands = {}
        for path in sorted(binary.iterdir()):
            load_commands[path.name] = {'binary': descriptor(path),
                'otool': subprocess.check_output(['/usr/bin/otool', '-l', str(path)], text=True),
                'architectures': subprocess.check_output(['/usr/bin/lipo', '-archs', str(path)], text=True).strip()}
            if load_commands[path.name]['architectures'] != 'arm64':
                raise ValueError('receiver helper architecture differs')
        atomic_json(output / 'helper-load-commands.json', load_commands)
        r['helper_archive'] = write_archive(binary, output, 'receiver-helpers')
        protected = {'sdk': inventory(sdk), 'package': inventory(package), 'helpers': inventory(binary),
                     'assets': {a['name']: descriptor(assets / a['name']) for a in chosen}}
        atomic_json(output / 'build-inputs.json', {'source_pins': p['repository_files'], 'helper_inputs': p['helper_inputs'],
                    'compiled_helpers': protected['helpers'], 'compiler_environment': compile_env,
                    'model_or_frontend_execution_during_build': False})

        def unchanged():
            frozen_source()
            if (inventory(sdk) != protected['sdk'] or inventory(package) != protected['package']
                    or inventory(binary) != protected['helpers'] or any(not matches(assets / a['name'], a) for a in chosen)):
                raise ValueError('receiver input changed after helper construction')

        home, temporary = output / 'home', output / 'tmp'
        home.mkdir()
        temporary.mkdir()
        runtime_policy = output / 'runtime.sb'
        denied = [sdk, producer, repo, target, helper_source, '/opt/homebrew', cargo_home, rustup_home,
                  args.producer_zip.resolve().parent]
        runtime_policy.write_text(policy(denied_reads=denied, readonly=[work, sdk, producer, repo]))
        runtime_env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/',
                       'LC_ALL': 'C', 'DYLD_PRINT_LIBRARIES': '1'}
        control = work / 'read-only-control.txt'
        control.write_bytes(b'not a model')
        target_control = target / 'runtime-denial-control.txt'
        target_control.write_bytes(b'not a model')
        denied_controls = [sdk / 'include/nemo_speech/diar.h', producer / 'protocol.json', repo / 'Cargo.toml',
                           target_control, helper_source / 'wav_reader.h', '/opt/homebrew/bin/brew']
        if not all(Path(path).is_file() for path in denied_controls):
            raise ValueError('runtime sandbox positive read controls missing')
        for path in denied_controls:
            Path(path).read_bytes()
        run('runtime-controls', [sys.executable, '-c', CONTROL,
            json.dumps({'reads': list(map(str, denied_controls)), 'writes': [str(control)]})], runtime_policy, runtime_env)
        if control.read_bytes() != b'not a model':
            raise ValueError('runtime readonly control was modified')
        version = run('version-only', [binary / 'load-probe', package / 'lib/libnemo_speech_asr_c.dylib',
                                      lock['runtime']['reported_version']], runtime_policy, runtime_env, runtime=True)
        if version.strip() != lock['runtime']['reported_version']:
            raise ValueError('receiver C ABI version differs')
        missing = output / 'missing-runtime'
        shutil.copytree(package, missing, symlinks=True)
        missing_name = p['missing_library_name']
        (missing / 'lib' / missing_name).unlink()
        stdout = run('missing-library', [binary / 'load-probe', missing / 'lib/libnemo_speech_asr_c.dylib',
            lock['runtime']['reported_version']], runtime_policy, runtime_env, expected=1)
        stderr = (output / 'missing-library.stderr.log').read_text()
        if stdout or 'Library not loaded:' not in stderr or missing_name not in stderr:
            raise ValueError('specified missing dependency did not cause rejection')
        listed = run('lifecycle-list', [binary / 'native-lifecycle', '--ignored', '--list'],
                     runtime_policy, runtime_env, runtime=False)
        names = {line.removesuffix(': test') for line in listed.splitlines() if line.endswith(': test')}
        if not set(p['lifecycle_tests']) <= names:
            raise ValueError('required compiled lifecycle tests are missing')
        unchanged()
        r['model_execution_admitted'] = True
        persist()
        run('cpu-rejects-metal', [binary / 'public-stream', '--metal', assets,
            package / 'lib/libnemo_speech_asr_c.dylib', wav_path, '1', output / 'unsupported.f32'],
            runtime_policy, runtime_env, expected=1, runtime=True, inference=True)
        if 'requested Metal device is unavailable in the native SDK' not in (output / 'cpu-rejects-metal.stderr.log').read_text():
            raise ValueError('CPU package did not explicitly reject Metal')
        unchanged()
        results = output / 'results'
        results.mkdir()
        native_path = results / 'native.json'
        run('native', [binary / 'native-stream', assets / 'Nemotron-3-Diarization.q8_0.gguf', wav_path,
            '1', '-1', native_path], runtime_policy, runtime_env, runtime=True, inference=True)
        unchanged()
        public = json.loads(run('public', [binary / 'public-stream', assets,
            package / 'lib/libnemo_speech_asr_c.dylib', wav_path, '1', results / 'public.f32'],
            runtime_policy, runtime_env, runtime=True, inference=True))
        atomic_json(results / 'public.json', public)
        native = json.loads(native_path.read_text())
        for record in [public, native]:
            if record['input_pcm_f32le_sha256'] != p['pcm_sha256'] or record['input_repeats'] != 1:
                raise ValueError('actual receiver PCM differs')
        if public['retained_probabilities'] != {'first_frame': 0, 'rows': 6000} or native['retained_rows'] != 6000:
            raise ValueError('complete receiver probability histories required')
        r['comparison'] = compare(public, native, Path(str(native_path) + '.f32').read_bytes(), (results / 'public.f32').read_bytes())
        reference = output / 'reference'
        reference.mkdir()
        atomic_json(reference / 'ami.json', {'segments': [{'start': t['start_seconds'], 'end': t['end_seconds'],
            'speaker': t['speaker']} for t in native['turns']]})
        shutil.copy2(str(native_path) + '.f32', reference / 'ami.json.f32')
        life_env = {**runtime_env, 'GEN2_NEMOTRON_ASSETS': str(assets),
                    'GEN2_NEMO_LIBRARY': str(package / 'lib/libnemo_speech_asr_c.dylib'),
                    'GEN2_NEMOTRON_REFERENCE': str(reference), 'GEN2_NEMO_EXECUTION': 'cpu'}
        r['lifecycle_tests_passed'] = []
        for index, name in enumerate(p['lifecycle_tests']):
            unchanged()
            text = run('lifecycle-' + str(index), [binary / 'native-lifecycle', '--ignored', '--exact', name,
                '--test-threads=1'], runtime_policy, life_env,
                runtime=name != 'verified_provisioned_cache_is_reused_offline',
                inference=name != 'verified_provisioned_cache_is_reused_offline')
            if (not re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', text, re.MULTILINE)
                    or '1 passed; 0 failed; 0 ignored;' not in text):
                raise ValueError('required lifecycle test did not pass: ' + name)
            r['lifecycle_tests_passed'].append(name)
            persist()
        unchanged()
        r['fixed_receiver_runtime_passed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release()
        r['storage'] = guard.record if guard is not None else {'admission_failed_before_child': True}
        persist()
    print(json.dumps({'fixed_receiver_runtime_passed': True, 'comparison': r['comparison']}))


if __name__ == '__main__':
    main()
