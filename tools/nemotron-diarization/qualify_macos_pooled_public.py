"""Owned-pool CPU public Rust parity and seven real-model lifecycle checks.

Historical receiver code remains frozen. This separate consumer admits exact
pooled construction/native deliveries and records a new current-source freeze.
No accuracy, long-session, mixed-model, app or release promotion follows.
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

from artifacts import matches
from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from compare_stream_pooled import compare
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from pooled_runtime import POOLED_PROFILE, load_profile, profile_provenance
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import CONTROL, cargo_executable, policy
from qualify_pooled_native import result_policy
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory, restore_archive, write_archive


def runtime_policies(denied, readonly, reference, results):
    denied = [Path(path).resolve() for path in denied]
    readonly = [Path(path).resolve() for path in readonly]
    standalone = result_policy(policy(denied_reads=[*denied, reference.resolve()], readonly=readonly), results)
    lifecycle = result_policy(policy(denied_reads=denied, readonly=readonly), results)
    return {'runtime': standalone, 'lifecycle': lifecycle, 'premodel': model_free_policy(standalone)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['producer-zip', 'producer-metadata', 'native-reference-zip', 'native-reference-metadata', 'assets', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name('macos-pooled-public.protocol.json'))
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac receiver required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh receiver output required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    output = args.output.resolve()
    inputs = [args.producer_zip.resolve(), args.producer_metadata.resolve(), args.native_reference_zip.resolve(),
              args.native_reference_metadata.resolve(), args.assets.resolve(strict=True)]
    if output.is_relative_to(repo) or any(p.is_relative_to(output) for p in inputs):
        parser.error('receiver output must be separate from checkout and delivery inputs')
    protocol_path = args.protocol.resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    p = json.loads(protocol_bytes)
    lock_path, lock = load_profile(POOLED_PROFILE)

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
        'pooled_public_runtime_passed': False, 'model_execution_admitted': False,
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
        if inference:
            # Trace only the target, after time/env launchers, as in the
            # separately qualified C ABI process. Preserve whole-process RSS.
            environment = dict(environment)
            trace = environment.pop('DYLD_PRINT_LIBRARIES')
            child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=' + trace, *argv]
        else:
            child = argv
        actual, parent_environment = sandboxed_command(child, profile, environment)
        command = {'name': name, 'argv': actual, 'cwd': str(output), 'environment': parent_environment,
                   'expected_returncode': expected, 'inference_admitted': inference}
        r['commands'].append(command)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=output, environment=parent_environment, stdout=stdout, stderr=stderr,
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
        if (not q['construction_and_negative_c_api_controls_passed'] or q['valid_models_loaded'] != 0
                or q['frontend_or_neural_calls'] != 0
                or q['source_commit'] != p['producer']['source_commit']):
            raise ValueError('qualified model-free producer receipt required')
        native_root = output / 'native-reference'
        r['native_reference_artifact'] = admit_delivery(args.native_reference_zip, args.native_reference_metadata,
                                                       p['native_reference'], native_root)
        native_qualification = json.loads((native_root / 'qualification.json').read_text())
        if (native_qualification['native_parity_passed'] is not True
                or native_qualification['source_commit'] != p['native_reference']['source_commit']
                or native_qualification['neural_commands_started'] != ['native-12s', 'native-60s']
                or any(c['returncode'] != 0 for c in native_qualification['commands'])):
            raise ValueError('completed pooled C ABI parity delivery required')
        origin = native_qualification['deliveries']['construction']
        if (origin['id'] != p['producer']['id'] or origin['workflow_run']['head_sha'] != p['producer']['source_commit']
                or origin['digest'] != 'sha256:' + p['producer']['zip']['sha256']):
            raise ValueError('native baseline used a different pooled construction')
        native_path = native_root / 'results/native-60s.json'
        native = json.loads(native_path.read_text())
        if (native['runtime_version'] != lock['runtime']['reported_version']
                or native['audio_samples'] != 960000 or native['retained_first_frame'] != 0
                or native['retained_rows'] != 6000 or native['native_frame_count'] != 6000
                or len(Path(str(native_path) + '.f32').read_bytes()) != 192000):
            raise ValueError('complete immutable 60-second pooled native reference required')
        work = output / 'work'
        work.mkdir()
        sdk, package = output / 'sdk', work / 'nemo'
        for stem, summary, destination in [('pooled-cpu-sdk', 'sdk_archive', sdk),
                                            ('pooled-cpu-runtime', 'package_archive', package)]:
            files = json.loads((producer / (stem + '-files.json')).read_text())
            if {k: v for k, v in files.items() if k != 'files'} != q[summary]:
                raise ValueError('producer filesystem archive summary differs')
            restore_archive(producer / files['archive'], files, destination)
        manifest = verify(package)  # Static otool/codesign inspection; no library execution.
        if manifest['runtime_profile'] != POOLED_PROFILE:
            raise ValueError('owned-pool runtime package required')
        atomic_json(output / 'package.json', manifest)
        host = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
                'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
                'xcode': subprocess.check_output(['/usr/bin/xcodebuild', '-version'], text=True)}
        atomic_json(output / 'host.json', host)
        if host['cpu'] not in p['allowed_receiver_cpus']:
            raise ValueError('receiver CPU is outside the frozen CPU host set')
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
            original = args.assets.resolve() / entry['name']
            if not matches(original, entry):
                raise ValueError('cached asset differs: ' + entry['name'])
            shutil.copy2(original, assets / entry['name'])
            if not matches(assets / entry['name'], entry):
                raise ValueError('copied asset differs: ' + entry['name'])
            acquisitions[entry['name']] = descriptor(assets / entry['name'])
            r['acquisition_attempts'][-1]['verified'] = acquisitions[entry['name']]
            persist()
        load_source = helper_source / 'load-probe.cpp'
        shutil.copy2(tools / 'macos_runtime_load_probe.cpp', load_source)
        if descriptor(load_source) != p['tools']['macos_runtime_load_probe.cpp']:
            raise ValueError('copied load helper differs')
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
        compile_policy.write_text(model_free_policy(policy(denied_reads=[assets, args.assets.resolve(), native_root],
            readonly=[sdk, package, producer, native_root, repo, helper_source, '/opt/homebrew'])))
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            if probe.read_bytes() != b'not a model':
                raise ValueError('format positive control failed')
        run('compile-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': [str(assets / a['name']) for a in chosen] + list(map(str, probes)),
                        'writes': [str(load_source)]})], compile_policy, compile_env)
        run('rustc-version', [str(Path(cargo).parent / 'rustc'), '-vV'], compile_policy, compile_env)
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-arch', 'arm64',
                 '-march=armv8.2-a+dotprod+fp16', '-mmacosx-version-min=26.0']
        run('load-helper-build', ['/usr/bin/c++', *flags, load_source,
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
        r['helper_archive'] = write_archive(binary, output, 'pooled-public-helpers')
        protected = {'sdk': inventory(sdk), 'package': inventory(package), 'helpers': inventory(binary),
                     'assets': {a['name']: descriptor(assets / a['name']) for a in chosen},
                     'helper_source': inventory(helper_source), 'native_reference': inventory(native_root),
                     'producer': inventory(producer)}
        atomic_json(output / 'build-inputs.json', {'source_pins': p['repository_files'], 'helper_sources': protected['helper_source'],
                    'compiled_helpers': protected['helpers'], 'compiler_environment': compile_env,
                    'model_or_frontend_execution_during_build': False})

        def unchanged():
            frozen_source()
            if 'reference' in protected and inventory(output / 'reference') != protected['reference']:
                raise ValueError('lifecycle reference changed')
            if (inventory(sdk) != protected['sdk'] or inventory(package) != protected['package']
                    or inventory(binary) != protected['helpers'] or inventory(helper_source) != protected['helper_source']
                    or inventory(native_root) != protected['native_reference'] or inventory(producer) != protected['producer']
                    or any(not matches(assets / a['name'], a) for a in chosen)):
                raise ValueError('receiver input changed after helper construction')

        home, temporary = output / 'home', output / 'tmp'
        home.mkdir()
        temporary.mkdir()
        results, reference = output / 'results', output / 'reference'
        results.mkdir()
        reference.mkdir()
        atomic_json(reference / 'ami.json', {'segments': [{'start': t['start_seconds'], 'end': t['end_seconds'],
            'speaker': t['speaker']} for t in native['turns']]})
        shutil.copy2(str(native_path) + '.f32', reference / 'ami.json.f32')
        protected['reference'] = inventory(reference)
        runtime_policy, lifecycle_policy, premodel_policy = (output / name for name in ['runtime.sb', 'lifecycle.sb', 'premodel.sb'])
        denied = [sdk, producer, native_root, repo, target, helper_source, '/opt/homebrew', cargo_home, rustup_home,
                  args.producer_zip.resolve().parent, args.native_reference_zip.resolve().parent, args.assets.resolve()]
        readonly = [work, sdk, producer, native_root, repo, helper_source, reference]
        for name, body in runtime_policies(denied, readonly, reference, results).items():
            (output / (name + '.sb')).write_text(body)
        runtime_env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/',
                       'LC_ALL': 'C', 'DYLD_PRINT_LIBRARIES': '1'}
        control = work / 'read-only-control.txt'
        control.write_bytes(b'not a model')
        target_control = target / 'runtime-denial-control.txt'
        target_control.write_bytes(b'not a model')
        prior_control = results / 'prior-output-control.txt'
        prior_control.write_bytes(b'not a model')
        denied_controls = [sdk / 'include/nemo_speech/diar.h', producer / 'protocol.json', repo / 'Cargo.toml',
                           target_control, load_source, native_path, prior_control, reference / 'ami.json.f32',
                           '/opt/homebrew/bin/brew']
        if not all(Path(path).is_file() for path in denied_controls):
            raise ValueError('runtime sandbox positive read controls missing')
        for path in denied_controls:
            Path(path).read_bytes()
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, denied_controls)), 'writes': [str(control)]})], runtime_policy, runtime_env)
        if control.read_bytes() != b'not a model':
            raise ValueError('runtime readonly control was modified')
        run('lifecycle-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': [str(path) for path in denied_controls if path != reference / 'ami.json.f32'],
                        'writes': [str(control), str(reference / 'ami.json.f32')]})], lifecycle_policy, runtime_env)
        allowed = run('lifecycle-reference-readable', ['/usr/bin/python3', '-c',
            'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())',
            reference / 'ami.json.f32'], lifecycle_policy, runtime_env)
        if allowed.strip() != descriptor(reference / 'ami.json.f32')['sha256']:
            raise ValueError('lifecycle assertion reference is not readable as frozen')
        version = run('version-only', [binary / 'load-probe', package / 'lib/libnemo_speech_asr_c.dylib',
                                      lock['runtime']['reported_version']], premodel_policy, runtime_env, runtime=True)
        if version.strip() != lock['runtime']['reported_version']:
            raise ValueError('receiver C ABI version differs')
        missing = output / 'missing-runtime'
        shutil.copytree(package, missing, symlinks=True)
        missing_name = p['missing_library_name']
        (missing / 'lib' / missing_name).unlink()
        stdout = run('missing-library', [binary / 'load-probe', missing / 'lib/libnemo_speech_asr_c.dylib',
            lock['runtime']['reported_version']], premodel_policy, runtime_env, expected=1)
        stderr = (output / 'missing-library.stderr.log').read_text()
        if stdout or 'Library not loaded:' not in stderr or missing_name not in stderr:
            raise ValueError('specified missing dependency did not cause rejection')
        listed = run('lifecycle-list', [binary / 'native-lifecycle', '--ignored', '--list'],
                     premodel_policy, runtime_env, runtime=False)
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
        public = json.loads(run('public', [binary / 'public-stream', assets,
            package / 'lib/libnemo_speech_asr_c.dylib', wav_path, '1', results / 'public.f32'],
            runtime_policy, runtime_env, runtime=True, inference=True))
        atomic_json(results / 'public.json', public)
        for record in [public, native]:
            if record['input_pcm_f32le_sha256'] != p['pcm_sha256'] or record['input_repeats'] != 1:
                raise ValueError('actual receiver PCM differs')
        if public['retained_probabilities'] != {'first_frame': 0, 'rows': 6000} or native['retained_rows'] != 6000:
            raise ValueError('complete receiver probability histories required')
        r['comparison'] = compare(public, native, Path(str(native_path) + '.f32').read_bytes(), (results / 'public.f32').read_bytes())
        life_env = {**runtime_env, 'GEN2_NEMOTRON_ASSETS': str(assets),
                    'GEN2_NEMO_LIBRARY': str(package / 'lib/libnemo_speech_asr_c.dylib'),
                    'GEN2_NEMOTRON_REFERENCE': str(reference), 'GEN2_NEMO_EXECUTION': 'cpu'}
        r['lifecycle_tests_passed'] = []
        for index, name in enumerate(p['lifecycle_tests']):
            unchanged()
            text = run('lifecycle-' + str(index), [binary / 'native-lifecycle', '--ignored', '--exact', name,
                '--test-threads=1'], lifecycle_policy, life_env,
                runtime=name != 'verified_provisioned_cache_is_reused_offline',
                inference=name != 'verified_provisioned_cache_is_reused_offline')
            if (not re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', text, re.MULTILINE)
                    or '1 passed; 0 failed; 0 ignored;' not in text):
                raise ValueError('required lifecycle test did not pass: ' + name)
            r['lifecycle_tests_passed'].append(name)
            persist()
        unchanged()
        r['pooled_public_runtime_passed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release()
        r['storage'] = guard.record if guard is not None else {'admission_failed_before_child': True}
        persist()
    print(json.dumps({'pooled_public_runtime_passed': True, 'comparison': r['comparison']}))


if __name__ == '__main__':
    main()
