"""Fresh pooled CPU residency checks with actual Bark, Laya and Metal chat.

Run each existing real-Laya load-order test once. Tiny silence exercises
residency, refusal, eviction and cleanup; it cannot qualify speech accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from pooled_runtime import POOLED_PROFILE, load_profile, profile_provenance
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import CONTROL, cargo_executable
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory, restore_archive, write_archive


def sandbox_path(path):
    return json.dumps(str(Path(path).resolve()), ensure_ascii=False)


def mixed_policy(*, denied_reads, readonly):
    # Seatbelt does not interpret JSON's \u escapes as Unicode paths.
    return ('(version 1)\n(allow default)\n(deny network*)\n'
        + '(deny file-read* ' + ' '.join('(subpath ' + sandbox_path(x) + ')' for x in denied_reads) + ')\n'
        + '(deny file-write* ' + ' '.join('(subpath ' + sandbox_path(x) + ')' for x in readonly) + ')\n')


def mixed_runtime_policy(denied_reads, readonly, results):
    return mixed_policy(denied_reads=denied_reads, readonly=readonly) + (
        '(deny file-read-data (subpath ' + sandbox_path(results) + '))\n')


def compiler_policy(base, mel_filter):
    # This coefficient table is an include_bytes! build input, not weights.
    # Deny every other matching format, including another file named nemo128.
    return base + ('(deny file-read* (require-all '
        '(regex #".*\\.(nemo|gguf|safetensors|onnx|npy|f32)$") '
        '(require-not (literal ' + sandbox_path(mel_filter.resolve(strict=True)) + '))))\n')


def verify_vendor_worktree(vendor, revision):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vendor, text=True).strip() != revision
            or subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=vendor)
            or subprocess.check_output(['git', 'ls-files', '--others', '--ignored', '--exclude-standard', '-z'], cwd=vendor)):
        raise ValueError('mixed vendor revision or worktree differs')


def admit_builtin_mel_table(repo, protocol):
    name = 'assets/bark/nemo128.f32'
    expected = protocol['repository_files'].get(name)
    if expected is None or descriptor(repo / name) != expected:
        raise ValueError('required built-in mel coefficient source pin missing or changed')
    return repo / name


def compiler_tools_environment(cargo, records):
    names = {'cmake', 'ninja', 'pkg-config'}
    if set(records) != names:
        raise ValueError('exact compiler tool pins required')
    paths = {}
    for name in sorted(names):
        path = Path(records[name]['path'])
        if (not path.is_absolute() or path.resolve(strict=True) != path
                or descriptor(path) != records[name]['file'] or not os.access(path, os.X_OK)):
            raise ValueError('frozen compiler tool differs: ' + name)
        paths[name] = path
    directories = dict.fromkeys([str(Path(cargo).resolve().parent),
        *[str(paths[name].parent) for name in sorted(names)], '/usr/bin', '/bin', '/usr/sbin', '/sbin'])
    return {'PATH': ':'.join(directories), 'CMAKE': str(paths['cmake']),
            'PKG_CONFIG': str(paths['pkg-config']), 'CMAKE_BUILD_PARALLEL_LEVEL': '2'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'producer-zip', 'producer-metadata', 'public-helpers',
                 'public-helpers-manifest', 'public-qualification', 'source-models', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh mixed-runtime output required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    inputs = [getattr(args, name).resolve(strict=True) for name in ['producer_zip', 'producer_metadata',
              'public_helpers', 'public_helpers_manifest', 'public_qualification', 'source_models']]
    if output.is_relative_to(repo) or any(path.is_relative_to(output) for path in inputs):
        parser.error('output must be separate from source and inputs')
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)
    lock_path, lock = load_profile(POOLED_PROFILE)

    def frozen():
        admit_builtin_mel_table(repo, p)
        compiler_tools_environment(shutil.which('cargo') or '/missing/cargo', p['compiler_tools'])
        original = json.loads((tools / 'valid-mel-mixed.protocol.json').read_text())
        vendor = repo / 'vendor/llama-cpp-sys-2/llama.cpp'
        if (args.protocol.read_bytes() != protocol_bytes
                or p['runtime_provenance'] != profile_provenance(lock_path, lock)
                or p['tests'] != [c['test'] for c in original['cases'] if c['backend'] == 'cpu']
                or p['compiled_test_names'] != original['compiled_test_names']
                or p['model_and_provider_files'] != {n: v for n, v in original['files'].items()
                    if n.startswith(('models/', 'providers/'))}):
            raise ValueError('mixed protocol/runtime identity changed')
        verify_vendor_worktree(vendor, p['vendor_revision'])
        for group, root in [('tools', tools), ('repository_files', repo),
                            ('vendor_sources', repo / 'vendor/llama-cpp-sys-2/llama.cpp')]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen mixed source differs: ' + name)
        for name in ['public_helpers', 'public_helpers_manifest', 'public_qualification']:
            if descriptor(getattr(args, name)) != p[name]:
                raise ValueError('qualified mixed prerequisite differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'protocol': descriptor(args.protocol),
        'pooled_cpu_mixed_residency_passed': False, 'commands': [], 'loaded_images': {},
        'cases': [], 'neural_commands_started': [], 'release_qualified': False, 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    guard = None

    def run(name, argv, sandbox, environment, inference=False, trace=False):
        frozen()
        guard.before_model()
        if inference:
            r['neural_commands_started'].append(name)
        child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)] if trace else list(map(str, argv))
        actual, environment = sandboxed_command(child, sandbox, environment)
        row = {'name': name, 'argv': actual, 'cwd': str(output), 'environment': environment,
               'inference_admitted': inference, 'load_before': os.getloadavg()}
        r['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=output, environment=environment, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
        row['load_after'] = os.getloadavg()
        persist()
        if row['returncode'] != 0:
            raise ValueError('mixed command failed: ' + name)
        return (output / (name + '.stdout.log')).read_text()

    try:
        guard = StorageGuard(output, **p['storage'])
        producer = output / 'producer'
        r['producer_artifact'] = admit_delivery(args.producer_zip, args.producer_metadata, p['producer'], producer)
        construction = json.loads((producer / 'qualification.json').read_text())
        public_q = json.loads(args.public_qualification.read_text())
        if (construction['construction_and_negative_c_api_controls_passed'] is not True
                or construction['valid_models_loaded'] or construction['frontend_or_neural_calls']
                or public_q['pooled_public_runtime_passed'] is not True
                or public_q['source_commit'] != p['public_source_commit']
                or public_q['producer_artifact']['id'] != p['producer']['id']
                or public_q['lifecycle_tests_passed'] != p['lifecycle_tests']):
            raise ValueError('completed construction/public lifecycle prerequisites required')
        host = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'platform': platform.platform(), 'load_before': os.getloadavg()}
        atomic_json(output / 'host.json', host)
        if host['cpu'] != p['expected_cpu']:
            raise ValueError('named physical M4 required')
        work = output / 'work'
        work.mkdir()
        package, binaries = work / 'nemo', work / 'bin'
        binaries.mkdir()
        m = json.loads((producer / 'pooled-cpu-runtime-files.json').read_text())
        restore_archive(producer / m['archive'], m, package)
        manifest = verify(package)
        if manifest['runtime_profile'] != POOLED_PROFILE:
            raise ValueError('pooled CPU private package required')
        atomic_json(output / 'package.json', manifest)
        for name, expected in p['model_and_provider_files'].items():
            source, destination = args.source_models / name, work / name
            if descriptor(source) != expected:
                raise ValueError('original mixed asset differs: ' + name)
            guard.before_model()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if descriptor(destination) != expected:
                raise ValueError('copied mixed asset differs: ' + name)
        helpers = output / 'public-helpers'
        helper_manifest = json.loads(args.public_helpers_manifest.read_text())
        if {k: v for k, v in helper_manifest.items() if k != 'files'} != public_q['helper_archive']:
            raise ValueError('public helper archive summary differs')
        restore_archive(args.public_helpers, helper_manifest, helpers)
        shutil.copy2(helpers / 'load-probe', binaries / 'load-probe')
        home, temporary = output / 'home', output / 'tmp'
        home.mkdir()
        temporary.mkdir()
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}
        cargo = shutil.which('cargo')
        if not cargo:
            raise ValueError('Cargo bootstrap missing')
        compile_env = {**env, **compiler_tools_environment(cargo, p['compiler_tools']),
            'HOME': str(Path.home()), 'CARGO_HOME': str(Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo'))).resolve()),
            'RUSTUP_HOME': str(Path(os.environ.get('RUSTUP_HOME', str(Path.home() / '.rustup'))).resolve()),
            'CARGO_TARGET_DIR': str(output / 'cargo-target'), 'MACOSX_DEPLOYMENT_TARGET': '26.0',
            'RUSTFLAGS': '-C target-cpu=generic -C link-arg=-mmacosx-version-min=26.0',
            'CC': '/usr/bin/clang', 'CXX': '/usr/bin/clang++',
            'CFLAGS': '-march=armv8.2-a+dotprod+fp16', 'CXXFLAGS': '-march=armv8.2-a+dotprod+fp16'}
        compile_sb = output / 'compile.sb'
        mel = repo / 'assets/bark/nemo128.f32'
        compile_sb.write_text(compiler_policy(mixed_policy(
            denied_reads=[work / 'models', work / 'providers', args.source_models.resolve()],
            readonly=[repo, work, producer, helpers, '/opt/homebrew', *inputs]), mel))
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            probe.read_bytes()
        controls = {'reads': list(map(str, probes)) + [str(work / 'models/laya/model.data'),
                    str(work / 'providers/libonnxruntime.dylib'), str(args.source_models.resolve() / 'models/laya/model.data')],
                    'writes': [str(mel)]}
        run('compile-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps(controls)], compile_sb, compile_env)
        run('mel-build-input-readable', ['/usr/bin/python3', '-c',
            'from pathlib import Path;import hashlib;'
            'assert hashlib.sha256(Path(' + repr(str(mel)) + ').read_bytes()).hexdigest() == ' +
            repr(p['repository_files']['assets/bark/nemo128.f32']['sha256'])], compile_sb, compile_env)
        run('rustc-version', [str(Path(cargo).parent / 'rustc'), '-vV'], compile_sb, compile_env)
        log = run('mixed-helper-build', [cargo, 'test', '--offline', '--locked', '--release', '--no-default-features',
            '--features', 'backend-llamacpp,metal,asr-dynamic,laya-dynamic,backend-diar-nemo',
            '--jobs', '2', '--message-format=json', '--manifest-path', repo / 'Cargo.toml',
            '--test', 'diarization_mixed_native', '--no-run'], compile_sb, compile_env)
        shutil.copy2(cargo_executable(log, 'diarization_mixed_native'), binaries / 'mixed-native-tests')
        r['helper_archive'] = write_archive(binaries, output, 'pooled-mixed-helpers')
        protected = {'work': inventory(work), 'producer': inventory(producer), 'public_helpers': inventory(helpers)}
        atomic_json(output / 'build-inputs.json', {'repository_sources': p['repository_files'],
            'vendor_sources': p['vendor_sources'], 'vendor_revision': p['vendor_revision'], 'compiler_environment': compile_env,
            'model_or_frontend_execution_during_build': False, 'protected_inputs': protected})

        def unchanged():
            frozen()
            for key, directory in [('work', work), ('producer', producer), ('public_helpers', helpers)]:
                if inventory(directory) != protected[key]:
                    raise ValueError('mixed protected input changed: ' + key)

        results = output / 'results'
        results.mkdir()
        prior = results / 'prior-output-control.txt'
        prior.write_bytes(b'not a model')
        denied = [repo, producer, helpers, args.source_models.resolve(), '/opt/homebrew',
            Path.home() / '.cargo', Path.home() / '.rustup', *[x.parent for x in inputs if x.is_file()],
            *p['denied_prior_paths']]
        runtime, premodel = output / 'runtime.sb', output / 'premodel.sb'
        runtime.write_text(mixed_runtime_policy(denied, [work, *denied], results))
        premodel.write_text(model_free_policy(runtime.read_text()) +
            '(deny file-read* (subpath ' + sandbox_path(work / 'models') + ') '
            '(subpath ' + sandbox_path(work / 'providers') + '))\n')
        reads = [repo / 'Cargo.toml', args.source_models.resolve() / 'models/laya/model.data',
                 producer / 'protocol.json', prior]
        for path in reads:
            path.read_bytes()
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, reads)), 'writes': [str(work / 'models/chat.gguf')]})], runtime, env)
        run('premodel-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, probes)) + [str(work / 'models/laya/model.data'),
                str(work / 'providers/libonnxruntime.dylib')], 'writes': []})], premodel, env)
        # The version probe may load Nemo, while the mixed listing must load no providers.
        version_sb = output / 'version.sb'
        version_sb.write_text(model_free_policy(runtime.read_text()) +
            '(deny file-read* (subpath ' + sandbox_path(work / 'models') + '))\n')
        version = run('version-only', [binaries / 'load-probe', package / 'lib/libnemo_speech_asr_c.dylib',
            lock['runtime']['reported_version']], version_sb, env, trace=True)
        if version.strip() != lock['runtime']['reported_version']:
            raise ValueError('mixed private runtime version differs')
        r['loaded_images']['version-only'] = audit_loaded_libraries(
            (output / 'version-only.stderr.log').read_text(), package, manifest, binaries / 'load-probe')
        helper = binaries / 'mixed-native-tests'
        listing = run('list', [helper, '--ignored', '--list'], premodel, env, trace=True)
        if {s.removesuffix(': test') for s in listing.splitlines() if s.endswith(': test')} != set(p['compiled_test_names']):
            raise ValueError('compiled mixed test set differs')
        r['loaded_images']['list'] = audit_loaded_libraries(
            (output / 'list.stderr.log').read_text(), package, manifest, helper, False)
        persist()
        for name in p['tests']:
            unchanged()
            case_env = {**env, 'GEN2_NEMO_EXECUTION': 'cpu', 'GEN2_NEMOTRON_ASSETS': str(work / 'models/nemotron'),
                'GEN2_NEMO_LIBRARY': str(package / 'lib/libnemo_speech_asr_c.dylib'),
                'GEN2_BARK_BUNDLE': str(work / 'models/bark'), 'GEN2_ORT_LIBRARY': str(work / 'providers/libonnxruntime.dylib'),
                'GEN2_CHAT_GGUF': str(work / 'models/chat.gguf'), 'GEN2_LAYA_BUNDLE': str(work / 'models/laya'),
                'GEN2_CHAT_EXPECT_BACKEND': 'MTL'}
            text = run(name, [helper, '--ignored', '--exact', name, '--nocapture'], runtime, case_env,
                       inference=True, trace=True)
            placement = {}
            for line in text.splitlines():
                if line.startswith('chat-placement '):
                    phase, value = line.removeprefix('chat-placement ').split(': ', 1)
                    if phase in placement:
                        raise ValueError('duplicate mixed chat placement phase')
                    placement[phase] = json.loads(value)
            if (not re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', text, re.MULTILINE)
                    or '1 passed; 0 failed; 0 ignored;' not in text
                    or set(placement) != {'initial', 'restored'}
                    or any(v['layers'] <= 0 or v['backend'] != 'MTL' or v['device'] != p['expected_cpu'] for v in placement.values())):
                raise ValueError('mixed test or chat placement failed')
            r['loaded_images'][name] = audit_with_auxiliary((output / (name + '.stderr.log')).read_text(),
                package, manifest, helper, {str(work / 'providers/libonnxruntime.dylib'):
                    p['model_and_provider_files']['providers/libonnxruntime.dylib']})
            r['cases'].append({'name': name, 'passed': True, 'diarization_backend': 'cpu', 'chat_placement': placement})
            persist()
        unchanged()
        if len(r['cases']) != 2:
            raise ValueError('both CPU load orders required')
        r['pooled_cpu_mixed_residency_passed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__':
    main()
