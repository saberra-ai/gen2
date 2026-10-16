"""Compare private helper pool lifetimes without changing the qualified runtime.

Four fresh processes, fixed ABBA order, four threads, identical twelve-second
input. Whole-output reference parity is required before timing interpretation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tarfile
import wave

from artifacts import acquire
from build_valid_mel_candidate import descriptor
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import CONTROL, policy
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory, restore_archive


SELF_TEST = {'model_constructed': False, 'frontend_executed': False,
             'backend_lifetimes': 20, 'persistent_pool_lifetimes': 10, 'threads': 4}
ORDER = ['disposable', 'persistent', 'persistent', 'disposable']


def model_free_policy(runtime_policy):
    return runtime_policy + r'(deny file-read* (regex #".*\.(nemo|gguf|safetensors|onnx|npy|f32)$"))' + '\n'


def compare_probabilities(actual, reference):
    """Use the established wrapper tolerance on every value, never only turns."""
    if len(actual) != 38400 or len(reference) != 38400:
        raise ValueError('complete 1200 by 8 float32 history required')
    pairs = zip(struct.iter_unpack('<f', actual), struct.iter_unpack('<f', reference))
    maximum = 0.0
    for (value,), (expected,) in pairs:
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in [value, expected]):
            raise ValueError('invalid probability')
        maximum = max(maximum, abs(value - expected))
    if maximum > 1e-6:
        raise ValueError('qualified C ABI probability parity failed: ' + str(maximum))
    return {'values_compared': 9600, 'max_absolute_error': maximum,
            'tolerance': 1e-6, 'bit_exact': actual == reference}


def materialize_headers(archive, native, audit, destination, apply_patch):
    """Replay only public GGML headers, then match all qualified compiler bytes."""
    expected = {n: d for n, d in audit['files'].items() if n.startswith('ggml/include/')}
    if len(expected) != 22 or descriptor(archive) != audit['archive']:
        raise ValueError('qualified GGML archive/header set differs')
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        for name in expected:
            member = bundle.getmember(name)
            path = Path(name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe GGML header')
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(member).read())
    for index, item in enumerate(audit['patch_series']):
        patch = native / 'patches' / item['name']
        if descriptor(patch) != {key: item[key] for key in ['bytes', 'sha256']}:
            raise ValueError('qualified GGML patch differs')
        normalized = destination / 'current.patch'
        normalized.write_bytes(patch.read_bytes().replace(b'\r\n', b'\n'))
        apply_patch(index, normalized)
    normalized.unlink()
    for name, expected_descriptor in expected.items():
        if descriptor(destination / name) != expected_descriptor:
            raise ValueError('replayed GGML header differs: ' + name)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'construction-zip', 'construction-metadata',
                 'reference-zip', 'reference-metadata', 'native-source', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual Apple Silicon receiver required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh output required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    native = args.native_source.resolve(strict=True)
    deliveries = [args.construction_zip, args.construction_metadata,
                  args.reference_zip, args.reference_metadata]
    if (output.is_relative_to(repo) or native.is_relative_to(output)
            or any(f.resolve().is_relative_to(output) for f in deliveries)):
        parser.error('output must be separate from source/delivery inputs')
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)

    def frozen():
        if args.protocol.read_bytes() != protocol_bytes or p['order'] != ORDER:
            raise ValueError('pool comparison protocol/order changed')
        for group, root in [('tools', tools), ('native_inputs', native), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen pool source differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'protocol': descriptor(args.protocol), 'comparison_completed': False,
        'inference_commands_started': [], 'commands': [], 'comparisons': {},
        'loaded_images': {}, 'release_qualified': False, 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    guard = None
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': os.environ['HOME'], 'LC_ALL': 'C'}

    def run(name, argv, sandbox, *, cwd=output, inference=False):
        frozen()
        guard.before_model()
        if inference:
            r['inference_commands_started'].append(name)
        actual = ['/usr/bin/sandbox-exec', '-f', str(sandbox), *map(str, argv)]
        command = {'name': name, 'argv': actual, 'environment': env, 'cwd': str(cwd),
                   'inference_admitted': inference}
        r['commands'].append(command)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=cwd, environment=env, stdout=stdout, stderr=stderr,
                            guard=guard, record=command, persist=persist,
                            timeout=p['command_timeout_seconds'])
        persist()
        if command['returncode'] != 0:
            raise ValueError('pool comparison command failed: ' + name)

    try:
        guard = StorageGuard(output, **p['storage'])
        r['deliveries'] = {}
        for name in ['construction', 'reference']:
            r['deliveries'][name] = admit_delivery(getattr(args, name + '_zip'),
                getattr(args, name + '_metadata'), p[name], output / name)
        construction, reference = output / 'construction', output / 'reference'
        if not json.loads((reference / 'qualification.json').read_text())['profile_completed']:
            raise ValueError('completed qualified reference required')
        work, sdk = output / 'work', output / 'sdk'
        work.mkdir()
        package = work / 'nemo'
        for kind, destination in [('sdk', sdk), ('runtime', package)]:
            manifest = json.loads((construction / ('portable-cpu-' + kind + '-files.json')).read_text())
            restore_archive(construction / manifest['archive'], manifest, destination)
        package_manifest = verify(package)
        atomic_json(output / 'package.json', package_manifest)
        protected = {'sdk': inventory(sdk), 'runtime': inventory(package)}
        assets, binaries = work / 'assets', work / 'bin'
        assets.mkdir()
        binaries.mkdir()
        host = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'cpu_counts': subprocess.check_output(['/usr/sbin/sysctl', 'hw.ncpu', 'hw.logicalcpu', 'hw.physicalcpu'], text=True),
                'platform': platform.platform(), 'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
                'load_before': os.getloadavg()}
        atomic_json(output / 'host.json', host)
        if host['cpu'] != p['expected_receiver_cpu']:
            raise ValueError('receiver CPU differs')
        compile_policy = output / 'compile.sb'
        compile_policy.write_text(policy(denied_reads=[assets, reference],
            readonly=[repo, native, sdk, package, construction]))
        run('compile-controls', [sys.executable, '-c', CONTROL, json.dumps({
            'reads': [str(reference / 'native.json.f32')], 'writes': [str(sdk / 'include/nemo_speech/asr.h')]})], compile_policy)
        headers = output / 'ggml-headers'
        audit = json.loads((construction / 'build/llama-materialized-audit.json').read_text())
        def apply_patch(index, patch):
            run('headers-patch-' + str(index), ['/usr/bin/git', '-c', 'core.hooksPath=/dev/null',
                'apply', '--include=ggml/include/*', '--allow-empty', '--whitespace=nowarn', patch],
                compile_policy, cwd=headers)
        r['ggml_headers'] = materialize_headers(construction / 'build/pristine-llama.tar', native, audit, headers, apply_patch)
        # Reuse the three exact previously successful compilation commands,
        # translating only isolated input/output roots. No runtime rebuild.
        replacements = {'@native@': str(native), '@headers@': str(headers),
                        '@sdk@': str(sdk), '@tools@': str(tools), '@output@': str(output)}
        for command in p['compile_commands']:
            argv = command['argv']
            for key, value in replacements.items():
                argv = [arg.replace(key, value) for arg in argv]
            run(command['name'], argv, compile_policy)
        executable = binaries / 'native-pool-probe'
        run('link', ['/usr/bin/clang++', '-arch', 'arm64', '-mmacosx-version-min=26.0',
            *[output / (name + '.o') for name in ['probe', 'backend', 'logging']],
            *[package / 'lib' / name for name in p['link_libraries']],
            '-Wl,-rpath,@loader_path/../nemo/lib', '-o', executable], compile_policy)
        r['binary'] = descriptor(executable)
        load_commands = {
            'binary': r['binary'], 'otool': subprocess.check_output(['/usr/bin/otool', '-l', executable], text=True),
            'architectures': subprocess.check_output(['/usr/bin/lipo', '-archs', executable], text=True).strip()}
        atomic_json(output / 'helper-load-commands.json', load_commands)
        if load_commands['architectures'] != 'arm64':
            raise ValueError('helper architecture differs')
        denied = [repo, native, sdk, headers, construction, reference,
                  *[f.resolve() for f in deliveries], '/opt/homebrew',
                  Path.home() / '.cargo', Path.home() / '.rustup']
        results = output / 'results'
        results.mkdir()
        runtime = output / 'runtime.sb'
        runtime.write_text(policy(denied_reads=[*denied, results], readonly=[work, *denied]))
        run('runtime-controls', [sys.executable, '-c', CONTROL, json.dumps({
            'reads': [str(repo / 'Cargo.toml'), str(reference / 'native.json.f32'),
                      str(native / 'src/runtime/ggml/backend.cpp')],
            'writes': [str(executable)]})], runtime)
        model_free = output / 'model-free.sb'
        model_free.write_text(model_free_policy(runtime.read_text()))
        probes = [assets / ('denied.' + suffix)
                  for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            if probe.read_bytes() != b'not a model':
                raise ValueError('positive format control differs')
        run('model-free-controls', [sys.executable, '-c', CONTROL,
            json.dumps({'reads': list(map(str, probes)), 'writes': []})], model_free)
        run('pool-self-test', ['/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', executable, '--self-test'], model_free)
        if json.loads((output / 'pool-self-test.stdout.log').read_text()) != SELF_TEST:
            raise ValueError('all backend/pool lifecycle controls must pass')
        r['loaded_images']['pool-self-test'] = audit_loaded_libraries(
            (output / 'pool-self-test.stderr.log').read_text(), package, package_manifest, executable)
        r['acquisition_attempts'] = []
        for entry in p['assets']:
            guard.before_model()
            row = {'expected': entry}
            r['acquisition_attempts'].append(row)
            persist()
            acquire(entry, assets)
            row['verified'] = descriptor(assets / entry['name'])
            persist()
        with wave.open(str(assets / 'ami_en2002d_2132.wav'), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, 960000):
                raise ValueError('input recording geometry differs')
            pcm = b''.join(struct.pack('<f', value[0] / 32768) for value in struct.iter_unpack('<h', wav.readframes(192000)))
        pcm_path = assets / 'input.f32'
        pcm_path.write_bytes(pcm)
        r['pcm'] = descriptor(pcm_path)
        if r['pcm']['sha256'] != p['pcm_f32le_sha256']:
            raise ValueError('cropped PCM differs from qualified reference')
        frozen_assets = inventory(assets)
        for index, mode in enumerate(ORDER):
            frozen()
            if (inventory(sdk) != protected['sdk'] or inventory(package) != protected['runtime']
                    or descriptor(executable) != r['binary'] or inventory(assets) != frozen_assets):
                raise ValueError('pool experiment input changed')
            name = str(index + 1) + '-' + mode
            target = results / (name + '.json')
            run(name, ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1',
                executable, mode, assets / p['model_name'], pcm_path, target], runtime, inference=True)
            r['loaded_images'][name] = audit_loaded_libraries((output / (name + '.stderr.log')).read_text(),
                package, package_manifest, executable)
            actual = json.loads(target.read_text())
            if (actual['audio_samples'] != 192000 or actual['frames'] != 1200 or actual['channels'] != 8
                    or actual['retained_first_frame'] != 0 or actual['threads'] != 4
                    or actual['persistent_pool'] != (mode == 'persistent') or actual['geometry'] != p['geometry']
                    or actual['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256']
                    or actual['elapsed_micros'] <= 0 or actual['load_micros'] <= 0):
                raise ValueError('private helper result identity/geometry differs')
            r['comparisons'][name] = compare_probabilities(Path(str(target) + '.f32').read_bytes(),
                                                          (reference / 'native.json.f32').read_bytes())
            r['comparisons'][name]['warm_rtf'] = actual['elapsed_micros'] / 12e6
            persist()
        if inventory(sdk) != protected['sdk'] or inventory(package) != protected['runtime']:
            raise ValueError('qualified SDK/package changed')
        r['load_after'] = os.getloadavg()
        r['comparison_completed'] = True
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
