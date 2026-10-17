"""First real-audio parity checks of the owned-pool public C ABI.

Keep the frozen original helper intact. Derive only its exact expected version
literal, and compare complete12s/60s histories and turns with qualified outputs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import wave

from artifacts import acquire
from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from pooled_runtime import POOLED_PROFILE, load_profile, profile_provenance
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import CONTROL, policy
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory, restore_archive, write_archive


OLD_VERSION = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
NEW_VERSION = 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1'
CASES = [{'name': 'native-12s', 'wav': 'crop-12s.wav', 'reference': 'short_reference/native.json'},
         {'name': 'native-60s', 'wav': 'ami_en2002d_2132.wav', 'reference': 'long_reference/results/native.json'}]


def result_policy(base, results):
    # The frozen helper checks output existence. Permit stat metadata while
    # denying payload reads, including previously published native outputs.
    # Seatbelt matches canonical paths (e.g. /private/var rather than /var).
    return base + '(deny file-read-data (subpath ' + json.dumps(str(results.resolve(strict=True))) + '))\n'


def derive_helper(original):
    before, after = json.dumps(OLD_VERSION), json.dumps(NEW_VERSION)
    if original.count(before) != 1:
        raise ValueError('exactly one original expected-version literal required')
    return original.replace(before, after)


def compare_history(actual, raw, reference, golden):
    samples = reference['audio_samples']
    frames = samples // 160
    for record, version in [(actual, NEW_VERSION), (reference, OLD_VERSION)]:
        if (record['runtime_version'] != version or record['audio_samples'] != samples
                or record['native_frame_count'] != frames or record['retained_first_frame'] != 0
                or record['retained_rows'] != frames or record['input_repeats'] != 1
                or record['requested_gpu_index'] != -1 or record['audio_duration_seconds'] != samples / 16000
                or record['geometry'] != reference['geometry']
                or record['input_pcm_f32le_sha256'] != reference['input_pcm_f32le_sha256']):
            raise ValueError('native result identity or geometry differs')
    if len(raw) != frames * 32 or len(golden) != frames * 32:
        raise ValueError('complete eight-channel history required')
    maximum = 0.0
    for (value,), (expected,) in zip(struct.iter_unpack('<f', raw), struct.iter_unpack('<f', golden)):
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in [value, expected]):
            raise ValueError('invalid probability')
        maximum = max(maximum, abs(value - expected))
    if maximum > 1e-6:
        raise ValueError('qualified probability parity failed: ' + str(maximum))
    if len(actual['turns']) != len(reference['turns']):
        raise ValueError('native turn count differs')
    largest_boundary = 0.0
    for turn, expected in zip(actual['turns'], reference['turns']):
        if turn['speaker'] != expected['speaker'] or not 1 <= turn['speaker'] <= 8:
            raise ValueError('native speaker differs')
        for name in ['start_seconds', 'end_seconds']:
            if not math.isfinite(turn[name]) or not math.isfinite(expected[name]):
                raise ValueError('invalid turn boundary')
            largest_boundary = max(largest_boundary, abs(turn[name] - expected[name]))
        if not 0 <= turn['start_seconds'] <= turn['end_seconds']:
            raise ValueError('invalid turn interval')
    if largest_boundary > 1e-6:
        raise ValueError('qualified turn parity failed')
    return {'frames': frames, 'values_compared': frames * 8, 'bit_exact': raw == golden,
            'max_absolute_probability_error': maximum, 'probability_tolerance': 1e-6,
            'turns': len(actual['turns']), 'max_boundary_error_seconds': largest_boundary,
            'boundary_tolerance_seconds': 1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'construction-zip', 'construction-metadata',
                 'short-reference-zip', 'short-reference-metadata',
                 'long-reference-zip', 'long-reference-metadata', 'upstream', 'model', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual Apple Silicon Mac required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh output required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    upstream = args.upstream.resolve(strict=True)
    deliveries = [getattr(args, n + '_' + kind) for n in ['construction', 'short_reference', 'long_reference']
                  for kind in ['zip', 'metadata']]
    if (output.is_relative_to(repo) or upstream.is_relative_to(output)
            or args.model.resolve().is_relative_to(output)
            or any(f.resolve().is_relative_to(output) for f in deliveries)):
        parser.error('output must be separate from source and delivery inputs')
    frozen_bytes = args.protocol.read_bytes()
    p = json.loads(frozen_bytes)
    lock_path, lock = load_profile(POOLED_PROFILE)

    def frozen():
        if (args.protocol.read_bytes() != frozen_bytes or p['runtime_provenance'] != profile_provenance(lock_path, lock)
                or p['cases'] != CASES):
            raise ValueError('pooled native protocol/profile changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen native qualification source differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(frozen_bytes)
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'protocol': descriptor(args.protocol),
        'native_parity_passed': False, 'neural_commands_started': [], 'commands': [],
        'comparisons': {}, 'loaded_images': {}, 'release_qualified': False, 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    guard = None
    home, temporary = output / 'home', output / 'tmp'
    home.mkdir()
    temporary.mkdir()
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}

    def run(name, argv, sandbox, *, inference=False):
        frozen()
        guard.before_model()
        if inference:
            r['neural_commands_started'].append(name)
        row = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(sandbox), *map(str, argv)],
               'cwd': str(output), 'environment': env, 'inference_admitted': inference}
        r['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(row['argv'], cwd=output, environment=env, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
        persist()
        if row['returncode'] != 0:
            raise ValueError('pooled native command failed: ' + name)

    try:
        guard = StorageGuard(output, **p['storage'])
        r['deliveries'] = {}
        for name in ['construction', 'short_reference', 'long_reference']:
            r['deliveries'][name] = admit_delivery(getattr(args, name + '_zip'),
                getattr(args, name + '_metadata'), p[name], output / name)
        construction = output / 'construction'
        q = json.loads((construction / 'qualification.json').read_text())
        if not q['construction_and_negative_c_api_controls_passed'] or q['frontend_or_neural_calls'] or q['valid_models_loaded']:
            raise ValueError('completed model-free pooled construction required')
        work, sdk = output / 'work', output / 'sdk'
        work.mkdir()
        package, assets, binaries = work / 'nemo', work / 'assets', work / 'bin'
        assets.mkdir()
        binaries.mkdir()
        for stem, destination in [('pooled-cpu-sdk', sdk), ('pooled-cpu-runtime', package)]:
            manifest = json.loads((construction / (stem + '-files.json')).read_text())
            restore_archive(construction / manifest['archive'], manifest, destination)
        manifest = verify(package)
        if manifest['runtime_profile'] != POOLED_PROFILE:
            raise ValueError('owned-pool runtime package required')
        atomic_json(output / 'package.json', manifest)
        helper_source = output / 'helper-source'
        helper_source.mkdir()
        source = helper_source / 'native-stream.cpp'
        source.write_text(derive_helper((tools / 'native_stream_probe_valid_mel.cpp').read_text()))
        if descriptor(source) != p['derived_helper']:
            raise ValueError('derived helper differs')
        r['helper_inputs'] = {}
        for name, entry in p['helper_inputs'].items():
            original = upstream / ('examples/wav_reader.h' if name == 'wav_reader.h' else 'llama.cpp/vendor/hash/' + name)
            if descriptor(original) != {'bytes': entry['size'], 'sha256': entry['sha256']}:
                raise ValueError('helper dependency differs: ' + name)
            destination = helper_source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            r['helper_inputs'][name] = descriptor(destination)
        denied = [repo, upstream, sdk, helper_source, *[output / n for n in r['deliveries']],
                  *[f.resolve() for f in deliveries], args.model.resolve(), '/opt/homebrew',
                  Path.home() / '.cargo', Path.home() / '.rustup']
        compile_policy = output / 'compile.sb'
        compile_policy.write_text(model_free_policy(policy(denied_reads=[assets, *[output / n for n in r['deliveries']]],
            readonly=[repo, upstream, sdk, package, helper_source])))
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-arch', 'arm64',
                 '-march=armv8.2-a+dotprod+fp16', '-mmacosx-version-min=26.0']
        for name, inputs in [('native-stream', [source, helper_source / 'sha256/sha256.c']),
                             ('failed-load-probe', [tools / 'native_failed_load_probe.cpp'])]:
            run('build-' + name, ['/usr/bin/clang++', *flags, '-x', 'c++', *inputs,
                '-I' + str(sdk / 'include'), '-I' + str(helper_source), '-L' + str(package / 'lib'),
                '-lnemo_speech_asr_c', '-Wl,-rpath,@loader_path/../nemo/lib', '-o', binaries / name], compile_policy)
        r['helper_archive'] = write_archive(binaries, output, 'pooled-native-helpers')
        protected = {'sdk': inventory(sdk), 'package': inventory(package), 'binaries': inventory(binaries)}
        results = output / 'results'
        results.mkdir()
        runtime = output / 'runtime.sb'
        runtime.write_text(result_policy(policy(denied_reads=denied,
            readonly=[work, repo, upstream, sdk, package, helper_source]), results))
        model_free = output / 'model-free.sb'
        model_free.write_text(model_free_policy(runtime.read_text()))
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            if probe.read_bytes() != b'not a model':
                raise ValueError('format positive control failed')
        run('model-free-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, probes)), 'writes': []})], model_free)
        failed = binaries / 'failed-load-probe'
        run('failed-loads', ['/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', failed,
            NEW_VERSION, work / 'missing-model'], model_free)
        r['failed_load_control'] = json.loads((output / 'failed-loads.stdout.log').read_text())
        control = r['failed_load_control']
        counts = control['threads_after_each_failure']
        if (control['failed_model_construction_attempts'] != 101 or control['valid_models_loaded'] != 0
                or control['frontend_or_neural_calls'] != 0 or control['thread_growth_allowance'] != 2
                or control['maximum_requested_settling_wait_milliseconds'] != 100
                or len(control['immediate_thread_counts']) != 101
                or len(control['requested_settling_wait_milliseconds']) != 101
                or any(w < 10 or w > 100 or w % 10 for w in control['requested_settling_wait_milliseconds'])
                or len(counts) != 101 or min(counts) < 1 or max(counts) > counts[0] + 2):
            raise ValueError('failed-load cleanup control differs')
        r['loaded_images']['failed-loads'] = audit_loaded_libraries((output / 'failed-loads.stderr.log').read_text(),
            package, manifest, failed)
        atomic_json(output / 'host.json', {'cpu': subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'platform': platform.platform(), 'cpu_counts': subprocess.check_output(
                ['/usr/sbin/sysctl', 'hw.ncpu', 'hw.logicalcpu', 'hw.physicalcpu'], text=True),
            'load_before': os.getloadavg()})
        r['acquisitions'] = {}
        for entry in p['assets']:
            target = assets / entry['name']
            if entry['role'] == 'model':
                if descriptor(args.model) != {'bytes': entry['size'], 'sha256': entry['sha256']}:
                    raise ValueError('cached model differs')
                shutil.copy2(args.model, target)
            elif entry['role'] == 'fixture':
                shutil.copy2(upstream / 'test_files/diar/ami_en2002d_2132.wav', target)
            acquire(entry, assets)
            r['acquisitions'][entry['name']] = descriptor(target)
        with wave.open(str(assets / 'ami_en2002d_2132.wav'), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, 960000):
                raise ValueError('fixture geometry differs')
            crop = wav.readframes(192000)
        crop_path = assets / 'crop-12s.wav'
        with wave.open(str(crop_path), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(crop)
        frozen_assets = inventory(assets)
        oracle_control = results / 'unreadable-control.txt'
        oracle_control.write_bytes(b'not a model')
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': [str(repo / 'Cargo.toml'), str(output / 'short_reference/native.json.f32'),
                                 str(oracle_control)], 'writes': [str(crop_path)]})], runtime)
        executable = binaries / 'native-stream'
        for case in p['cases']:
            frozen()
            if (inventory(sdk) != protected['sdk'] or inventory(package) != protected['package']
                    or inventory(binaries) != protected['binaries'] or inventory(assets) != frozen_assets):
                raise ValueError('native qualification input changed')
            target = results / (case['name'] + '.json')
            run(case['name'], ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', executable,
                assets / p['model_name'], assets / case['wav'], '1', '-1', target], runtime, inference=True)
            r['loaded_images'][case['name']] = audit_loaded_libraries((output / (case['name'] + '.stderr.log')).read_text(),
                package, manifest, executable)
            actual = json.loads(target.read_text())
            reference = output / case['reference']
            expected = json.loads(reference.read_text())
            r['comparisons'][case['name']] = compare_history(actual, Path(str(target) + '.f32').read_bytes(),
                expected, Path(str(reference) + '.f32').read_bytes())
            r['comparisons'][case['name']]['warm_rtf'] = actual['elapsed_micros'] / 1e6 / actual['audio_duration_seconds']
            persist()
        frozen()
        if (inventory(sdk) != protected['sdk'] or inventory(package) != protected['package']
                or inventory(binaries) != protected['binaries'] or inventory(assets) != frozen_assets):
            raise ValueError('native qualification input changed after inference')
        r['load_after'] = os.getloadavg()
        r['native_parity_passed'] = True
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
