"""Fixed relocated Metal admission: actual loader closure, 60s parity and lifecycle."""
import hashlib
import json
from pathlib import Path
import re
import os
import signal
import shutil
import subprocess
import time
import wave

import numpy as np

from artifacts import digest
from qualify_valid_mel_macos_package import compare_complete_60s
from macos_loaded_libraries import audit_loaded_libraries
from macos_multiple_loaded_libraries import audit_multiple_loaded_libraries
from package_macos import verify_files, verify_package_provenance


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parent.parent
    protocol_path = tools / 'valid-mel-metal-package.protocol.json'
    p = json.loads(protocol_path.read_text())
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    output = Path(p['output_path'])
    prior_long = json.loads(Path(p['prior_cpu_long_result_path']).read_text())
    if not prior_long.get('fixed_cpu_30m_1h_compaction_parity_passed'):
        raise ValueError('CPU long-session attempt must finish before Metal neural admission')
    if output.exists():
        raise ValueError('refusing previous packaged runtime attempt overwrite')
    for names, root in [(p['tools'], tools), (p['rust_sources'], repo)]:
        for name, expected in names.items():
            if descriptor(root / name) != expected:
                raise ValueError('qualification source differs: ' + name)
    for name, expected in p['inputs'].items():
        if descriptor(Path(name)) != expected:
            raise ValueError('packaged qualification input differs: ' + name)
    def check_trees():
        for name, expected in p['tree_files'].items():
            root = Path(name)
            actual = {str(path.relative_to(root)): {'sha256': digest(path), 'bytes': path.stat().st_size,
                       'symlink_target': str(path.readlink()) if path.is_symlink() else None}
                      for path in root.rglob('*') if path.is_file()}
            if actual != expected:
                raise ValueError('packaged qualification input tree differs: ' + name)
    check_trees()
    if Path(p['missing_library_path']).exists():
        raise ValueError('missing-library negative control no longer has its fixed defect')
    work = Path(p['work_path'])
    build = json.loads((work.parent / 'build.json').read_text())
    load_build = json.loads((Path(p['load_probe_path']).parent / 'build.json').read_text())
    prior_public = json.loads(Path(p['public_prior_build_path']).read_text())
    if (build['models_loaded'] is not False or build['neural_calls'] != 0
            or build['rust_source_files'] != p['rust_sources']
            or build['fixed_lifecycle_tests'] != p['lifecycle_tests']
            or load_build['source'] != p['tools']['macos_runtime_load_probe.cpp']
            or load_build['binary'] != descriptor(Path(p['load_probe_path']))
            or load_build['probe_executed'] is not False
            or prior_public['binaries']['nemotron-valid-mel-stream'] != descriptor(work / 'bin/public-stream')):
        raise ValueError('compiled probe/source/binary admission differs')
    for name, expected in build['helper_source_files'].items():
        if p['inputs'].get(name) != expected:
            raise ValueError('compiled native helper source binding differs')
    for name, expected in prior_public['source_files'].items():
        if descriptor(repo / name) != expected:
            raise ValueError('prior public helper source differs: ' + name)
    package = work / 'nemo'
    manifest = json.loads((package / 'package.json').read_text())
    verify_files(package, manifest['files'])
    verify_package_provenance(package, manifest)
    with wave.open(str(work / 'assets/ami_en2002d_2132.wav'), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, 960000):
            raise ValueError('fixed 60s PCM16 input geometry differs')
        pcm = np.frombuffer(wav.readframes(960000), dtype='<i2').astype('<f4') / 32768
    if hashlib.sha256(pcm.tobytes()).hexdigest() != p['pcm_sha256']:
        raise ValueError('normalized 60s PCM differs')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    profile = output / 'runtime.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n'
        + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(name) + ')' for name in p['denied_paths']) + ')\n'
        + '(deny file-write* ' + ' '.join('(subpath ' + json.dumps(name) + ')' for name in p['readonly_paths']) + ')\n')
    temporary = output / 'tmp'
    temporary.mkdir()
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'TMPDIR': str(temporary) + '/'}
    commands, loaded = [], {}

    def run(name, args, extra=None, expected=0, require_runtime=True, runtime_package=None, both_packages=False):
        argv = ['/usr/bin/sandbox-exec', '-f', str(profile), '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, args)]
        env = {**environment, **(extra or {})}
        command = {'name': name, 'argv': argv, 'environment': env, 'expected_returncode': expected,
                   'timeout_seconds': p['command_timeout_seconds']}
        commands.append(command)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=repo, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                command['process_group_id'] = process.pid
                (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
                try:
                    process.wait(timeout=p['command_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    command['forced_process_group_termination'] = True
                    process.wait()
                    command['returncode'] = process.returncode
                    raise
            command['returncode'] = process.returncode
        except subprocess.TimeoutExpired:
            command['timed_out'] = True
            raise
        finally:
            command['elapsed_seconds'] = time.monotonic() - started
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(json.dumps({'command': name, 'returncode': process.returncode}), flush=True)
        if process.returncode != expected:
            raise ValueError('packaged command failed: ' + name)
        if require_runtime is not None:
            runtime_package = Path(runtime_package) if runtime_package else package
            runtime_manifest = json.loads((runtime_package / 'package.json').read_text())
            trace = (output / (name + '.stderr.log')).read_text()
            if both_packages:
                cpu_package = Path(p['cpu_package_path'])
                cpu_manifest = json.loads((cpu_package / 'package.json').read_text())
                loaded[name] = audit_multiple_loaded_libraries(trace,
                    {'cpu': (cpu_package, cpu_manifest), 'metal': (package, manifest)}, args[0])
            else:
                loaded[name] = audit_loaded_libraries(trace, runtime_package, runtime_manifest, args[0], require_runtime)
            (output / 'loaded-libraries.json').write_text(json.dumps(loaded, indent=2) + '\n')
        return (output / (name + '.stdout.log')).read_text()

    # These two commands query a version or fail at dlopen; neither creates a model.
    version = run('version-only', [p['load_probe_path'], package / 'lib/libnemo_speech_asr_c.dylib', p['runtime_version']])
    if version.strip() != p['runtime_version']:
        raise ValueError('relocated actual C ABI version differs')
    negative = run('missing-library', [p['load_probe_path'], p['missing_entry_path'], p['runtime_version']],
                   expected=1, require_runtime=None)
    error = (output / 'missing-library.stderr.log').read_text()
    if negative or 'Library not loaded:' not in error or Path(p['missing_library_path']).name not in error:
        raise ValueError('missing private dependency did not fail for the specified reason')

    native = output / 'native.json'
    run('native', [work / 'bin/native-stream', work / 'assets/Nemotron-3-Diarization.q8_0.gguf',
                   work / 'assets/ami_en2002d_2132.wav', '1', '0', native])
    public = json.loads(run('public', [work / 'bin/public-stream', '--metal', work / 'assets',
                       package / 'lib/libnemo_speech_asr_c.dylib', work / 'assets/ami_en2002d_2132.wav',
                       '1', output / 'public.f32']))
    (output / 'public.json').write_text(json.dumps(public, indent=2) + '\n')
    reference = json.loads(native.read_text())
    if public['backend'] != 'metal' or public['device'] != p['expected_metal_device'] or reference['requested_gpu_index'] != 0:
        raise ValueError('actual Metal placement differs')
    for record in [public, reference]:
        if record['input_pcm_f32le_sha256'] != p['pcm_sha256'] or record['input_repeats'] != 1:
            raise ValueError('actual packaged input identity differs')
    comparison = compare_complete_60s(public, reference, Path(str(native) + '.f32').read_bytes(), (output / 'public.f32').read_bytes())
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    refs = output / 'reference'
    refs.mkdir()
    (refs / 'ami.json').write_text(json.dumps({'segments': [
        {'start': s['start_seconds'], 'end': s['end_seconds'], 'speaker': s['speaker']} for s in reference['turns']]}) + '\n')
    shutil.copy2(str(native) + '.f32', refs / 'ami.json.f32')
    lifecycle_env = {'GEN2_NEMOTRON_ASSETS': str(work / 'assets'),
                     'GEN2_NEMO_LIBRARY': str(package / 'lib/libnemo_speech_asr_c.dylib'),
                     'GEN2_NEMOTRON_REFERENCE': str(refs), 'GEN2_NEMO_EXECUTION': 'metal'}
    for index, test in enumerate(p['lifecycle_tests']):
        name = 'lifecycle-' + str(index)
        stdout = run(name, [work / 'bin/native-lifecycle', '--ignored', '--exact', test, '--test-threads=1'],
                     lifecycle_env, require_runtime=test != 'verified_provisioned_cache_is_reused_offline')
        if not re.search(r'^test ' + re.escape(test) + r' \.\.\. ok$', stdout, re.MULTILINE) or '1 passed; 0 failed; 0 ignored;' not in stdout:
            raise ValueError('required lifecycle test did not execute and pass: ' + test)
    # Isolated negative control: explicit Metal against the qualified CPU package
    # must fail at device admission, before model creation, without fallback.
    cpu_package = Path(p['cpu_package_path'])
    unavailable = run('cpu-metal-unavailable', [work / 'bin/public-stream', '--metal', work / 'assets',
        cpu_package / 'lib/libnemo_speech_asr_c.dylib', work / 'assets/ami_en2002d_2132.wav', '1',
        output / 'unexpected-cpu.f32'], expected=1, runtime_package=cpu_package)
    if unavailable or (output / 'unexpected-cpu.f32').exists() or 'requested Metal device is unavailable' not in (output / 'cpu-metal-unavailable.stderr.log').read_text():
        raise ValueError('explicit Metal CPU-SDK control did not fail before inference')
    metal_reference = refs / 'metal.json'
    metal_reference.write_text(json.dumps({'frames': reference['native_frame_count'], 'segments': [
        {'start': t['start_seconds'], 'end': t['end_seconds'], 'speaker': t['speaker']} for t in reference['turns']]}) + '\n')
    shutil.copy2(str(native) + '.f32', refs / 'metal.json.f32')
    combined_env = {**lifecycle_env, 'GEN2_NEMO_LIBRARY': str(cpu_package / 'lib/libnemo_speech_asr_c.dylib'),
        'GEN2_NEMO_METAL_LIBRARY': str(package / 'lib/libnemo_speech_asr_c.dylib'),
        'GEN2_NEMO_METAL_REFERENCE': str(metal_reference), 'GEN2_NEMO_METAL_DEVICE': p['expected_metal_device']}
    combined_test = p['combined_metal_test']
    stdout = run('combined-cpu-metal', [work / 'bin/native-lifecycle', '--ignored', '--exact', combined_test, '--test-threads=1'],
                 combined_env, both_packages=True)
    if not re.search(r'^test ' + re.escape(combined_test) + r' \.\.\. ok$', stdout, re.MULTILINE) or '1 passed; 0 failed; 0 ignored;' not in stdout:
        raise ValueError('combined CPU rejection and actual Metal parity test did not execute and pass')
    check_trees()
    for name, expected in p['inputs'].items():
        if descriptor(Path(name)) != expected:
            raise ValueError('frozen input changed during qualification: ' + name)
    for records, root in [(p['tools'], tools), (p['rust_sources'], repo)]:
        for name, expected in records.items():
            if descriptor(root / name) != expected:
                raise ValueError('frozen qualification source changed: ' + name)

    result = {'source_commit': source_commit,
              'repository_head_at_completion': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'protocol_sha256': digest(protocol_path), 'comparison': comparison,
              'actual_package_private_library_count': len(manifest['libraries']),
              'loaded_image_audits': loaded, 'lifecycle_tests_passed': p['lifecycle_tests'],
              'fixed_packaged_metal_parity_and_lifecycle_passed': True,
              'new_neural_inference_performed': True, 'explicit_metal_cpu_sdk_rejected_without_fallback': True,
              'combined_cpu_rejection_and_metal_parity_passed': True, 'release_qualified': False, 'limits': p['limits']}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'fixed_packaged_metal_parity_and_lifecycle_passed': True, 'lifecycle_tests': 7,
                      'rows': comparison['native_frame_count'], 'private_libraries': len(manifest['libraries'])}), flush=True)


if __name__ == '__main__':
    main()
