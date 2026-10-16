"""Four fixed relocated CPU streams: independent native/Rust at 30m and one hour."""
import json
import os
from pathlib import Path
import subprocess
import signal
import time

from artifacts import digest
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files, verify_package_provenance
from qualify_valid_mel_macos_package import descriptor
from valid_mel_long_metrics import audit_case


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parent.parent
    protocol_path = tools / 'valid-mel-cpu-long.protocol.json'
    p = json.loads(protocol_path.read_text())
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    output = Path(p['output_path'])
    if output.exists():
        raise ValueError('refusing previous long-session attempt overwrite')
    for records, root in [(p['tools'], tools), (p['rust_sources'], repo)]:
        for name, expected in records.items():
            if descriptor(root / name) != expected:
                raise ValueError('long-session qualification source differs: ' + name)
    for name, expected in p['inputs'].items():
        if descriptor(Path(name)) != expected:
            raise ValueError('long-session qualification input differs: ' + name)
    work = Path(p['work_path'])
    expected_tree = p['work_files']
    def check_tree():
        actual = {str(path.relative_to(work)): {**descriptor(path),
                  'symlink_target': str(path.readlink()) if path.is_symlink() else None}
                  for path in work.rglob('*') if path.is_file()}
        if actual != expected_tree:
            raise ValueError('long-session staged input tree differs')
    check_tree()
    prior = json.loads(Path(p['prior_package_qualification_path']).read_text())
    if prior['fixed_packaged_cpu_parity_and_lifecycle_passed'] is not True or prior['all6000rows_native_public_byte_identical'] is not True:
        raise ValueError('required fixed package qualification did not pass')
    build = json.loads(Path(p['build_metadata_path']).read_text())
    if build['rust_source_files'] != p['rust_sources'] or build['models_loaded'] is not False:
        raise ValueError('compiled long-session helper source admission differs')
    for name, expected in build['helper_source_files'].items():
        if p['inputs'].get(name) != expected:
            raise ValueError('native helper compiler input binding differs')
    for name in ['native-stream', 'public-stream']:
        if descriptor(work / 'bin' / name) != {k: build['inputs']['bin/' + name][k] for k in ['bytes', 'sha256']}:
            raise ValueError('long-session executable differs from its compile receipt')
    package = work / 'nemo'
    manifest = json.loads((package / 'package.json').read_text())
    verify_files(package, manifest['files'])
    verify_package_provenance(package, manifest)
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    profile = output / 'runtime.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n'
        + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(name) + ')' for name in p['denied_paths']) + ')\n'
        + '(deny file-write* ' + ' '.join('(subpath ' + json.dumps(name) + ')' for name in p['readonly_paths']) + ')\n')
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}
    commands, loaded, comparisons = [], {}, {}

    def run(name, executable, args):
        argv = ['/usr/bin/time', '-l', '/usr/bin/sandbox-exec', '-f', str(profile),
                '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', str(executable), *map(str, args)]
        record = {'name': name, 'argv': argv, 'environment': environment,
                  'timeout_seconds': p['command_timeout_seconds']}
        commands.append(record)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=repo, env=environment, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                record['process_group_id'] = process.pid
                (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
                try:
                    process.wait(timeout=p['command_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    record['forced_process_group_termination'] = True
                    process.wait()
                    record['returncode'] = process.returncode
                    raise
            record['returncode'] = process.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            record['execution_error'] = str(error)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(json.dumps({'command': name, 'returncode': process.returncode}), flush=True)
        if process.returncode:
            raise ValueError('long-session native command failed: ' + name)
        loaded[name] = audit_loaded_libraries((output / (name + '.stderr.log')).read_text(),
                                              package, manifest, executable)
        (output / 'loaded-libraries.json').write_text(json.dumps(loaded, indent=2) + '\n')
        return (output / (name + '.stdout.log')).read_text()

    for case in p['cases']:
        name = case['name']
        native_json, public_raw = output / (name + '-native.json'), output / (name + '-public.f32')
        run(name + '-native', work / 'bin/native-stream', [work / 'assets/Nemotron-3-Diarization.q8_0.gguf',
             work / 'assets/ami_en2002d_2132.wav', str(case['repeats']), '-1', native_json])
        public = json.loads(run(name + '-public', work / 'bin/public-stream', [work / 'assets',
             package / 'lib/libnemo_speech_asr_c.dylib', work / 'assets/ami_en2002d_2132.wav',
             str(case['repeats']), public_raw]))
        (output / (name + '-public.json')).write_text(json.dumps(public, indent=2) + '\n')
        native = json.loads(native_json.read_text())
        comparison = audit_case(public, native, Path(str(native_json) + '.f32').read_bytes(), public_raw.read_bytes(),
             (output / (name + '-native.stderr.log')).read_text(),
             (output / (name + '-public.stderr.log')).read_text(), case)
        comparisons[name] = comparison
        (output / (name + '-comparison.json')).write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps({'case': name, 'passed': True, 'rows': comparison['native_frame_count'],
                          'retained_first_frame': comparison['retained_first_frame'],
                          'gen2_rtf': comparison['gen2_rtf']}), flush=True)
    check_tree()
    for name, expected in p['inputs'].items():
        if descriptor(Path(name)) != expected:
            raise ValueError('frozen long-session input changed during execution: ' + name)
    for records, root in [(p['tools'], tools), (p['rust_sources'], repo)]:
        for name, expected in records.items():
            if descriptor(root / name) != expected:
                raise ValueError('frozen qualification source changed during execution: ' + name)
    result = {'source_commit': source_commit,
              'repository_head_at_completion': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'protocol_sha256': digest(protocol_path), 'comparisons': comparisons,
              'loaded_image_audits': loaded, 'fixed_cpu_30m_1h_compaction_parity_passed': True,
              'neural_inference_performed': True, 'release_qualified': False, 'limits': p['limits']}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'fixed_cpu_30m_1h_compaction_parity_passed': True}), flush=True)


if __name__ == '__main__':
    main()
