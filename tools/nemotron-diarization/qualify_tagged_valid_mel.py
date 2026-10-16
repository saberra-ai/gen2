"""Fixed tagged CPU admission: F32/Q8 calibration controls and public/native parity."""
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np

from artifacts import digest
from frontend_admission import require_linked_sdk, run_probe
from frontend_metrics import difference
from compare_stream_valid_mel import compare
from valid_mel_finish_metrics import audit_outputs


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parent.parent
    protocol_path = tools / 'tagged-valid-mel-qualification.protocol.json'
    p = json.loads(protocol_path.read_text())
    output = Path(p['output_path'])
    if output.exists():
        raise ValueError('refusing previous tagged neural evidence overwrite')
    for name, expected in p['tools'].items():
        if descriptor(tools / name) != expected:
            raise ValueError('tagged qualification tool differs: ' + name)
    for name, expected in p['inputs'].items():
        if descriptor(Path(name)) != expected:
            raise ValueError('tagged model/runtime/input differs: ' + name)
    for name, expected in p['rust_sources'].items():
        if descriptor(repo / name) != expected:
            raise ValueError('Rust qualification source differs: ' + name)
    metadata = json.loads(Path(p['sdk_metadata_path']).read_text())
    if (metadata['runtime_profile'] != 'gen2-valid-mel-v1'
            or metadata['runtime_reported_version'] != p['runtime_version']
            or metadata['metadata_recovery_only'] is not True
            or metadata['model_instantiated_or_inference_performed'] is not False):
        raise ValueError('tagged SDK construction admission differs')
    sdk = Path(p['sdk_path'])
    for name, expected in metadata['installed_library_paths'].items():
        path = sdk / 'lib' / name
        if descriptor(path) != {k: expected[k] for k in ['bytes', 'sha256']} or (str(path.readlink()) if path.is_symlink() else None) != expected['symlink_target']:
            raise ValueError('tagged SDK alias or binary differs')
    audit = json.loads(Path(p['materialized_audit_path']).read_text())
    for name, expected in p['materialized_headers'].items():
        if audit['files'].get(name) != expected or descriptor(Path(audit['compiled_tree_path']) / name) != expected:
            raise ValueError('actual materialized compiler header differs')
    native = Path(p['native_source_path'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=native, text=True).strip() != p['native_base_revision']:
        raise ValueError('tagged native source base differs')
    if subprocess.check_output(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'], cwd=native) != (tools / 'patches/v3-finish-valid-mel.patch').read_bytes():
        raise ValueError('tagged native source patch differs')
    build = json.loads(Path(p['probe_build_path']).read_text())
    if build['neural_calls'] != 0 or build['probes_executed'] is not False or not build['pcm_container_conversion_lossless']:
        raise ValueError('probe construction is not the admitted compile-only step')
    for name, expected in build['binaries'].items():
        if p['inputs'][str(Path(p['probe_directory']) / name)] != expected:
            raise ValueError('probe binary construction binding differs')
    for name, expected in build['source_files'].items():
        if descriptor(repo / name) != expected:
            raise ValueError('actual compiled source differs: ' + name)
    for name in ['valid-mel-finish-probe', 'native-stream-probe-valid-mel']:
        load = subprocess.check_output(['/usr/bin/otool', '-l', str(Path(p['probe_directory']) / name)], text=True)
        lines = load.splitlines()
        rpaths = [lines[i + 2].strip().removeprefix('path ').split(' (offset ')[0]
                  for i, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']
        require_linked_sdk(sdk, p['sdk_path'], rpaths)
    for case in p['public_cases']:
        with wave.open(case['wav_path'], 'rb') as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()) != (1, 2, 16000, case['samples']):
                raise ValueError('fixed PCM16 WAV geometry differs')
            pcm = np.frombuffer(reader.readframes(case['samples']), dtype='<i2').astype('<f4') / 32768
        if hashlib.sha256(pcm.tobytes()).hexdigest() != case['pcm_sha256']:
            raise ValueError('fixed WAV changes actual normalized PCM')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (output / 'probe-build.json').write_text(json.dumps(build, indent=2) + '\n')
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}
    suites = {}
    summary = {'fixed_candidate_finish_regression_completed': True, 'stream_sessions': 28,
               'direct_graph_calls': 4, 'boundary_cases': 21, 'tolerance': 0.0001}
    mel = np.fromfile(p['mel_path'], dtype='<f4').reshape(112, 128)[:111]
    for name, model in [('f32', p['f32_model_path']), ('q8', p['q8_model_path'])]:
        directory = output / name
        directory.mkdir()
        stdout = run_probe([Path(p['probe_directory']) / 'valid-mel-finish-probe', model,
                            p['pcm_path'], p['mel_path'], directory / 'probe'], directory, 'suite', environment)
        if json.loads(stdout) != summary:
            raise ValueError('tagged native fixed-suite completion/count differs')
        suites[name] = audit_outputs(directory / 'probe')
        for left in [0, 1]:
            for stage, rows in [('first', mel[:104]), ('final', mel[72 - left * 8:])]:
                if (directory / 'probe' / ('direct-left' + str(left) + '-' + stage + '.mel.f32')).read_bytes() != rows.tobytes():
                    raise ValueError('direct graph valid-Mel input differs')
    previous = np.fromfile(p['previous_f32_whole_path'], dtype='<f4').reshape(111, 8)
    current = np.fromfile(output / 'f32/probe/whole.f32', dtype='<f4').reshape(111, 8)
    f32_rebuild = difference(current, previous, 0.0001)
    (output / 'f32-rebuild-comparison.json').write_text(json.dumps(f32_rebuild, indent=2) + '\n')
    if f32_rebuild['maximum_absolute_error'] > 0.0001:
        raise ValueError('tagged F32 rebuild differs beyond its preregistered policy')
    public = []
    for case in p['public_cases']:
        directory = output / ('public-' + case['name'])
        directory.mkdir()
        native_json, probabilities = directory / 'native.json', directory / 'gen2.f32'
        run_probe([Path(p['probe_directory']) / 'native-stream-probe-valid-mel', p['q8_model_path'],
                   case['wav_path'], '1', '-1', native_json], directory, 'native', environment)
        stdout = run_probe([Path(p['probe_directory']) / 'nemotron-valid-mel-stream', p['assets_path'],
                            sdk / 'lib/libnemo_speech_asr_c.dylib', case['wav_path'], '1', probabilities],
                           directory, 'gen2', environment)
        gen2 = json.loads(stdout)
        reference = json.loads(native_json.read_text())
        (directory / 'gen2.json').write_text(json.dumps(gen2, indent=2) + '\n')
        for record in [gen2, reference]:
            if record['input_pcm_f32le_sha256'] != case['pcm_sha256'] or record['input_repeats'] != 1:
                raise ValueError('public/native actual input bytes differ')
        result = compare(gen2, reference, Path(str(native_json) + '.f32').read_bytes(), probabilities.read_bytes())
        if gen2['accepted_samples'] != case['samples'] or gen2['backend'] != 'cpu':
            raise ValueError('public sample count/placement differs')
        value = np.fromfile(str(native_json) + '.f32', dtype='<f4').reshape(case['samples'] // 160, 8)
        pipeline = np.fromfile(output / 'q8/probe' / case['cpp_reference'], dtype='<f4').reshape(case['samples'] // 160, 8)
        result['c_api_vs_cpp_pipeline'] = difference(value, pipeline, 0.000001)
        (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        if result['c_api_vs_cpp_pipeline']['maximum_absolute_error'] > 0.000001:
            raise ValueError('default-thread C API differs from fixed-thread C++ pipeline')
        public.append({'case': case['name'], **result})
        (directory / 'comparison.json').write_text(json.dumps(public[-1], indent=2) + '\n')
    for name, expected in p['inputs'].items():
        if descriptor(Path(name)) != expected:
            raise ValueError('frozen model/runtime/input changed during qualification: ' + name)
    record = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'protocol_sha256': digest(protocol_path), 'suites': suites, 'f32_rebuild': f32_rebuild,
              'public_native_comparisons': public, 'fixed_tagged_cpu_qualification_completed': True,
              'native_finish_logical_cases': 56, 'explicit_direct_graph_calls': 8,
              'public_rust_stream_runs': 2, 'independent_c_api_stream_runs': 2,
              'models_instantiated': 6, 'neural_inference_performed': True,
              'hf_original_nemo_or_ami_target_executed': False, 'der_scoring_performed': False,
              'original_quality_gate_changed': False, 'all_platform_or_release_qualified': False,
              'limits': p['limitations']}
    record['files'] = {path.relative_to(output).as_posix(): descriptor(path) for path in output.rglob('*') if path.is_file()}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'fixed_tagged_cpu_qualification_completed': True, 'native_finish_logical_cases': 56,
                      'explicit_direct_graph_calls': 8, 'public_rust_stream_runs': 2, 'independent_c_api_stream_runs': 2}))


if __name__ == '__main__':
    main()
