"""Fixed CPU native F32/Q8/HF probability control; never replaces quality gates."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from artifacts import digest
from frontend_admission import require_linked_sdk, run_probe
from native_f32_metrics import comparisons
from package_macos import verify


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['evidence', 'corpus', 'construction', 'model', 'helper', 'sdk', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'native-f32-execution.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if platform.system() != 'Darwin' or 'Python ' + platform.python_version() != protocol['python_version']:
        raise ValueError('fixed native F32 control requires the frozen macOS/Python host')
    if args.output.exists():
        parser.error('refusing to overwrite native F32 evidence')
    for name, expected in protocol['tools'].items():
        if digest(tools / name) != expected:
            raise ValueError('numerical control tool changed: ' + name)
    if importlib.metadata.version('numpy') != protocol['numpy_version']:
        raise ValueError('numerical control NumPy changed')
    for name, expected in protocol['construction_files'].items():
        if descriptor(args.construction / name) != expected:
            raise ValueError('reviewed F32 construction evidence differs: ' + name)
    construction = json.loads((args.construction / 'qualification.json').read_text())
    if not (construction['all_mapped_names_shapes_dtypes_bytes_equal_independent_source_casts']
            and construction['all_stored_values_finite']
            and construction['all_source_keys_accounted_including_three_intentional_exclusions']
            and protocol['fixed_cpu_diagnostic_admission'] is True):
        raise ValueError('F32 artifact not admitted for this fixed CPU diagnostic')
    if descriptor(args.model) != construction['artifact'] or descriptor(args.model) != protocol['model']:
        raise ValueError('native F32 model bytes differ from reviewed construction')
    if descriptor(args.helper) != protocol['helper']:
        raise ValueError('native numerical helper differs from frozen producer')
    provenance = protocol['original_helper_provenance']
    build_path = tools.parents[1] / provenance['build_manifest_path']
    if descriptor(build_path) != provenance['build_manifest']:
        raise ValueError('original helper build provenance differs')
    build = json.loads(build_path.read_text())
    if (build['source_commit'] != provenance['source_commit']
            or build['binaries']['native-cpu'] != protocol['helper']['sha256']
            or build['package_manifests']['cpu'] != protocol['package_manifest']['sha256']
            or build['source_sha256']['native_stream_probe.cpp'] != protocol['tools']['native_stream_probe.cpp']):
        raise ValueError('original native helper source/package binding differs')
    load_commands = subprocess.check_output(['/usr/bin/otool', '-l', str(args.helper)], text=True)
    lines = load_commands.splitlines()
    rpaths = [lines[index + 2].strip().removeprefix('path ').split(' (offset ')[0]
              for index, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']
    require_linked_sdk(args.sdk, protocol['linked_sdk_path'], rpaths)
    if descriptor(args.sdk / 'package.json') != protocol['package_manifest']:
        raise ValueError('numerical helper package manifest differs')
    package = verify(args.sdk)
    if package['backend'] != 'cpu':
        raise ValueError('numerical control requires the qualified CPU package')
    for name, expected in protocol['sdk_libraries'].items():
        if descriptor(args.sdk / 'lib' / name) != expected:
            raise ValueError('numerical helper library differs: ' + name)
    if descriptor(args.corpus / 'manifest.json') != protocol['corpus_manifest']:
        raise ValueError('original corpus manifest changed')
    for name, expected in protocol['original_inputs'].items():
        if descriptor(args.evidence / name) != expected:
            raise ValueError('original Q8/HF inference evidence changed: ' + name)
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'helper-load-commands.txt').write_text(load_commands)
    profile = args.output / 'offline.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* ' +
                       ' '.join('(subpath ' + json.dumps(path) + ')' for path in protocol['denied_paths']) + ')\n')
    results = []
    for case in protocol['cases']:
        root = args.evidence if case['calibration'] else args.corpus
        wav = root / case['wav']
        if descriptor(wav) != case['wav_identity']:
            raise ValueError('fixed numerical input WAV differs: ' + case['wav'])
        for name in [case['q8_metadata'], case['hf_metadata']]:
            previous = json.loads((args.evidence / name).read_text())
            if (previous['audio_samples'] != case['samples'] or previous['input_repeats'] != 1
                    or previous['input_pcm_f32le_sha256'] != case['pcm_sha256']
                    or previous['geometry'] != protocol['geometry']):
                raise ValueError('saved numerical input metadata differs: ' + name)
        hf_metadata = json.loads((args.evidence / case['hf_metadata']).read_text())
        if hf_metadata['device'] != 'cpu' or hf_metadata['dtype'] != 'float32':
            raise ValueError('saved HF reference is not the official CPU FP32 run')
        directory = args.output / case['id']
        directory.mkdir()
        output = directory / 'native.json'
        argv = ['/usr/bin/sandbox-exec', '-f', str(profile), str(args.helper), str(args.model),
                str(wav), '1', '-1', str(output)]
        run_probe(argv, directory, 'native', {'PATH': '/usr/bin:/bin'})
        observed = json.loads(output.read_text())
        expected = {'audio_samples': case['samples'], 'input_pcm_f32le_sha256': case['pcm_sha256'],
                    'input_repeats': 1, 'geometry': protocol['geometry'], 'requested_gpu_index': -1,
                    'native_frame_count': case['samples'] // 160 + 1, 'retained_first_frame': 0,
                    'retained_rows': case['samples'] // 160 + 1}
        if any(observed[key] != value for key, value in expected.items()):
            raise ValueError('native F32 actual PCM/config/frame observation differs')
        f32 = np.fromfile(str(output) + '.f32', dtype='<f4').reshape(-1, 8)
        q8 = np.fromfile(args.evidence / case['q8_probabilities'], dtype='<f4').reshape(-1, 8)
        hf = np.load(args.evidence / case['hf_probabilities'], allow_pickle=False)
        report = comparisons(f32, q8, hf, case['samples'])
        (directory / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        results.append({'case': case['id'], 'calibration': case['calibration'], 'comparison': report,
                        'native_output': descriptor(output), 'native_probabilities': descriptor(Path(str(output) + '.f32'))})
    if descriptor(args.model) != protocol['model'] or verify(args.sdk) != package:
        raise ValueError('native F32 model or CPU package changed during execution')
    for name, expected in protocol['original_inputs'].items():
        if descriptor(args.evidence / name) != expected:
            raise ValueError('original inference output changed during control')
    for case in protocol['cases']:
        root = args.evidence if case['calibration'] else args.corpus
        if descriptor(root / case['wav']) != case['wav_identity']:
            raise ValueError('fixed WAV changed during control')
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tools, text=True).strip(),
              'protocol_sha256': digest(protocol_path), 'scope': protocol['scope'],
              'native_f32_neural_inference_performed': True, 'original_gate_status_changed': False,
              'release_qualified': False, 'results': results, 'limitations': protocol['limitations']}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'fixed_native_f32_control_completed': True, 'cases': len(results), 'original_gate_status_changed': False}))


if __name__ == '__main__':
    main()
