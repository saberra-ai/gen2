"""One fixed session-context contrast on already-failed AMI targets.

Process [0,420), score only [300,420), compare with the existing cold crop.
No parameter search, release-gate replacement, or warm FP32/Q8 quality claim.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from ami_prefix_metrics import contrast, load_readonly_scorer
from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from compare_stream_valid_mel import compare
from diagnose_ami_reference import calibrate_engine
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_pooled_mixed import mixed_runtime_policy, sandbox_path
from qualify_macos_portable_runtime import CONTROL
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory

IDS = ['ami-ES2004a-Mix-Headset', 'ami-ES2004a-Array1-01']
VERSION = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw_protocol = args.protocol.read_bytes()
    p = json.loads(raw_protocol)
    work, output = Path(p['work_path']), Path(p['output_path'])
    if (output.exists() or output.is_symlink() or output.is_relative_to(work)
            or work.is_relative_to(output) or p['channels'] != IDS
            or p['source_interval_seconds'] != [0, 420] or p['scored_interval_seconds'] != [300, 420]):
        raise ValueError('fresh disjoint output and exact fixed two-channel contrast required')

    def frozen():
        if args.protocol.read_bytes() != raw_protocol:
            raise ValueError('prefix diagnostic protocol changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('prefix source changed: ' + name)
        if inventory(work) != p['work_inventory']:
            raise ValueError('prefix input inventory changed')
        for name, expected in p['preparation_records'].items():
            if descriptor(Path(name)) != expected:
                raise ValueError('prefix prerequisite changed')
        actual = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'interpreter': descriptor(Path(sys.executable).resolve(strict=True))}
        if actual != p['scoring_environment']:
            raise ValueError('prefix scoring environment changed')

    frozen()
    prior = json.loads((work / 'baseline/qualification.json').read_text())
    if (prior['source_commit'] != p['cold_source_commit']
            or prior['all_fixed_pairs_completed'] is not True or prior['all_slice_quality_gates_passed'] is not False):
        raise ValueError('completed original qualification with retained quality failures required')
    package = work / 'nemo'
    manifest = verify(package)
    if manifest['runtime_profile'] != 'gen2-valid-mel-v1' or manifest['backend'] != 'cpu':
        raise ValueError('exact previously qualified CPU runtime required')
    engine = load_readonly_scorer(work / 'scoring/md_eval.py', p['independent_scorer'])
    calibrations = calibrate_engine(engine)
    cold = {}
    for name in IDS:
        directory = work / 'baseline'
        native = json.loads((directory / (name + '-cpu-native.json')).read_text())
        public = json.loads((directory / (name + '-cpu-public.json')).read_text())
        raw = (directory / (name + '-cpu-public.f32')).read_bytes()
        comparison = compare(public, native, (directory / (name + '-cpu-native.json.f32')).read_bytes(), raw)
        if raw != (directory / (name + '-cpu-native.json.f32')).read_bytes() or comparison['native_frame_count'] != 12000:
            raise ValueError('previous cold crop parity differs')
        cold[name] = (public, raw)
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    atomic_json(output / 'scorer-calibrations.json', calibrations)
    results = output / 'results'
    results.mkdir()
    home, temporary = output / 'home', output / 'tmp'
    home.mkdir()
    temporary.mkdir()
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}
    runtime = output / 'runtime.sb'
    denied = [Path(x) for x in p['denied_paths']]
    runtime.write_text(mixed_runtime_policy([*denied, work / 'baseline', work / 'references', work / 'scoring'],
        [work, *denied], results))
    premodel = output / 'premodel.sb'
    premodel.write_text(model_free_policy(runtime.read_text()) +
        '(deny file-read* (subpath ' + sandbox_path(work / 'assets') + '))\n')
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'diagnostic_completed': False, 'release_qualified': False, 'original_quality_gate_remains_failed': True,
         'original_gate_status_changed': False, 'fresh_fp32_calls': 0, 'protocol': descriptor(args.protocol),
         'commands': [], 'neural_commands_started': [], 'comparisons': {}, 'results': {}, 'loaded_images': {},
         'limits': p['limits']}
    guard = None
    persist = lambda: atomic_json(output / 'qualification.json', r)

    def run(name, argv, *, sandbox=runtime, inference=False, trace=False):
        frozen()
        guard.before_model()
        if inference:
            r['neural_commands_started'].append(name)
        child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)] if trace else list(map(str, argv))
        actual, environment = sandboxed_command(child, sandbox, env)
        row = {'name': name, 'argv': actual, 'cwd': str(output), 'environment': environment, 'inference_admitted': inference}
        r['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=output, environment=environment, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
        if row['returncode'] != 0:
            raise ValueError('prefix diagnostic command failed: ' + name)
        return (output / (name + '.stdout.log')).read_text()

    try:
        guard = StorageGuard(output, **p['storage'])
        prior_output = results / 'prior-output-control.txt'
        prior_output.write_bytes(b'not a model')
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for path in probes:
            path.write_bytes(b'not a model')
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps({
            'reads': list(map(str, [repo / 'Cargo.toml', prior_output, work / 'baseline/qualification.json',
                work / 'references/transcriber.json', work / 'scoring/md_eval.py'])),
            'writes': [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf')]})])
        run('premodel-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps({
            'reads': [*map(str, probes), str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf')], 'writes': []})], sandbox=premodel)
        version = run('version-only', [work / 'bin/load-probe', package / 'lib' / manifest['entry_library'], VERSION], sandbox=premodel, trace=True)
        if version.strip() != VERSION:
            raise ValueError('prefix native version differs')
        r['loaded_images']['version-only'] = audit_loaded_libraries((output / 'version-only.stderr.log').read_text(), package, manifest, work / 'bin/load-probe')
        references = {name: json.loads((work / 'references' / (name + '.json')).read_text())
                      for name in ['transcriber', 'published_lexical']}
        for name in IDS:
            wav = work / 'assets' / (name + '-prefix.wav')
            native_path = results / (name + '-native.json')
            public_raw = results / (name + '-public.f32')
            run(name + '-native', [work / 'bin/native-stream', work / 'assets/Nemotron-3-Diarization.q8_0.gguf', wav, '1', '-1', native_path], inference=True, trace=True)
            public = json.loads(run(name + '-public', [work / 'bin/public-stream', work / 'assets',
                package / 'lib' / manifest['entry_library'], wav, '1', public_raw], inference=True, trace=True))
            atomic_json(results / (name + '-public.json'), public)
            native = json.loads(native_path.read_text())
            raw, actual_raw = Path(str(native_path) + '.f32').read_bytes(), public_raw.read_bytes()
            comparison = compare(public, native, raw, actual_raw)
            if (comparison['native_frame_count'] != 42000 or comparison['retained_first_frame'] != 0
                    or comparison['retained_rows'] != 42000):
                raise ValueError('prefix full frame coverage differs')
            for value in [public, native]:
                if value['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256'][name] or value['input_repeats'] != 1:
                    raise ValueError('prefix actual PCM identity differs')
            for suffix, executable in [('native', 'native-stream'), ('public', 'public-stream')]:
                case = name + '-' + suffix
                r['loaded_images'][case] = audit_loaded_libraries((output / (case + '.stderr.log')).read_text(), package, manifest, work / 'bin' / executable)
            r['comparisons'][name] = comparison
            result = contrast(engine, cold[name][0], public, cold[name][1], actual_raw, references)
            previous = json.loads((work / 'baseline' / (name + '-cpu-score.json')).read_text())
            scores = result['reference_policies']['transcriber']['scores']
            if (scores['cold_shared']['frame'] != previous['q8_shared_threshold']
                    or scores['cold_production']['frame'] != previous['q8_production_default']):
                raise ValueError('unchanged original cold failure not reproduced')
            r['results'][name] = result
            persist()
        frozen()
        r['diagnostic_completed'] = len(r['results']) == 2
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
