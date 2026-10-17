"""One preregistered low/large context contrast on training-exposed AMI dev."""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import scipy

from ami_prefix_metrics import load_readonly_scorer
from build_context_probe import descriptor
from compare import segments
from compare_stream_valid_mel import compare
from context_development_metrics import production_turns
from diagnose_ami_reference import calibrate_engine, official_der
from frozen_git_sources import verify_git_sources
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from qualification_storage import StorageGuard, atomic_json
from score import der
from sdk_archive import inventory

VERSION = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
GEOMETRIES = {'low_latency': {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222},
              'large_context': {'chunk': 340, 'right': 40, 'left': 0, 'fifo': 40, 'cache': 264, 'update': 300}}
CONTROL = '''import json,sys
for path, mode in json.loads(sys.argv[1]):
 try:
  open(path, mode).close()
 except PermissionError:
  continue
 raise RuntimeError('unexpected sandbox access: ' + path)
print('all sandbox controls denied')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual macOS arm64 required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    work, out = Path(p['work_path']), Path(p['output_path'])
    if (out.exists() or out.is_symlink() or out.is_relative_to(repo) or out.is_relative_to(work)
            or work.is_relative_to(out) or p['meeting'] != 'ES2011a' or p['crop_seconds'] != [0, 300]
            or p['channels'] != ['Mix-Headset', 'Array1-01'] or p['geometries'] != GEOMETRIES):
        raise ValueError('fresh disjoint output and exact fixed contrast required')

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('development execution protocol changed')
        for name, identity in p['tools'].items():
            if descriptor(tools / name) != identity:
                raise ValueError('development source changed: ' + name)
        if inventory(work) != p['work_inventory']:
            raise ValueError('development staged input inventory changed')
        for path, identity in p['prerequisite_records'].items():
            if descriptor(path) != identity:
                raise ValueError('development prerequisite changed: ' + path)
        if {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'interpreter': descriptor(Path(sys.executable).resolve(strict=True))} != p['scoring_environment']:
            raise ValueError('scoring environment differs')

    frozen()
    # Bind the reused public binary to its historical compilation sources too.
    old_build = json.loads((work / 'provenance/public-build.json').read_text())
    verify_git_sources(repo, old_build['source_commit'], old_build['rust_source_files'])
    if descriptor(work / 'bin/public-stream') != {key: old_build['inputs']['bin/public-stream'][key]
                                                 for key in ['bytes', 'sha256']}:
        raise ValueError('public helper compilation identity differs')
    package = work / 'nemo'
    manifest = json.loads((package / 'package.json').read_text())
    if (manifest['runtime_reported_version'] != VERSION or manifest['backend'] != 'cpu'
            or manifest['runtime_profile'] != 'gen2-valid-mel-v1'):
        raise ValueError('exact old CPU runtime required')
    engine = load_readonly_scorer(work / 'scoring/md_eval.py', p['independent_scorer'])
    calibrations = calibrate_engine(engine)
    refs = {name: json.loads((work / 'references' / (name + '.json')).read_text())
            for name in ['transcriber', 'published-ntt-mfa']}
    out.mkdir()
    (out / 'protocol.json').write_bytes(raw)
    atomic_json(out / 'scorer-calibrations.json', calibrations)
    results = out / 'results'
    results.mkdir()
    home, temporary = out / 'home', out / 'tmp'
    home.mkdir()
    temporary.mkdir()
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}
    quote = lambda path: json.dumps(str(Path(path).resolve()), ensure_ascii=False)
    runtime = out / 'runtime.sb'
    runtime.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n'
        + '(allow file-write* ' + ' '.join('(subpath ' + quote(path) + ')' for path in [results, home, temporary])
        + ' (literal "/dev/null"))\n'
        + '(deny file-read-data (subpath ' + quote(results) + '))\n'
        + ''.join('(deny file-read* (subpath ' + quote(path) + '))\n'
                  for path in [repo, *map(Path, p['denied_paths']), work / 'references', work / 'scoring', work / 'provenance'])
        + '(deny file-read* (require-all (subpath ' + quote(p['source_cache_root']) + ') '
          '(require-not (subpath ' + quote(work) + ')) (require-not (subpath ' + quote(out) + '))))\n')
    premodel = out / 'premodel.sb'
    premodel.write_text(runtime.read_text() + '(deny file-read* (subpath ' + quote(work / 'assets') + '))\n'
        + '(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'diagnostic_completed': False, 'training_exposed_development': True, 'commands': [],
         'neural_commands_started': [], 'loaded_images': {}, 'comparisons': {}, 'scores': {},
         'native_metrics': {},
         'fresh_fp32_calls': 0, 'original_gate_status_changed': False, 'release_qualified': False,
         'limits': p['limits']}
    guard = None
    persist = lambda: atomic_json(out / 'qualification.json', r)

    def run(name, argv, *, sandbox=runtime, inference=False, trace=False, expected=0):
        frozen()
        guard.before_model()
        row = {'name': name, 'inference_admitted': inference, 'expected_returncode': expected}
        if inference:
            r['neural_commands_started'].append(name)
        target_executable = Path(argv[0])
        if trace:
            argv = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)]
        actual, parent = sandboxed_command(argv, sandbox, env)
        row.update(argv=actual, cwd=str(out), environment=parent)
        r['commands'].append(row)
        persist()
        started = time.monotonic()
        try:
            with (out / (name + '.stdout.log')).open('xb') as stdout, (out / (name + '.stderr.log')).open('xb') as stderr:
                process = subprocess.Popen(actual, cwd=out, env=parent, stdout=stdout, stderr=stderr, start_new_session=True)
                row['returncode'] = guard.wait(process, row, 600)
            row['elapsed_seconds'] = time.monotonic() - started
            if row['returncode'] != expected:
                raise ValueError('development command failed: ' + name)
            if trace:
                text = (out / (name + '.stderr.log')).read_text()
                row['maximum_resident_set_size_bytes'] = int(re.search(r'(\d+)\s+maximum resident set size', text)[1])
                r['loaded_images'][name] = audit_loaded_libraries(text, package, manifest, target_executable)
            return (out / (name + '.stdout.log')).read_text()
        finally:
            persist()

    try:
        guard = StorageGuard(out, 5 * 2**30, 2**30, 256 * 2**20)
        marker = results / 'prior-output.txt'
        marker.write_bytes(b'not a model')
        run('runtime-controls', ['/usr/bin/python3', '-B', '-c', CONTROL, json.dumps([
            [str(repo / 'Cargo.toml'), 'rb'], [str(marker), 'rb'],
            [str(work / 'references/transcriber.json'), 'rb'], [str(work / 'scoring/md_eval.py'), 'rb'],
            [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf'), 'wb']])])
        run('premodel-controls', ['/usr/bin/python3', '-B', '-c', CONTROL, json.dumps([
            [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf'), 'rb']])], sandbox=premodel)
        executable = work / 'bin/load-probe'
        version = run('version-only', [executable, package / 'lib' / manifest['entry_library'], VERSION], sandbox=premodel, trace=True)
        if version.strip() != VERSION:
            raise ValueError('runtime version differs')
        executable = work / 'bin/native-context'
        run('invalid-profile', [executable, 'unused-model', 'unused-audio', '1', '-1', 'invalid', results / 'invalid.json'],
            sandbox=premodel, trace=True, expected=1)
        if 'unknown fixed context profile' not in (out / 'invalid-profile.stderr.log').read_text() or (results / 'invalid.json').exists():
            raise ValueError('invalid-profile control differs')
        for channel in p['channels']:
            wav = work / 'assets' / ('ES2011a-' + channel + '.wav')
            natives = {}
            for profile in GEOMETRIES:
                name = channel + '-' + profile
                target = results / (name + '.json')
                executable = work / 'bin/native-context'
                run(name, [executable, work / 'assets/Nemotron-3-Diarization.q8_0.gguf', wav, '1', '-1', profile, target],
                    inference=True, trace=True)
                native = json.loads(target.read_text())
                raw_values = Path(str(target) + '.f32').read_bytes()
                if (native['native_frame_count'] != 30000 or native['retained_first_frame'] != 0
                        or native['retained_rows'] != 30000 or len(raw_values) != 30000 * 8 * 4
                        or native['geometry'] != GEOMETRIES[profile] or native['runtime_version'] != VERSION
                        or native['audio_samples'] != 4800000 or native['input_repeats'] != 1
                        or native['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256'][channel]):
                    raise ValueError('fixed native geometry/input/frame coverage differs')
                probabilities = np.frombuffer(raw_values, dtype='<f4').reshape(30000, 8)
                if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
                    raise ValueError('invalid development probabilities')
                hypotheses = {'shared': segments(probabilities), 'production': production_turns(native['turns'])}
                r['scores'][name] = {policy: {mode: {'frame': der(reference, hyp, 300.),
                    'official': official_der(engine, reference, hyp, 300.)} for mode, hyp in hypotheses.items()}
                    for policy, reference in refs.items()}
                natives[profile] = (native, raw_values)
                r['native_metrics'][name] = {'rtf': native['elapsed_micros'] / 1e6 / 300.,
                    'cold_load_seconds': native['cold_native_load_micros'] / 1e6,
                    'push_median_seconds': native['native_push_median_micros'] / 1e6,
                    'push_p95_seconds': native['native_push_p95_micros'] / 1e6,
                    'finish_seconds': native['finish_micros'] / 1e6}
                persist()
            name = channel + '-public-low'
            public_raw = results / (name + '.f32')
            executable = work / 'bin/public-stream'
            public = json.loads(run(name, [executable, work / 'assets', package / 'lib' / manifest['entry_library'],
                                          wav, '1', public_raw], inference=True, trace=True))
            atomic_json(results / (name + '.json'), public)
            native, raw_values = natives['low_latency']
            actual = public_raw.read_bytes()
            if (actual != raw_values or public['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256'][channel]
                    or public['input_repeats'] != 1):
                raise ValueError('low-latency public/native actual PCM/output differs')
            r['comparisons'][channel] = compare(public, native, raw_values, actual)
            persist()
        frozen()
        if [row['name'] for row in r['commands']] != p['command_order'] or len(r['neural_commands_started']) != 6:
            raise ValueError('fixed development execution sequence differs')
        r['diagnostic_completed'] = len(r['comparisons']) == 2 and len(r['scores']) == 4 and len(r['loaded_images']) == 8
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
