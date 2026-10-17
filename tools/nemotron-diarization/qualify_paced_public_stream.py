"""One endpoint-paced public CPU trial with a frozen prior independent C ABI reference."""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from build_context_probe import descriptor
from compare_stream_valid_mel import compare
from frozen_git_sources import verify_git_sources
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from paced_stream_metrics import paced_timing
from qualification_storage import StorageGuard, atomic_json
from sdk_archive import inventory

VERSION = 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1'
OLD_VERSION = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'


def compare_reference(public, native, expected_raw, actual_raw):
    if (public['runtime_version'] != VERSION or native['runtime_version'] != VERSION
            or public['backend'] != 'cpu' or native['requested_gpu_index'] != -1):
        raise ValueError('exact pooled CPU runtime required for both records')
    result = compare({**public, 'runtime_version': OLD_VERSION}, {**native, 'runtime_version': OLD_VERSION}, expected_raw, actual_raw)
    return {**result, 'runtime_profile': 'gen2-valid-mel-pool-v1', 'runtime_version': VERSION,
            'probabilities_bit_exact': expected_raw == actual_raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    work, out = Path(p['work_path']), Path(p['output_path'])
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64' or out.exists() or out.is_symlink()
            or out.is_relative_to(repo) or out.is_relative_to(work) or work.is_relative_to(out)):
        parser.error('fresh disjoint output on actual macOS arm64 required')

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('paced execution protocol changed')
        for name, identity in p['tools'].items():
            if descriptor(tools / name) != identity:
                raise ValueError('paced execution tool changed: ' + name)
        if inventory(work) != p['work_inventory']:
            raise ValueError('paced staged input inventory changed')
        for path, identity in p['prerequisite_records'].items():
            if descriptor(path) != identity:
                raise ValueError('paced prerequisite changed: ' + path)
        if {'python': platform.python_version(), 'interpreter': descriptor(Path(sys.executable).resolve(strict=True))} != p['controller_environment']:
            raise ValueError('paced controller interpreter changed')

    frozen()
    build_protocol = json.loads((work / 'baseline/build-protocol.json').read_text())
    build = json.loads((work / 'baseline/construction.json').read_text())
    if build['constructed'] is not True or build['model_calls'] != 0 or descriptor(work / 'bin/paced-public') != build['binary']:
        raise ValueError('successful model-free paced construction required')
    verify_git_sources(repo, build['source_commit'], build_protocol['repository_files'])
    package = work / 'nemo'
    manifest = json.loads((package / 'package.json').read_text())
    if manifest['runtime_reported_version'] != VERSION or manifest['backend'] != 'cpu':
        raise ValueError('exact qualified pooled CPU package required')
    native = json.loads((work / 'baseline/native-80s.json').read_text())
    expected_raw = (work / 'baseline/native-80s.json.f32').read_bytes()
    if native['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256'] or native['audio_samples'] != 1280640:
        raise ValueError('saved independent native PCM differs')
    out.mkdir()
    (out / 'protocol.json').write_bytes(raw)
    results, home, temporary = out / 'results', out / 'home', out / 'tmp'
    for path in [results, home, temporary]:
        path.mkdir()
    quote = lambda path: json.dumps(str(Path(path).resolve()), ensure_ascii=False)
    runtime = out / 'runtime.sb'
    runtime.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n'
        + '(allow file-write* ' + ' '.join('(subpath ' + quote(path) + ')' for path in [results, home, temporary]) + ' (literal "/dev/null"))\n'
        + '(deny file-read-data (subpath ' + quote(results) + '))\n'
        + ''.join('(deny file-read* (subpath ' + quote(path) + '))\n'
                  for path in [repo, work / 'baseline', '/opt/homebrew', '/Users/victor/workspace/pnola', '/Users/victor/workspace/gen2'])
        + '(deny file-read-data (require-all (subpath ' + quote(p['source_cache_root']) + ') '
          '(require-not (subpath ' + quote(work) + ')) (require-not (subpath ' + quote(out) + '))))\n')
    premodel = out / 'premodel.sb'
    premodel.write_text(runtime.read_text() + '(deny file-read* (subpath ' + quote(work / 'assets') + '))\n'
        + '(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'trial_completed': False, 'commands': [], 'neural_commands_started': [], 'loaded_images': {},
         'physical_microphone_used': False, 'gui_used': False, 'release_qualified': False, 'limits': p['limits']}
    guard = None
    persist = lambda: atomic_json(out / 'qualification.json', r)

    def run(name, argv, *, sandbox=runtime, trace=False, inference=False, expected=0, require_runtime=True):
        frozen()
        guard.before_model()
        target = Path(argv[0])
        if trace:
            argv = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)]
        actual, environment = sandboxed_command(argv, sandbox, env)
        row = {'name': name, 'argv': actual, 'environment': environment, 'cwd': str(out),
               'inference_admitted': inference, 'expected_returncode': expected}
        r['commands'].append(row)
        if inference:
            r['neural_commands_started'].append(name)
        persist()
        started = time.monotonic()
        try:
            with (out / (name + '.stdout.log')).open('xb') as stdout, (out / (name + '.stderr.log')).open('xb') as stderr:
                process = subprocess.Popen(actual, cwd=out, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
                row['returncode'] = guard.wait(process, row, 600)
            row['elapsed_seconds'] = time.monotonic() - started
            if row['returncode'] != expected:
                raise ValueError('paced trial command failed: ' + name)
            if trace:
                text = (out / (name + '.stderr.log')).read_text()
                row['maximum_resident_set_size_bytes'] = int(re.search(r'(\d+)\s+maximum resident set size', text)[1])
                r['loaded_images'][name] = audit_loaded_libraries(text, package, manifest, target, require_runtime=require_runtime)
            return (out / (name + '.stdout.log')).read_text()
        finally:
            persist()

    try:
        guard = StorageGuard(out, 5 * 2**30, 2**30, 256 * 2**20)
        marker = results / 'prior-output.txt'
        marker.write_bytes(b'not a model')
        control = '''import ctypes,json,pathlib,sys
for path,mode in json.loads(sys.argv[1]):
 try:open(path,mode).close()
 except PermissionError:continue
 raise RuntimeError('unexpected protected access: '+path)
lib=ctypes.CDLL(None,use_errno=True);lib.realpath.argtypes=[ctypes.c_char_p,ctypes.c_void_p];lib.realpath.restype=ctypes.c_void_p
for path in sys.argv[2:]:
 buffer=ctypes.create_string_buffer(4096)
 if not lib.realpath(path.encode(),buffer):raise OSError(ctypes.get_errno(),'canonicalization failed',path)
print('protected reads/writes denied; selected bundle libc paths admitted')
'''
        model = work / 'assets/Nemotron-3-Diarization.q8_0.gguf'
        run('runtime-controls', ['/usr/bin/python3', '-B', '-c', control, json.dumps([
            [str(repo / 'Cargo.toml'), 'rb'], [str(marker), 'rb'], [str(work / 'baseline/native-80s.json'), 'rb'],
            [str(work / 'baseline/native-80s.json.f32'), 'rb'], [str(model), 'wb']]), str(model.parent), str(model)])
        run('premodel-controls', ['/usr/bin/python3', '-B', '-c', control,
            json.dumps([[str(model), 'rb']])], sandbox=premodel)
        version = run('version-only', [work / 'bin/load-probe', package / 'lib' / manifest['entry_library'], VERSION], sandbox=premodel, trace=True)
        if version.strip() != VERSION:
            raise ValueError('paced runtime version differs')
        error_target = results / 'error-before-model.json'
        run('error-before-model', [work / 'bin/paced-public', work / 'assets', package / 'lib' / manifest['entry_library'],
            work / 'assets/live-80s.wav', error_target], sandbox=premodel, trace=True, expected=1, require_runtime=False)
        error_value = json.loads(error_target.read_text())
        if (error_value['complete'] is not False or 'Operation not permitted (os error 1)' not in error_value['error'] or error_value['pushes']
                or error_value['observations'] or error_value['checkpoints'] or Path(str(error_target) + '.f32').exists()):
            raise ValueError('pre-model partial evidence control differs')
        r['pre_model_error_control'] = error_value
        target = results / 'paced-public.json'
        run('paced-public', [work / 'bin/paced-public', work / 'assets', package / 'lib' / manifest['entry_library'],
                             work / 'assets/live-80s.wav', target], trace=True, inference=True)
        public = json.loads(target.read_text())
        actual_raw = Path(str(target) + '.f32').read_bytes()
        if public['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256'] or public['input_repeats'] != 1:
            raise ValueError('paced actual normalized input identity differs')
        r['comparison'] = compare_reference(public, native, expected_raw, actual_raw)
        r['timing'] = paced_timing(public)
        r['output_files'] = {path.name: descriptor(path) for path in results.iterdir() if path.is_file()}
        frozen()
        if [row['name'] for row in r['commands']] != p['command_order'] or r['neural_commands_started'] != ['paced-public'] or len(r['loaded_images']) != 3:
            raise ValueError('paced fixed execution sequence differs')
        r['trial_completed'] = True
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
