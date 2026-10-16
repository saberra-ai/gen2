"""Fresh tagged native/public corpus runs with immutable cached FP32 oracles.

Every quality failure remains a result; no slice, reference or threshold selection.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from build_valid_mel_candidate import descriptor
from corpus import pcm_header
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json
from qualify_corpus import validate_corpus, validate_oracle_environment
from qualify_valid_mel_pnola import execute_guarded
from score_valid_mel_corpus import score_slice
from sdk_archive import inventory


def admit_reference(item, work, legacy_protocol, legacy_receipt):
    """Bind reused oracle bytes to the original completed, unmodified qualification."""
    name = item['id'] + '-fp32'
    array_path = work / 'oracle' / (name + '.npy')
    metadata_path = work / 'oracle' / (name + '.json')
    expected = legacy_receipt['raw_artifacts'][name + '.npy']
    if (descriptor(array_path) != {key: expected[key] for key in ['bytes', 'sha256']}
            or descriptor(metadata_path)['sha256'] != legacy_receipt['artifacts'][name + '.json']):
        raise ValueError('cached original FP32 oracle differs')
    metadata = json.loads(metadata_path.read_text())
    for key, tool in [('script_sha256', 'reference.py'), ('lock_sha256', 'runtime.lock.json'),
                      ('environment_lock_sha256', 'reference/uv.lock')]:
        if metadata[key] != legacy_protocol['tool_sha256'][tool]:
            raise ValueError('cached FP32 script/model/environment provenance differs')
    validate_oracle_environment(metadata)
    return np.load(array_path, allow_pickle=False), metadata


def main():
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    protocol_path = tools / 'valid-mel-corpus.protocol.json'
    raw_protocol = protocol_path.read_bytes(); p = json.loads(raw_protocol)
    work = Path(p['work_path']); output = Path(p['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError('fresh tagged corpus output required')
    def frozen():
        if protocol_path.read_bytes() != raw_protocol:
            raise ValueError('tagged corpus protocol changed')
        for records, root in [(p['tools'], tools), (p['repository_files'], repo)]:
            for name, expected in records.items():
                if descriptor(root / name) != expected:
                    raise ValueError('tagged corpus source differs: ' + name)
        if inventory(work) != p['inventory']:
            raise ValueError('tagged corpus input inventory differs')
        for name, expected in p['preparation_records'].items():
            if descriptor(Path(name)) != expected:
                raise ValueError('tagged corpus preparation record differs')
        environment = {'python': platform.python_version(), 'numpy': np.__version__,
                       'interpreter': descriptor(Path(sys.executable).resolve(strict=True))}
        if environment != p['scoring_environment']:
            raise ValueError('tagged corpus scoring environment differs')
    frozen()
    legacy_protocol = json.loads((tools / 'corpus.protocol.json').read_text())
    legacy_receipt = json.loads(Path(p['legacy_receipt_path']).read_text())
    manifest = validate_corpus(work / 'corpus')
    if (descriptor(work / 'corpus/manifest.json')['sha256'] != legacy_protocol['corpus_manifest_sha256']
            or [item['id'] for item in manifest['slices']] != legacy_protocol['slice_ids']
            or p['quality_gates'] != legacy_protocol['scoring']['per_slice_gates']
            or p['legacy_scoring'] != legacy_protocol['scoring']
            or p['scoring'] != {**legacy_protocol['scoring'],
                                'native_frame_rule': 'floor(N/160); retain every valid-Mel row; full audio UEM'}):
        raise ValueError('original corpus/scoring/gates must remain unchanged')
    references = {item['id']: admit_reference(item, work, legacy_protocol, legacy_receipt) for item in manifest['slices']}
    packages = {backend: verify(work / backend / 'nemo') for backend in ['cpu', 'metal']}
    for backend in ['cpu', 'metal']:
        build = json.loads(Path(p['helper_build_records'][backend]).read_text())
        if (build['models_loaded'] is not False or build['neural_calls'] != 0
                or build['rust_source_files'] != p['repository_files']):
            raise ValueError('tagged corpus helper compilation source differs')
        for name, expected in build['helper_source_files'].items():
            if p['preparation_records'].get(name) != expected:
                raise ValueError('tagged corpus native compiler input differs')
        for name in ['native-stream', 'public-stream']:
            if descriptor(work / backend / 'bin' / name) != {key: build['inputs']['bin/' + name][key] for key in ['bytes', 'sha256']}:
                raise ValueError('tagged corpus helper binary differs from compilation receipt')
    output.mkdir(parents=True); (output / 'protocol.json').write_bytes(raw_protocol)
    profile = output / 'runtime.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write* (subpath '
                       + json.dumps(str(work)) + '))\n(deny file-read* '
                       + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n')
    temp = output / 'tmp'; temp.mkdir()
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'TMPDIR': str(temp) + '/'}
    commands = []; scores = []; guard = None
    r = {'schema_version': 1, 'all_fixed_pairs_completed': False, 'all_slice_quality_gates_passed': False,
         'release_qualified': False, 'scope': p['scope'], 'cached_original_fp32_oracles_reused': 8,
         'fresh_hf_or_original_nemo_calls': 0, 'protocol': descriptor(protocol_path), 'results': scores}
    def run(name, backend, executable, arguments):
        frozen(); guard.before_model()
        argv = ['/usr/bin/sandbox-exec', '-f', str(profile), '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', str(executable), *map(str, arguments)]
        command = {'name': name, 'argv': argv, 'environment': environment, 'cwd': str(output)}
        commands.append(command); atomic_json(output / 'commands.json', commands); started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                execute_guarded(argv, cwd=output, environment=environment, stdout=stdout, stderr=stderr, guard=guard,
                                record=command, persist=lambda: atomic_json(output / 'commands.json', commands), timeout=p['command_timeout_seconds'])
        finally:
            command['elapsed_seconds'] = time.monotonic() - started; atomic_json(output / 'commands.json', commands)
        if command['returncode'] != 0:
            raise RuntimeError('tagged corpus inference failed: ' + name)
        images = audit_loaded_libraries((output / (name + '.stderr.log')).read_text(), work / backend / 'nemo', packages[backend], executable)
        atomic_json(output / (name + '.images.json'), images)
        return (output / (name + '.stdout.log')).read_text()
    try:
        r['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        guard = StorageGuard(output, **p['storage'])
        for item in manifest['slices']:
            wav = work / 'corpus' / item['wav']; reference_path = work / 'corpus' / item['reference']
            audio = wav.read_bytes(); offset, samples = pcm_header(audio[:65536], len(audio))
            if samples != item['audio_samples']:
                raise ValueError('corpus PCM sample count differs')
            pcm = np.frombuffer(audio, dtype='<i2', offset=offset, count=samples).astype(np.float32) / np.float32(32768)
            pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
            fp32, metadata = references[item['id']]
            if metadata['wav_sha256'] != manifest['files'][item['wav']]['sha256']:
                raise ValueError('oracle WAV binding differs')
            for backend in ['cpu', 'metal']:
                name = item['id'] + '-' + backend
                native_path = output / (name + '-native.json'); public_raw = output / (name + '-public.f32')
                run(name + '-native', backend, work / backend / 'bin/native-stream',
                    [work / 'assets/Nemotron-3-Diarization.q8_0.gguf', wav, '1', '-1' if backend == 'cpu' else '0', native_path])
                public = json.loads(run(name + '-public', backend, work / backend / 'bin/public-stream',
                    (['--metal'] if backend == 'metal' else [])
                    + [work / 'assets', work / backend / 'nemo/lib/libnemo_speech_asr_c.dylib', wav, '1', public_raw]))
                public_path = output / (name + '-public.json'); atomic_json(public_path, public)
                score = score_slice(item, public, json.loads(native_path.read_text()), public_raw.read_bytes(),
                                    Path(str(native_path) + '.f32').read_bytes(), fp32, metadata,
                                    json.loads(reference_path.read_text()), pcm_sha)
                score.update(all_slice_gates_passed=all(score['gates'].values()), input_pcm_f32le_sha256=pcm_sha,
                             oracle_array=descriptor(work / 'oracle' / (item['id'] + '-fp32.npy')),
                             oracle_metadata=descriptor(work / 'oracle' / (item['id'] + '-fp32.json')))
                atomic_json(output / (name + '-score.json'), score)
                scores.append(score); atomic_json(output / 'scores.json', scores)
        frozen()
        r['all_fixed_pairs_completed'] = len(scores) == 16
        r['all_slice_quality_gates_passed'] = len(scores) == 16 and all(s['all_slice_gates_passed'] for s in scores)
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release(); r['storage'] = guard.record
        atomic_json(output / 'qualification.json', r)


if __name__ == '__main__':
    main()
