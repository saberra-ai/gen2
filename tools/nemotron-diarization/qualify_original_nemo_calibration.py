"""Admit one source-pinned original-NeMo calibration forward and compare saved HF.

No target corpus, new HF/native execution, parameter selection or quality waiver.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from build_valid_mel_candidate import descriptor
from frontend_metrics import difference
from original_nemo_metrics import compare_original
from qualification_storage import StorageGuard, atomic_json
from qualify_valid_mel_pnola import execute_guarded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)
    tools = Path(__file__).resolve().parent
    env = Path(p['environment_root'])
    site = env / 'venv/lib/python3.12/site-packages'
    output = Path(p['output_path'])
    if output.exists() or output.is_symlink():
        raise FileExistsError('fresh original-NeMo output required')

    def frozen():
        if args.protocol.read_bytes() != protocol_bytes:
            raise ValueError('original-NeMo protocol changed')
        for records, root in [(p['tools'], tools), (p['environment_files'], env),
                              (p['installed_nemo_files'], site)]:
            for name, expected in records.items():
                if descriptor(root / name) != expected:
                    raise ValueError('original-NeMo frozen source/environment differs: ' + name)
        for name, expected in p['inputs'].items():
            if descriptor(Path(name)) != expected:
                raise ValueError('original-NeMo frozen input differs: ' + name)
        actual = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(site)])}
        if actual != p['packages']:
            raise ValueError('original-NeMo installed dependency versions differ')

    frozen()
    metadata = json.loads(Path(p['reference_metadata_path']).read_text())
    if (metadata['shape'] != [111, 8] or metadata['audio_samples'] != 17872
            or metadata['input_pcm_f32le_sha256'] != p['pcm']['sha256']
            or metadata['geometry'] != dict(zip(['chunk', 'right', 'left', 'fifo', 'cache', 'update'], [9, 4, 0, 264, 264, 222]))
            or metadata['model_revision'] != 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
            or metadata['device'] != 'cpu' or metadata['dtype'] != 'float32'):
        raise ValueError('saved original calibration reference differs')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_bytes)
    home = output / 'home'
    temp = output / 'tmp'
    home.mkdir()
    temp.mkdir()
    policy = output / 'runtime.sb'
    policy.write_text('(version 1)\n(allow default)\n(deny network*)\n'
                      + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n'
                      + '(deny file-write* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['readonly_paths']) + ')\n')
    environment = {'HOME': str(home), 'TMPDIR': str(temp) + '/', 'PATH': '/usr/bin:/bin',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                   'WANDB_MODE': 'disabled', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    argv = ['/usr/bin/sandbox-exec', '-f', str(policy), str(env / 'venv/bin/python'),
            str(tools / 'original_nemo_calibration.py'), '--protocol', str(output / 'protocol.json'), '--output', str(output)]
    result = {'argv': argv, 'environment': environment, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=tools.parents[1], text=True).strip(),
        'protocol': descriptor(args.protocol), 'release_qualified': False, 'quality_qualified': False}
    persist = lambda: atomic_json(output / 'qualification.json', result)
    guard = None
    started = time.monotonic()
    try:
        guard = StorageGuard(output, 5 * 1024**3, 1024**3, 256 * 1024**2)
        guard.before_model()
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            execute_guarded(argv, cwd=output, environment=environment, stdout=stdout, stderr=stderr,
                            guard=guard, record=result, persist=persist, timeout=180)
        if result['returncode'] != 0:
            raise ValueError('original-NeMo calibration child failed')
        forward = json.loads((output / 'forward-completed.json').read_text())
        if (forward['passed'] is not True or forward['calls'] != 1 or forward['strict_restore'] is not True
                or forward['shapes']['probabilities'] != [1, 111, 8]
                or forward['frontend_valid_lengths'] != [111]):
            raise ValueError('original-NeMo calibration observation differs')
        original = np.fromfile(output / 'probabilities.f32', dtype='<f4').reshape(-1, 8)
        reference = np.load(p['reference_path'], allow_pickle=False)
        result['full_valid_probability_comparison'] = compare_original(original, reference, 17872)
        mel = np.fromfile(output / 'frontend-valid.f32', dtype='<f4').reshape(111, 128)
        first = np.fromfile(p['hf_first_mel_path'], dtype='<f4').reshape(105, 128)
        last = np.fromfile(p['hf_final_mel_path'], dtype='<f4').reshape(39, 128)
        result['frontend_comparisons'] = {
            'first_chunk_104_valid_rows': difference(mel[:104], first[:104], 1e-3),
            'final_chunk_39_valid_rows': difference(mel[72:], last, 1e-3)}
        frozen()
        result['fixed_calibration_execution_passed'] = True
        result['numerical_agreement_promoted_to_quality_gate'] = False
    except BaseException as error:
        result.update(fixed_calibration_execution_passed=False, error=repr(error))
        raise
    finally:
        if guard is not None:
            guard.release()
        result['elapsed_seconds'] = time.monotonic() - started
        result['storage'] = guard.record if guard is not None else {'admission_failed_before_child': True}
        persist()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
