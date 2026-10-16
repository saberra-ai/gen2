"""Two fixed original-NeMo CPU AMI diagnostics; no quality-gate replacement."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from build_valid_mel_candidate import descriptor
from compare import segments
from original_nemo_metrics import compare_original
from qualification_storage import StorageGuard, atomic_json
from qualify_valid_mel_pnola import execute_guarded
from score import der


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
        raise FileExistsError('fresh original-NeMo AMI output required')

    def frozen():
        if args.protocol.read_bytes() != protocol_bytes:
            raise ValueError('original-NeMo AMI protocol changed')
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
            raise ValueError('original-NeMo dependency versions differ')

    frozen()
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_bytes)
    results = []
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tools.parents[1],
                                                     text=True).strip(),
              'protocol': descriptor(args.protocol), 'results': results,
              'quality_qualified': False, 'release_qualified': False}
    persist = lambda: atomic_json(output / 'qualification.json', result)
    guard = None
    try:
        guard = StorageGuard(output, 5 * 1024**3, 1024**3, 256 * 1024**2)
        for case in p['cases']:
            directory = output / case['id']
            directory.mkdir()
            for name in ['home', 'tmp']:
                (directory / name).mkdir()
            child = {key: p[key] for key in ['nemo_source_commit', 'checkpoint_path', 'checkpoint']}
            child.update({key: case[key] for key in ['samples', 'pcm_path', 'pcm']})
            atomic_json(directory / 'protocol.json', child)
            policy = directory / 'runtime.sb'
            policy.write_text('(version 1)\n(allow default)\n(deny network*)\n'
                              + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n'
                              + '(deny file-write* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['readonly_paths']) + ')\n')
            environment = {'HOME': str(directory / 'home'), 'TMPDIR': str(directory / 'tmp') + '/',
                           'PATH': '/usr/bin:/bin', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                           'HF_DATASETS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
                           'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
            argv = ['/usr/bin/sandbox-exec', '-f', str(policy), str(env / 'venv/bin/python'),
                    str(tools / 'original_nemo_calibration.py'), '--protocol', str(directory / 'protocol.json'),
                    '--output', str(directory)]
            item = {'case': case['id'], 'argv': argv, 'environment': environment}
            results.append(item)
            persist()
            guard.before_model()
            started = time.monotonic()
            with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
                execute_guarded(argv, cwd=directory, environment=environment, stdout=stdout, stderr=stderr,
                                guard=guard, record=item, persist=persist, timeout=600)
            item['elapsed_seconds'] = time.monotonic() - started
            if item['returncode'] != 0:
                raise ValueError('original-NeMo fixed AMI child failed')
            forward = json.loads((directory / 'forward-completed.json').read_text())
            if (forward['passed'] is not True or forward['calls'] != 1 or forward['strict_restore'] is not True
                    or forward['shapes']['probabilities'] != [1, 12000, 8]
                    or forward['frontend_valid_lengths'] != [12000]):
                raise ValueError('original-NeMo fixed AMI observation differs')
            original = np.fromfile(directory / 'probabilities.f32', dtype='<f4').reshape(-1, 8)
            hf = np.load(case['reference_path'], allow_pickle=False)
            metadata = json.loads(Path(case['reference_metadata_path']).read_text())
            if (metadata['audio_samples'] != case['samples'] or metadata['shape'] != [12000, 8]
                    or metadata['input_pcm_f32le_sha256'] != case['pcm']['sha256']
                    or metadata['geometry'] != dict(zip(['chunk', 'right', 'left', 'fifo', 'cache', 'update'], [9, 4, 0, 264, 264, 222]))
                    or metadata['device'] != 'cpu' or metadata['dtype'] != 'float32'
                    or metadata['model_revision'] != 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'):
                raise ValueError('original-NeMo cached HF identity differs')
            item['probability_comparison'] = compare_original(original, hf, case['samples'])
            reference = json.loads(Path(case['labels_path']).read_text())
            item['original_shared_der'] = der(reference, segments(original), 120)
            item['cached_hf_shared_der'] = der(reference, segments(hf), 120)
            item['original_shared_der_at_most_original_0_20'] = item['original_shared_der']['der'] <= .20
            frozen()
            persist()
        result['both_fixed_original_model_calls_completed'] = True
        result['original_ami_shared_gates_passed'] = all(i['original_shared_der_at_most_original_0_20'] for i in results)
    except BaseException as error:
        result.update(both_fixed_original_model_calls_completed=False, error=repr(error))
        raise
    finally:
        if guard is not None:
            guard.release()
        result['storage'] = guard.record if guard is not None else {'admission_failed_before_child': True}
        persist()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
