"""Score every frozen slice; never use a score to replace or tune a case."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib

import numpy as np

from compare import segments
from compare_stream import compare
from corpus import pcm_header
from score import der


GEOMETRY = {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222}
MODEL_REVISION = 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
PROTOCOL_TOOLS = ['qualify_corpus.py', 'reference.py', 'score.py', 'compare.py', 'compare_stream.py', 'corpus.py',
                  'native_stream_probe.cpp', '../../examples/nemotron_stream.rs', 'runtime.lock.json', 'reference/uv.lock']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_corpus(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['inference_performed'] is not False or len(manifest['slices']) != 8:
        raise ValueError('not the frozen eight-slice acquisition manifest')
    for name, expected in manifest['files'].items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError('invalid frozen corpus member')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('frozen corpus member differs: ' + name)
    return manifest


def validate_protocol(protocol_path, corpus_root):
    protocol = json.loads(protocol_path.read_text())
    if digest(corpus_root / 'manifest.json') != protocol['corpus_manifest_sha256']:
        raise ValueError('corpus differs from the committed frozen protocol')
    if set(protocol['tool_sha256']) != set(PROTOCOL_TOOLS):
        raise ValueError('protocol does not pin all qualification tools')
    for name, expected in protocol['tool_sha256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError('qualification tool changed after protocol freeze: ' + name)
    return protocol


def validate_oracle_environment(metadata):
    direct = metadata['transformers_direct_url']
    if direct['vcs_info']['commit_id'] != '683b721971417b3b2c64fba9001f525607547e5a':
        raise ValueError('wrong actual Transformers source commit')
    frozen = tomllib.loads(Path(__file__).with_name('reference').joinpath('uv.lock').read_text())
    normalize = lambda name: re.sub('[-_.]+', '-', name.lower())
    packages = {normalize(p['name']): p['version'] for p in frozen['package']}
    installed = {normalize(k): v for k, v in metadata['installed_packages'].items()}
    if not {'torch', 'transformers', 'numpy', 'soundfile', 'librosa', 'scipy'} <= installed.keys():
        raise ValueError('incomplete actual oracle environment')
    for name, version in installed.items():
        if packages.get(name) != version:
            raise ValueError('unlocked actual oracle package: ' + name)


def score_slice(item, gen2, native, q8_raw, native_raw, fp32, metadata, reference, expected_pcm_digest):
    samples = item['audio_samples']
    duration = samples / 16000
    if type(samples) is not int or not 320 <= samples <= 600 * 16000 or item['uem_seconds'] != [0, duration]:
        raise ValueError('invalid frozen UEM')
    if gen2['accepted_samples'] != samples or metadata['audio_samples'] != samples:
        raise ValueError('inference input differs from frozen slice')
    for output in [gen2, native, metadata]:
        if output.get('input_pcm_f32le_sha256') != expected_pcm_digest or output.get('input_repeats') != 1:
            raise ValueError('actual input PCM digest/repeat semantics differ from frozen slice')
    if metadata['geometry'] != GEOMETRY or metadata['model_revision'] != MODEL_REVISION:
        raise ValueError('FP32 reference identity/geometry differs')
    if metadata['device'] != 'cpu' or metadata['dtype'] != 'float32':
        raise ValueError('reference must use original CPU FP32 weights')
    parity = compare(gen2, native, native_raw, q8_raw)
    if gen2['retained_probabilities']['first_frame'] != 0:
        raise ValueError('short corpus slice unexpectedly compacted')
    q8 = np.frombuffer(q8_raw, dtype='<f4').reshape(-1, 8)
    if fp32.dtype != np.dtype('float32') or fp32.shape != (samples // 160, 8) or metadata['shape'] != list(fp32.shape):
        raise ValueError('FP32 output differs from pinned valid-frame rule')
    if not np.isfinite(fp32).all() or not ((0 <= fp32) & (fp32 <= 1)).all():
        raise ValueError('invalid FP32 probabilities')
    if len({s['speaker'] for s in reference}) != item['statistics']['speakers']:
        raise ValueError('reference speaker count differs')
    # Preserve both raw arrays and their explained one-frame difference. Shared
    # thresholding sees every emitted frame; the full audio UEM clips padding.
    original = der(reference, segments(fp32), duration)
    quantized = der(reference, segments(q8), duration)
    deployed = der(reference, [{'start': s['start_seconds'], 'end': s['end_seconds'], 'speaker': s['speaker']}
                              for s in gen2['turns']], duration)
    delta = quantized['der'] - original['der']
    return {'slice': item['id'], 'backend': gen2['backend'], 'device': gen2['device'],
            'scope': 'local tuning-heldout integration and quality, not guaranteed checkpoint-unseen',
            'wrapper_parity': parity, 'fp32': original, 'q8_shared_threshold': quantized,
            'q8_production_default': deployed, 'q8_minus_fp32_absolute_der': delta,
            'gates': {'wrapper_parity': True, 'q8_degradation_at_most_0_02': delta <= .02 + 1e-12,
                      'q8_shared_der_at_most_0_20': quantized['der'] <= .20,
                      'q8_production_der_at_most_0_20': deployed['der'] <= .20},
            'postprocessing': 'shared probability >0.5, no hysteresis/padding/minimum-duration; production default reported separately',
            'shape_difference': {'native': list(q8.shape), 'fp32': list(fp32.shape),
                                 'native_extra_terminal_frame': 1, 'frame_seconds': .01,
                                 'rule': 'native floor(N/160)+1; FP32 mask floor(N/160); full audio UEM clips final padding'},
            'channel_mae_common_frames': float(np.abs(q8[:-1] - fp32).mean()),
            'channel_max_error_common_frames': float(np.abs(q8[:-1] - fp32).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--slice', required=True)
    parser.add_argument('--gen2', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('corpus.protocol.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite qualification evidence')
    protocol = validate_protocol(args.protocol, args.corpus)
    manifest = validate_corpus(args.corpus)
    selected = [s for s in manifest['slices'] if s['id'] == args.slice]
    if len(selected) != 1:
        parser.error('slice is not uniquely present in frozen corpus')
    item = selected[0]
    reference_path = args.corpus / item['reference']
    # Every file consumed must be covered by the immutable acquisition hashes.
    for name in [item['wav'], item['reference']]:
        if name not in manifest['files']:
            raise ValueError('missing frozen input hash')
    gen2 = json.loads(args.gen2.read_text())
    metadata_path = args.reference.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    if metadata['script_sha256'] != protocol['tool_sha256']['reference.py'] or metadata['lock_sha256'] != protocol['tool_sha256']['runtime.lock.json'] or metadata['environment_lock_sha256'] != protocol['tool_sha256']['reference/uv.lock']:
        raise ValueError('FP32 reference used a different frozen script/asset/environment lock')
    validate_oracle_environment(metadata)
    if metadata['wav_sha256'] != manifest['files'][item['wav']]['sha256']:
        raise ValueError('FP32 audio digest differs')
    audio = (args.corpus / item['wav']).read_bytes()
    offset, samples = pcm_header(audio[:65536], len(audio))
    if samples != item['audio_samples']:
        raise ValueError('frozen PCM sample count differs')
    normalized = np.frombuffer(audio, dtype='<i2', offset=offset, count=samples).astype(np.float32) / np.float32(32768)
    pcm_digest = hashlib.sha256(normalized.astype('<f4').tobytes()).hexdigest()
    native_path = Path(str(args.native) + '.f32')
    q8_path = args.gen2.with_suffix('.f32')
    fp32_path = args.reference.with_suffix('.npy')
    result = score_slice(item, gen2, json.loads(args.native.read_text()), q8_path.read_bytes(), native_path.read_bytes(),
                         np.load(fp32_path, allow_pickle=False), metadata, json.loads(reference_path.read_text()), pcm_digest)
    result['input_pcm_f32le_sha256'] = pcm_digest
    result['all_slice_gates_passed'] = all(result['gates'].values())
    result['sha256'] = {str(p): digest(p) for p in [args.protocol, args.corpus / 'manifest.json', reference_path, args.gen2, q8_path,
                                                args.native, native_path, fp32_path, metadata_path, Path(__file__),
                                                Path(__file__).with_name('score.py'), Path(__file__).with_name('compare.py'),
                                                Path(__file__).with_name('compare_stream.py')]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    # A failed quality gate is retained evidence, not a reason to stop the set.
    print(json.dumps({'slice': args.slice, 'gates': result['gates']}))


if __name__ == '__main__':
    main()
