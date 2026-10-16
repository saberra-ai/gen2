"""Frozen CPU frontend/filterbank control. Loads no neural model or checkpoint."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from artifacts import digest
from frontend_admission import require_linked_sdk, run_probe
from frontend_metrics import difference, valid_prefix


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'corpus', 'native-source', 'sdk', 'probe', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'frontend-diagnostic.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.output.exists():
        parser.error('refusing to overwrite frontend evidence')
    if 'Python ' + platform.python_version() != protocol['python_version']:
        raise ValueError('frontend Python version differs')
    for name, expected in protocol['tools'].items():
        if digest(tools / name) != expected:
            raise ValueError('frontend tool differs from frozen protocol: ' + name)
    if descriptor(args.probe) != protocol['probe']:
        raise ValueError('compiled frontend probe differs from frozen identity')
    # The exact probe binary has one fixed LC_RPATH. Bind admission to the tree
    # it executes, rather than accepting an identical copy at another path.
    load_commands = subprocess.check_output(['/usr/bin/otool', '-l', str(args.probe)], text=True)
    lines = load_commands.splitlines()
    rpaths = [lines[index + 2].strip().removeprefix('path ').split(' (offset ')[0]
              for index, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']
    require_linked_sdk(args.sdk, protocol['linked_sdk_path'], rpaths)
    for group, root in [('native_source_files', args.native_source), ('sdk_libraries', args.sdk / 'lib')]:
        for name, expected in protocol[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('frontend source/SDK differs: ' + name)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.native_source, text=True).strip() != protocol['native_revision']:
        raise ValueError('native source revision differs')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.native_source, text=True):
        raise ValueError('native source has tracked modifications')
    if descriptor(args.corpus / 'manifest.json') != protocol['corpus_manifest']:
        raise ValueError('frozen corpus manifest differs')
    for name, expected in protocol['assets'].items():
        if descriptor(args.assets / name) != expected:
            raise ValueError('frontend asset differs: ' + name)
    for name, version in protocol['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('frontend dependency differs: ' + name)
    dist = importlib.metadata.distribution('transformers')
    direct = json.loads(dist.read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != protocol['transformers_revision']:
        raise ValueError('frontend Transformers source revision differs')
    for name, expected in protocol['transformers_files'].items():
        if descriptor(Path(dist.locate_file(name))) != expected:
            raise ValueError('frontend Transformers file differs: ' + name)
    gguf_source = args.native_source / 'llama.cpp/gguf-py'
    for name, expected in protocol['gguf_files'].items():
        if descriptor(gguf_source / name) != expected:
            raise ValueError('frontend GGUF reader file differs: ' + name)
    sys.path.insert(0, str(gguf_source))
    from gguf import GGMLQuantizationType, GGUFReader
    import soundfile as sf
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(args.assets, local_files_only=True)
    extractor = processor.feature_extractor
    if [extractor.feature_size, extractor.sampling_rate, extractor.hop_length, extractor.n_fft, extractor.win_length] != [128, 16000, 160, 512, 400]:
        raise ValueError('HF frontend geometry differs')
    reader = GGUFReader(str(args.assets / 'Nemotron-3-Diarization.q8_0.gguf'))
    bank, = [tensor for tensor in reader.tensors if tensor.name == 'preprocessor.fb']
    if bank.tensor_type != GGMLQuantizationType.F32 or tuple(bank.data.shape) != (128, 257):
        raise ValueError('native filterbank type/layout differs')
    banks = {'native': np.array(bank.data, dtype='<f4', order='C', copy=True),
             'hf': np.array(extractor.mel_filters.cpu().numpy(), dtype='<f4', order='C', copy=True)}
    if any(value.shape != (128, 257) or not np.isfinite(value).all() or (value < 0).any() for value in banks.values()):
        raise ValueError('invalid frontend bank')
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    for name, value in banks.items():
        value.tofile(args.output / (name + '-bank.f32'))
    rows = []
    child_environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}

    def run_case(name, audio, origin):
        if audio.dtype != np.float32 or audio.ndim != 1 or not np.isfinite(audio).all() or not 160 <= len(audio) <= 1920000:
            raise ValueError('invalid fixed frontend PCM')
        case = args.output / name
        case.mkdir()
        pcm = case / 'audio.f32'
        audio.astype('<f4').tofile(pcm)
        if 'pcm_sha256' in origin and digest(pcm) != origin['pcm_sha256']:
            raise ValueError('frontend PCM differs from previous actual inference')
        outputs, native_metadata, masks = {}, {}, {}
        for bank_name, value in banks.items():
            native_path = case / ('native-' + bank_name + '.f32')
            argv = [str(args.probe), str(pcm), str(args.output / (bank_name + '-bank.f32')), str(native_path)]
            stdout = run_probe(argv, case, 'native-' + bank_name, child_environment)
            metadata = json.loads(stdout)
            if metadata != {'audio_samples': len(audio), 'frames': len(audio) // 160 + 1,
                            'mel_bins': 128, 'center': True, 'normalize': False, 'uses_gpu': False}:
                raise ValueError('native frontend observation differs')
            native = np.fromfile(native_path, dtype='<f4').reshape(-1, 128)
            extractor.mel_filters = torch.from_numpy(value.copy())
            with torch.inference_mode():
                features = extractor(audio.copy(), sampling_rate=16000, center=True,
                                     return_attention_mask=True, return_tensors='pt', device='cpu')
            hf = features.input_features[0].cpu().numpy().astype('<f4')
            mask = features.attention_mask[0].cpu().numpy().astype(np.uint8)
            if hf.dtype != np.float32 or not np.isfinite(hf).all():
                raise ValueError('invalid HF frontend tensor')
            valid_left, valid_right, valid = valid_prefix(hf, mask, native)
            if valid != len(audio) // 160:
                raise ValueError('HF valid-frame count differs from fixed PCM geometry')
            hf.tofile(case / ('hf-' + bank_name + '.f32'))
            mask.tofile(case / ('hf-' + bank_name + '.mask.u8'))
            outputs['native-' + bank_name], outputs['hf-' + bank_name] = valid_left, valid_right
            native_metadata[bank_name], masks[bank_name] = {**metadata, 'argv': argv}, {'valid': valid, 'raw': len(mask), 'masked': len(mask) - valid}
        comparisons = {label: difference(outputs[left], outputs[right], protocol['absolute_tolerance'])
                       for label, left, right in [('own_banks', 'native-native', 'hf-hf'),
                                                  ('same_native_bank', 'native-native', 'hf-native'),
                                                  ('same_hf_bank', 'native-hf', 'hf-hf')]}
        return {'case': name, 'origin': origin, 'pcm': descriptor(pcm),
                'native_observations': native_metadata, 'masks': masks, 'comparisons': comparisons}

    # Controls execute before target WAV decoding. No threshold is selected from
    # the AMI data. These tolerances diagnose FE numerics, never DER/model parity.
    controls = {'silence': np.zeros(16000, dtype=np.float32)}
    t = np.arange(32003, dtype=np.float64) / 16000
    controls['fixed_tones_and_impulses'] = (.2 * np.sin(2 * np.pi * 440 * t) + .1 * np.cos(2 * np.pi * 1234 * t)).astype(np.float32)
    controls['fixed_tones_and_impulses'][[0, 16000, 32002]] += np.array([.125, -.125, .0625], dtype=np.float32)
    for name, audio in controls.items():
        row = run_case(name, audio, {'synthetic': True})
        rows.append(row)
        for label in ['same_native_bank', 'same_hf_bank']:
            metrics = row['comparisons'][label]
            if metrics['maximum_absolute_error'] > protocol['absolute_tolerance'] or metrics['mean_absolute_error'] > protocol['mean_absolute_tolerance']:
                (args.output / 'failed-control.json').write_text(json.dumps(row, indent=2) + '\n')
                raise ValueError('synthetic same-bank frontend control exceeds frozen tolerances')
    for name, expected in protocol['target_wavs'].items():
        wav = args.corpus / name
        if descriptor(wav) != {key: expected[key] for key in ['bytes', 'sha256']}:
            raise ValueError('target WAV differs from frozen corpus')
        info = sf.info(wav)
        if info.samplerate != 16000 or info.channels != 1 or info.frames != 1920000:
            raise ValueError('target WAV geometry differs')
        audio, rate = sf.read(wav, dtype='float32', frames=info.frames + 1)
        if rate != 16000 or len(audio) != info.frames:
            raise ValueError('target WAV changed during read')
        rows.append(run_case(Path(name).stem, audio, {'wav': name, **expected}))
    for name, expected in protocol['sdk_libraries'].items():
        if descriptor(args.sdk / 'lib' / name) != expected:
            raise ValueError('frontend SDK changed during execution')
    for name, expected in protocol['assets'].items():
        if descriptor(args.assets / name) != expected:
            raise ValueError('frontend asset changed during execution')
    for name, expected in protocol['target_wavs'].items():
        if descriptor(args.corpus / name) != {key: expected[key] for key in ['bytes', 'sha256']}:
            raise ValueError('target WAV changed during execution')
    files = {path.relative_to(args.output).as_posix(): descriptor(path) for path in args.output.rglob('*') if path.is_file()}
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tools, text=True).strip(),
              'scope': protocol['scope'], 'release_qualified': False, 'original_gate_status_changed': False,
              'neural_model_inference_performed': False, 'checkpoint_deserialization_performed': False,
              'protocol_sha256': digest(protocol_path), 'bank_comparison': difference(banks['native'], banks['hf'], 0),
              'transformers_direct_url': direct, 'results': rows, 'files': files, 'limitations': protocol['limitations']}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'frontend_diagnostic_completed': True, 'cases': len(rows), 'original_gate_status_changed': False}))


if __name__ == '__main__':
    main()
