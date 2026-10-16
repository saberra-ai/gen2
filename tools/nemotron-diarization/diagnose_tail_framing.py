"""Fixed CPU streaming-FE/tail control; no learned projection or neural model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from artifacts import digest
from frontend_admission import require_linked_sdk, run_probe
from frontend_metrics import difference
from tail_framing import native_schedule, stack_inputs


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'corpus', 'evidence', 'frontend-evidence', 'native-source', 'sdk', 'probe', 'probe-build', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'tail-framing.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.output.exists():
        parser.error('refusing to overwrite framing evidence')
    if platform.python_version() != protocol['python_version']:
        raise ValueError('framing Python version differs')
    for name, expected in protocol['tools'].items():
        if descriptor(tools / name) != expected:
            raise ValueError('framing tool differs: ' + name)
    if descriptor(args.probe) != protocol['probe']:
        raise ValueError('framing probe differs')
    if descriptor(args.probe_build) != protocol['probe_build_manifest']:
        raise ValueError('framing probe build manifest differs')
    build = json.loads(args.probe_build.read_text())
    if (build['returncode'] != 0 or build['source'] != descriptor(tools / 'frontend_tail_probe.cpp')
            or build['binary'] != protocol['probe'] or build['argv'] != protocol['probe_build_command']
            or build['argv'][-1] != str(args.probe)
            or str(tools / 'frontend_tail_probe.cpp') not in build['argv']
            or '-L' + str(args.sdk / 'lib') not in build['argv']
            or '-Wl,-rpath,' + str(args.sdk / 'lib') not in build['argv']):
        raise ValueError('framing compile/source/binary/SDK binding differs')
    load_commands = subprocess.check_output(['/usr/bin/otool', '-l', str(args.probe)], text=True)
    lines = load_commands.splitlines()
    rpaths = [lines[i + 2].strip().removeprefix('path ').split(' (offset ')[0]
              for i, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']
    require_linked_sdk(args.sdk, protocol['linked_sdk_path'], rpaths)
    for group, root in [('native_source_files', args.native_source), ('sdk_libraries', args.sdk / 'lib'),
                        ('assets', args.assets), ('frontend_inputs', args.frontend_evidence)]:
        for name, expected in protocol[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('framing input/source differs: ' + name)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.native_source, text=True).strip() != protocol['native_revision']:
        raise ValueError('native revision differs')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.native_source, text=True):
        raise ValueError('native source has tracked modifications')
    for name, version in protocol['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('framing dependency differs: ' + name)
    dist = importlib.metadata.distribution('transformers')
    direct = json.loads(dist.read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != protocol['transformers_revision']:
        raise ValueError('Transformers revision differs')
    for name, expected in protocol['transformers_files'].items():
        if descriptor(Path(dist.locate_file(name))) != expected:
            raise ValueError('Transformers source differs: ' + name)
    import soundfile as sf
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(args.assets, local_files_only=True)
    processor.set_streaming_mode('low_latency')
    if [processor.num_mel_frames_per_audio_chunk, processor.num_mel_frames_per_step,
            processor.num_samples_first_audio_chunk, processor.num_samples_per_audio_chunk] != [104, 72, 16680, 17040]:
        raise ValueError('HF processor geometry differs')
    bank_path = args.frontend_evidence / 'native-bank.f32'
    bank = np.fromfile(bank_path, dtype='<f4').reshape(128, 257)
    if not np.isfinite(bank).all() or (bank < 0).any():
        raise ValueError('invalid retained shared bank')
    processor.feature_extractor.mel_filters = torch.from_numpy(bank.copy())
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'probe-build.json').write_bytes(args.probe_build.read_bytes())
    (args.output / 'helper-load-commands.txt').write_text(load_commands)

    def run_case(name, audio, origin):
        if audio.dtype != np.float32 or audio.ndim != 1 or not 320 <= len(audio) <= 1920000 or not np.isfinite(audio).all():
            raise ValueError('invalid fixed framing PCM')
        case = args.output / name
        case.mkdir()
        pcm = case / 'audio.f32'
        audio.astype('<f4').tofile(pcm)
        if 'pcm_sha256' in origin and digest(pcm) != origin['pcm_sha256']:
            raise ValueError('actual framing PCM differs')
        output = case / 'native-stream.f32'
        argv = [str(args.probe), str(pcm), str(bank_path), str(output)]
        metadata = json.loads(run_probe(argv, case, 'native-stream', {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}))
        # Row readiness is measured by the upstream FE, not inferred from final
        # output. The scheduler description below remains an integer replay.
        endpoints = list(range(16640, len(audio), 16640)) + [len(audio)]
        expected_ready = [0 if n < 256 else (n - 256) // 160 + 1 for n in endpoints]
        expected = {'audio_samples': len(audio), 'raw_frames': len(audio) // 160 + 1,
                    'before_finish_frames': expected_ready[-1], 'mel_bins': 128,
                    'push_samples': 16640, 'uses_gpu': False, 'available_after_push': expected_ready}
        if metadata != expected:
            raise ValueError('actual native streaming FE readiness differs')
        native = np.fromfile(output, dtype='<f4').reshape(-1, 128)
        if native.shape != (len(audio) // 160 + 1, 128) or not np.isfinite(native).all():
            raise ValueError('invalid native streaming features')
        schedule = native_schedule(metadata['available_after_push'], len(native))
        hf_calls = []
        cursor = 0
        final_features = final_mask = None
        while True:
            first = cursor == 0
            start = 0 if first else processor.audio_chunk_start(cursor)
            size = processor.num_samples_first_audio_chunk if first else processor.num_samples_per_audio_chunk
            end = min(start + size, len(audio))
            last = start + size >= len(audio)
            if start < 0 or start >= end:
                raise ValueError('invalid HF raw chunk interval')
            chunk = audio[start:end].copy()
            with torch.inference_mode():
                inputs = processor(chunk, sampling_rate=16000, is_streaming=True,
                                   is_first_audio_chunk=first, is_last_audio_chunk=last, return_tensors='pt')
            features = inputs.input_features[0].cpu().numpy().astype('<f4')
            mask = inputs.attention_mask[0].cpu().numpy().astype(np.uint8)
            if not np.isfinite(features).all() or not mask.all() or len(features) != len(mask):
                raise ValueError('HF trimmed features/mask differ')
            lookahead = inputs.get('num_lookahead_frames')
            if lookahead != (None if last else 4):
                raise ValueError('HF lookahead key differs')
            emitted = len(features) if last else 72
            row = {'start': cursor, 'input_end': cursor + len(features), 'emitted_end': cursor + emitted,
                   'raw_start': start, 'raw_end': end, 'first': first, 'last': last,
                   'num_lookahead_frames': lookahead}
            hf_calls.append(row)
            if first or last:
                label = 'first' if first else 'final'
                # Capture raw extractor output before the processor trims it.
                # This repeats only FE arithmetic, never a neural operation.
                with torch.inference_mode():
                    raw = processor.feature_extractor(chunk, sampling_rate=16000, center=first,
                                                      is_last_audio_chunk=last, return_tensors='pt')
                raw_features = raw.input_features[0].cpu().numpy().astype('<f4')
                raw_mask = raw.attention_mask[0].cpu().numpy().astype(np.uint8)
                if (int(raw_mask.sum()) != len(features) or not np.array_equal(raw_features[:len(features)], features)
                        or not np.array_equal(raw_mask[:len(mask)], mask) or raw_mask[len(mask):].any()):
                    raise ValueError('HF raw/trimmed frontend capture differs')
                raw_features.tofile(case / ('hf-' + label + '-raw.f32'))
                raw_mask.tofile(case / ('hf-' + label + '-raw.mask.u8'))
                row['raw_feature_rows'] = len(raw_features)
                row['valid_feature_rows'] = len(features)
            if last:
                final_features, final_mask = features, mask
                break
            cursor += 72
        if hf_calls[-1]['emitted_end'] != len(audio) // 160:
            raise ValueError('HF final valid output endpoint differs')
        hf_final = hf_calls[-1]
        native_final = schedule[-1]
        native_features = native[native_final['start']:native_final['input_end']]
        native_mask = np.ones(len(native_features), np.uint8)
        padded = {}
        for label, features, mask in [('native', native_features, native_mask), ('hf', final_features, final_mask)]:
            features.tofile(case / (label + '-final.f32'))
            mask.tofile(case / (label + '-final.mask.u8'))
            stack, encoder_mask, padding = stack_inputs(features, mask)
            stack.tofile(case / (label + '-final-stack.f32'))
            encoder_mask.tofile(case / (label + '-final-encoder.mask.u8'))
            padded[label] = {'real_mel_rows': len(features), 'zero_feature_padding_rows': padding,
                             'encoder_tokens': len(stack), 'all_encoder_tokens_valid': bool(encoder_mask.all())}
        common_start = max(native_final['start'], hf_final['start'])
        common_end = min(native_final['input_end'], hf_final['input_end'])
        if common_start >= common_end:
            raise ValueError('no common final frontend rows')
        comparison = difference(native[common_start:common_end],
                                final_features[common_start - hf_final['start']:common_end - hf_final['start']],
                                protocol['absolute_tolerance'])
        row = {'case': name, 'origin': origin, 'pcm': descriptor(pcm), 'native_frontend': metadata,
               'native_source_derived_schedule': schedule, 'hf_actual_calls': hf_calls, 'final_stack_inputs': padded,
               'final_common_mel_comparison': comparison, 'comparison_global_start': common_start,
               'comparison_global_end': common_end,
               'native_extra_computed_rows': len(native) - hf_final['emitted_end']}
        (case / 'framing.json').write_text(json.dumps(row, indent=2) + '\n')
        return row

    # Two fixed model-free controls precede all real WAV decoding. The grouping
    # tests and exact structural contracts are prerequisites. Numeric differences
    # are descriptive; the existing absolute flag is retained, never tuned.
    rows = [run_case('silence', np.zeros(16000, np.float32), {'synthetic': True})]
    t = np.arange(32003, dtype=np.float64) / 16000
    audio = (.2 * np.sin(2 * np.pi * 440 * t) + .1 * np.cos(2 * np.pi * 1234 * t)).astype(np.float32)
    audio[[0, 16000, 32002]] += np.array([.125, -.125, .0625], np.float32)
    rows.append(run_case('fixed_tones_and_impulses', audio, {'synthetic': True}))
    for row in rows:
        expected = protocol['synthetic_contracts'][row['case']]
        actual = {'native_calls_derived': len(row['native_source_derived_schedule']),
                  'hf_calls_observed': len(row['hf_actual_calls']),
                  'native_final_start': row['native_source_derived_schedule'][-1]['start'],
                  'hf_final_start': row['hf_actual_calls'][-1]['start'],
                  'final_stack_inputs': row['final_stack_inputs']}
        if actual != expected:
            raise ValueError('synthetic framing contract differs')
    for case in protocol['cases']:
        root = args.evidence if case['calibration'] else args.corpus
        wav = root / case['wav']
        if descriptor(wav) != case['wav_identity']:
            raise ValueError('framing WAV identity differs')
        info = sf.info(wav)
        if info.samplerate != 16000 or info.channels != 1 or info.frames != case['samples']:
            raise ValueError('framing WAV geometry differs')
        audio, rate = sf.read(wav, dtype='float32', frames=info.frames + 1)
        if rate != 16000 or len(audio) != info.frames:
            raise ValueError('framing WAV changed during read')
        row = run_case(case['id'], audio, {'pcm_sha256': case['pcm_sha256'], 'wav': case['wav'],
                                         'wav_identity': case['wav_identity'], 'calibration': case['calibration']})
        if {key: row['final_stack_inputs'][key] for key in ['native', 'hf']} != case['expected_final_stacks']:
            raise ValueError('fixed target stack geometry differs')
        if (row['hf_actual_calls'][-1]['start'] != case['expected_final_start']
                or row['native_source_derived_schedule'][-1]['start'] != case['expected_final_start']):
            raise ValueError('fixed target final cursor differs')
        rows.append(row)
    for group, root in [('sdk_libraries', args.sdk / 'lib'), ('assets', args.assets),
                        ('frontend_inputs', args.frontend_evidence)]:
        for name, expected in protocol[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('framing input changed: ' + name)
    for case in protocol['cases']:
        root = args.evidence if case['calibration'] else args.corpus
        if descriptor(root / case['wav']) != case['wav_identity']:
            raise ValueError('framing WAV changed during execution')
    files = {p.relative_to(args.output).as_posix(): descriptor(p) for p in args.output.rglob('*') if p.is_file()}
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tools, text=True).strip(),
              'protocol_sha256': digest(protocol_path), 'neural_model_instantiation_or_inference': False,
              'learned_projection_or_cache_executed': False, 'checkpoint_deserialization_performed': False,
              'native_scheduler_executed': False, 'native_scheduler_source_replayed': True,
              'hf_frontend_processor_and_native_streaming_frontend_executed': True,
              'release_qualified': False, 'original_gate_status_changed': False,
              'transformers_direct_url': direct, 'results': rows, 'files': files, 'limitations': protocol['limitations']}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'tail_framing_completed': True, 'cases': len(rows), 'neural_inference': False}))


if __name__ == '__main__':
    main()
