"""Fixed excluded-calibration HF/native graph control, with teacher-forced FIFO."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from artifacts import digest
from calibration_graph_metrics import probability_comparisons, variants
from frontend_admission import require_linked_sdk, run_probe
from frontend_metrics import difference


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'model', 'evidence', 'native-source', 'sdk', 'probe', 'probe-build', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'calibration-graph.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.output.exists():
        parser.error('refusing to overwrite graph evidence')
    if platform.python_version() != protocol['python_version']:
        raise ValueError('graph Python version differs')
    for group, root in [('tools', tools), ('native_source_files', args.native_source),
                        ('sdk_libraries', args.sdk / 'lib'), ('assets', args.assets), ('inputs', args.evidence),
                        ('evidence_receipts', tools.parent.parent)]:
        for name, expected in protocol[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('graph input/source differs: ' + name)
    for path, expected in [(args.probe, protocol['probe']), (args.probe_build, protocol['probe_build_manifest']),
                           (args.model, protocol['native_f32_model'])]:
        if descriptor(path) != expected:
            raise ValueError('graph binary/build/model differs')
    build = json.loads(args.probe_build.read_text())
    if (build['returncode'] != 0 or build['argv'] != protocol['probe_build_command']
            or build['source'] != descriptor(tools / 'calibration_graph_probe.cpp')
            or build['binary'] != protocol['probe'] or build['argv'][-1] != str(args.probe)
            or str(tools / 'calibration_graph_probe.cpp') not in build['argv']
            or '-L' + str(args.sdk / 'lib') not in build['argv']
            or '-Wl,-rpath,' + str(args.sdk / 'lib') not in build['argv']):
        raise ValueError('graph compile/source/binary/SDK binding differs')
    for name, expected in build['runtime_sources'].items():
        if descriptor(args.native_source / name) != expected or protocol['native_source_files'][name] != expected:
            raise ValueError('helper embedded runtime source differs')
    load_commands = subprocess.check_output(['/usr/bin/otool', '-l', str(args.probe)], text=True)
    lines = load_commands.splitlines()
    rpaths = [lines[i + 2].strip().removeprefix('path ').split(' (offset ')[0]
              for i, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']
    require_linked_sdk(args.sdk, protocol['linked_sdk_path'], rpaths)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.native_source, text=True).strip() != protocol['native_revision']:
        raise ValueError('native revision differs')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.native_source, text=True):
        raise ValueError('native tracked source modified')
    for name, version in protocol['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('graph dependency differs: ' + name)
    dist = importlib.metadata.distribution('transformers')
    direct = json.loads(dist.read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != protocol['transformers_revision']:
        raise ValueError('Transformers revision differs')
    for name, expected in protocol['transformers_files'].items():
        if descriptor(Path(dist.locate_file(name))) != expected:
            raise ValueError('Transformers source differs: ' + name)
    first_raw = np.fromfile(args.evidence / 'calibration/hf-first-raw.f32', dtype='<f4').reshape(105, 128)
    hf_final = np.fromfile(args.evidence / 'calibration/hf-final.f32', dtype='<f4').reshape(39, 128)
    native_final = np.fromfile(args.evidence / 'calibration/native-final.f32', dtype='<f4').reshape(40, 128)
    first_mask = np.fromfile(args.evidence / 'calibration/hf-first-raw.mask.u8', dtype=np.uint8)
    final_mask = np.fromfile(args.evidence / 'calibration/hf-final.mask.u8', dtype=np.uint8)
    if not np.array_equal(first_mask, np.r_[np.ones(104, np.uint8), np.zeros(1, np.uint8)]) or not np.array_equal(final_mask, np.ones(39, np.uint8)):
        raise ValueError('retained calibration masks differ')
    mel_inputs = variants(first_raw, hf_final, native_final)
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'probe-build.json').write_bytes(args.probe_build.read_bytes())
    (args.output / 'helper-load-commands.txt').write_text(load_commands)
    for name, values in mel_inputs.items():
        values.tofile(args.output / (name + '.mel.f32'))
    np.zeros((0, 512), np.float32).tofile(args.output / 'empty.fifo.f32')

    import torch
    from transformers import AutoModelForAudioFrameClassification

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    write_json = lambda name, value: (args.output / name).write_text(json.dumps(value, indent=2) + '\n')
    write_json('hf-load.started.json', {'device': 'cpu', 'dtype': 'float32', 'local_files_only': True})
    model, loading_info = AutoModelForAudioFrameClassification.from_pretrained(
        args.assets, local_files_only=True, dtype=torch.float32, output_loading_info=True)
    write_json('hf-loading-info.json', loading_info)
    if any(loading_info.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise ValueError('HF load reports missing/unexpected/mismatched/error entries')
    model = model.to('cpu').eval()
    if model.__class__.__name__ != 'Nemotron3DiarizationForAudioFrameClassification':
        raise ValueError('unexpected HF graph class')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('HF model parameters are not CPU F32')
    cfg = model.config
    if ([cfg.audio_config.hidden_size, cfg.audio_config.num_mel_bins, cfg.audio_config.subsampling_factor,
         cfg.audio_config.num_hidden_layers, cfg.head_config.num_speakers, cfg.head_config.subsampling_factor] != [512, 128, 8, 31, 8, 8]
            or any(getattr(cfg.streaming_config, key) != value
                   for key, value in protocol['hf_streaming_config'].items())):
        raise ValueError('HF V3/streaming configuration differs')
    write_json('hf-load.completed.json', {'class': model.__class__.__name__, 'device': 'cpu', 'dtype': 'float32',
                                        'attention_implementation': cfg._attn_implementation})
    captured_logits, captured_embeds, captured_inputs = [], [], []

    def logits_hook(_module, _args, output):
        captured_logits.append(output.detach().clone())

    def embed_hook(_module, _args, output):
        captured_embeds.append(output.detach().clone())

    def input_hook(_module, _args, kwargs):
        captured_inputs.append({key: kwargs[key].detach().clone() for key in ['inputs_embeds', 'attention_mask', 'position_ids']})

    def persist_capture(index, label):
        metadata = {}
        def save(suffix, tensor, dtype):
            metadata[suffix] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
            tensor.cpu().contiguous().numpy().astype(dtype).tofile(args.output / ('hf-' + label + '.' + suffix))
        if index < len(captured_logits):
            logits = captured_logits[index]
            save('full-logits.f32', logits, '<f4')
            save('full-probabilities.f32', logits.sigmoid(), '<f4')
        if index < len(captured_embeds):
            save('embeddings.f32', captured_embeds[index], '<f4')
        if index < len(captured_inputs):
            inputs = captured_inputs[index]
            save('graph-input.f32', inputs['inputs_embeds'], '<f4')
            save('graph-mask.u8', inputs['attention_mask'], np.uint8)
            save('positions.i64', inputs['position_ids'], '<i8')
        write_json('hf-' + label + '.capture.json', metadata)

    handles = [model.classifier.register_forward_hook(logits_hook),
               model.model.audio_tower.embedder.register_forward_hook(embed_hook),
               model.model.register_forward_pre_hook(input_hook, with_kwargs=True)]
    try:
        write_json('hf-first.started.json', {'mel_rows': 104, 'num_lookahead_frames': 4, 'cache': None})
        try:
            with torch.inference_mode():
                first = model(input_features=torch.from_numpy(mel_inputs['first104'].copy())[None],
                              attention_mask=torch.ones((1, 104), dtype=torch.long), num_lookahead_frames=4)
        finally:
            persist_capture(0, 'first')
        cache = first.speaker_cache
        if cache is None or (cache.num_cache_frames, cache.num_fifo_frames, cache.is_compressed) != (0, 9, False):
            raise ValueError('HF first cache occupancy differs')
        # Clone before the second forward mutates cache buffers in place.
        fifo = cache.fifo[:, :9].detach().clone()
        if not torch.equal(fifo, captured_embeds[0][:, :9]):
            raise ValueError('teacher-forced FIFO differs from first pre-encoder embeddings')
        fifo[0].numpy().astype('<f4').tofile(args.output / 'hf-first.fifo.f32')
        write_json('hf-first.completed.json', {'returned_logit_rows': first.logits.shape[1],
                                             'speaker_cache_rows': 0, 'fifo_rows': 9, 'compressed': False})
        write_json('hf-final.started.json', {'mel_rows': 39, 'num_lookahead_frames': None, 'speaker_cache_rows': 0, 'fifo_rows': 9})
        try:
            with torch.inference_mode():
                final = model(input_features=torch.from_numpy(mel_inputs['valid39'].copy())[None],
                              attention_mask=torch.ones((1, 39), dtype=torch.long), speaker_cache=cache)
        finally:
            persist_capture(1, 'final')
    finally:
        for handle in handles:
            handle.remove()
    if len(captured_logits) != 2 or len(captured_embeds) != 2 or len(captured_inputs) != 2:
        raise ValueError('HF hook count differs from exactly two fixed forwards')
    if (tuple(first.logits.shape) != (1, 72, 8) or tuple(final.logits.shape) != (1, 39, 8)
            or not torch.equal(first.logits, captured_logits[0][:, :72])
            or not torch.equal(final.logits, captured_logits[1][:, 72:111])):
        raise ValueError('HF returned/full final logit slice differs')
    hf_probs, hf_embeds = [], []
    for index, (label, tokens) in enumerate([('first', 13), ('final', 14)]):
        logits, embeds, inputs = captured_logits[index], captured_embeds[index], captured_inputs[index]
        if tuple(logits.shape) != (1, tokens * 8, 8) or tuple(embeds.shape) != (1, 13 if index == 0 else 5, 512):
            raise ValueError('HF hook tensor shape differs')
        if (tuple(inputs['inputs_embeds'].shape) != (1, tokens, 512)
                or not torch.equal(inputs['attention_mask'], torch.ones((1, tokens), dtype=torch.bool))
                or not torch.equal(inputs['position_ids'], torch.arange(tokens)[None])):
            raise ValueError('HF actual graph mask/positions differ')
        if index == 1 and not torch.equal(inputs['inputs_embeds'][:, :9], fifo):
            raise ValueError('HF final graph did not consume the saved FIFO prefix')
        probs = logits.sigmoid()[0].numpy().astype('<f4')
        embedding = embeds[0].numpy().astype('<f4')
        hf_probs.append(probs)
        hf_embeds.append(embedding)
    hf_returned = final.logits.sigmoid()[0].numpy().astype('<f4')
    hf_returned.tofile(args.output / 'hf-final.returned-probabilities.f32')
    write_json('hf-final.completed.json', {'returned_logit_rows': 39, 'full_logit_rows': 112,
                                         'prefix_rows': 72, 'valid_end': 111, 'speaker_cache_rows': cache.num_cache_frames,
                                         'fifo_rows_after': cache.num_fifo_frames})
    del model, first, final, cache
    native, embeddings = {}, {}
    for name in ['first104', 'valid39', 'zero40', 'extra40']:
        case = args.output / name
        case.mkdir()
        fifo_path = args.output / ('empty.fifo.f32' if name == 'first104' else 'hf-first.fifo.f32')
        prefix = case / 'native'
        run_probe([str(args.probe), str(args.model), str(args.output / (name + '.mel.f32')), str(fifo_path), str(prefix)],
                  case, 'native', {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
        metadata = json.loads(prefix.with_suffix('.json').read_text())
        tokens, rows = (13, 104) if name == 'first104' else (5, 112)
        if metadata != {'mel_rows': len(mel_inputs[name]), 'fifo_rows': 0 if name == 'first104' else 9,
                        'speaker_cache_rows': 0, 'chunk_encoder_rows': tokens,
                        'total_encoder_rows': 13 if name == 'first104' else 14, 'native_probability_rows': rows,
                        'speakers': 8, 'hidden_size': 512, 'cpu_threads': 4, 'gpu_requested': False, 'batching': False}:
            raise ValueError('actual native graph metadata differs')
        native[name] = np.fromfile(str(prefix) + '.predictions.f32', dtype='<f4').reshape(rows, 8)
        embeddings[name] = np.fromfile(str(prefix) + '.embeddings.f32', dtype='<f4').reshape(tokens, 512)
    reports = probability_comparisons(native, hf_probs[0], hf_probs[1], hf_returned)
    embedding_reports = {name: difference(value, hf_embeds[0 if name == 'first104' else 1], 0)
                         for name, value in embeddings.items()}
    for group, root in [('sdk_libraries', args.sdk / 'lib'), ('assets', args.assets), ('inputs', args.evidence)]:
        for name, expected in protocol[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('graph input changed during execution: ' + name)
    if descriptor(args.model) != protocol['native_f32_model'] or descriptor(args.probe) != protocol['probe']:
        raise ValueError('native graph artifact changed during execution')
    files = {p.relative_to(args.output).as_posix(): descriptor(p) for p in args.output.rglob('*') if p.is_file()}
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tools, text=True).strip(),
              'protocol_sha256': digest(protocol_path), 'new_hf_forward_calls': 2, 'new_native_graph_calls': 4,
              'excluded_calibration_only': True, 'hf_fifo_cloned_before_mutating_final_forward': True,
              'native_fifo_teacher_forced_for_all_final_cases': True, 'native_aosc_executed': False,
              'frontend_or_processor_executed': False, 'neural_model_inference_performed': True,
              'original_nemo_executed': False, 'target_ami_inputs_executed': False, 'der_scoring_performed': False,
              'original_gate_status_changed': False, 'release_qualified': False, 'numeric_equivalence_claimed': False,
              'probability_comparisons': reports, 'pre_encoder_embedding_comparisons': embedding_reports,
              'transformers_direct_url': direct, 'files': files, 'limitations': protocol['limitations']}
    write_json('result.json', result)
    print(json.dumps({'calibration_graph_control_completed': True, 'hf_calls': 2, 'native_calls': 4, 'original_gate_changed': False}))


if __name__ == '__main__':
    main()
