"""One original-NeMo CPU forward on the excluded fixed calibration PCM.

The parent admits all source/environment/checkpoint identities before launch.
This process cannot read saved reference probabilities or target corpus data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output = args.output
    write = lambda name, value: (output / name).write_text(json.dumps(value, indent=2) + '\n')

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    pcm = np.fromfile(protocol['pcm_path'], dtype='<f4')
    if (pcm.shape != (17872,) or not np.isfinite(pcm).all()
            or hashlib.sha256(pcm.tobytes()).hexdigest() != protocol['pcm']['sha256']):
        raise ValueError('fixed original-NeMo calibration PCM differs')
    write('restore-started.json', {'strict': True, 'map_location': 'cpu', 'checkpoint': protocol['checkpoint']})
    started = time.monotonic()
    model = SortformerEncLabelModel.restore_from(protocol['checkpoint_path'],
                                                 map_location=torch.device('cpu'), strict=True)
    model = model.to(device='cpu', dtype=torch.float32).eval()
    config = OmegaConf.to_container(model.cfg, resolve=True)
    write('restored-config.json', config)
    if (not model.streaming_mode or model.async_streaming or not model.high_resolution
            or model.output_subsampling_factor != 1 or model.encoder.subsampling_factor != 8
            or config['encoder']['self_attention_model'] != 'rope'
            or config['encoder']['subsampling'] != 'feature_stacking'
            or config['sortformer_modules']['num_spks'] != 8
            or not config['sortformer_modules']['use_learnable_sil_emb']):
        raise ValueError('original-NeMo restored V3 identity differs')
    names = ['chunk_len', 'chunk_right_context', 'chunk_left_context', 'fifo_len',
             'spkcache_len', 'spkcache_update_period']
    for name, value in zip(names, [9, 4, 0, 264, 264, 222]):
        setattr(model.sortformer_modules, name, value)
    model._check_streaming_parameters()
    geometry = {name: getattr(model.sortformer_modules, name) for name in names}
    if list(geometry.values()) != [9, 4, 0, 264, 264, 222]:
        raise ValueError('original-NeMo runtime geometry differs')
    parameters = {}
    for name, value in model.named_parameters():
        if value.device.type != 'cpu' or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('original-NeMo parameter is not finite CPU FP32: ' + name)
        parameters[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                            'sha256': hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()}
    write('restored-parameters.json', parameters)
    write('restore-completed.json', {'strict': True, 'missing_keys': [], 'unexpected_keys': [],
                                    'seconds': time.monotonic() - started, 'geometry': geometry,
                                    'parameter_tensors': len(parameters), 'async_streaming': model.async_streaming})
    frontends = []

    def capture_frontend(_module, _inputs, result):
        features, lengths = result
        mel = features.detach().cpu().numpy().copy()
        valid_lengths = lengths.detach().cpu().numpy().copy()
        frontends.append((mel, valid_lengths))
        # Retain preprocessing evidence even if the later CPU graph raises.
        index = len(frontends)
        mel.astype('<f4').tofile(output / f'frontend-capture-{index}.f32')
        write(f'frontend-capture-{index}.json', {'shape': list(mel.shape),
                                              'valid_lengths': valid_lengths.tolist()})

    handle = model.preprocessor.register_forward_hook(capture_frontend)
    write('forward-started.json', {'calls': 1, 'samples': len(pcm), 'return_logits': True})
    started = time.monotonic()
    try:
        with torch.inference_mode():
            probabilities, logits, activity = model(audio_signal=torch.from_numpy(pcm)[None],
                                                   audio_signal_length=torch.tensor([len(pcm)], dtype=torch.long),
                                                   return_logits=True)
    finally:
        handle.remove()
    elapsed = time.monotonic() - started
    shapes, arrays, dtypes = {}, {}, {}
    for name, tensor in [('probabilities', probabilities), ('logits', logits), ('activity', activity)]:
        if tensor is None:
            shapes[name] = None
            dtypes[name] = None
            continue
        values = tensor.detach().cpu().numpy()
        values.astype('<f4').tofile(output / (name + '.f32'))
        arrays[name] = values
        shapes[name] = list(values.shape)
        dtypes[name] = str(values.dtype)
    write('output-shapes.json', {'shapes': shapes, 'dtypes': dtypes})
    if len(frontends) != 1 or frontends[0][1].tolist() != [111]:
        raise ValueError('original-NeMo frontend valid length differs')
    mel = frontends[0][0]
    if mel.ndim != 3 or mel.shape[:2] != (1, 128) or mel.shape[2] < 111 or not np.isfinite(mel).all():
        raise ValueError('original-NeMo frontend shape differs')
    mel.astype('<f4').tofile(output / 'frontend-physical.f32')
    mel[0, :, :111].T.copy().astype('<f4').tofile(output / 'frontend-valid.f32')
    for name, values in arrays.items():
        if values.shape != ((1, 111, 3) if name == 'activity' else (1, 111, 8)) or not np.isfinite(values).all():
            raise ValueError('original-NeMo output shape/finiteness differs: ' + name)
        if name == 'probabilities' and not ((0 <= values) & (values <= 1)).all():
            raise ValueError('original-NeMo probabilities outside unit interval')
    write('forward-completed.json', {'passed': True, 'calls': 1, 'seconds': elapsed,
                                    'shapes': shapes, 'frontend_physical_shape': list(mel.shape),
                                    'frontend_valid_lengths': [111], 'geometry': geometry,
                                    'strict_restore': True, 'device': 'cpu', 'dtype': 'float32',
                                    'official_source_commit': protocol['nemo_source_commit'],
                                    'quality_qualified': False, 'release_qualified': False})


if __name__ == '__main__':
    main()
