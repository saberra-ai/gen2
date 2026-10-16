"""Replay candidate finish regression from saved arrays, without model execution."""
import json
from pathlib import Path

import numpy as np

from frontend_metrics import difference


BOUNDARIES = [0, 1, 159, 160, 161, 319, 320, 321, 11519, 11520, 11521,
              16639, 16640, 16679, 16680, 16735, 16736, 16799, 16800, 17872, 17920]


def audit_outputs(output):
    output = Path(output)
    def tensor(name, rows, columns=8, probabilities=True):
        value = np.fromfile(output / name, dtype='<f4')
        if value.size != rows * columns or not np.isfinite(value).all():
            raise ValueError('invalid candidate saved shape/value: ' + name)
        if probabilities and ((value < 0).any() or (value > 1).any()):
            raise ValueError('invalid candidate probability: ' + name)
        return value.reshape(rows, columns)
    events = [json.loads(line) for line in (output / 'cases.jsonl').read_text().splitlines()]
    sessions = {'whole': 17872, 'transport16640': 17872, 'uneven': 17872,
                'provisional': 17872, 'reset-replay': 17872, 'left1-whole': 17872,
                'left1-uneven': 17872, **{'boundary-' + str(n): n for n in BOUNDARIES}}
    event_rows = [row for row in events if 'event' in row]
    expected_events = {'model': ['load-started', 'load-completed']}
    expected_events.update({name: ['started', 'returned', 'completed']
                           for name in ['direct-left0-first', 'direct-left0-final',
                                        'direct-left1-first', 'direct-left1-final']})
    expected_events.update({name: ['started', 'finish-started', 'finish-returned', 'completed']
                           for name in sessions})
    actual_events = {}
    for row in event_rows:
        actual_events.setdefault(row['case'], []).append(row['event'])
    if actual_events != expected_events:
        raise ValueError('candidate cases incomplete or duplicated')
    traces = {}
    for name, samples in sessions.items():
        final = tensor(name + '.f32', samples // 160)
        snapshots = sorted(output.glob(name + '-push-*.json'),
                           key=lambda p: int(p.stem.rsplit('-', 1)[1]))
        stable = np.empty((0, 8), np.float32)
        saw_preview = False
        for path in snapshots + [output / (name + '-finish.json'), output / (name + '-repeated-late.json')]:
            meta = json.loads(path.read_text())
            frames, frontier = meta['frames'], meta['stable_frames']
            if not 0 <= frontier <= frames or meta['retained_base'] != 0 or meta['elements'] != frames * 8:
                raise ValueError('invalid candidate saved frontier')
            current = tensor(path.with_suffix('.f32').name, frames)
            if frontier < len(stable) or len(current) < len(stable) or current[:len(stable)].tobytes() != stable.tobytes():
                raise ValueError('candidate changed stable output')
            stable = current[:frontier].copy()
            saw_preview |= frontier < frames
            if path.name.endswith(('-finish.json', '-repeated-late.json')):
                if frontier != samples // 160 or frames != frontier or current.tobytes() != final.tobytes():
                    raise ValueError('candidate final/lifecycle output differs')
        traces[name] = {'samples': samples, 'frames': len(final), 'pushes': len(snapshots),
                        'saw_provisional': saw_preview}
    if not traces['provisional']['saw_provisional']:
        raise ValueError('candidate never exercised provisional output')
    if (output / 'reset-replay.f32').read_bytes() != (output / 'whole.f32').read_bytes():
        raise ValueError('candidate reset replay differs')
    for left in [0, 1]:
        label = 'direct-left' + str(left)
        first = tensor(label + '-first.predictions.f32', 104)
        first_emb = tensor(label + '-first.embeddings.f32', 13, 512, False)
        fifo = tensor(label + '-final.fifo.f32', 9, 512, False)
        if fifo.tobytes() != first_emb[:9].tobytes():
            raise ValueError('candidate direct FIFO differs')
        count = 39 + left * 8
        last = tensor(label + '-final.predictions.f32', (9 + (count + 7) // 8) * 8)
        tensor(label + '-final.embeddings.f32', (count + 7) // 8, 512, False)
        tensor(label + '-first.mel.f32', 104, 128, False)
        tensor(label + '-final.mel.f32', count, 128, False)
        expected = np.concatenate([first[:72], last[(9 + left) * 8:(9 + left) * 8 + 39]])
        name = 'left1-direct.f32' if left else 'direct-valid111.f32'
        if tensor(name, 111).tobytes() != expected.tobytes():
            raise ValueError('candidate direct output assembly differs')
    comparisons = {}
    for name in ['whole', 'transport16640', 'uneven', 'provisional', 'left1-whole', 'left1-uneven']:
        reference = tensor('left1-direct.f32' if name.startswith('left1-') else 'direct-valid111.f32', 111)
        comparisons[name] = difference(tensor(name + '.f32', 111), reference, 0.0001)
        if comparisons[name]['maximum_absolute_error'] > 0.0001:
            raise ValueError('candidate same-runtime probability policy failed')
    return {'sessions': traces, 'probability_comparisons': comparisons,
            'reset_replay_bitwise_equal': True, 'all_saved_stable_prefixes_unchanged': True}
