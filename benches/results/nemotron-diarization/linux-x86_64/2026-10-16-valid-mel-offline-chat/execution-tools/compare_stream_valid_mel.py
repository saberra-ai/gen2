"""Compare valid-Mel V1 public/native output; preserve every valid row."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"invalid {name}")
    return value


def compare(gen2, native, raw, actual_raw):
    samples = integer(gen2['accepted_samples'], 'accepted samples', 320)
    if samples > 3600 * 16000 or gen2['complete'] is not True:
        raise ValueError('incomplete or oversized Gen2 session')
    duration = samples / 16000
    frames = samples // 160
    if integer(native['audio_samples'], 'native audio samples') != samples:
        raise ValueError('native/Gen2 input length differs')
    for result in [gen2, native]:
        if abs(finite(result['audio_duration_seconds'], 'audio duration') - duration) > 1e-9:
            raise ValueError('audio duration differs')
        if integer(result['native_frame_count'], 'frame count') != frames:
            raise ValueError('native terminal frame rule differs')
    geometry = {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222}
    if gen2['geometry'] != geometry or native['geometry'] != geometry:
        raise ValueError('streaming geometry differs')
    if gen2['model_revision'] != 'f667ed73aee57d40cc39428eb768b4fd87a0a29e':
        raise ValueError('unexpected model revision')
    if any(result.get('runtime_version') != 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1' for result in [gen2, native]):
        raise ValueError('unexpected runtime version')
    backend = gen2['backend']
    gpu = integer(native['requested_gpu_index'], 'requested GPU index', -1)
    if backend == 'cpu':
        if gpu != -1 or gen2.get('device') is not None:
            raise ValueError('CPU placement differs')
    elif backend == 'metal':
        if gpu < 0 or not isinstance(gen2.get('device'), str) or not gen2['device']:
            raise ValueError('Metal placement is missing')
    else:
        raise ValueError('unexpected backend')
    first = integer(native['retained_first_frame'], 'retained first frame')
    rows = integer(native['retained_rows'], 'retained rows', 1)
    if first + rows != frames or rows > 130000:
        raise ValueError('native retained frame bounds differ')
    if gen2['retained_probabilities'] != {'first_frame': first, 'rows': rows}:
        raise ValueError('Gen2 retained frame bounds differ')
    if duration >= 1800 and first == 0:
        raise ValueError('long-session probability compaction did not occur')
    if len(raw) != rows * 8 * 4:
        raise ValueError('retained native probability length differs')
    if len(actual_raw) != len(raw):
        raise ValueError('Gen2 probability length differs')
    if gen2.get('retained_probability_sha256') != hashlib.sha256(actual_raw).hexdigest():
        raise ValueError('Gen2 probability digest differs')
    probability_error = 0.0
    for (expected,), (actual,) in zip(struct.iter_unpack('<f', raw), struct.iter_unpack('<f', actual_raw)):
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in [expected, actual]):
            raise ValueError('invalid native/Gen2 probabilities')
        probability_error = max(probability_error, abs(expected-actual))
    if probability_error > 1e-6:
        raise ValueError('native/Gen2 retained probability values differ')

    def turns(result, clip):
        values = result['turns']
        if not isinstance(values, list) or len(values) > 100000:
            raise ValueError('invalid native turn count')
        normalized = []
        for turn in values:
            start = finite(turn['start_seconds'], 'turn start')
            end = finite(turn['end_seconds'], 'turn end')
            speaker = integer(turn['speaker'], 'speaker', 1)
            if start < 0 or end < start or (not clip and end == start) or speaker > 8 or end > (frames * .01 + 1e-4 if clip else duration):
                raise ValueError('invalid speaker turn')
            if clip:
                start, end = min(start, duration), min(end, duration)
            if end > start:
                normalized.append((start, end, speaker))
        return normalized

    actual, expected = turns(gen2, False), turns(native, True)
    if len(actual) != len(expected):
        raise ValueError('native/Gen2 turn count differs')
    maximum = 0.0
    for a, b in zip(actual, expected):
        error = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        if a[2] != b[2] or error > 1e-6:
            raise ValueError('native/Gen2 turn identity or boundary differs')
        maximum = max(maximum, error)
    elapsed = integer(gen2['elapsed_micros'], 'Gen2 elapsed time', 1)
    native_elapsed = integer(native['elapsed_micros'], 'native elapsed time', 1)
    return {'passed': True, 'scope': 'wrapper/reference correctness at the recorded duration; not held-out quality or full release qualification',
            'runtime_profile': 'gen2-valid-mel-v1', 'frame_contract': 'v3-streaming-valid-mel-v1',
            'audio_duration_seconds': duration, 'backend': backend, 'device': gen2.get('device'),
            'native_frame_count': frames, 'retained_first_frame': first, 'retained_rows': rows,
            'turns': len(actual), 'maximum_turn_error_seconds': maximum, 'turn_tolerance_seconds': 1e-6,
            'raw_native_and_gen2_probabilities_finite_and_bounded': True,
            'retained_probabilities_match': True, 'maximum_probability_error': probability_error, 'probability_tolerance': 1e-6,
            'gen2_rtf': elapsed / 1e6 / duration, 'native_rtf': native_elapsed / 1e6 / duration}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gen2', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--gen2-probabilities', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = Path(str(args.native) + '.f32')
    try:
        result = compare(json.loads(args.gen2.read_text()), json.loads(args.native.read_text()), raw.read_bytes(), args.gen2_probabilities.read_bytes())
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        args.output.write_text(json.dumps({'passed': False, 'error': str(error)}, indent=2) + '\n')
        raise SystemExit(str(error))
    result['sha256'] = {str(p): digest(p) for p in [args.gen2, args.native, raw, args.gen2_probabilities, Path(__file__)]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
