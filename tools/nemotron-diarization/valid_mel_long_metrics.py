"""Qualify complete long-session accounting, retained tails and bounded process RSS."""
import math
import re

from compare_stream_valid_mel import compare, integer


def peak_rss(log):
    values = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', log, re.MULTILINE)
    if len(values) != 1 or int(values[0]) <= 0:
        raise ValueError('missing or ambiguous macOS process peak RSS')
    return int(values[0])


def audit_case(public, native, raw, actual_raw, native_time, public_time, case, backend='cpu', device=None):
    samples = case['samples']
    for record, key in [(public, 'accepted_samples'), (native, 'audio_samples')]:
        if (record[key] != samples or record['input_repeats'] != case['repeats']
                or record['input_pcm_f32le_sha256'] != case['pcm_sha256']):
            raise ValueError('fixed long-session actual input differs')
    if public['backend'] != backend or public.get('device') != device:
        raise ValueError('fixed long-session backend or device differs')
    if integer(native['retained_rows'], 'retained rows') < 60000:
        raise ValueError('long-session compaction retained less than the prescribed ten-minute tail')
    result = compare(public, native, raw, actual_raw)
    if result['gen2_rtf'] >= 1 or result['native_rtf'] >= 1:
        raise ValueError('long-session computation did not remain faster than real time')
    memory = {'native': peak_rss(native_time), 'public': peak_rss(public_time)}
    if any(value > 1024 * 1024 * 1024 for value in memory.values()):
        raise ValueError('long-session process RSS exceeds the 1024 MiB reservation')
    snapshots = public['snapshots']
    duration = samples / 16000
    if not isinstance(snapshots, list) or not 2 <= len(snapshots) <= int(duration / 5) + 1:
        raise ValueError('missing or unbounded provisional revision history')
    revision, processed, wall = 0, 0., 0
    for snapshot in snapshots:
        next_revision = integer(snapshot['revision'], 'revision', 1)
        next_wall = integer(snapshot['wall_micros'], 'snapshot wall time', 1)
        next_processed = snapshot['processed_audio_seconds']
        integer(snapshot['turn_count'], 'snapshot turn count', 0)
        if (type(next_processed) not in (int, float) or not math.isfinite(next_processed)
                or not processed < next_processed <= duration or next_revision <= revision
                or next_wall <= wall or snapshot['turn_count'] > 100000):
            raise ValueError('invalid provisional revision sequence')
        revision, processed, wall = next_revision, next_processed, next_wall
    if snapshots[0]['processed_audio_seconds'] > 60 or processed < duration - 20:
        raise ValueError('provisional observations do not cover the long session')
    integer(public['backpressure_count'], 'backpressure count', 1)
    result.update(peak_rss_bytes=memory, process_rss_within_1024_mib=True,
                  provisional_snapshots=len(snapshots), first_provisional_audio_seconds=snapshots[0]['processed_audio_seconds'],
                  last_provisional_audio_seconds=processed, backpressure_count=public['backpressure_count'])
    return result
