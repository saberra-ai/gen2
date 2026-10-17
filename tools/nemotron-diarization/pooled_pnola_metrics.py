"""Pooled app admission with unchanged historical session/numerical gates.

Only metadata on copies is adapted after exact pooled CPU identity admission.
Live turn comparison uses an independent C ABI run on the exact paced PCM,
not an old app output with a different native implementation.
"""
import copy
import math

from qualify_pooled_native import NEW_VERSION, OLD_VERSION
from qualify_valid_mel_pnola import timeline_matches as original_timeline_matches
from qualify_valid_mel_pnola_app import recorded_matches as original_recorded_matches

MODEL_REVISION = 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
GEOMETRY = {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222}


def native_reference(native, duration):
    frames = round(duration * 100)
    if (native['runtime_version'] != NEW_VERSION or native['requested_gpu_index'] != -1
            or native['geometry'] != GEOMETRY or native['input_repeats'] != 1
            or native['audio_samples'] != round(duration * 16000)
            or not math.isfinite(native['audio_duration_seconds'])
            or abs(native['audio_duration_seconds'] - duration) > 1e-6
            or native['native_frame_count'] != frames or native['retained_first_frame'] != 0
            or native['retained_rows'] != frames):
        raise ValueError('independent pooled CPU native reference differs')
    return {**native, 'runtime_version': OLD_VERSION,
            'backend': 'cpu', 'model_revision': MODEL_REVISION}


def timeline_matches(actual, native, duration, *, compact=False):
    if actual['runtime_version'] != NEW_VERSION or actual['backend'] != 'cpu':
        raise ValueError('exact pooled CPU app timeline required')
    return original_timeline_matches({**actual, 'runtime_version': OLD_VERSION},
        native_reference(native, duration), duration, 'cpu', compact=compact)


def recorded_matches(actual, native, bark_text, wav_sha256, *, unavailable=False):
    if unavailable:
        return original_recorded_matches(actual, {}, bark_text, 'cpu', wav_sha256, unavailable=True)
    adapted = copy.deepcopy(actual)
    timeline = adapted['attribution']['timeline']
    comparison = timeline_matches(timeline, native, 60.)
    timeline['runtime_version'] = OLD_VERSION
    # Retain every original Bark/word/availability admission rule as well.
    original_recorded_matches(adapted, native_reference(native, 60.), bark_text, 'cpu', wav_sha256)
    return comparison


def live_matches(value, native):
    # The same complete-app session, revision and export criteria remain frozen.
    if (value['passed'] is not True or value['microphone_used'] is not False
            or value['speaker_qualification'] is not True or value['sections'] != 11
            or value['queued'] != 0 or value['words'] != 152 or value['export_bytes'] != 803
            or not math.isfinite(value['duration_seconds']) or abs(value['duration_seconds'] - 80.04) > 1e-6
            or not value['provisional_revisions'] or value['speakers']['status'] != 'complete'
            or not math.isfinite(value['speakers']['processed_audio_seconds'])
            or abs(value['speakers']['processed_audio_seconds'] - 80.04) > 1e-6):
        raise ValueError('complete pooled app live session/recovery/export differs')
    revisions = value['provisional_revisions']
    if len(revisions) > 17:
        raise ValueError('unbounded complete pooled app live revisions')
    previous = 0
    for revision in revisions:
        if (type(revision['revision']) is not int or revision['revision'] <= previous
                or not math.isfinite(revision['elapsed_seconds']) or revision['elapsed_seconds'] < 0
                or not math.isfinite(revision['processed_audio_seconds'])
                or not 0 < revision['processed_audio_seconds'] <= 80.04):
            raise ValueError('invalid complete pooled app live revision')
        previous = revision['revision']
    comparison = timeline_matches(value['speakers']['timeline'], native, 80.04, compact=True)
    return {**comparison, 'native_reference_frame_count': native['native_frame_count'],
            'reference': 'Independent pinned public C ABI on the exact 80.04s repeated PCM; app final timeline compared within 1e-6. Internal public API parity also required by the app helper.'}
