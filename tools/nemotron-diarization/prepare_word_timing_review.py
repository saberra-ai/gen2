"""Prepare model-blind HCRC timing calibration inputs, never human gold.

The exact crop and conversion are frozen before any recognizer/aligner output.
Audio stays outside Git. Source timed units remain explicitly unreviewed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import stat
import wave
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from corpus import fetch


CONVERSION = {'source_pcm': 'stereo PCM16LE, 20000 Hz',
              'mix': 'float64 equal-channel arithmetic mean divided by 32768',
              'resample': {'up': 4, 'down': 5, 'window': ['kaiser', 5.0], 'padtype': 'constant', 'cval': 0.0},
              'output': 'clip [-1,32767/32768], round-to-even PCM16LE mono 16000 Hz',
              'output_samples': 960000}


NOTICE = """This package is UNREVIEWED timing-calibration material, not human gold.

Selection: q1ec1 [0,60), fixed before Bark/native/aligner output. This is a
close-microphone two-person dialogue, not representative meeting acceptance.
Publisher training exposure is unknown; do not claim unseen test data.

Use mono16k.wav as the exact future model/review timebase. stereo20k.wav retains
both source channels for identity checking; left/right role assignment has NOT
been verified. Source participant IDs are metadata hints, not acoustic proof.

Have one human inspect the complete waveform/spectrogram and audio, then a
second human independently check every word start/end, lexical text and speaker
identity. Neither may view Bark, aligner or diarizer outputs before freezing the
adjudicated annotation. Review the entire crop, including omissions or errors
in source text. Add missing words; retain uncertainty, partial words at crop
edges, noise and disagreements explicitly. Do not distribute a multiword timed
unit's duration proportionally. Enter individual lexical boundaries only when
actually reviewed. Existing source timings have unresolved human-review origin.

Copy review-template.json elsewhere and fill the copy with actual review
identities/procedure and individual words. Preserve this original package and
its receipts unchanged. Source units are aids only. Merely editing a status field or
running this preparer does not establish human review, timing accuracy or gold
admission. Freeze and hash a separately adjudicated annotation before any model
comparison. Count missing/substituted/inserted words as well as matched-word
boundary errors. This package has no acceptance score or inferred labels.

Current publisher download page and separate audio notice grant CC BY 4.0.
The original annotation archive embeds an older CC BY-NC-SA 2.5 notice; both
notices are retained verbatim. Preserve attribution and this discrepancy.
"""


def descriptor(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read_pinned(path, expected):
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or not 0 <= expected['bytes'] <= 20 * 1024 * 1024
            or info.st_size != expected['bytes']):
        raise ValueError('pinned input must be a regular file of the exact bounded size')
    with path.open('rb') as stream:
        data = stream.read(expected['bytes'] + 1)
    if descriptor(data) != expected:
        raise ValueError('pinned input identity differs')
    return data


def wav_bytes(pcm, channels, rate):
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as wav:
        wav.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
        wav.writeframes(pcm)
    return stream.getvalue()


def canonical_audio(header, pcm, source):
    from scipy.signal import resample_poly
    if descriptor(header) != source['header']:
        raise ValueError('source header identity changed')
    with wave.open(io.BytesIO(header), 'rb') as wav:
        geometry = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype())
    if geometry != (2, 2, 20000, 5301780, 'NONE') or len(pcm) != 60 * 20000 * 4:
        raise ValueError('fixed stereo source/crop geometry differs')
    stereo = np.frombuffer(pcm, dtype='<i2').reshape(-1, 2)
    # Exact equal-channel mean in float64, followed by the explicitly fixed filter.
    mono = stereo.astype(np.float64).mean(axis=1) / 32768.
    converted = resample_poly(mono, 4, 5, window=('kaiser', 5.0), padtype='constant', cval=0.)
    if converted.shape != (960000,) or not np.isfinite(converted).all():
        raise ValueError('canonical crop resampling failed')
    quantized = np.rint(np.clip(converted, -1., 32767. / 32768.) * 32768.).astype('<i2')
    return wav_bytes(pcm, 2, 20000), wav_bytes(quantized.tobytes(), 1, 16000)


def source_units(archive, expected_members):
    units = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        metadata_name = 'maptaskv2-1/Data/corpus-resources/maptask-corpus.xml'
        metadata = zipped.read(metadata_name)
        if descriptor(metadata) != expected_members[metadata_name]:
            raise ValueError('participant metadata identity changed')
        conversations = [row for row in ET.fromstring(metadata) if row.attrib.get('id') == 'q1ec1']
        if len(conversations) != 1:
            raise ValueError('fixed conversation metadata missing or duplicated')
        roles = {row.attrib['role']: row.attrib['href'] for row in conversations[0]}
        if roles != {'g': 'maptask-participants.xml#id(q1eta1)',
                     'f': 'maptask-participants.xml#id(q1etb1)'}:
            raise ValueError('fixed participant metadata differs')
        for role, speaker in [('g', 'q1eta1'), ('f', 'q1etb1')]:
            name = f'maptaskv2-1/Data/timed-units/q1ec1.{role}.timed-units.xml'
            member = zipped.read(name)
            if descriptor(member) != expected_members[name]:
                raise ValueError('annotation member identity changed')
            for element in ET.fromstring(member):
                if element.tag != 'tu':
                    continue
                start, end = float(element.attrib['start']), float(element.attrib['end'])
                if not np.isfinite([start, end]).all() or not 0 <= start < end:
                    raise ValueError('invalid source timing unit')
                if start < 60 and end > 0:
                    text = ''.join(element.itertext()).strip()
                    if not text:
                        raise ValueError('empty source timing unit')
                    units.append({'id': element.attrib['id'], 'role': role, 'participant_id_hint': speaker,
                                  'text': text, 'source_start_seconds': start, 'source_end_seconds': end,
                                  'crosses_crop_edge': start < 0 or end > 60,
                                  'contains_multiple_space_separated_words': len(text.split()) > 1,
                                  'human_acoustic_boundary_review_verified': False})
    if len({row['id'] for row in units}) != len(units):
        raise ValueError('duplicated source timing unit')
    return sorted(units, key=lambda row: (row['source_start_seconds'], row['role'], row['id']))


def review_template(audio, units):
    return {'schema_version': 1, 'status': 'unreviewed', 'purpose': 'timing_calibration_only',
            'recording': 'HCRC/q1ec1', 'source_crop_seconds': [0, 60],
            'audio': audio, 'sample_rate': 16000, 'samples': 960000,
            'source_timed_units': len(units),
            'source_timing_human_review_origin': 'unverified',
            'bark_aligner_or_diarizer_outputs_used': False,
            'annotators': [], 'adjudication': None, 'words': [], 'uncertainties': [],
            'word_fields_required_after_actual_review': ['id', 'text', 'start_seconds', 'end_seconds',
                                                        'speaker', 'source_unit_ids', 'review_notes'],
            'gold_admitted': False, 'release_qualified': False}


def main():
    import scipy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    out = args.output.resolve()
    if out.exists() or args.output.is_symlink() or out.is_relative_to(repo):
        parser.error('fresh output outside the repository required')
    if p['crop_seconds'] != [0, 60] or p['recording'] != 'q1ec1' or p['conversion'] != CONVERSION:
        parser.error('only the fixed first60s calibration is admitted')
    actual_environment = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    if actual_environment != p['environment']:
        parser.error('fixed resampling environment required')
    for name, expected in p['tools'].items():
        read_pinned(tools / name, expected)
    archive = read_pinned(args.annotations, p['annotations']['archive'])
    units = source_units(archive, p['annotations']['members'])
    if len(units) != 125 or sum(row['role'] == 'g' for row in units) != 111:
        parser.error('fixed selected source units differ')
    notices = {}
    if set(p['notices']) != {'publisher-license.html', 'audio-license.html', 'archive-license.html',
                            'archive-readme.txt', 'archive-changelog.txt'}:
        parser.error('all five original publisher/archive notices required')
    for name, entry in p['notices'].items():
        data = read_pinned(Path(entry['path']), entry['identity'])
        notices[name] = data
    source = p['audio_source']
    header = fetch(source['url'], 64, start=0, end=63, total=source['bytes'], etag=source['etag'])
    if descriptor(header) != source['header']:
        parser.error('frozen source header differs')
    pcm = fetch(source['url'], 4800000, start=44, end=4800043,
                total=source['bytes'], etag=source['etag'])
    stereo, mono = canonical_audio(header, pcm, source)
    if args.protocol.read_bytes() != raw:
        raise ValueError('protocol changed during acquisition')
    out.mkdir(parents=True)
    for name, data in {'stereo20k.wav': stereo, 'mono16k.wav': mono, 'source-header.bin': header,
                       **notices}.items():
        (out / name).write_bytes(data)
    (out / 'protocol.json').write_bytes(raw)
    (out / 'README.txt').write_text(NOTICE)
    (out / 'source-units-unreviewed.json').write_text(json.dumps(units, indent=2) + '\n')
    template = review_template(descriptor(mono), units)
    (out / 'review-template.json').write_text(json.dumps(template, indent=2) + '\n')
    receipt = {'schema_version': 1, 'prepared': True, 'human_gold_admitted': False,
               'source_crop_seconds': [0, 60], 'model_or_aligner_calls': 0,
               'annotation_archive': p['annotations']['archive'], 'source_pcm': descriptor(pcm),
               'canonical_wav': descriptor(mono), 'source_units': len(units),
               'source_units_crossing_crop_edge': sum(row['crosses_crop_edge'] for row in units),
               'source_multiword_units': sum(row['contains_multiple_space_separated_words'] for row in units),
               'role_channel_assignment_verified': False, 'release_qualified': False,
               'environment': actual_environment, 'protocol': descriptor(raw),
               'limits': p['limits']}
    (out / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    files = {path.name: descriptor(path.read_bytes()) for path in out.iterdir() if path.is_file()}
    (out / 'files.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
