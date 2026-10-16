"""Acquire the model-blind corpus in docs/research/nemotron-heldout-corpus.md.

No inference or result-dependent selection occurs here. Audio stays outside Git;
the emitted manifest, references and provenance are frozen before qualification.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
import zlib


NITE = '{http://nite.sourceforge.net/}'
MAX_MEMBER = 20 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, limit, *, start=None, end=None, total=None, etag=None):
    """Reject whole-archive responses before reading their bodies."""
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'Gen2-corpus-qualification/1'}
    if start is not None:
        if not 0 <= start <= end < total or end - start + 1 > limit:
            raise ValueError('invalid bounded HTTP range')
        headers['Range'] = f'bytes={start}-{end}'
        if etag is not None:
            headers['If-Match'] = etag
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('unexpected HTTP content encoding')
        expected = end - start + 1 if start is not None else None
        if start is not None:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                raise ValueError('server did not honor the exact frozen range')
            if etag is not None and response.headers.get('ETag') != etag:
                raise ValueError('source ETag changed')
        elif response.status != 200:
            raise ValueError('unexpected HTTP status')
        length = response.headers.get('Content-Length')
        if length is not None and (int(length) > limit or (expected is not None and int(length) != expected)):
            raise ValueError('unexpected HTTP content length')
        data = response.read(limit + 1)
        if len(data) > limit or (expected is not None and len(data) != expected):
            raise ValueError('HTTP body length differs')
        return data


def pinned_download(entry, local=None):
    if local:
        with Path(local).open('rb') as source:
            data = source.read(entry['bytes'] + 1)
    else:
        data = fetch(entry['url'], entry['bytes'])
    if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
        raise ValueError('annotation source digest/length differs')
    return data


def zip_audio(archive, member):
    """Read one frozen ZIP member without fetching the multi-GB archive."""
    def read(start, count):
        return fetch(archive['url'], count, start=start, end=start + count - 1,
                     total=archive['bytes'], etag=archive['etag'])

    offset = member['offset']
    header = read(offset, 30)
    signature, version, flags, method, _, _, crc, compressed, expanded, names, extra = struct.unpack('<4s5H3I2H', header)
    if signature != b'PK\x03\x04' or version > 20 or flags & ~0x808 or method != 8:
        raise ValueError('unexpected ZIP local header')
    if not 0 < member['compressed_bytes'] <= MAX_MEMBER or not 0 < member['bytes'] <= MAX_MEMBER:
        raise ValueError('ZIP member exceeds qualification bounds')
    expected_crc = int(member['crc32'], 16)
    # Data-descriptor writers may know the uncompressed size in advance while
    # deferring compressed size/CRC; each populated field must still match.
    actual = (crc, compressed, expanded)
    expected = (expected_crc, member['compressed_bytes'], member['bytes'])
    if any(a != e and not (flags & 8 and a == 0) for a, e in zip(actual, expected)):
        raise ValueError('ZIP local sizes/CRC differ from frozen directory')
    if names > 1024 or extra > 65535:
        raise ValueError('oversized ZIP name')
    name = read(offset + 30, names).decode('utf-8')
    if name != 'voxconverse_test_wav/' + member['id'] + '.wav':
        raise ValueError('unexpected ZIP member path')
    payload = read(offset + 30 + names + extra, member['compressed_bytes'])
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(payload, member['bytes'] + 1)
    if len(data) != member['bytes'] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('invalid/oversized ZIP deflate payload')
    if zlib.crc32(data) != expected_crc:
        raise ValueError('ZIP CRC mismatch')
    return data


def pcm_header(prefix, total):
    """Validate PCM geometry and locate data, including non-44-byte headers."""
    if len(prefix) < 12 or prefix[:4] != b'RIFF' or prefix[8:12] != b'WAVE' or struct.unpack_from('<I', prefix, 4)[0] + 8 != total:
        raise ValueError('invalid RIFF extent')
    position, fmt = 12, None
    while position + 8 <= min(len(prefix), 65536):
        kind, length = struct.unpack_from('<4sI', prefix, position)
        start = position + 8
        if start + length > total:
            raise ValueError('RIFF chunk exceeds source')
        if kind == b'fmt ':
            if fmt is not None or not 16 <= length <= 40 or start + length > len(prefix):
                raise ValueError('invalid PCM format chunk')
            fmt = struct.unpack_from('<HHIIHH', prefix, start)
            if fmt != (1, 1, 16000, 32000, 2, 16):
                raise ValueError('corpus requires mono PCM16 16 kHz')
        if kind == b'data':
            if fmt is None or not length or length % 2:
                raise ValueError('invalid PCM data')
            return start, length // 2
        position = start + length + length % 2
    raise ValueError('PCM header exceeds bounded prefix or has no data')


def crop_audio(entry):
    prefix = fetch(entry['url'], 4096, start=0, end=4095, total=entry['bytes'], etag=entry['etag'])
    offset, samples = pcm_header(prefix, entry['bytes'])
    first, last = [int(t * 16000) for t in entry['crop_seconds']]
    if not 0 <= first < last <= samples or last - first > 600 * 16000:
        raise ValueError('crop outside source audio')
    data = fetch(entry['url'], (last - first) * 2, start=offset + first * 2,
                 end=offset + last * 2 - 1, total=entry['bytes'], etag=entry['etag'])
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(data)
    return out.getvalue(), {'source_pcm_samples': samples, 'data_offset': offset,
                            'source_header_sha256': sha(prefix), 'downloaded_pcm_sha256': sha(data)}


def interval(start, end, speaker, crop):
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError('invalid reference interval')
    first, last = max(start, crop[0]), min(end, crop[1])
    return {'start': first - crop[0], 'end': last - crop[0], 'speaker': speaker} if first < last else None


def rttm_reference(data, recording, duration):
    result = []
    for line in data.decode('utf-8').splitlines():
        if not line.strip():
            continue
        f = line.split()
        if len(f) != 10 or f[0] != 'SPEAKER' or f[1] != recording or f[2] != '1':
            raise ValueError('unexpected RTTM row')
        start, length = float(f[3]), float(f[4])
        item = interval(start, start + length, f[7], [0, duration])
        if item:
            result.append(item)
    return result


def xml_root(data):
    # The pinned ICSI MRT declares this external DTD. Never fetch it, and reject
    # internal subsets/entities; core XML syntax is sufficient for extraction.
    data = data.replace(b'<!DOCTYPE Meeting SYSTEM "http://www.icsi.berkeley.edu/Speech/mr/Meeting1.dtd">', b'')
    if len(data) > MAX_MEMBER or b'\0' in data or b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('oversized or entity-bearing annotation XML')
    return ET.fromstring(data)


def ami_reference(archive, crop):
    references, words, sources = [], [], {}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for speaker in 'ABCD':
            name = f'ES2004a.{speaker}.words.xml'
            word_bytes = annotation_member(z, 'words/' + name)
            segment_bytes = annotation_member(z, f'segments/ES2004a.{speaker}.segments.xml')
            sources['words/' + name] = sha(word_bytes)
            sources[f'segments/ES2004a.{speaker}.segments.xml'] = sha(segment_bytes)
            nodes = list(xml_root(word_bytes))
            ids = {w.attrib[NITE + 'id']: i for i, w in enumerate(nodes)}
            if len(ids) != len(nodes):
                raise ValueError('duplicate AMI word IDs')

            def lexical(node):
                return node.tag == 'w' and node.get('punc') != 'true' and any(c.isalpha() for c in ''.join(node.itertext()))

            for node in nodes:
                if lexical(node) and float(node.get('endtime')) > float(node.get('starttime')):
                    item = interval(node.get('starttime'), node.get('endtime'), speaker, crop)
                    if item:
                        words.append({**item, 'text': ''.join(node.itertext()), 'source_id': node.get(NITE + 'id')})
            for segment in xml_root(segment_bytes):
                linked = []
                for child in segment:
                    match = re.fullmatch(re.escape(name) + r'#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?', child.get('href', ''))
                    if child.tag != NITE + 'child' or not match:
                        raise ValueError('unexpected AMI NXT link')
                    a, b = ids[match[1]], ids[match[2] or match[1]]
                    if a > b:
                        raise ValueError('reversed AMI child range')
                    linked.extend(nodes[a:b + 1])
                if any(lexical(w) for w in linked):
                    item = interval(segment.get('transcriber_start'), segment.get('transcriber_end'), speaker, crop)
                    if item:
                        references.append(item)
    return references, words, sources


def annotation_member(archive, name):
    info = archive.getinfo(name)
    if not 0 < info.file_size <= MAX_MEMBER:
        raise ValueError('oversized annotation member')
    with archive.open(info) as source:
        data = source.read(MAX_MEMBER + 1)
        if len(data) != info.file_size or source.read(1):
            raise ValueError('annotation member length differs')
    return data


def icsi_reference(archive, crop):
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        data = annotation_member(z, 'transcripts/Bmr005.mrt')
    result, texts, noises = [], [], []
    for segment in xml_root(data).iter('Segment'):
        text = ' '.join(''.join(segment.itertext()).split())
        speaker = segment.get('Participant')
        start, end = float(segment.get('StartTime')), float(segment.get('EndTime'))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('invalid ICSI source interval')
        if speaker and re.search('[A-Za-z]', text) and end > start:
            item = interval(segment.get('StartTime'), segment.get('EndTime'), speaker, crop)
            if item:
                result.append(item)
                texts.append({**item, 'text': text})
        for noise in segment.iter('NonVocalSound'):
            if float(segment.get('EndTime')) > crop[0] and float(segment.get('StartTime')) < crop[1]:
                noises.append({'source_start': segment.get('StartTime'), 'source_end': segment.get('EndTime'),
                               'attributes': noise.attrib})
    return result, texts, {'transcripts/Bmr005.mrt': sha(data)}, noises


def summarize(reference):
    """Exact interval union per speaker; inter-speaker overlap is preserved."""
    unions = {}
    for item in sorted(reference, key=lambda x: (x['speaker'], x['start'], x['end'])):
        spans = unions.setdefault(item['speaker'], [])
        if spans and item['start'] <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], item['end'])
        else:
            spans.append([item['start'], item['end']])
    events = {}
    for spans in unions.values():
        for a, b in spans:
            events[a] = events.get(a, 0) + 1
            events[b] = events.get(b, 0) - 1
    active, previous, speech, overlap = 0, 0, 0., 0.
    for t, change in sorted(events.items()):
        speech += (t - previous) if active else 0
        overlap += (t - previous) if active >= 2 else 0
        active += change
        previous = t
    return {'speakers': len(unions), 'speaker_seconds': {s: round(sum(b-a for a, b in spans), 6) for s, spans in unions.items()},
            'speech_union_seconds': round(speech, 6), 'overlap_seconds': round(overlap, 6),
            'intervals': len(reference), 'short_intervals_at_most_1s': sum(v['end'] - v['start'] <= 1 + 1e-9 for v in reference)}


def acquire(config, output, ami_annotations=None):
    if output.exists():
        raise ValueError('refusing to replace a frozen corpus directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.', dir=output.parent))
    try:
        manifest = {'schema_version': 1, 'scope': 'local tuning-heldout, not guaranteed checkpoint-unseen',
                    'selection': 'docs/research/nemotron-heldout-corpus.md', 'inference_performed': False,
                    'sources_sha256': sha(json.dumps(config, sort_keys=True).encode()),
                    'extractor_sha256': sha(Path(__file__).read_bytes()), 'slices': []}
        ami = pinned_download(config['ami_annotations'], ami_annotations)
        icsi = pinned_download(config['icsi_annotations'])
        (staging/'AMI-annotation-source.zip').write_bytes(ami)
        (staging/'ICSI-annotation-source.zip').write_bytes(icsi)

        def save(name, audio, reference, provenance, expected_speakers, secondary=None):
            _, samples = pcm_header(audio[:65536], len(audio))
            if not 320 <= samples <= 600 * 16000:
                raise ValueError('corpus WAV duration outside gate bounds')
            stats = summarize(reference)
            if stats['speakers'] != expected_speakers:
                raise ValueError('frozen reference speaker count differs')
            wav = name + '.wav'
            ref = name + '.reference.json'
            (staging/wav).write_bytes(audio)
            (staging/ref).write_text(json.dumps(reference, indent=2) + '\n')
            if secondary is not None:
                (staging/(name + '.transcript.json')).write_text(json.dumps(secondary, indent=2) + '\n')
            manifest['slices'].append({'id': name, 'wav': wav, 'audio_samples': samples,
                                       'uem_seconds': [0, samples / 16000], 'reference': ref,
                                       'statistics': stats, 'provenance': provenance})

        for member in config['vox_members']:
            print('Acquiring Vox ' + member['id'], flush=True)
            audio = zip_audio(config['vox_archive'], member)
            _, samples = pcm_header(audio[:65536], len(audio))
            annotation = pinned_download(member['annotation'])
            (staging/(member['id'] + '.rttm')).write_bytes(annotation)
            save('vox-' + member['id'], audio, rttm_reference(annotation, member['id'], samples / 16000),
                 {'audio_archive': config['vox_archive'], 'member': member,
                  'reference_policy': 'provided RTTM clipped to complete decoded WAV UEM; original RTTM retained unchanged',
                  'original_reference_last_endpoint_seconds': max(float(f[3]) + float(f[4]) for line in annotation.decode().splitlines() if (f := line.split())),
                  'training_exposure': 'VoxConverse v0.3 dev/test explicitly declared training'}, member['speakers'])
        crop = [300, 420]
        refs, words, sources = ami_reference(ami, crop)
        for entry in config['ami_audio']:
            print('Acquiring AMI ' + entry['id'], flush=True)
            audio, metadata = crop_audio(entry)
            save(entry['id'], audio, refs, {'source': entry, 'metadata': metadata, 'annotation_members': sources,
                 'reference_policy': 'human-reviewed lexical transcriber segments, clipped; word timing is forced aligned',
                 'training_exposure': 'outside declared AMI train/dev under official partition, not guaranteed zero exposure'}, 4,
                 {'timing': 'automatic forced alignment of human words; secondary only', 'words': words})
        entry = config['icsi_audio']
        print('Acquiring ICSI Bmr005', flush=True)
        audio, metadata = crop_audio(entry)
        refs, texts, sources, noises = icsi_reference(icsi, entry['crop_seconds'])
        save(entry['id'], audio, refs, {'source': entry, 'metadata': metadata, 'annotation_members': sources,
             'reference_policy': 'positive-duration participant MRT segments containing alphabetic transcript text',
             'training_exposure': 'ICSI Full explicitly declared training', 'incident_noise_annotations': noises}, 8,
             {'timing': 'manual utterance boundaries, no word-level timing claim', 'utterances': texts})
        (staging/'NOTICE.md').write_text(config['notice'])
        manifest['files'] = {p.name: {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size} for p in sorted(staging.iterdir())}
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(output)
        print(output/'manifest.json')
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path(__file__).with_name('corpus.sources.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ami-annotations', type=Path)
    args = parser.parse_args()
    acquire(json.loads(args.sources.read_text()), args.output, args.ami_annotations)


if __name__ == '__main__':
    main()
