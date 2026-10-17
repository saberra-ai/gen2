"""AMI development references with the frozen corpus's lexical-segment semantics.

Historical corpus.ami_reference remains unchanged and specific to ES2004a.
Published NTT MFA RTTM is a separate reference, not a renamed word union.
"""
import io
import math
import re
import zipfile

from corpus import NITE, annotation_member, interval, sha, xml_root
from diagnose_ami_reference import crop_reference


def check_crop(crop):
    if (len(crop) != 2 or not all(math.isfinite(x) for x in crop)
            or not 0 <= crop[0] < crop[1]):
        raise ValueError('invalid AMI development crop')


def transcriber_reference(archive, meeting, crop):
    if not re.fullmatch(r'[A-Z]{2}\d{4}[a-z]', meeting):
        raise ValueError('invalid AMI meeting identity')
    check_crop(crop)
    references, words, sources, zero_duration = [], [], {}, []
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for speaker in 'ABCD':
            name = f'{meeting}.{speaker}.words.xml'
            word_path, segment_path = 'words/' + name, f'segments/{meeting}.{speaker}.segments.xml'
            word_bytes, segment_bytes = [annotation_member(zipped, path) for path in [word_path, segment_path]]
            sources.update({word_path: sha(word_bytes), segment_path: sha(segment_bytes)})
            nodes = list(xml_root(word_bytes))
            ids = {node.attrib[NITE + 'id']: index for index, node in enumerate(nodes)}
            if len(ids) != len(nodes):
                raise ValueError('duplicate AMI word IDs')

            def lexical(node):
                return node.tag == 'w' and node.get('punc') != 'true' and any(c.isalpha() for c in ''.join(node.itertext()))

            for node in nodes:
                if not lexical(node):
                    continue
                start, end = float(node.get('starttime')), float(node.get('endtime'))
                if not all(math.isfinite(x) for x in [start, end]) or not 0 <= start <= end:
                    raise ValueError('invalid lexical word timing')
                if start == end:
                    if crop[0] <= start < crop[1]:
                        zero_duration.append({'source_id': node.get(NITE + 'id'), 'speaker': speaker,
                                              'source_seconds': start, 'text': ''.join(node.itertext())})
                    continue
                clipped = interval(start, end, speaker, crop)
                if clipped:
                    words.append({**clipped, 'text': ''.join(node.itertext()), 'source_id': node.get(NITE + 'id')})
            for segment in xml_root(segment_bytes):
                linked = []
                for child in segment:
                    match = re.fullmatch(re.escape(name) + r'#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?', child.get('href', ''))
                    if child.tag != NITE + 'child' or not match:
                        raise ValueError('unexpected AMI NXT link')
                    first, last = ids[match[1]], ids[match[2] or match[1]]
                    if first > last:
                        raise ValueError('reversed AMI child range')
                    linked.extend(nodes[first:last + 1])
                if any(lexical(node) for node in linked):
                    clipped = interval(segment.get('transcriber_start'), segment.get('transcriber_end'), speaker, crop)
                    if clipped:
                        references.append(clipped)
    return references, words, sources, zero_duration


def published_reference(text, meeting, crop):
    check_crop(crop)
    # Keep the archived publisher-linked RTTM separate from manual1.6.2 word times.
    for line in text.splitlines():
        fields = line.split()
        if fields and (len(fields) != 10 or fields[2] != '1'):
            raise ValueError('unexpected published RTTM channel')
    return crop_reference(text, meeting, *crop)
