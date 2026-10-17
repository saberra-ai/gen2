"""Freeze one training-exposed development recording before context inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import wave

import numpy as np

from context_development_reference import published_reference, transcriber_reference
from corpus import fetch, pcm_header, pinned_download, summarize


def descriptor(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read_source(entry):
    path = Path(entry['path'])
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_size != entry['bytes']
            or not 0 < entry['bytes'] <= 32 * 1024 * 1024):
        raise ValueError('fixed regular bounded source input required')
    return pinned_download(entry, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    out = args.output.resolve()
    if out.exists() or args.output.is_symlink() or out.is_relative_to(repo):
        parser.error('fresh output outside the repository required')
    if (p['meeting'] != 'ES2011a' or p['crop_seconds'] != [0, 300]
            or p['channels'] != ['Mix-Headset', 'Array1-01'] or np.__version__ != p['numpy_version']):
        parser.error('exact fixed development selection/environment required')

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('development input protocol changed')
        for name, expected in p['tools'].items():
            if descriptor(tools / name) != expected:
                raise ValueError('development preparer source changed: ' + name)
        for entry in p['sources'].values():
            read_source(entry)

    frozen()
    inputs = {name: read_source(entry) for name, entry in p['sources'].items()}
    tree = json.loads(inputs['ntt_tree'])
    ids = sorted(row['path'][len('AMI/dev/'):-len('.rttm')] for row in tree['tree']
                 if row['path'].startswith('AMI/dev/') and row['path'].endswith('.rttm'))
    if (tree['truncated'] is not False or tree['sha'] != p['ntt_revision'] or ids != p['development_ids']
            or min(name for name in ids if not name.startswith(('EN2002', 'ES2004'))) != p['meeting']):
        raise ValueError('publisher-linked development identity selection differs')
    primary, words, source_members, zero = transcriber_reference(inputs['ami_archive'], p['meeting'], p['crop_seconds'])
    published = published_reference(inputs['ntt_rttm'].decode(), p['meeting'], p['crop_seconds'])
    if source_members != p['annotation_members'] or summarize(primary) != p['transcriber_summary']:
        raise ValueError('frozen development annotation extraction differs')
    reference_bytes = {name: (json.dumps(value, indent=2) + '\n').encode() for name, value in
                       [('transcriber', primary), ('published-ntt-mfa', published)]}
    if {name: hashlib.sha256(data).hexdigest() for name, data in reference_bytes.items()} != p['reference_file_sha256']:
        raise ValueError('frozen development reference bytes differ')
    out.mkdir(parents=True)
    (out / 'protocol.json').write_bytes(raw)
    references = out / 'references'
    references.mkdir()
    for name, data in reference_bytes.items():
        (references / (name + '.json')).write_bytes(data)
    (out / 'source-NTT.rttm').write_bytes(inputs['ntt_rttm'])
    (out / 'NTT-LICENSE.txt').write_bytes(inputs['ntt_license'])
    (out / 'source-word-metadata.json').write_text(json.dumps({'positive_duration_words': words,
        'zero_duration_source_words': zero, 'source_member_sha256': source_members,
        'human_word_boundary_gold': False}, indent=2) + '\n')
    (out / 'NOTICE.txt').write_text(p['notice'] + '\n')
    result = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'prepared': False, 'meeting': p['meeting'], 'crop_seconds': p['crop_seconds'],
        'training_exposed_development': True, 'model_calls': 0, 'release_qualified': False,
        'reference_summaries': {'transcriber': summarize(primary), 'published_ntt_mfa': summarize(published)},
        'channels': {}, 'limits': p['limits']}
    persist = lambda: (out / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    persist()
    try:
        for name in p['channels']:
            source = p['audio'][name]
            header = fetch(source['url'], 64, start=0, end=63, total=source['bytes'], etag=source['etag'])
            if {'bytes': len(header), 'sha256': hashlib.sha256(header).hexdigest()} != source['header']:
                raise ValueError('frozen development audio header changed')
            offset, source_samples = pcm_header(header, source['bytes'])
            if offset != 44 or source_samples != 17821526:
                raise ValueError('development WAV source geometry differs')
            pcm = fetch(source['url'], 9600000, start=44, end=9600043,
                        total=source['bytes'], etag=source['etag'])
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(pcm)
            wav_path = out / (p['meeting'] + '-' + name + '.wav')
            wav_path.write_bytes(buffer.getvalue())
            (out / (name + '.source-header.bin')).write_bytes(header)
            actual = np.frombuffer(pcm, dtype='<i2').astype('<f4') / np.float32(32768.)
            result['channels'][name] = {'wav': descriptor(wav_path), 'samples': len(actual),
                'source_pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                'input_pcm_f32le_sha256': hashlib.sha256(actual.astype('<f4').tobytes()).hexdigest(),
                'source_samples': source_samples, 'downloaded_pcm_bytes': len(pcm)}
            persist()
        frozen()
        result['prepared'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        persist()
        files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
        (out / 'files.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
