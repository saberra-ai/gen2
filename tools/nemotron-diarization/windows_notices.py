"""Exact supplemental SentencePiece texts from verified pinned upstream sources."""
import json
from pathlib import Path

from windows_archive import digest

SOURCE = Path(__file__).with_name('notices') / 'sentencepiece'
TARGET = Path('notices/nemo-speech/third_party/vcpkg/sentencepiece')
EXPECTED = {
    'darts-clone-LICENSE': (1481, '155f59997298ee336602c49f9c1110f268ac394ca2197eb02647a3555935ad52'),
    'esaxx-LICENSE': (1112, '7c28553d1d3312d65fe309f76a22ebaf33a3d76c8c1e3b98a88ee4654ecb53db'),
    'sais-lite-NOTICE.txt': (1174, '162a72e52ad69921a67676af4fb59911233566dce1637e2e33db27659fcac523'),
}


def policy():
    record = json.loads((SOURCE / 'provenance.json').read_text(encoding='utf-8'))
    if (record['sentencepiece_version'] != '0.2.1'
            or record['vcpkg_baseline'] != '9e593bb18ea69cc5095e012465dcd675a822ed0d'
            or record['source_archive']['sha256'] != 'c1a59e9259c9653ad0ade653dadff074cd31f0a6ff2a11316f67bee4189a8f1b'
            or set(record['files']) != set(EXPECTED)
            or any((row['bytes'], row['sha256']) != EXPECTED[name] for name, row in record['files'].items())):
        raise ValueError('supplemental notice policy differs from verified source selection')
    return record


def verify_supplement(root, record):
    if record != policy():
        raise ValueError('package supplemental notice provenance differs from pinned policy')
    for name, (size, sha256) in EXPECTED.items():
        path = Path(root) / TARGET / name
        if path.is_symlink() or path.stat().st_size != size or digest(path) != sha256:
            raise ValueError('supplemental source notice bytes differ: ' + name)
    if (Path(root) / TARGET / 'gen2-supplement-provenance.json').read_bytes() != (SOURCE / 'provenance.json').read_bytes():
        raise ValueError('retained supplemental source provenance differs')


def install_supplement(root):
    record = policy()
    destination = Path(root) / TARGET
    destination.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        target = destination / name
        with target.open('xb') as output:
            output.write((SOURCE / name).read_bytes())
    with (destination / 'gen2-supplement-provenance.json').open('xb') as output:
        output.write((SOURCE / 'provenance.json').read_bytes())
    verify_supplement(root, record)
    return record
