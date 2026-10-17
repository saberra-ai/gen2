"""Replay calibration preparation, never label gold or run any speech model."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import sys
import wave
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = cache / 'hcrc-word-review-v1'
read = lambda path: json.loads(path.read_text())
descriptor = lambda data: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
p = read(root / 'protocol.json')
for name, expected in p['tools'].items():
    assert descriptor((tools / name).read_bytes()) == expected
sys.path.insert(0, str(tools))
from prepare_word_timing_review import canonical_audio, review_template, source_units, read_pinned, NOTICE
from frozen_git_sources import verify_git_sources
source = sys.argv[1]
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/hcrc-word-review.protocol.json'], cwd=repo)
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: expected for name, expected in p['tools'].items()})
files = read(root / 'files.json')['files']
assert {path.name for path in root.iterdir()} == set(files) | {'files.json'}
assert all(path.is_file() and not path.is_symlink() for path in root.iterdir())
for name, expected in files.items():
    assert descriptor((root / name).read_bytes()) == expected
q = read(root / 'preparation.json')
assert q['prepared'] is True and q['human_gold_admitted'] is False and q['model_or_aligner_calls'] == 0
assert q['source_crop_seconds'] == [0, 60] and q['role_channel_assignment_verified'] is False
assert q['release_qualified'] is False and q['protocol'] == descriptor((root / 'protocol.json').read_bytes())
research = cache / 'research/manual-word-boundaries-2026-10-16'
archive = read_pinned(research / 'hcrcmaptask.nxtformatv2-1.zip', p['annotations']['archive'])
assert descriptor(archive) == p['annotations']['archive'] == q['annotation_archive']
units = source_units(archive, p['annotations']['members'])
assert units == read(root / 'source-units-unreviewed.json') and len(units) == q['source_units'] == 125
for name, entry in p['notices'].items():
    assert descriptor(Path(entry['path']).read_bytes()) == entry['identity']
    assert (root / name).read_bytes() == Path(entry['path']).read_bytes()
with wave.open(io.BytesIO((root / 'stereo20k.wav').read_bytes())) as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 2, 20000, 1200000)
    pcm = wav.readframes(1200000)
assert descriptor(pcm) == q['source_pcm']
stereo, mono = canonical_audio((root / 'source-header.bin').read_bytes(), pcm, p['audio_source'])
assert stereo == (root / 'stereo20k.wav').read_bytes() and mono == (root / 'mono16k.wav').read_bytes()
assert descriptor(mono) == q['canonical_wav']
assert review_template(descriptor(mono), units) == read(root / 'review-template.json')
assert (root / 'README.txt').read_text() == NOTICE
assert sum(row['crosses_crop_edge'] for row in units) == q['source_units_crossing_crop_edge']
assert sum(row['contains_multiple_space_separated_words'] for row in units) == q['source_multiword_units']
output = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-hcrc-word-review-preparation'
assert not output.exists()
output.mkdir()
for name in ['protocol.json', 'preparation.json', 'files.json', 'review-template.json', 'README.txt']:
    shutil.copy2(root / name, output / name)
for path in [Path(__file__), cache / 'hcrc-word-review-v1.controller.log',
             cache / 'platform-ci/model-free-240-word-review-local.log', cache / 'word-timing-review-numpy-only.log']:
    shutil.copy2(path, output / path.name)
summary = {'schema_version': 1, 'source_commit': source, 'prepared': True,
           'all_package_members_rehashed': len(files) + 1,
           'both_audio_conversions_replayed_exactly': True,
           'all_unreviewed_source_units_replayed': 125,
           'source_pins_verified': len(p['tools']),
           'human_annotators': 0, 'human_words': 0, 'human_gold_admitted': False,
           'new_model_or_aligner_calls': 0, 'release_qualified': False,
           'source_audio_and_annotation_payloads_outside_git': True,
           'limits': p['limits']}
(output / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
manifest = {path.name: descriptor(path.read_bytes()) for path in output.iterdir() if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
print(output, len(manifest) + 1, 'members')
