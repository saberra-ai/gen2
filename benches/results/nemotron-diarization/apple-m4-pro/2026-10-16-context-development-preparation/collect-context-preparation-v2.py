"""Replay fixed input preparation and model-free C ABI probe construction."""
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
def desc(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

read = lambda path: json.loads(path.read_text())
inputs = cache / 'context-development-inputs-v1'
build = cache / 'context-probe-construction-v1'
source = sys.argv[1]
p, b = read(inputs / 'protocol.json'), read(build / 'protocol.json')
# Admit current replay modules before importing them. Historical Git hashes alone
# do not establish that current modules executing this collection are unchanged.
current_tools = dict(p['tools'])
for name, identity in b['tools'].items():
    assert name not in current_tools or current_tools[name] == identity
    current_tools[name] = identity
for name, identity in current_tools.items():
    assert desc(tools / name) == identity
verifier_bytes = subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/frozen_git_sources.py'], cwd=repo)
assert (tools / 'frozen_git_sources.py').read_bytes() == verifier_bytes
sys.path.insert(0, str(tools))
import numpy as np
from context_development_reference import transcriber_reference, published_reference
from corpus import summarize, pcm_header
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from sdk_archive import inventory
assert np.__version__ == p['numpy_version']
for root, name, protocol in [(inputs, 'context-development-inputs.protocol.json', p),
                             (build, 'context-probe-construction.protocol.json', b)]:
    assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
        'git', 'show', source + ':tools/nemotron-diarization/' + name], cwd=repo)
    verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: identity
                                      for name, identity in protocol['tools'].items()})
    records = read(root / 'files.json')['files']
    for name, identity in records.items():
        assert desc(root / name) == identity
q, c = read(inputs / 'preparation.json'), read(build / 'construction.json')
assert q['prepared'] is True and c['constructed'] is True
assert q['source_commit'] == c['source_commit'] == source
assert q['model_calls'] == c['model_calls'] == 0
assert q['release_qualified'] is c['release_qualified'] is False
for entry in p['sources'].values():
    assert desc(entry['path']) == {key: entry[key] for key in ['bytes', 'sha256']}
archive = Path(p['sources']['ami_archive']['path']).read_bytes()
primary, words, members, zero = transcriber_reference(archive, p['meeting'], p['crop_seconds'])
published = published_reference((inputs / 'source-NTT.rttm').read_text(), p['meeting'], p['crop_seconds'])
assert members == p['annotation_members'] and summarize(primary) == p['transcriber_summary']
assert q['reference_summaries'] == {'transcriber': summarize(primary), 'published_ntt_mfa': summarize(published)}
assert read(inputs / 'source-word-metadata.json') == {'positive_duration_words': words,
    'zero_duration_source_words': zero, 'source_member_sha256': members, 'human_word_boundary_gold': False}
for name, reference in [('transcriber', primary), ('published-ntt-mfa', published)]:
    path = inputs / 'references' / (name + '.json')
    assert read(path) == reference and desc(path)['sha256'] == p['reference_file_sha256'][name]
for channel in p['channels']:
    record = q['channels'][channel]
    path = inputs / (p['meeting'] + '-' + channel + '.wav')
    assert desc(path) == record['wav']
    header = (inputs / (channel + '.source-header.bin')).read_bytes()
    assert {'bytes': len(header), 'sha256': hashlib.sha256(header).hexdigest()} == p['audio'][channel]['header']
    assert pcm_header(header, p['audio'][channel]['bytes']) == (44, 17821526)
    with wave.open(io.BytesIO(path.read_bytes())) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, 4800000)
        pcm = wav.readframes(4800000)
    actual = np.frombuffer(pcm, dtype='<i2').astype('<f4') / np.float32(32768.)
    assert len(pcm) == record['downloaded_pcm_bytes'] == 9600000
    assert record['samples'] == len(actual) == 4800000
    assert hashlib.sha256(pcm).hexdigest() == record['source_pcm_sha256']
    assert hashlib.sha256(actual.astype('<f4').tobytes()).hexdigest() == record['input_pcm_f32le_sha256']
assert inventory(build / 'work/nemo') == b['package_inventory'] == inventory(Path(b['package_path']))
assert c['staged_source_identity'] == {name: entry['identity'] for name, entry in b['source_files'].items()}
assert c['non_system_dependencies'] == sorted(str(build / 'work/sources' / name) for name in b['source_files'])
assert [row['name'] for row in c['commands']] == ['sandbox-controls', 'compiler-version', 'compile-probe', 'compile-sha256', 'link', 'invalid-profile']
assert all(row['returncode'] == row['expected_returncode'] for row in c['commands'])
assert c['storage_guard']['released'] is True
binary = build / 'work/bin/native-context'
assert desc(binary) == c['binary']
audit = audit_loaded_libraries((build / 'invalid-profile.stderr').read_text(), build / 'work/nemo',
                              read(build / 'work/nemo/package.json'), binary)
assert audit == c['invalid_profile_loaded_images'] and audit['private_library_count'] == 92
output = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-context-development-preparation'
assert not output.exists()
output.mkdir()
for root, prefix, names in [
    (inputs, 'input-', ['protocol.json', 'preparation.json', 'files.json', 'NOTICE.txt']),
    (build, 'build-', ['protocol.json', 'construction.json', 'files.json', 'model-free.sb'])]:
    for name in names:
        shutil.copy2(root / name, output / (prefix + name))
for path in (inputs / 'references').iterdir():
    shutil.copy2(path, output / ('reference-' + path.name))
for pattern in ['*.stdout', '*.stderr', 'work/*.d']:
    for path in build.glob(pattern):
        shutil.copy2(path, output / path.name)
for path in [Path(__file__), cache / 'context-development-inputs-v1.controller.log',
             cache / 'context-probe-construction-v1.controller.log',
             cache / 'platform-ci/model-free-244-context-construction-local.log']:
    shutil.copy2(path, output / path.name)
summary = {'schema_version': 1, 'source_commit': source, 'prepared': True, 'constructed': True,
    'current_tools_verified_before_import': len(current_tools),
    'historical_verifier_identity': desc(tools / 'frozen_git_sources.py'),
    'all_saved_files_rehashed': True, 'exact_reference_policies_replayed': 2,
    'canonical_audio_samples_replayed_per_channel': 4800000, 'channels': 2,
    'staged_public_source_and_header_inputs': 7, 'runtime_inventory_entries': len(b['package_inventory']),
    'actual_invalid_profile_private_images': 92, 'new_neural_calls': 0,
    'training_exposed_development': True, 'weak_speaker_B_coverage_seconds': 1.792,
    'large_context_inference_performed': False, 'release_qualified': False,
    'limits': p['limits'] + b['limits']}
(output / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {path.name: desc(path) for path in output.iterdir() if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(output, len(files) + 1, 'members')
