"""Independently replay saved bounded corpus evidence; no helper/model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--admission', action='store_true')
args = parser.parse_args()
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / ('bounded-corpus-runtime-v1-admission' if args.admission else 'bounded-corpus-runtime-v1')
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro' / (
    '2026-10-17-bounded-corpus-admission' if args.admission else '2026-10-17-bounded-corpus')
read = lambda path: json.loads(path.read_bytes())


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def regular_files(directory):
    result = {}
    for path in directory.rglob('*'):
        mode = path.lstat().st_mode
        assert stat.S_ISDIR(mode) or stat.S_ISREG(mode), path
        if stat.S_ISREG(mode):
            result[path.relative_to(directory).as_posix()] = descriptor(path)
    return result


assert not out.exists() and not out.is_symlink()
before = regular_files(root)


def captured_json(name):
    raw = (root / name).read_bytes()
    assert before[name] == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)


p, q = captured_json('protocol.json'), captured_json('qualification.json')
source = q['source_commit']
assert descriptor(root / 'protocol.json') == q['protocol'] == {
    'bytes': 132337, 'sha256': 'c849c8a244d7b12f97e0f3c8593c9416d13df5407f48fcba6de7d6eec0a1c7d1'}
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/bounded-corpus.protocol.json'], cwd=repo)
for key, base in [('tools', tools), ('repository_files', repo), ('preparation_records', Path('/'))]:
    for name, value in p[key].items():
        assert descriptor(base / name) == value, name
sys.path.insert(0, str(tools))
import numpy as np
from bounded_corpus_metrics import score_slice
from compare_macos_cpu_pools import model_free_policy
from corpus import pcm_header
from frozen_git_sources import verify_git_sources
from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files, verify_package_provenance
from qualify_corpus import validate_corpus
from qualify_macos_portable_runtime import CONTROL
from qualify_valid_mel_corpus import admit_reference
from sdk_archive import inventory

pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
assert {'python': platform.python_version(), 'numpy': np.__version__,
        'interpreter': descriptor(Path(sys.executable).resolve(strict=True))} == p['scoring_environment']
work = Path(p['work_path'])
assert inventory(work) == p['inventory']
manifest = validate_corpus(work / 'corpus')
old = read(tools / 'corpus.protocol.json')
assert [i['id'] for i in manifest['slices']] == old['slice_ids']
assert p['quality_gates'] == old['scoring']['per_slice_gates']
assert p['legacy_scoring'] == old['scoring']
legacy = read(Path(p['legacy_receipt_path']))
references = {i['id']: admit_reference(i, work, old, legacy) for i in manifest['slices']}
package = work / 'cpu/nemo'
package_manifest = read(package / 'package.json')
verify_files(package, package_manifest['files'])
verify_package_provenance(package, package_manifest)
assert package_manifest['runtime_profile'] == q['runtime_profile'] == 'gen2-valid-mel-pool-cpu-v2'
assert q['runtime_version'] == 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2'
assert q['input_admission_passed'] is True and q['admit_only'] is args.admission
assert q['release_qualified'] is False and 'error' not in q
assert q['cached_original_fp32_oracles_reused'] == 8 and q['fresh_hf_or_original_nemo_calls'] == 0
assert q['storage']['released'] is True
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert not (root / 'diagnostic-reserve.bin').exists()
scores, usage = [], {}
if args.admission:
    assert set(before) == {'qualification.json', 'protocol.json'}
    assert q['commands'] == q['results'] == q['neural_commands_started'] == []
    assert q['loaded_images'] == {}
    assert q['all_fixed_pairs_completed'] is q['all_slice_quality_gates_passed'] is False
else:
    results = root / 'results'
    home, temporary = root / 'home', root / 'tmp'
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': str(home), 'TMPDIR': str(temporary) + '/'}
    policy, premodel = root / 'runtime.sb', root / 'premodel.sb'
    denied = [*map(Path, p['denied_paths']), work / 'oracle',
        *[work / 'corpus' / n for n in manifest['files'] if not n.endswith('.wav')],
        work / 'corpus/manifest.json']
    policy_text = '(version 1)\n(allow default)\n(deny network*)\n'
    policy_text += '(deny file-write* (subpath ' + json.dumps(str(work)) + '))\n'
    policy_text += '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(str(path)) + ')' for path in denied) + ')\n'
    policy_text += '(deny file-read-data (subpath ' + json.dumps(str(results)) + '))\n'
    assert policy.read_text() == policy_text
    assert premodel.read_text() == model_free_policy(policy_text)
    prior = results / 'prior-output-control.txt'
    assert prior.read_bytes() == b'not an oracle'
    reads = [work / 'oracle' / (manifest['slices'][0]['id'] + '-fp32.npy'),
        work / 'corpus' / manifest['slices'][0]['reference'], repo / 'Cargo.toml', prior]
    commands = [
        ('runtime-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps({'reads': list(map(str, reads)),
            'writes': [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf')]})], policy, False),
        ('premodel-controls', ['/usr/bin/python3', '-c', CONTROL, json.dumps({
            'reads': [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf')], 'writes': []})], premodel, False),
        ('version-only', [work / 'cpu/bin/load-probe', package / 'lib' / package_manifest['entry_library'],
            q['runtime_version']], premodel, True)]
    for item in manifest['slices']:
        name = item['id'] + '-cpu'
        wav = work / 'corpus' / item['wav']
        commands.extend([
            (name + '-native', [work / 'cpu/bin/native-stream', work / 'assets/Nemotron-3-Diarization.q8_0.gguf',
                wav, '1', '-1', results / (name + '-native.json')], policy, True),
            (name + '-public', [work / 'cpu/bin/public-stream', work / 'assets',
                package / 'lib' / package_manifest['entry_library'], wav, '1', results / (name + '-public.f32')], policy, True)])
    expected = ['runtime-controls', 'premodel-controls', 'version-only']
    expected += [i['id'] + '-cpu-' + k for i in manifest['slices'] for k in ['native', 'public']]
    assert [c['name'] for c in q['commands']] == expected
    assert q['neural_commands_started'] == expected[3:]
    assert q['all_fixed_pairs_completed'] is True and len(q['results']) == 8
    assert read(root / 'host.json')['cpu'] == p['expected_cpu'] == 'Apple M4 Pro'
    assert set(q['loaded_images']) == set(expected[2:])
    for c, (name, argv, sandbox, trace) in zip(q['commands'], commands, strict=True):
        assert c['returncode'] == 0 and c['inference_admitted'] is (c['name'] in expected[3:])
        child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)] if trace else argv
        actual, variables = sandboxed_command(child, sandbox, environment)
        assert c['name'] == name and c['argv'] == actual and c['environment'] == variables and c['cwd'] == str(root)
        assert not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes')
    for name in expected[:2]:
        assert (root / (name + '.stdout.log')).read_bytes() == b''
        assert (root / (name + '.stderr.log')).read_bytes() == b''
    for name, original in q['loaded_images'].items():
        exe = work / 'cpu/bin' / ('load-probe' if name == 'version-only' else
            'native-stream' if name.endswith('-native') else 'public-stream')
        assert audit_loaded_libraries((root / (name + '.stderr.log')).read_text(), package, package_manifest, exe) == original
    assert (root / 'version-only.stdout.log').read_text().strip() == q['runtime_version']
    for item, original in zip(manifest['slices'], q['results'], strict=True):
        name = item['id'] + '-cpu'
        audio = (work / 'corpus' / item['wav']).read_bytes()
        offset, samples = pcm_header(audio[:65536], len(audio))
        assert samples == item['audio_samples']
        pcm = np.frombuffer(audio, dtype='<i2', offset=offset, count=samples).astype(np.float32) / np.float32(32768)
        pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
        native_path = root / 'results' / (name + '-native.json')
        public = read(root / 'results' / (name + '-public.json'))
        assert public == read(root / (name + '-public.stdout.log'))
        fp32, metadata = references[item['id']]
        replay = score_slice(item, public, read(native_path),
            (root / 'results' / (name + '-public.f32')).read_bytes(),
            Path(str(native_path) + '.f32').read_bytes(), fp32, metadata,
            read(work / 'corpus' / item['reference']), pcm_sha)
        replay.update(all_slice_gates_passed=all(replay['gates'].values()), input_pcm_f32le_sha256=pcm_sha,
            oracle_array=descriptor(work / 'oracle' / (item['id'] + '-fp32.npy')),
            oracle_metadata=descriptor(work / 'oracle' / (item['id'] + '-fp32.json')))
        assert replay == original == read(root / 'results' / (name + '-score.json'))
        scores.append(replay)
        for kind in ['native', 'public']:
            log = (root / (name + '-' + kind + '.stderr.log')).read_text()
            times = re.findall(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$', log, re.M)
            assert len(times) == 1
            wall, user, system = map(float, times[0])
            usage[name + '-' + kind] = {'real_seconds': wall, 'user_seconds': user,
                'system_seconds': system, 'rtf': wall / (samples / 16000)}
    assert q['all_slice_quality_gates_passed'] is all(s['all_slice_gates_passed'] for s in scores)
assert inventory(work) == p['inventory']
assert before == regular_files(root)
for key, base in [('tools', tools), ('repository_files', repo), ('preparation_records', Path('/'))]:
    for name, value in p[key].items():
        assert descriptor(base / name) == value, name
out.mkdir()
for name in before:
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, destination)
    assert descriptor(destination) == before[name]
shutil.copy2(Path(__file__), out / Path(__file__).name)
record = {'schema_version': 1, 'source_commit': source, 'admission_only': args.admission,
    'saved_outputs_independently_replayed': True, 'new_helper_or_model_calls': 0,
    'historical_git_sources_verified': len(pins), 'staged_input_inventory_entries_verified': len(p['inventory']),
    'original_fp32_oracles_revalidated': len(references), 'raw_controller_files': before,
    'results': scores, 'time_l': usage, 'release_qualified': False,
    'limits': ['Original eight integration slices; no independent unseen-data or human timing qualification.',
               'No Metal, other-machine corpus, physical GUI/microphone, fresh-machine or public release claim.']}
(out / 'collection-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = regular_files(out)
assert {n: files[n] for n in before} == before
assert before == regular_files(root)
for key, base in [('tools', tools), ('repository_files', repo), ('preparation_records', Path('/'))]:
    for name, value in p[key].items():
        assert descriptor(base / name) == value, name
(out / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
assert regular_files(out) == {**files, 'collection.json': descriptor(out / 'collection.json')}
print(out, len(files) + 1, 'members;', len(scores), 'scores replayed; no helper/model calls')
