"""Independently replay the fixed prefix contrast; no native/model execution."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
read = lambda path: json.loads(path.read_text())
root = cache / 'ami-prefix-diagnostic-v2'
p = read(root / 'protocol.json')
q = read(root / 'qualification.json')
source = 'e2b026791d55274d5c54147309c3b46e0b5749c6'
# Admit every replay dependency before importing qualification modules.
for name, expected in p['tools'].items():
    path = tools / name
    assert {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} == expected
from ami_prefix_metrics import contrast, load_readonly_scorer
from build_valid_mel_candidate import descriptor
from compare_stream_valid_mel import compare
from diagnose_ami_reference import calibrate_engine
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from sdk_archive import inventory
assert q['source_commit'] == source and q['diagnostic_completed'] is True and 'error' not in q
assert q['release_qualified'] is False and q['original_gate_status_changed'] is False
assert q['original_quality_gate_remains_failed'] is True and q['fresh_fp32_calls'] == 0
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/ami-prefix-diagnostic-v2.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
work = Path(p['work_path'])
assert inventory(work) == p['work_inventory']
for name, expected in p['preparation_records'].items():
    assert descriptor(Path(name)) == expected
build = read(cache / 'valid-mel-macos-runtime-inputs-v1/build.json')
verify_git_sources(repo, build['source_commit'], build['rust_source_files'])
assert p['source_interval_seconds'] == [0, 420] and p['scored_interval_seconds'] == [300, 420]
ids = ['ami-ES2004a-Mix-Headset', 'ami-ES2004a-Array1-01']
cases = [name + '-' + suffix for name in ids for suffix in ['native', 'public']]
assert q['neural_commands_started'] == cases
assert [r['name'] for r in q['commands']] == ['runtime-controls', 'premodel-controls', 'version-only', *cases]
assert all(r['returncode'] == 0 for r in q['commands'])
manifest = verify(work / 'nemo')
assert manifest['runtime_profile'] == 'gen2-valid-mel-v1' and manifest['backend'] == 'cpu'
images = {'version-only': audit_loaded_libraries((root / 'version-only.stderr.log').read_text(), work / 'nemo', manifest, work / 'bin/load-probe')}
engine = load_readonly_scorer(work / 'scoring/md_eval.py', p['independent_scorer'])
calibrations = calibrate_engine(engine)
assert len(calibrations) == 4 and calibrations == read(root / 'scorer-calibrations.json')
references = {n: read(work / 'references' / (n + '.json')) for n in ['transcriber', 'published_lexical']}
replayed = {}
for name in ids:
    native = read(root / 'results' / (name + '-native.json'))
    public = read(root / 'results' / (name + '-public.json'))
    native_raw = (root / 'results' / (name + '-native.json.f32')).read_bytes()
    public_raw = (root / 'results' / (name + '-public.f32')).read_bytes()
    comparison = compare(public, native, native_raw, public_raw)
    assert comparison == q['comparisons'][name]
    assert comparison['native_frame_count'] == comparison['retained_rows'] == 42000
    assert comparison['retained_first_frame'] == 0
    for value in [public, native]:
        assert value['input_pcm_f32le_sha256'] == p['pcm_f32le_sha256'][name] and value['input_repeats'] == 1
    for suffix, executable in [('native', 'native-stream'), ('public', 'public-stream')]:
        case = name + '-' + suffix
        images[case] = audit_loaded_libraries((root / (case + '.stderr.log')).read_text(), work / 'nemo', manifest, work / 'bin' / executable)
    cold = read(work / 'baseline' / (name + '-cpu-public.json'))
    cold_native = read(work / 'baseline' / (name + '-cpu-native.json'))
    cold_raw = (work / 'baseline' / (name + '-cpu-public.f32')).read_bytes()
    cold_native_raw = (work / 'baseline' / (name + '-cpu-native.json.f32')).read_bytes()
    assert cold_raw == cold_native_raw
    assert compare(cold, cold_native, cold_native_raw, cold_raw)['native_frame_count'] == 12000
    replayed[name] = contrast(engine, cold, public, cold_raw, public_raw, references)
    assert replayed[name] == q['results'][name]
    scores = replayed[name]['reference_policies']['transcriber']['scores']
    previous = read(work / 'baseline' / (name + '-cpu-score.json'))
    assert scores['cold_shared']['frame'] == previous['q8_shared_threshold']
    assert scores['cold_production']['frame'] == previous['q8_production_default']
assert images == q['loaded_images'] and len(images) == 5
assert inventory(work) == p['work_inventory']
assert q['storage']['released'] is True and q['storage']['minimum_observed_free_bytes'] >= 1024**3
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-ami-prefix-diagnostic'
assert not out.exists()
out.mkdir()
for path in root.rglob('*'):
    if path.is_file() and path.suffix in {'.json', '.log', '.sb', '.f32'} and not path.is_relative_to(root / 'home') and not path.is_relative_to(root / 'tmp'):
        destination = out / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
shutil.copy2(__file__, out / Path(__file__).name)
shutil.copy2(cache / 'ami-prefix-diagnostic-v2.controller.log', out / 'controller.log')
for name in ['ami-prefix-readonly-scorer-preflight-v2.json', 'ami-prefix-cold-files-admission-v1.json', 'ami-prefix-host-v1.json', 'ami-prefix-baseline-replay-v1.json']:
    shutil.copy2(cache / name, out / name)
summary = {
    'schema_version': 1, 'source_commit': source, 'diagnostic_completed': True,
    'release_qualified': False, 'original_quality_gate_remains_failed': True,
    'original_gate_status_changed': False, 'source_pins_verified': len(pins),
    'historical_rust_source_pins_verified': len(build['rust_source_files']),
    'all_151_protected_input_entries_unchanged': True,
    'actual_commands': 7, 'fresh_neural_processes': 4,
    'full_native_public_rows_per_channel': 42000, 'scored_target_rows_per_channel': 12000,
    'official_controls_replayed': 4, 'strict_private_library_image_audits_replayed': 5,
    'all_results_replayed_exactly': True, 'new_native_or_neural_calls_during_collection': 0,
    'contrasts': {name: value['reference_policies'] for name, value in replayed.items()},
    'limits': p['limits'],
}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members')
for name, value in replayed.items():
    print(name, {policy: {mode: results['scores'][mode]['official']['der'] for mode in ['cold_shared', 'prefix_shared', 'cold_production', 'prefix_production']} for policy, results in value['reference_policies'].items()})
