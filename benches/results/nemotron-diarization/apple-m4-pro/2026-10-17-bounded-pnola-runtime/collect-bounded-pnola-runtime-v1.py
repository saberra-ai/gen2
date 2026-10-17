"""Independently retain saved bounded app admission/runtime; never invoke neural code."""
from pathlib import Path
import argparse
import array
import hashlib
import json
import math
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--admit-only', action='store_true')
args = parser.parse_args()
suffix = '-admission' if args.admit_only else ''
root = cache / ('bounded-pnola-complete-app-runtime-v1' + suffix)
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
source = q['source_commit']
assert q['admit_only'] == args.admit_only and q['input_admission_passed'] is True
assert q['automatic_packaged_resolution_gates_passed'] == (not args.admit_only)
assert q['release_qualified'] is False and 'error' not in q
assert q['microphone_used'] is False and q['artifact_environment_overrides_used'] is False
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-pnola-app.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
for name, expected in pins.items():
    assert descriptor(repo / name) == expected
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory
from package_macos import verify
from bounded_pnola_metrics import VERSION, native_reference, recorded_matches, live_matches
from macos_loaded_libraries import audit_loaded_libraries
from verify_valid_mel_pnola_app import audit_app_images
from verify_bounded_pnola_app import verify_app
verify_git_sources(repo, source, pins)
work, app = Path(p['work_path']), Path(p['app_path'])
assert inventory(work) == p['work_inventory'] and inventory(app) == p['app_inventory']
for name, expected in p['preparation_records'].items():
    assert descriptor(Path(name)) == expected
construction_path = Path(p['construction']['path'])
construction = read(construction_path)
assert descriptor(construction_path) == p['construction']['file']
assert construction['source_commit'] == p['construction']['source_commit']
assert construction['app_construction_passed'] is True and construction['native_version_queries_or_neural_calls'] == 0
assert construction['models_loaded_for_inference'] is False and 'error' not in construction
assert construction['storage']['released'] and all(c['returncode'] == 0 for c in construction['commands'])
assert inventory(app) == construction['apps']['cpu']['files']
verified = verify_app(app)
package = work / 'nemo'
manifest = verify(package)
assert manifest['runtime_profile'] == 'gen2-valid-mel-pool-cpu-v2' and manifest['backend'] == 'cpu'
assert manifest == verify(app / 'Contents/Resources/nemo')
assert p['command_timeout_seconds'] == 600
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert not (root / 'diagnostic-reserve.bin').exists()
comparisons, images, natives = {}, {}, {}
CASES = ['native-60s', 'native-80s', 'cpu-recorded', 'cpu-live', 'cpu-rejects-metal']
assert p['cases'] == CASES
if args.admit_only:
    assert q['commands'] == q['cases'] == q['neural_commands_started'] == [] and q['loaded_images'] == {}
    assert not any((root / name).exists() for name in ['host.json', 'home', 'tmp', 'results', 'runtime.sb', 'premodel.sb'])
    assert {v.name for v in root.iterdir()} == {'protocol.json', 'qualification.json'}
else:
    assert read(root / 'host.json')['cpu'] == p['expected_cpu'] == 'Apple M4 Pro'
    assert q['neural_commands_started'] == CASES and [v['name'] for v in q['cases']] == CASES
    assert [v['name'] for v in q['commands']] == ['runtime-controls', 'premodel-controls', 'version-only', *CASES]
    assert all(v['returncode'] == 0 for v in q['commands'])
    assert all(v['inference_admitted'] == (v['name'] in CASES) for v in q['commands'])
    assert (root / 'version-only.stdout.log').read_text().strip() == VERSION
    images['version-only'] = audit_loaded_libraries((root / 'version-only.stderr.log').read_text(), package, manifest, work / 'bin/load-probe')
    for name, wav, duration in [('native-60s', 'ami.wav', 60.), ('native-80s', 'live-80s.wav', 80.04)]:
        result = root / 'results' / (name + '.json')
        native = read(result)
        native_reference(native, duration)
        assert native['input_pcm_f32le_sha256'] == p['pcm_sha256'][wav]
        raw = array.array('f')
        raw.frombytes(Path(str(result) + '.f32').read_bytes())
        assert len(raw) == round(duration * 100) * 8 and all(math.isfinite(v) and 0 <= v <= 1 for v in raw)
        row = next(v for v in q['cases'] if v['name'] == name)
        assert row['passed'] and row['result'] == descriptor(result)
        assert row['probabilities'] == descriptor(Path(str(result) + '.f32'))
        images[name] = audit_loaded_libraries((root / (name + '.stderr.log')).read_text(), package, manifest, work / 'bin/native-stream')
        natives[name] = native
    for name in CASES[2:]:
        result = root / 'results' / (name + '.json')
        actual = read(result)
        comparisons[name] = (live_matches(actual, natives['native-80s']) if name == 'cpu-live' else
            recorded_matches(actual, natives['native-60s'], p['bark_text'], p['wav_sha256'], unavailable=name == 'cpu-rejects-metal'))
        row = next(v for v in q['cases'] if v['name'] == name)
        assert row['passed'] and row['result'] == descriptor(result) and row['comparison'] == comparisons[name]
        images[name] = audit_app_images((root / (name + '.stderr.log')).read_text(), app, verified)
        assert images[name]['process_local_add_image_history_observed']
    assert images == q['loaded_images']
for log, count in [('bounded-pnola-metrics-local.log', 3), ('platform-ci/model-free-280-bounded-pnola-metrics-local.log', 280)]:
    text = (cache / log).read_text()
    assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro' / ('2026-10-17-bounded-pnola-runtime' + suffix)
assert not out.exists()
out.mkdir()
for path in root.rglob('*'):
    if path.is_file() and path.suffix in {'.json', '.log', '.sb', '.f32', '.txt'} and not path.is_relative_to(root / 'home') and not path.is_relative_to(root / 'tmp'):
        dest = out / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
for path in [Path(__file__), cache / 'prepare-bounded-pnola-app-runtime-v1.py',
             root.with_suffix('.controller.log'), cache / 'bounded-pnola-metrics-local.log',
             cache / 'platform-ci/model-free-280-bounded-pnola-metrics-local.log']:
    shutil.copy2(path, out / path.name)
summary = {'schema_version': 1, 'controller_source_commit': source,
    'app_construction_source_commit': construction['source_commit'],
    'input_admission_passed': True, 'admit_only': args.admit_only,
    'automatic_packaged_resolution_gates_passed': not args.admit_only, 'release_qualified': False,
    'source_pins_verified': len(pins), 'work_entries_verified': len(p['work_inventory']),
    'app_entries_verified': len(p['app_inventory']), 'preparation_records_verified': len(p['preparation_records']),
    'actual_commands': len(q['commands']), 'fresh_neural_processes': len(q['neural_commands_started']),
    'independent_native_reference_frames': {n: v['native_frame_count'] for n, v in natives.items()},
    'comparisons_replayed': comparisons, 'native_private_image_audits': len([n for n in images if n in ['version-only', 'native-60s', 'native-80s']]),
    'hardened_app_callback_image_audits': len([n for n in images if n in CASES[2:]]),
    'all_protected_app_and_work_entries_rehashed': True, 'storage_reserve_released': True,
    'model_free_tests_passed': 280, 'new_native_or_app_or_neural_calls_during_collection': 0,
    'limits': ['Actual static-only admission.' if args.admit_only else 'Actual same-host hardened ad-hoc CPU app automatic packaged recording/live/export and explicit Metal rejection. No artifact-path overrides.',
        'Static collection may inspect signatures and temporary unsigned copies; no app/SDK/helper/model code executes.',
        'Native references retain all frames/probabilities. Live app has compact turns; independent final-turn comparison remains1e-6, not observed app probability arrays.',
        'No GUI, physical microphone, fresh machine, other-platform app, acoustic timing, held-out accuracy, human word gold, public signing/notarization, default promotion or release claim.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, len(files) + 1, 'members', comparisons)
