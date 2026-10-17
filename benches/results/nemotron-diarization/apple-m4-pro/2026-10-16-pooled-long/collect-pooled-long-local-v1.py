"""Replay saved four-stream long qualification; never execute model helpers."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'pooled-long-local-v1'
source = 'aeb6ba813d61653ebdac7273f2a4b404c606f142'
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from pooled_long_metrics import audit_case
from sdk_archive import inventory, restore_archive

p = json.loads((root / 'protocol.json').read_text())
q = json.loads((root / 'qualification.json').read_text())
assert q['source_commit'] == source and q['pooled_30m_1h_passed'] is True
assert q['release_qualified'] is False and 'error' not in q
assert descriptor(root / 'protocol.json') == q['protocol'] == {
    'bytes': 94430, 'sha256': '11fcc9b32198ff8572a7cb224bf8feb143e7079e179db724c4a4ef30ff64c1af'}
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-pooled-long.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
old = json.loads((tools / 'valid-mel-cpu-long.protocol.json').read_text())
for key in ['acceptance', 'cases', 'command_timeout_seconds']:
    assert p[key] == old[key]
assert json.loads((root / 'host.json').read_text())['cpu'] == p['expected_cpu'] == 'Apple M4 Pro'
expected = ['model-free-controls', 'runtime-controls', 'version-only',
            '30m-native', '30m-public', '60m-native', '60m-public']
assert [c['name'] for c in q['commands']] == expected
assert all(c['returncode'] == 0 for c in q['commands'])
assert q['neural_commands_started'] == expected[3:]
assert q['producer_artifact']['id'] == p['producer']['id'] == 12082417995
assert q['native_artifact']['id'] == p['native']['id'] == 12083481818
public_root = cache / 'pooled-public-local-v1'
for key, filename in [('public_helpers', 'pooled-public-helpers.tar.gz'),
                      ('public_helpers_manifest', 'pooled-public-helpers-files.json'),
                      ('public_qualification', 'qualification.json')]:
    assert descriptor(public_root / filename) == p[key]
public_q = json.loads((public_root / 'qualification.json').read_text())
assert public_q['pooled_public_runtime_passed'] is True
assert public_q['source_commit'] == p['compiled_public_source_commit']
assert public_q['lifecycle_tests_passed'] == p['lifecycle_tests']
protected = json.loads((root / 'protected-inputs.json').read_text())
for key, directory in [('package', 'work/nemo'), ('binaries', 'work/bin'), ('assets', 'work/assets'),
                       ('producer', 'producer'), ('native_delivery', 'native-delivery'),
                       ('native_helpers', 'native-helpers'), ('public_helpers', 'public-helpers')]:
    assert inventory(root / directory) == protected[key]
helpers = json.loads((root / 'pooled-long-helpers-files.json').read_text())
assert inventory(root / 'work/bin') == helpers['files']
assert {k: v for k, v in helpers.items() if k != 'files'} == q['helper_archive']
restore_archive(root / helpers['archive'], helpers, root / 'independent-helper-restore')
assert inventory(root / 'independent-helper-restore') == helpers['files']
package = root / 'work/nemo'
manifest = verify(package)
assert json.loads((root / 'package.json').read_text()) == manifest
assert set(q['loaded_images']) == set(expected[2:])
for name, original in q['loaded_images'].items():
    executable = root / 'work/bin' / ('load-probe' if name == 'version-only' else
        'native-stream' if name.endswith('-native') else 'public-stream')
    assert audit_loaded_libraries((root / (name + '.stderr.log')).read_text(),
                                 package, manifest, executable) == original
    assert original['private_library_count'] == 92
comparisons = {}
for case in p['cases']:
    name = case['name']
    native_path = root / 'results' / (name + '-native.json')
    public_path = root / 'results' / (name + '-public.json')
    native = json.loads(native_path.read_text())
    public = json.loads(public_path.read_text())
    replay = audit_case(public, native, Path(str(native_path) + '.f32').read_bytes(),
        (root / 'results' / (name + '-public.f32')).read_bytes(),
        (root / (name + '-native.stderr.log')).read_text(),
        (root / (name + '-public.stderr.log')).read_text(), case)
    assert replay == q['comparisons'][name]
    assert replay['native_frame_count'] == case['final_rows']
    assert replay['retained_first_frame'] > 0 and 60000 <= replay['retained_rows'] <= 130000
    comparisons[name] = replay
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= 1024 ** 3
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-long'
assert not out.exists()
out.mkdir()
def retain(path, relative):
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    assert descriptor(path) == descriptor(destination)
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        retain(path, path.name)
for path in (root / 'results').rglob('*'):
    if path.is_file():
        retain(path, path.relative_to(root))
retain(cache / 'platform-ci/model-free-227-pooled-long-local.log', 'model-free-227.log')
retain(cache / 'pooled-long-metrics-preflight.log', 'metrics-preflight.log')
retain(Path(__file__), Path(__file__).name)
record = {'schema_version': 1, 'source_commit': source, 'pooled_30m_1h_passed': True,
    'historical_source_pins_verified': len(pins), 'four_fresh_neural_processes': expected[3:],
    'five_strict_private_library_audits': 92, 'protected_input_inventories_unchanged': True,
    'qualified_helper_payloads_independently_restored_and_rehashed': True,
    'comparisons': comparisons, 'release_qualified': False, 'limits': p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, 'members', len(files) + 1, 'comparisons', comparisons)
