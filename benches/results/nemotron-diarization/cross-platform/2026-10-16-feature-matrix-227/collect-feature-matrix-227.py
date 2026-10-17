"""Collect completed feature CI; do not execute models or replace old evidence."""
import hashlib
import json
from pathlib import Path
import shutil

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci')
ci = json.loads((cache / 'feature-38989434879.current.json').read_text())
source = 'aeb6ba813d61653ebdac7273f2a4b404c606f142'
def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
assert ci['databaseId'] == 38989434879 and ci['headSha'] == source
assert ci['status'] == 'completed' and ci['conclusion'] == 'success' and len(ci['jobs']) == 13
assert all(j['status'] == 'completed' and j['conclusion'] == 'success' for j in ci['jobs'])
platforms = ['ubuntu-latest', 'windows-latest', 'macos-latest']
features = ['', 'backend-diar-nemo', 'asr-dynamic,backend-diar-nemo', 'backend-llamacpp,asr-dynamic,backend-diar-nemo']
assert {j['name'] for j in ci['jobs']} == {'evidence-tools'} | {
    f'features ({os}, {flags})' if flags else f'features ({os})' for os in platforms for flags in features}
for job in ci['jobs']:
    names = ['Run python -m unittest discover -s tools/nemotron-diarization',
             'Run python -m unittest discover -s integrations/pnola',
             'Run node --test integrations/pnola/overlay/tests/desktop-assets.test.mjs'] if job['name'] == 'evidence-tools' else [
             'Run cargo fmt', 'Run cargo clippy', 'Run cargo test']
    for name in names:
        steps = [s for s in job['steps'] if s['name'].startswith(name)]
        assert len(steps) == 1 and steps[0]['conclusion'] == 'success'
log = cache / 'feature-38989434879.log'
lines = [line for line in log.read_text().splitlines() if line.startswith(
    'evidence-tools\tRun python -m unittest discover -s tools/nemotron-diarization')]
assert any('Ran 227 tests' in line for line in lines) and any('OK (skipped=7)' in line for line in lines)
out = repo / 'benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-227'
assert not out.exists()
out.mkdir()
shutil.copy2(cache / 'feature-38989434879.current.json', out / 'ci-success.json')
shutil.copy2(__file__, out / Path(__file__).name)
(out / 'evidence-test-summary.log').write_text('\n'.join(
    line for line in lines if 'Ran 227 tests' in line or 'OK (skipped=7)' in line) + '\n')
q = {'schema_version': 1, 'source_commit': source, 'ci_url': ci['url'],
    'all_thirteen_feature_jobs_passed': True, 'model_free_evidence_tests_reported': 227,
    'model_free_evidence_tests_passed': 220, 'model_free_evidence_tests_skipped': 7,
    'platforms': platforms, 'feature_sets': features,
    'all_twelve_feature_jobs_ran_fmt_clippy_tests_successfully': True,
    'evidence_and_pnola_python_and_desktop_asset_test_steps_passed': True,
    'combined_log_retained_outside_git': descriptor(log), 'release_qualified': False,
    'limits': ['Source aeb6ba8, before the later mixed qualification tools. All12OS/feature combinations and evidence-tools pass. Linux227tests comprise220passes and7platformskips; the separate same-source Mac227suite has no skips.',
        'Feature compatibility only; this workflow executes no diarization model or desktop app. It doesnot qualify accuracy, packaging, real-time/capture latency, notarization, or other real-model gates. Historical receipts remain unchanged.']}
(out / 'qualification.json').write_text(json.dumps(q, indent=2) + '\n')
files = {p.relative_to(out).as_posix(): descriptor(p) for p in out.rglob('*') if p.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, 'members', len(files) + 1)
