"""Collect saved model-free Pnola staging regression; never run app/native code."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory

before = cache / 'pooled-pnola-profile-staging-red-v1/.desktop-assets'
after = cache / 'pooled-pnola-profile-staging-green-v1/.desktop-assets'
legacy = cache / 'pooled-pnola-profile-staging-legacy-v1/.desktop-assets'
old = json.loads((before / 'nemotron/bundle.json').read_text())
new = json.loads((after / 'nemotron/bundle.json').read_text())
native = json.loads((after / 'nemo/package.json').read_text())
assert old['runtime_profile'] == 'gen2-valid-mel-v1'
assert native['runtime_profile'] == new['runtime_profile'] == 'gen2-valid-mel-pool-v1'
old['runtime_profile'] = new['runtime_profile']
assert old == new
a, b = inventory(before), inventory(after)
assert a.keys() == b.keys()
changed = [n for n in a if a[n] != b[n]]
assert changed == ['nemotron/bundle.json']
old_package = json.loads((legacy / 'nemo/package.json').read_text())
old_bundle = json.loads((legacy / 'nemotron/bundle.json').read_text())
assert old_package['runtime_profile'] == old_bundle['runtime_profile'] == 'gen2-valid-mel-v1'
source = repo / 'integrations/pnola'
manifest = json.loads((source / 'manifest.json').read_text())
backup = Path('/Users/victor/workspace/pnola/.nemotron-backups/20261017T013128126488Z')
fixed = ['scripts/speaker-assets.mjs', 'tests/desktop-assets.test.mjs']
for name in fixed:
    assert descriptor(source / 'overlay' / name) == descriptor(Path('/Users/victor/workspace/pnola') / name)
    assert descriptor(source / 'overlay' / name)['sha256'] == manifest['files'][name]['overlay_sha256']
    original = subprocess.check_output(['git', 'show', '4083013d56c59f67ad766ade1c331c3dd274eb67:integrations/pnola/overlay/' + name], cwd=repo)
    assert (backup / name).read_bytes() == original
    assert descriptor(backup / name)['sha256'] in manifest['files'][name]['compatible_base_sha256']
assert 'tests 5' in (cache / 'pooled-pnola-profile-node-tests.log').read_text()
assert 'pass 5' in (cache / 'pooled-pnola-profile-node-tests.log').read_text()
assert 'Ran 2 tests' in (cache / 'pooled-pnola-profile-overlay-tests.log').read_text()
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-profile-staging'
assert not out.exists()
out.mkdir()
for suffix in ['red-v1', 'green-v1', 'legacy-v1']:
    name = 'pooled-pnola-profile-staging-' + suffix + '.log'
    shutil.copy2(cache / name, out / name)
for name in ['pooled-pnola-profile-node-tests.log', 'pooled-pnola-profile-overlay-tests.log']:
    shutil.copy2(cache / name, out / name)
for label, directory in [('before', before), ('after', after), ('legacy', legacy)]:
    shutil.copy2(directory / 'nemotron/bundle.json', out / (label + '-bundle.json'))
shutil.copy2(__file__, out / Path(__file__).name)
q = {'schema_version': 1, 'source_base_commit': '4083013d56c59f67ad766ade1c331c3dd274eb67',
    'model_free_staging_regression_passed': True, 'initial_real_staging_failed_profile_equality': True,
    'changed_staged_files': changed, 'both_supported_profiles_staged_with_matching_identity': True,
    'all_other_staged_payloads_byte_and_mode_identical': True,
    'staged_inventory_entries_compared': len(a),
    'fixed_source_files': {name: descriptor(source / 'overlay' / name) for name in fixed},
    'overlay_manifest': descriptor(source / 'manifest.json'), 'overlay_upgrade_files': fixed,
    'overlay_upgrade_backups_verified_at_base_commit': True,
    'node_tests_passed': 5, 'python_overlay_tests_passed': 2,
    'models_loaded_for_inference': False, 'app_or_native_version_queries_executed': False,
    'release_qualified': False,
    'limits': ['Initial real staging incorrectly labels the pooled runtime as the old profile. Exact profile/lock/version selection fixes only model-bundle provenance; no neural behavior is relabelled.',
        'Both actual supported packages stage successfully; all other before/after candidate payload hashes/modes are unchanged. Fixture tests reject unknown, suffix, missing-version and mismatched identities.',
        'Only the two known overlay files were upgraded with verified backups; unrelated Pnola files are protected by the existing complete-plan hash guard.',
        'Static packaging/model checksum checks only. No complete app rebuild/execution, GUI, physical microphone, fresh-machine app, quality, public signing/notary, or release gate is established.']}
(out / 'qualification.json').write_text(json.dumps(q, indent=2) + '\n')
files = {p.relative_to(out).as_posix(): descriptor(p) for p in out.rglob('*') if p.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, 'members', len(files) + 1)
