from pathlib import Path
import hashlib
import json
import shutil
import subprocess

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'bounded-public-admission-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


q, p = read(root / 'qualification.json'), read(root / 'protocol.json')
source = q['source_commit']
assert source == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert q['admit_only'] is True and q['deliveries_admitted'] is True
assert q['bounded_public_runtime_passed'] is False and q['public_realtime_gate_passed'] is False
assert q['release_qualified'] is False and q['model_execution_admitted'] is False
assert q['commands'] == q['inference_commands_started'] == [] and q['loaded_images'] == {} and 'error' not in q
assert not any((root / name).exists() for name in ['sdk', 'work', 'helper-source', 'cargo-target', 'results'])
assert all(key not in q for key in ['acquisitions', 'acquisition_attempts', 'helper_archive', 'comparison'])
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-public.protocol.json'], cwd=repo)
deliveries = [('producer', 'producer_artifact', 'bounded-macos-construction-ci-v1'),
              ('native_reference', 'native_reference_artifact', 'bounded-native-ci-v2')]
for key, admitted_key, original in deliveries:
    directory = root / ('native-reference' if key == 'native_reference' else key)
    files = {str(path.relative_to(directory)): descriptor(path) for path in directory.rglob('*') if path.is_file()}
    assert files == p[key]['files']
    api = q[admitted_key]
    assert api['id'] == p[key]['id'] and api['name'] == p[key]['name']
    assert api['workflow_run']['id'] == p[key]['run_id']
    assert api['workflow_run']['head_sha'] == p[key]['source_commit']
    assert api['digest'] == 'sha256:' + p[key]['zip']['sha256']
    assert descriptor(cache / original / 'artifact.zip') == p[key]['zip']
    assert read(cache / original / 'artifacts.json')['artifacts'][0] == api
assert q['storage']['released']
assert q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] == p['storage']['reserve_bytes']
assert 'FAILED' in (cache / 'bounded-public-loader-red.log').read_text()
assert '1 passed; 0 failed' in (cache / 'bounded-public-loader-green.log').read_text()
assert 'Ran 268 tests' in (cache / 'platform-ci/model-free-268-bounded-public-local.log').read_text()
assert (cache / 'platform-ci/model-free-268-bounded-public-local.log').read_text().rstrip().endswith('OK')
assert 'Finished `dev` profile' in (cache / 'bounded-public-clippy.log').read_text()
rust = (cache / 'bounded-public-rust-tests.log').read_text()
for count in ['1030 passed', '5 passed', '4 passed', '1 passed']:
    assert count in rust
assert 'test result: FAILED' not in rust
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-public-admission'
assert not out.exists()
out.mkdir()
for path in [root / 'protocol.json', root / 'qualification.json',
             cache / 'bounded-public-admission-v1.controller.log', Path(__file__),
             cache / 'bounded-public-loader-red.log', cache / 'bounded-public-loader-green.log',
             cache / 'bounded-public-rust-tests.log', cache / 'bounded-public-clippy.log',
             cache / 'platform-ci/model-free-268-bounded-public-local.log',
             *sorted((cache / 'bounded-public-inputs-v1').iterdir())]:
    shutil.copy2(path, out / path.name)
summary = {'schema_version': 1, 'source_commit': source,
           'both_full_deliveries_admitted': True,
           'members_verified': {'producer': 200, 'native_reference': 44},
           'sdk_restored': False, 'helpers_compiled_or_executed_by_preflight': 0,
           'model_assets_acquired': 0, 'model_calls': 0, 'commands_started': 0,
           'complete_declared_identity_contract_verified': True,
           'storage_reserve_released': True,
           'local_model_free_rust_passed': 1041, 'local_model_free_rust_ignored': 13,
           'local_model_free_python_passed': 268, 'clippy_warnings_denied_passed': True,
           'release_qualified': False,
           'limits': ['Actual local archive admission and model-free tests only. Synthetic version-only C libraries in the Rust loader regression are not SDK/model execution. Public bounded-SDK compilation, numerical/realtime/lifecycle execution and original roadmap release gates remain pending.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members')
