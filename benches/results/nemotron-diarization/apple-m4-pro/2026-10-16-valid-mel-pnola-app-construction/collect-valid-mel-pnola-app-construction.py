"""Collect complete CPU/Metal app construction and static metadata, with no app execution."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory

read = lambda path: json.loads(path.read_text())
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-app-construction'
assert not out.exists()
prior = cache / 'valid-mel-pnola-app-build-v1'
metal = cache / 'valid-mel-pnola-metal-app-build-v2'
cpu = cache / 'valid-mel-pnola-app-cpu-snapshot-v1'
first = read(prior / 'build.json'); last = read(metal / 'build.json')
assert not first['app_construction_passed'] and first['commands'][0]['returncode'] == 0
assert len(first['commands']) == 1 and first['commands'][0]['name'] == 'cpu-build'
assert last['app_construction_passed'] and len(last['commands']) == 2 and all(c['returncode'] == 0 for c in last['commands'])
assert first['models_loaded_for_inference'] is False and last['models_loaded_for_inference'] is False
assert first['native_version_queries_or_neural_calls'] == last['native_version_queries_or_neural_calls'] == 0
assert first['storage']['released'] and last['storage']['released']
p = read(prior / 'protocol.json'); continuation = read(metal / 'protocol.json')
assert descriptor(prior / 'protocol.json') == continuation['prior_protocol']
assert descriptor(prior / 'build.json') == continuation['prior_failed_controller']
assert p['gen2_files'] == continuation['gen2_files'] and p['pnola_source_files'] == continuation['pnola_source_files']
assert inventory(cpu / 'Pnola.app') == read(cpu / 'snapshot.json')['files'] == continuation['prior_cpu_inventory']
assert inventory(metal / 'Pnola.app') == last['app']['files']
static = {'cpu': read(cpu / 'static-verification-v5.json'), 'metal': read(metal / 'static-verification.json')}
for backend, data in static.items():
    root = Path(data['app'])
    assert data['app_structure_verified'] and not data['release_qualified'] and data['backend'] == backend
    assert data['runtime_profile'] == 'gen2-valid-mel-v1' and data['logical_runtime_library_count'] == (92 if backend == 'cpu' else 93)
    assert len(data['frameworks']) == data['logical_runtime_library_count'] + 2
    assert descriptor(root / 'Contents/MacOS/pnola-desktop') == data['executable']
    for name, expected in data['frameworks'].items():
        assert descriptor(root / 'Contents/Frameworks' / name) == expected, name
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
for name, expected in p['gen2_files'].items():
    assert descriptor(repo / name) == expected, name
out.mkdir()
for directory in ['valid-mel-pnola-app-build-v1', 'valid-mel-pnola-app-cpu-snapshot-v1',
                  'valid-mel-pnola-metal-app-build-v1', 'valid-mel-pnola-metal-app-build-v2']:
    origin = cache / directory; dest = out / directory; dest.mkdir()
    for path in sorted(origin.iterdir()):
        if path.is_file() and path.suffix in {'.json', '.log', '.sb'} and (path.suffix != '.json' or path.stat().st_size):
            shutil.copy2(path, dest / path.name)
for name in ['build-valid-mel-pnola-apps.py', 'build-valid-mel-pnola-metal-app.py', 'build-valid-mel-pnola-metal-app-v2.py',
             'cleanup-stale-rust-objects-app-build-v1.py', 'cleanup-stale-rust-objects-app-build-v2.py',
             'cleanup-stale-rust-objects-pnola-app-build-v1.py', 'cleanup-stale-cmake-objects-app-build-v1.py']:
    shutil.copy2(cache / name, out / name)
for stem in ['compiler-object-cleanup-app-build-v1', 'compiler-object-cleanup-app-build-v2',
             'compiler-object-cleanup-pnola-app-build-v1', 'compiler-object-cleanup-cmake-app-build-v1']:
    for suffix in ['.json', '.events.jsonl']:
        shutil.copy2(cache / (stem + suffix), out / (stem + suffix))
for name, expected in p['pnola_source_files'].items():
    origin = Path('/Users/victor/workspace/pnola') / name
    assert descriptor(origin) == expected, name
    dest = out / 'pnola-source' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(origin, dest)
for name in ['verify_valid_mel_pnola_app.py', 'package_macos.py', 'sdk_archive.py', 'runtime_profiles.py',
             'build_valid_mel_candidate.py', 'qualification_storage.py', 'qualify_valid_mel_pnola.py']:
    dest = out / 'collection-tools' / name; dest.parent.mkdir(exist_ok=True)
    shutil.copy2(repo / 'tools/nemotron-diarization' / name, dest)
shutil.copy2(Path(__file__), out / Path(__file__).name)
r = {'schema_version': 1, 'complete_cpu_and_metal_app_construction_passed': True, 'release_qualified': False,
     'scope': 'Same-host ad-hoc complete app construction and static package/payload/model/license/Bark verification only.',
     'collection_source_commit': source, 'cpu_build_source_commit': first['source_commit'], 'metal_build_source_commit': last['source_commit'],
     'gen2_source_pins': len(p['gen2_files']), 'pnola_source_files_retained': len(p['pnola_source_files']),
     'app_framework_entries': {backend: len(data['frameworks']) for backend, data in static.items()},
     'app_logical_native_components': {backend: data['logical_runtime_library_count'] for backend, data in static.items()},
     'native_version_queries_or_neural_calls': 0, 'cpu_successful_build_repeated': False,
     'original_cpu_then_metal_controller_passed': False, 'first_metal_only_admission_passed': False,
     'limits': ['Original controller failed at the post-build storage admission before CPU cloning or Metal build. CPU build itself returned0 and was separately cloned and verified.',
                'First Metal-only attempt failed storage admission after reserve allocation and before any child command. Both failures remain unchanged.',
                'Metal-only continuation uses a separately declared non-neural construction storage policy:3GiB start,1GiB running,256MiB reserve. It does not alter5GiB admission for model execution.',
                'Framework code signatures are stripped only from temporary copies for canonical payload comparison. The Tauri public entry copy and ORT map to exact source package payloads.',
                'App/native/model binaries are retained locally and hash-described here, not included in Git. No application launch, automatic resolver inference, GUI, microphone, Developer ID, notarization, fresh-machine, quality, performance or release qualification.']}
r['files'] = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
(out / 'qualification.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({'target': str(out), 'members': len(r['files']) + 1, 'native_or_model_calls': 0}))
