"""Collect hardened instrumented app construction and exact payload-preserving storage recovery."""
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory

base = cache / 'valid-mel-pnola-instrumented-app-build-v1'
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-instrumented-pnola-apps'
assert not out.exists(); read = lambda path: json.loads(path.read_text())
r = read(base / 'build.json'); p = read(base / 'protocol.json')
assert r['app_construction_passed'] and r['native_version_queries_or_neural_calls'] == 0
assert r['models_loaded_for_inference'] is False and r['storage']['released']
assert len(r['commands']) == 6 and all(c['returncode'] == 0 for c in r['commands'])
reclone = read(cache / 'identical-app-model-reclone-v1.json')
assert reclone['both_app_inventories_and_deep_signatures_unchanged'] and all(f['replaced'] for f in reclone['files'])
out.mkdir(); actual = {}
for backend in ['cpu', 'metal']:
    app = base / backend / 'Pnola.app'; static = read(base / (backend + '-static-verification.json'))
    assert inventory(app) == r['apps'][backend]['files']
    assert static['app_structure_verified'] and static['backend'] == backend and not static['release_qualified']
    signature = subprocess.run(['/usr/bin/codesign', '--display', '--verbose=4', str(app)], capture_output=True, check=True)
    assert b'(adhoc,runtime)' in signature.stderr
    entitlement = subprocess.run(['/usr/bin/codesign', '--display', '--entitlements', ':-', str(app)], capture_output=True, check=True)
    assert plistlib.loads(entitlement.stdout) == {'com.apple.security.cs.disable-library-validation': True, 'com.apple.security.device.audio-input': True}
    (out / (backend + '-signature.log')).write_bytes(signature.stdout + signature.stderr)
    (out / (backend + '-entitlements.plist')).write_bytes(entitlement.stdout)
    actual[backend] = {'executable': static['executable'], 'logical_native_components': static['logical_runtime_library_count'], 'framework_files': len(static['frameworks'])}
for path in base.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
for name, expected in p['pnola_source_files'].items():
    source = Path('/Users/victor/workspace/pnola') / name; assert descriptor(source) == expected, name
    target = out / 'pnola-source' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
for name, expected in p['gen2_files'].items():
    assert descriptor(repo / name) == expected, name
for name in ['build-valid-mel-pnola-instrumented-apps-v1.py', 'reclone-identical-app-models-v1.py', 'identical-app-model-reclone-v1.json']:
    shutil.copy2(cache / name, out / name)
for name in ['verify_valid_mel_pnola_app.py', 'qualification_storage.py', 'qualify_valid_mel_pnola.py']:
    shutil.copy2(repo / 'tools/nemotron-diarization' / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'instrumented_cpu_metal_app_construction_passed': True, 'release_qualified': False,
          'source_commit': r['source_commit'], 'scope': __doc__, 'app_or_model_or_native_version_calls': 0,
          'source_pins': {'gen2': len(p['gen2_files']), 'pnola': len(p['pnola_source_files'])}, 'apps': actual,
          'hardened_runtime_and_exact_local_entitlements_verified': True,
          'prior_failed_app_attempt': p['trace_failure'], 'model_free_callback_proof': p['callback_proof'],
          'payload_identical_model_cow_replacements': len(reclone['files']),
          'limits': ['Complete new same-host ad-hoc CPU/Metal apps and callback code are built and statically verified. No new app-model process has run.',
                     'Storage recovery atomically substitutes byte-identical APFS model clones. New app inventories and deep signatures remain exact; originals remain unchanged.',
                     'Construction uses3GiB/1GiB/256MiB storage policy; neural admission remains5GiB after256MiB reserve. No GUI, microphone, quality, performance, Developer ID, notarization, fresh-machine or release qualification.']}
record['files'] = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
(out / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'members': len(record['files']) + 1, 'apps': actual}))
