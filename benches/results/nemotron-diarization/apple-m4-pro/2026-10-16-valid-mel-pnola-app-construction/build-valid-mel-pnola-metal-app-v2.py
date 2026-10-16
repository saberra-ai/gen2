"""Continue complete Metal app construction; preserve successful CPU and failed controller."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
app = Path('/Users/victor/workspace/pnola')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from qualification_storage import StorageGuard, atomic_json
from qualify_valid_mel_pnola import execute_guarded
from verify_valid_mel_pnola_app import verify_app
from sdk_archive import inventory

out = cache / 'valid-mel-pnola-metal-app-build-v2'
assert not out.exists()
prior = cache / 'valid-mel-pnola-app-build-v1'
p = json.loads((prior / 'protocol.json').read_text())
before = json.loads((prior / 'build.json').read_text())
assert not before['app_construction_passed'] and len(before['commands']) == 1
assert before['commands'][0]['name'] == 'cpu-build' and before['commands'][0]['returncode'] == 0
cpu = cache / 'valid-mel-pnola-app-cpu-snapshot-v1/Pnola.app'
cpu_inventory = inventory(cpu)
env = {**p['environment'], 'GEN2_NEMO_PACKAGE': str(cache / 'valid-mel-pnola-inputs-v2/work/metal/nemo')}
out.mkdir()
policy = out / 'construction.sb'
policy.write_bytes((prior / 'construction.sb').read_bytes())
protocol = {'scope': __doc__, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'prior_protocol': descriptor(prior / 'protocol.json'), 'prior_failed_controller': descriptor(prior / 'build.json'),
            'prior_cpu_snapshot': str(cpu), 'prior_cpu_inventory': cpu_inventory, 'gen2_files': p['gen2_files'],
            'pnola_source_files': p['pnola_source_files'], 'environment': env,
            'metal_package_manifest': p['model_package_manifests']['metal'], 'launcher': descriptor(Path(__file__)),
            'app_verifier': descriptor(repo / 'tools/nemotron-diarization/verify_valid_mel_pnola_app.py'),
            'model_payloads_read_for_copy_and_hash_only': True,
            'storage_policy': {'construction_only_start_bytes': 3 * 1024**3, 'running_bytes': 1024**3, 'reserve_bytes': 256 * 1024**2, 'neural_admission_policy_unchanged': '5 GiB after allocating 256 MiB reserve; this launcher executes no models'}}
atomic_json(out / 'protocol.json', protocol)
commands = []
r = {'scope': __doc__, 'source_commit': protocol['source_commit'], 'app_construction_passed': False,
     'native_version_queries_or_neural_calls': 0, 'models_loaded_for_inference': False, 'commands': commands}
guard = None

def frozen():
    for root, records in [(repo, p['gen2_files']), (app, p['pnola_source_files'])]:
        for name, expected in records.items():
            assert descriptor(root / name) == expected, name
    assert inventory(cpu) == cpu_inventory
    assert descriptor(Path(env['GEN2_NEMO_PACKAGE']) / 'package.json') == p['model_package_manifests']['metal']
    assert descriptor(prior / 'build.json') == protocol['prior_failed_controller']

def run(name, argv):
    command = {'name': name, 'argv': argv, 'cwd': str(app)}
    commands.append(command); atomic_json(out / 'commands.json', commands)
    started = time.monotonic()
    try:
        with (out / (name + '.stdout.log')).open('xb') as stdout, (out / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(argv, cwd=app, environment=env, stdout=stdout, stderr=stderr, guard=guard,
                            record=command, persist=lambda: atomic_json(out / 'commands.json', commands), timeout=1800)
    finally:
        command['elapsed_seconds'] = time.monotonic() - started
        atomic_json(out / 'commands.json', commands)
    if command['returncode'] != 0:
        raise RuntimeError('construction command failed: ' + name)

try:
    frozen()
    guard = StorageGuard(out, 3 * 1024**3, 1024**3, 256 * 1024**2)
    guard.before_model()
    run('metal-build', ['/usr/bin/sandbox-exec', '-f', str(policy), 'npm', 'run', 'desktop:build'])
    built = Path(env['CARGO_TARGET_DIR']) / 'release/bundle/macos/Pnola.app'
    dest = out / 'Pnola.app'
    # Pure copy/static inspection uses the running storage floor, with no model admission.
    run('metal-clone', ['/bin/cp', '-cR', str(built), str(dest)])
    actual = verify_app(dest)
    assert actual['backend'] == 'metal' and actual['logical_runtime_library_count'] == 93
    atomic_json(out / 'static-verification.json', actual)
    r['app'] = {'path': str(dest), 'static_verification': descriptor(out / 'static-verification.json'), 'files': inventory(dest)}
    frozen()
    r['app_construction_passed'] = True
except BaseException as error:
    r['error'] = repr(error)
    raise
finally:
    if guard is not None:
        guard.release(); r['storage'] = guard.record
    atomic_json(out / 'build.json', r)
print(json.dumps({'app_construction_passed': r['app_construction_passed'], 'neural_calls': 0}))
