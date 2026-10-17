"""Install the pinned macOS restore image onto an enforced clone of the seed.

Explicit setup networking is permitted. No normal guest boot, mount/provisioning,
private app/model invocation or host privilege/settings/identity change.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import signal
import subprocess
import time


def descriptor(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw = args.protocol.read_bytes()
    p = json.loads(raw)

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('install protocol changed')
        for name, want in p['tools'].items():
            if descriptor(tools / name) != want:
                raise ValueError('install source changed: ' + name)
        for name, want in p['external_files'].items():
            if descriptor(name) != want:
                raise ValueError('install tool/SDK/prerequisite metadata changed: ' + name)

    frozen()
    from frozen_git_sources import verify_git_sources
    from qualification_storage import StorageGuard, atomic_json
    from qualify_macos_vm_host_v4 import execute_child, policy
    from macos_vm_restore_metrics import seed_metrics
    from macos_vm_seed_clone import clone_regular, file_identity
    from macos_vm_install_supervision import execute_install_owner
    from macos_vm_install_metrics import owner_metrics, installed_seed_metrics
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    verify_git_sources(repo, source, {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()})
    if raw != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-vm-install.protocol.json'], cwd=repo):
        raise ValueError('committed install protocol required')
    prior = Path(p['restore_prerequisite']['receipt'])
    manifest = json.loads((prior / 'collection.json').read_text())
    for name, want in manifest['files'].items():
        if descriptor(prior / name) != want:
            raise ValueError('prior restore receipt changed: ' + name)
    pq = json.loads((prior / 'qualification.json').read_text())
    if pq['no_boot_restore_passed'] is not True or pq['source_commit'] != p['restore_prerequisite']['source_commit']:
        raise ValueError('passing actual no-boot qualification required')
    origin = Path(p['restore_prerequisite']['local_root'])
    image = origin / p['restore_image']['name']
    image_pin = {key: p['restore_image'][key] for key in ['bytes', 'sha256']}
    if descriptor(image) != image_pin:
        raise ValueError('complete pinned restore image required')

    def original_seed():
        result = seed_metrics(origin / 'vm-seed', pq['prepared'], pq['image'], uid=os.getuid())
        if result != pq['seed']:
            raise ValueError('original paired seed changed')
        return {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()}

    original_identity = original_seed()
    parent = Path(p['output_parent']).resolve(strict=True)
    if Path(p['output_name']).name != p['output_name']:
        raise ValueError('single fresh install output required')
    output = parent / p['output_name']
    if output.exists() or output.is_symlink():
        raise ValueError('fresh install output required; no automatic retry/reuse')
    output.mkdir(mode=0o700)
    (output / 'protocol.json').write_bytes(raw)
    q = {'schema_version': 1, 'source_commit': source, 'protocol': descriptor(args.protocol),
        'installation_passed': False, 'fresh_machine_qualified': False, 'release_qualified': False,
        'virtual_machine_created': False, 'installation_started': False, 'normal_guest_boot_started': False,
        'private_app_or_models_executed': False, 'setup_networking_permitted': True,
        'installed_disk_quarantined': True, 'commands': [], 'scope': __doc__,
        'prior_restore_receipt_members_verified': len(manifest['files']), 'original_seed_identity': original_identity}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    guard, lock = None, None
    home, temporary = output / 'home', output / 'tmp'
    offline, setup = output / 'offline.sb', output / 'setup.sb'
    environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}

    def interrupted(number, frame):
        raise InterruptedError('qualification signal ' + str(number))

    previous_handler = signal.signal(signal.SIGTERM, interrupted)

    def run(name, arguments, *, installing=False, setup_policy=False):
        frozen()
        if guard.check() < guard.running_minimum:
            raise OSError('install child storage floor exceeded')
        row = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(setup if installing or setup_policy else offline), *map(str, arguments)],
            'cwd': str(output), 'environment': environment}
        q['commands'].append(row)
        if installing:
            # Owner event stream, not stale false flags, is authoritative while active.
            q['virtual_machine_created'] = None
            q['installation_started'] = None
        persist()
        started = time.monotonic()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            if installing:
                execute_install_owner(row['argv'], cwd=output, environment=environment, record=row,
                    persist=persist, guard=guard, timeout=p['supervision']['operation_seconds'],
                    cleanup_timeout=p['supervision']['cleanup_seconds'], stdout=stdout, stderr=stderr)
            else:
                execute_child(row['argv'], cwd=output, environment=environment, record=row, persist=persist,
                    timeout=p['command_timeout_seconds'], guard=guard, stdout=stdout, stderr=stderr)
        row['wall_seconds'] = time.monotonic() - started
        persist()
        if row['returncode'] != 0:
            raise ValueError('install qualification child failed: ' + name)
        return (output / (name + '.stdout.log')).read_bytes()

    try:
        guard = StorageGuard(output, **p['storage'])
        q['storage'] = guard.record
        home.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        offline_text = policy(p['denied_paths']) + '(deny file-write* (subpath ' + json.dumps(str(origin)) + '))\n'
        if offline_text.count('(deny network*)\n') != 1:
            raise ValueError('exact offline/network policy split required')
        offline.write_text(offline_text)
        setup.write_text(offline_text.replace('(deny network*)\n', ''))
        q['host'] = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'platform': platform.platform(), 'uid': os.getuid()}
        if q['host']['cpu'] != p['expected_cpu']:
            raise ValueError('named physical host required')
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for path in probes:
            path.write_bytes(b'not a model')
        read_control = ('import pathlib,socket,sys,os\n'
            'for name in sys.argv[2:]:\n'
            ' try:pathlib.Path(name).read_bytes()\n'
            ' except PermissionError:pass\n'
            ' else:raise RuntimeError("model read allowed")\n'
            'try:fd=os.open(sys.argv[1],os.O_WRONLY|os.O_NOFOLLOW)\n'
            'except PermissionError:pass\n'
            'else:os.close(fd);raise RuntimeError("original seed write-open allowed")\n')
        control = read_control + ('s=socket.socket()\n'
            'try:s.connect(("127.0.0.1",9))\n'
            'except PermissionError:pass\n'
            'else:raise RuntimeError("network allowed")\n')
        original_control = origin / 'vm-seed/hardware-model.bin'
        run('offline-controls', ['/usr/bin/python3', '-c', control, original_control, *probes])
        run('setup-controls', ['/usr/bin/python3', '-c', read_control, original_control, *probes], setup_policy=True)
        executable = output / 'vm-install-owner'
        run('build', [p['swiftc'], '-sdk', p['sdk'], '-target', 'arm64-apple-macosx26.0',
            '-swift-version', '5', '-warnings-as-errors', '-j', '2', '-module-cache-path', output / 'module-cache',
            tools / 'macos_vm_install_owner.swift', '-o', executable])
        run('sign', ['/usr/bin/codesign', '--force', '--sign', '-', '--timestamp=none', '--options', 'runtime',
            '--entitlements', tools / 'macos_vm_host.entitlements', executable])
        run('signature-verify', ['/usr/bin/codesign', '--verify', '--strict', '--deep', executable])
        run('signature-metadata', ['/usr/bin/codesign', '--display', '--verbose=4', executable])
        q['entitlements'] = plistlib.loads(run('entitlements', ['/usr/bin/codesign', '--display', '--xml', '--entitlements', '-', executable]))
        if q['entitlements'] != {'com.apple.security.virtualization': True}:
            raise ValueError('exact unrestricted virtualization entitlement required')
        signature = (output / 'signature-metadata.stderr.log').read_text()
        if 'Signature=adhoc' not in signature or not re.search(r'flags=0x[0-9a-f]+\([^)]*runtime', signature):
            raise ValueError('actual ad-hoc hardened signature required')
        if run('architecture', ['/usr/bin/lipo', '-archs', executable]).strip() != b'arm64':
            raise ValueError('arm64 owner required')
        if not re.search(r'\bminos 26\.0\s', run('deployment', ['/usr/bin/otool', '-l', executable]).decode()):
            raise ValueError('macOS26.0 deployment required')
        run('dependencies', ['/usr/bin/otool', '-L', executable])
        q['owner_executable'] = descriptor(executable)
        graph = json.loads(run('error-control', [executable, 'error-control']))
        if (len(graph) != 5 or sorted(row['code'] for row in graph if 'code' in row) != [1, 2, 3, 4]
                or sum('reference_node' in row for row in graph) != 1
                or {row['relation'] for row in graph} != {'root', 'underlying', 'multiple[0]', 'multiple[1]'}):
            raise ValueError('complete independent/shared underlying NSError graph required')
        q['error_graph_control'] = graph
        lock = os.open(output / 'owner.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q['exclusive_lock_path'] = str(output / 'owner.lock')
        seed = output / 'vm-seed'
        seed.mkdir(mode=0o700)
        q['clones'] = {}
        for name in sorted(pq['seed']):
            q['clones'][name] = clone_regular(origin / 'vm-seed' / name, seed, name)
            persist()
        q['cloned_seed_before_install'] = seed_metrics(seed, pq['prepared'], pq['image'], uid=os.getuid())
        for name, metadata in q['cloned_seed_before_install'].items():
            # APFS may account cloned metadata blocks differently; compare actual content/ownership.
            if any(metadata[key] != pq['seed'][name][key] for key in ['bytes', 'sha256', 'mode', 'uid', 'links']):
                raise ValueError('cloned seed content/ownership changed: ' + name)
        if original_seed() != original_identity:
            raise ValueError('exact paired cloned seed and unchanged original required')
        config = {'seed': str(seed), 'image': str(image), 'result': str(output / 'owner-result.json'),
            'mac_address': pq['prepared']['mac_address'], **p['owner_limits']}
        atomic_json(output / 'owner-config.json', config)
        persist()
        try:
            result = run('install-owner', [executable, 'install', output / 'owner-config.json'], installing=True)
        finally:
            if (output / 'owner-result.json').is_file():
                q['owner_result'] = json.loads((output / 'owner-result.json').read_text())
                for key in ['virtual_machine_created', 'installation_started']:
                    q[key] = q['owner_result'].get(key)
        events = [json.loads(line) for line in result.splitlines()]
        owner_metrics(q['owner_result'], q['commands'][-1], events)
        q['installed_seed'] = installed_seed_metrics(seed, pq['seed'], uid=os.getuid())
        if original_seed() != original_identity or descriptor(image) != image_pin:
            raise ValueError('original image/seed changed during installation')
        frozen()
        q['installation_passed'] = True
        q['installed_disk_quarantined'] = False
    except BaseException as error:
        q['error'] = repr(error)
        q['installation_passed'] = False
        q['installed_disk_quarantined'] = True
        raise
    finally:
        if lock is not None:
            os.close(lock)
            q['exclusive_lock_released'] = True
        if guard is not None:
            guard.release()
            q['storage'] = guard.record
        signal.signal(signal.SIGTERM, previous_handler)
        persist()


if __name__ == '__main__':
    main()
