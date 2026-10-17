"""Normal-start a fresh paired clone, stop it, acquire scoped readonly disk metadata.

Setup networking is permitted. No mount/provisioning, SSH, app/model or guest OS
readiness qualification; volume-group schema is retained, not guessed.
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
import stat
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
            raise ValueError('boot protocol changed')
        for name, want in p['tools'].items():
            if descriptor(tools / name) != want:
                raise ValueError('boot source changed: ' + name)
        for name, want in p['external_files'].items():
            if descriptor(name) != want:
                raise ValueError('boot tool/SDK/prerequisite metadata changed: ' + name)

    frozen()
    from frozen_git_sources import verify_git_sources
    from qualification_storage import StorageGuard, atomic_json
    from qualify_macos_vm_host_v4 import execute_child, policy, cleanup_metadata
    from macos_vm_host_metrics import attached_image_device
    from macos_vm_seed_clone import clone_regular, file_identity
    from macos_vm_install_supervision import execute_install_owner
    from macos_vm_install_metrics import installed_seed_metrics
    from macos_vm_boot_metrics import boot_metrics, observed_partitions, scoped_container
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    verify_git_sources(repo, source, {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()})
    if raw != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-vm-boot.protocol.json'], cwd=repo):
        raise ValueError('committed boot protocol required')
    prior = Path(p['installed_prerequisite']['receipt'])
    manifest = json.loads((prior / 'collection.json').read_text())
    for name, want in manifest['files'].items():
        if descriptor(prior / name) != want:
            raise ValueError('prior install receipt changed: ' + name)
    pq = json.loads((prior / 'qualification.json').read_text())
    if (pq['installation_passed'] is not True or pq['source_commit'] != p['installed_prerequisite']['source_commit']
            or pq['owner_result']['cleanup_confirmed'] is not True or pq['owner_result']['final_state'] != 0
            or pq['installed_disk_quarantined'] is not False or pq['exclusive_lock_released'] is not True):
        raise ValueError('passing installed-stopped owner qualification required')
    origin = Path(p['installed_prerequisite']['local_root'])

    def original_seed():
        result = installed_seed_metrics(origin / 'vm-seed', pq['installed_seed'], uid=os.getuid())
        if result != pq['installed_seed']:
            raise ValueError('original paired seed changed')
        return {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()}

    original_identity = original_seed()
    if original_identity != p['installed_prerequisite']['seed_identity']:
        raise ValueError('preregistered installed seed identity required')
    parent = Path(p['output_parent']).resolve(strict=True)
    if Path(p['output_name']).name != p['output_name']:
        raise ValueError('single fresh boot output required')
    output = parent / p['output_name']
    if output.exists() or output.is_symlink():
        raise ValueError('fresh boot output required; no automatic retry/reuse')
    output.mkdir(mode=0o700)
    (output / 'protocol.json').write_bytes(raw)
    q = {'schema_version': 1, 'source_commit': source, 'protocol': descriptor(args.protocol),
        'boot_discovery_passed': False, 'boot_owner_qualified': False, 'fresh_machine_qualified': False, 'release_qualified': False,
        'virtual_machine_created': False, 'installation_started': False, 'normal_guest_boot_started': False,
        'guest_os_ready_qualified': False, 'system_data_group_verified': False,
        'private_app_or_models_executed': False, 'setup_networking_permitted': True,
        'scoped_metadata_acquired': False, 'owned_image_detached': False,
        'installed_disk_quarantined': True, 'commands': [], 'scope': __doc__,
        'prior_install_receipt_members_verified': len(manifest['files']), 'original_seed_identity': original_identity}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    guard, lock, source_lock = None, None, None
    attachment_attempted = False
    image = output / 'vm-seed/disk.img'
    home, temporary = output / 'home', output / 'tmp'
    offline, setup = output / 'offline.sb', output / 'setup.sb'
    environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}

    def interrupted(number, frame):
        raise InterruptedError('qualification signal ' + str(number))

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    previous_interrupt_handler = signal.getsignal(signal.SIGINT)

    def run(name, arguments, *, owning=False, setup_policy=False, cleanup=False):
        if not cleanup:
            frozen()
        if not cleanup and guard.check() < guard.running_minimum:
            raise OSError('boot child storage floor exceeded')
        row = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(setup if owning or setup_policy else offline), *map(str, arguments)],
            'cwd': str(output), 'environment': environment}
        q['commands'].append(row)
        if owning:
            # Owner event stream, not stale false flags, is authoritative while active.
            q['virtual_machine_created'] = None
            q['normal_guest_boot_started'] = None
        if not cleanup:
            persist()
        started = time.monotonic()
        if cleanup:
            captured = execute_child(row['argv'], cwd=output, environment=environment, record=row,
                persist=persist, timeout=p['command_timeout_seconds'])
            for label, data in zip(['stdout', 'stderr'], captured):
                cleanup_metadata(lambda: (output / (name + '.' + label + '.log')).write_bytes(data), q)
            result = captured[0]
        else:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                if owning:
                    execute_install_owner(row['argv'], cwd=output, environment=environment, record=row,
                        persist=persist, guard=guard, timeout=p['supervision']['operation_seconds'],
                        cleanup_timeout=p['supervision']['cleanup_seconds'], stdout=stdout, stderr=stderr)
                else:
                    execute_child(row['argv'], cwd=output, environment=environment, record=row, persist=persist,
                        timeout=p['command_timeout_seconds'], guard=guard, stdout=stdout, stderr=stderr)
            result = (output / (name + '.stdout.log')).read_bytes()
        row['wall_seconds'] = time.monotonic() - started
        if cleanup:
            cleanup_metadata(persist, q)
        else:
            persist()
        if row['returncode'] != 0:
            raise ValueError('boot qualification child failed: ' + name)
        return result

    def own_attachment(name, *, cleanup=False):
        inventory = plistlib.loads(run(name, ['/usr/bin/hdiutil', 'info', '-plist'], cleanup=cleanup))
        whole = attached_image_device(inventory, image)
        value = {'image': str(image), 'device': whole,
            'matching_images': [row for row in inventory.get('images', []) if row.get('image-path') == str(image)]}
        if cleanup:
            cleanup_metadata(lambda: atomic_json(output / (name + '.own-image.json'), value), q)
        else:
            atomic_json(output / (name + '.own-image.json'), value)
        return whole, inventory

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
        executable = output / 'vm-boot-owner'
        run('build', [p['swiftc'], '-sdk', p['sdk'], '-target', 'arm64-apple-macosx26.0',
            '-swift-version', '5', '-warnings-as-errors', '-j', '2', '-module-cache-path', output / 'module-cache',
            tools / 'macos_vm_boot_owner.swift', '-o', executable])
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
        q['shutdown_timing_control'] = json.loads(run('shutdown-control', [executable, 'shutdown-control']))
        if q['shutdown_timing_control'] != {'checks': 5, 'passed': True}:
            raise ValueError('actual late graceful-request timing control required')
        source_lock = os.open(origin / 'owner.lock', os.O_RDONLY | os.O_NOFOLLOW)
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_stat = os.fstat(source_lock)
        if (not stat.S_ISREG(lock_stat.st_mode) or stat.S_IMODE(lock_stat.st_mode) != 0o600
                or lock_stat.st_uid != os.getuid() or lock_stat.st_nlink != 1
                or lock_stat.st_ino != (origin / 'owner.lock').lstat().st_ino):
            raise ValueError('owned original exclusive lock required')
        q['source_exclusive_lock_path'] = str(origin / 'owner.lock')
        if original_seed() != original_identity:
            raise ValueError('installed seed changed before exclusive clone')
        lock = os.open(output / 'owner.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q['exclusive_lock_path'] = str(output / 'owner.lock')
        seed = output / 'vm-seed'
        seed.mkdir(mode=0o700)
        q['clones'] = {}
        for name in sorted(pq['installed_seed']):
            q['clones'][name] = clone_regular(origin / 'vm-seed' / name, seed, name)
            persist()
        q['cloned_seed_before_boot'] = installed_seed_metrics(seed, pq['installed_seed'], uid=os.getuid())
        for name, metadata in q['cloned_seed_before_boot'].items():
            # APFS may account cloned metadata blocks differently; compare actual content/ownership.
            if any(metadata[key] != pq['installed_seed'][name][key] for key in ['bytes', 'sha256', 'mode', 'uid', 'links']):
                raise ValueError('cloned seed content/ownership changed: ' + name)
        if original_seed() != original_identity:
            raise ValueError('exact paired cloned seed and unchanged original required')
        prior_config = json.loads((prior / 'owner-config.json').read_text())
        if p['mac_address'] != prior_config['mac_address']:
            raise ValueError('persisted installed platform MAC required')
        config = {'seed': str(seed), 'result': str(output / 'owner-result.json'),
            'mac_address': p['mac_address'], **p['owner_limits']}
        atomic_json(output / 'owner-config.json', config)
        persist()
        try:
            result = run('boot-owner', [executable, 'boot', output / 'owner-config.json'], owning=True)
        finally:
            if (output / 'owner-result.json').is_file():
                q['owner_result'] = json.loads((output / 'owner-result.json').read_text())
                q['virtual_machine_created'] = q['owner_result'].get('virtual_machine_created')
                q['normal_guest_boot_started'] = q['owner_result'].get('normal_start_called')
        events = [json.loads(line) for line in result.splitlines()]
        boot_metrics(q['owner_result'], q['commands'][-1], events)
        q['boot_owner_qualified'] = True
        q['booted_seed'] = installed_seed_metrics(seed, pq['installed_seed'], uid=os.getuid())
        if original_seed() != original_identity:
            raise ValueError('original installed seed changed during normal boot')
        frozen()
        before, _ = own_attachment('before-image')
        if before is not None:
            raise ValueError('new clone unexpectedly attached')
        attachment_attempted = True
        run('attach-readonly', ['/usr/bin/hdiutil', 'attach', '-imagekey', 'diskimage-class=CRawDiskImage',
            '-readonly', '-nomount', '-nobrowse', '-plist', image])
        whole, inventory = own_attachment('attached-image')
        partitions = observed_partitions(inventory, image, whole)
        run('whole-list', ['/usr/sbin/diskutil', 'list', '-plist', whole])
        info = plistlib.loads(run('whole-info', ['/usr/sbin/diskutil', 'info', '-plist', whole]))
        if info.get('DeviceIdentifier') != whole[5:] or info.get('DeviceNode') != whole or info.get('MountPoint'):
            raise ValueError('observed unmounted whole device required')
        q['layout'] = {'whole': whole, 'partitions': [], 'containers': []}
        containers = set()
        roles = set()
        for index, partition in enumerate(partitions):
            info = plistlib.loads(run(f'partition-{index}-info', ['/usr/sbin/diskutil', 'info', '-plist', partition]))
            if (info.get('DeviceIdentifier') != partition[5:] or info.get('DeviceNode') != partition
                    or info.get('ParentWholeDisk') != whole[5:] or info.get('MountPoint')):
                raise ValueError('observed physical partition binding required')
            q['layout']['partitions'].append({'device': partition, 'info': info})
            container = info.get('APFSContainerReference')
            if container is None:
                continue
            if not re.fullmatch(r'disk\d+', container) or container in containers:
                raise ValueError('unique observed APFS container required')
            containers.add(container)
            listing = plistlib.loads(run(f'container-{index}-list', ['/usr/sbin/diskutil', 'apfs', 'list', '-plist', container]))
            container_info = scoped_container(listing, container, partition)
            # The manual promises plist output, not a stable group dictionary schema.
            # Retain exact scoped raw metadata for later schema review, never infer a pair.
            group = run(f'container-{index}-groups', ['/usr/sbin/diskutil', 'apfs', 'listVolumeGroups', '-plist', container])
            plistlib.loads(group)
            record = {'device': container, 'physical_store': partition, 'volumes': [],
                'volume_groups_raw': descriptor(output / f'container-{index}-groups.stdout.log')}
            q['layout']['containers'].append(record)
            for volume_index, volume in enumerate(container_info['Volumes']):
                device = '/dev/' + volume['DeviceIdentifier']
                info = plistlib.loads(run(f'volume-{index}-{volume_index}-info', ['/usr/sbin/diskutil', 'info', '-plist', device]))
                if (info.get('DeviceIdentifier') != device[5:] or info.get('DeviceNode') != device
                        or info.get('MountPoint') or info.get('APFSContainerReference', container) != container):
                    raise ValueError('observed unmounted scoped APFS volume required')
                record['volumes'].append({'device': device, 'roles': volume['Roles'], 'info': info})
                roles.update(volume['Roles'])
            persist()
        if not containers or not {'System', 'Data'} <= roles:
            raise ValueError('System and Data role observations required; no group inference')
        q['scoped_metadata_acquired'] = True
        if original_seed() != original_identity:
            raise ValueError('original installed seed changed during readonly discovery')
        frozen()
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        # Once an owner has been reaped, bounded exact-image cleanup must finish
        # even if cancellation arrives for the first time during detach.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Admission/persistence errors must never veto exact owned-image detach.
        if guard is not None:
            cleanup_metadata(guard.release, q)
        if attachment_attempted and q['boot_owner_qualified']:
            try:
                attached, _ = own_attachment('cleanup-image', cleanup=True)
                if attached is not None:
                    run('detach-image', ['/usr/bin/hdiutil', 'detach', attached], cleanup=True)
                after, _ = own_attachment('after-image', cleanup=True)
                if after is not None:
                    raise ValueError('owned image remains attached')
                q['owned_image_detached'] = True
            except BaseException as error:
                q['cleanup_error'] = repr(error)
        if lock is not None:
            os.close(lock)
            q['exclusive_lock_released'] = True
        if source_lock is not None:
            os.close(source_lock)
            q['source_exclusive_lock_released'] = True
        if guard is not None:
            q['storage'] = guard.record
        q['boot_discovery_passed'] = (q['boot_owner_qualified'] and q['scoped_metadata_acquired']
            and q['owned_image_detached'] and not any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors']))
        q['installed_disk_quarantined'] = not q['boot_discovery_passed']
        persist()
        signal.signal(signal.SIGTERM, previous_handler)
        signal.signal(signal.SIGINT, previous_interrupt_handler)
    if not q['boot_discovery_passed']:
        raise ValueError('boot/discovery or cleanup failed; see qualification record')


if __name__ == '__main__':
    main()
