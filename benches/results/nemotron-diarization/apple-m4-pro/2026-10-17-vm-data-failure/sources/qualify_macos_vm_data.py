"""Qualify bounded offline Data access on a fresh stopped paired clone.

No VM/network/SSH/account/private app/model execution, host sudo or ownership DB
changes. A passing access probe never establishes stored guest UID/GID correctness.
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
    digest, size = hashlib.sha256(), 0
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
            raise ValueError('Data-access protocol changed')
        for name, want in p['tools'].items():
            if descriptor(tools / name) != want:
                raise ValueError('Data-access source changed: ' + name)
        for name, want in p['external_files'].items():
            if descriptor(name) != want:
                raise ValueError('Data-access prerequisite/tool metadata changed: ' + name)

    frozen()
    from frozen_git_sources import verify_git_sources
    from qualification_storage import StorageGuard, atomic_json
    from qualify_macos_vm_host_v4 import execute_child, policy, cleanup_metadata
    from macos_vm_host_metrics import attached_image_device
    from macos_vm_seed_clone import clone_regular, file_identity
    from macos_vm_install_metrics import installed_seed_metrics
    from macos_vm_boot_metrics import scoped_container
    from macos_vm_data_metrics import system_data_pair, mounted_member
    from macos_vm_data_probe import directory
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    verify_git_sources(repo, source, {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()})
    if raw != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-vm-data.protocol.json'], cwd=repo):
        raise ValueError('committed Data-access protocol required')
    prior = Path(p['boot_prerequisite']['receipt'])
    manifest = json.loads((prior / 'collection.json').read_text())
    for name, want in manifest['files'].items():
        if descriptor(prior / name) != want:
            raise ValueError('prior boot receipt changed: ' + name)
    pq = json.loads((prior / 'qualification.json').read_text())
    if (pq['boot_discovery_passed'] is not True or pq['boot_owner_qualified'] is not True
            or pq['source_commit'] != p['boot_prerequisite']['source_commit']
            or pq['owner_result']['cleanup_confirmed'] is not True or pq['owner_result']['final_state'] != 0
            or pq['owned_image_detached'] is not True or pq['installed_disk_quarantined'] is not False
            or pq['exclusive_lock_released'] is not True or pq['source_exclusive_lock_released'] is not True):
        raise ValueError('passing stopped/detached boot prerequisite required')
    origin = Path(p['boot_prerequisite']['local_root'])

    def original_seed():
        result = installed_seed_metrics(origin / 'vm-seed', pq['booted_seed'], uid=os.getuid())
        if result != pq['booted_seed']:
            raise ValueError('original stopped booted seed changed')
        return {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()}

    original_identity = original_seed()
    if original_identity != p['boot_prerequisite']['seed_identity']:
        raise ValueError('preregistered stopped seed identities required')
    parent = Path(p['output_parent']).resolve(strict=True)
    if Path(p['output_name']).name != p['output_name']:
        raise ValueError('single fresh Data-access output required')
    output = parent / p['output_name']
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output required; no retry/reuse')
    output.mkdir(mode=0o700)
    (output / 'protocol.json').write_bytes(raw)
    q = {'schema_version': 1, 'source_commit': source, 'protocol': descriptor(args.protocol),
        'data_access_passed': False, 'system_data_group_verified': False, 'stored_guest_ownership_verified': False,
        'fresh_machine_qualified': False, 'release_qualified': False, 'guest_os_ready_qualified': False,
        'virtual_machine_created': False, 'guest_boot_started': False, 'provisioning_performed': False,
        'ssh_or_private_app_or_models_executed': False, 'networking_permitted': False,
        'clone_quarantined': True, 'owned_image_detached': False, 'commands': [], 'scope': __doc__,
        'prior_boot_receipt_members_verified': len(manifest['files']), 'original_seed_identity': original_identity}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    guard, lock, source_lock = None, None, None
    attachment_attempted = False
    mounted = {}
    image = output / 'vm-seed/disk.img'
    home, temporary, sandbox = output / 'home', output / 'tmp', output / 'offline.sb'
    environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/',
        'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}

    def interrupted(number, frame):
        raise InterruptedError('Data-access qualification signal ' + str(number))

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    previous_interrupt = signal.getsignal(signal.SIGINT)

    def run(name, arguments, *, cleanup=False):
        if not cleanup:
            frozen()
            if guard.check() < guard.running_minimum:
                raise OSError('Data-access running storage floor exceeded')
        row = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(sandbox), *map(str, arguments)],
            'cwd': str(output), 'environment': environment}
        q['commands'].append(row)
        began = time.monotonic()
        if cleanup:
            captured = execute_child(row['argv'], cwd=output, environment=environment, record=row,
                persist=persist, timeout=p['command_timeout_seconds'])
            for label, data in zip(['stdout', 'stderr'], captured):
                cleanup_metadata(lambda: (output / (name + '.' + label + '.log')).write_bytes(data), q)
            result = captured[0]
        else:
            persist()
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                execute_child(row['argv'], cwd=output, environment=environment, record=row, persist=persist,
                    timeout=p['command_timeout_seconds'], guard=guard, stdout=stdout, stderr=stderr)
            result = (output / (name + '.stdout.log')).read_bytes()
        row['wall_seconds'] = time.monotonic() - began
        if cleanup:
            cleanup_metadata(persist, q)
        else:
            persist()
        if row['returncode'] != 0:
            raise ValueError('Data-access command failed: ' + name)
        return result

    def own_attachment(name, *, cleanup=False):
        inventory = plistlib.loads(run(name, ['/usr/bin/hdiutil', 'info', '-plist'], cleanup=cleanup))
        whole = attached_image_device(inventory, image)
        value = {'image': str(image), 'device': whole,
            'matching_images': [row for row in inventory['images'] if row.get('image-path') == str(image)]}
        if cleanup:
            cleanup_metadata(lambda: atomic_json(output / (name + '.own-image.json'), value), q)
        else:
            atomic_json(output / (name + '.own-image.json'), value)
        return whole, value

    try:
        guard = StorageGuard(output, **p['storage'])
        q['storage'] = guard.record
        home.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        sandbox.write_text(policy(p['denied_paths']) + '(deny file-write* (subpath ' + json.dumps(str(origin)) + '))\n')
        q['host'] = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'platform': platform.platform(), 'uid': os.getuid()}
        if q['host']['cpu'] != p['expected_cpu']:
            raise ValueError('named physical host required')
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for path in probes:
            path.write_bytes(b'not a model')
        control = ('import pathlib,socket,sys,os\n'
            'for name in sys.argv[2:]:\n'
            ' try:pathlib.Path(name).read_bytes()\n'
            ' except PermissionError:pass\n'
            ' else:raise RuntimeError("model read allowed")\n'
            'try:fd=os.open(sys.argv[1],os.O_WRONLY|os.O_NOFOLLOW)\n'
            'except PermissionError:pass\n'
            'else:os.close(fd);raise RuntimeError("original seed write-open allowed")\n'
            's=socket.socket()\n'
            'try:s.connect(("127.0.0.1",9))\n'
            'except PermissionError:pass\n'
            'else:raise RuntimeError("network allowed")\n')
        run('offline-controls', ['/usr/bin/python3', '-B', '-c', control, origin / 'vm-seed/hardware-model.bin', *probes])
        source_lock = os.open(origin / 'owner.lock', os.O_RDONLY | os.O_NOFOLLOW)
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_stat = os.fstat(source_lock)
        if (not stat.S_ISREG(source_stat.st_mode) or stat.S_IMODE(source_stat.st_mode) != 0o600
                or source_stat.st_uid != os.getuid() or source_stat.st_nlink != 1
                or source_stat.st_ino != (origin / 'owner.lock').lstat().st_ino):
            raise ValueError('owned original exclusive lock required')
        q['source_exclusive_lock_path'] = str(origin / 'owner.lock')
        if original_seed() != original_identity:
            raise ValueError('source seed changed before exclusive clone')
        lock = os.open(output / 'owner.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q['exclusive_lock_path'] = str(output / 'owner.lock')
        seed = output / 'vm-seed'
        seed.mkdir(mode=0o700)
        q['clones'] = {}
        for name in sorted(pq['booted_seed']):
            q['clones'][name] = clone_regular(origin / 'vm-seed' / name, seed, name)
            persist()
        q['cloned_seed_before_attach'] = installed_seed_metrics(seed, pq['booted_seed'], uid=os.getuid())
        for name, metadata in q['cloned_seed_before_attach'].items():
            if any(metadata[key] != pq['booted_seed'][name][key] for key in ['bytes', 'sha256', 'mode', 'uid', 'links']):
                raise ValueError('cloned paired content/ownership differs: ' + name)
        if original_seed() != original_identity:
            raise ValueError('original stopped seed changed after clone')
        if own_attachment('before-image')[0] is not None:
            raise ValueError('fresh clone unexpectedly attached')
        attachment_attempted = True
        run('attach-readwrite', ['/usr/bin/hdiutil', 'attach', '-imagekey', 'diskimage-class=CRawDiskImage',
            '-readwrite', '-nomount', '-nobrowse', '-plist', image])
        whole, inventory = own_attachment('attached-image')
        if whole is None or len(inventory['matching_images']) != 1 or inventory['matching_images'][0].get('writeable') is not True:
            raise ValueError('exact fresh raw image readwrite attachment required')
        partitions = [row['dev-entry'] for row in inventory['matching_images'][0]['system-entities']
            if re.fullmatch(re.escape(whole) + r's[1-9][0-9]*', row.get('dev-entry', ''))]
        if not partitions or len(set(partitions)) != len(partitions):
            raise ValueError('unique actual own physical partitions required')
        q['layout'] = {'whole': whole, 'partitions': [], 'containers': []}
        pairs, containers = [], set()
        for index, partition in enumerate(partitions):
            info = plistlib.loads(run(f'partition-{index}-info', ['/usr/sbin/diskutil', 'info', '-plist', partition]))
            if (info.get('DeviceNode') != partition or info.get('DeviceIdentifier') != partition[5:]
                    or info.get('ParentWholeDisk') != whole[5:] or info.get('MountPoint')):
                raise ValueError('unmounted exact observed physical partition required')
            q['layout']['partitions'].append({'device': partition, 'info': info})
            physical_info = info
            container = info.get('APFSContainerReference')
            if container is None:
                continue
            if not re.fullmatch(r'disk[0-9]+', container) or container in containers:
                raise ValueError('unique observed APFS container required')
            containers.add(container)
            listing = plistlib.loads(run(f'container-{index}-list', ['/usr/sbin/diskutil', 'apfs', 'list', '-plist', container]))
            observed = scoped_container(listing, container, partition)
            groups = plistlib.loads(run(f'container-{index}-groups', ['/usr/sbin/diskutil', 'apfs', 'listVolumeGroups', '-plist', container]))
            infos = {}
            for volume_index, volume in enumerate(observed['Volumes']):
                name = volume['DeviceIdentifier']
                info = plistlib.loads(run(f'volume-{index}-{volume_index}-info', ['/usr/sbin/diskutil', 'info', '-plist', '/dev/' + name]))
                if (info.get('DeviceIdentifier') != name or info.get('DeviceNode') != '/dev/' + name
                        or info.get('APFSContainerReference') != container or info.get('MountPoint')):
                    raise ValueError('exact unmounted scoped volume required')
                infos[name] = info
            q['layout']['containers'].append({'device': container, 'physical_store': partition,
                'listing': listing, 'groups': groups, 'infos': infos})
            if any(role in {'System', 'Data'} for volume in observed['Volumes'] for role in volume['Roles']):
                pairs.append(system_data_pair(listing, groups, container, partition, infos, physical_info))
        if len(pairs) != 1:
            raise ValueError('one unambiguous System/Data group across the exact image required')
        pair = q['system_data_pair'] = pairs[0]
        q['system_data_group_verified'] = True
        q['mounts'] = {}
        for role in ['data', 'system']:
            if own_attachment('before-' + role + '-mount')[0] != whole:
                raise ValueError('owned image changed before mount')
            mount = output / (role + '-mount')
            mount.mkdir(mode=0o700)
            mounted[role] = {'mount': mount, 'member': pair[role]}
            command = ['/usr/sbin/diskutil', 'mount'] + (['readOnly'] if role == 'system' else [])
            run('mount-' + role, command + ['nobrowse', '-mountOptions', 'noowners', '-mountPoint', mount, pair[role]['device']])
            info = plistlib.loads(run('mounted-' + role + '-info', ['/usr/sbin/diskutil', 'info', '-plist', pair[role]['device']]))
            status = {'is_mount': mount.is_mount(), 'stat_device': mount.stat().st_dev,
                'parent_stat_device': mount.parent.stat().st_dev, 'readonly': bool(os.statvfs(mount).f_flag & os.ST_RDONLY)}
            q['mounts'][role] = {'info': info, 'status': status,
                'validated': mounted_member(info, mount, pair[role], pair, status, readonly=role == 'system')}
        if q['mounts']['system']['status']['stat_device'] == q['mounts']['data']['status']['stat_device']:
            raise ValueError('separate System and Data filesystems required')
        config = {role: str(mounted[role]['mount']) for role in ['system', 'data']}
        config.update({role + '_device': q['mounts'][role]['status']['stat_device'] for role in ['system', 'data']})
        config.update(p['probe'])
        config['version_output'] = str(output / 'guest-SystemVersion.plist')
        atomic_json(output / 'probe-config.json', config)
        try:
            result = run('data-probe', ['/usr/bin/python3', '-B', tools / 'macos_vm_data_probe.py', '--config', output / 'probe-config.json'])
        finally:
            probe_log = output / 'data-probe.stdout.log'
            if probe_log.exists() and probe_log.stat().st_size:
                q['data_probe'] = json.loads(probe_log.read_text())
        if json.loads(result) != q['data_probe']:
            raise ValueError('terminal probe output changed')
        if (q['data_probe']['access_passed'] is not True or q['data_probe']['stored_guest_ownership_verified'] is not False
                or q['data_probe']['provisioning_performed'] is not False or len(q['data_probe']['writes']) != 7
                or len(q['data_probe']['existing_write_opens']) != 2):
            raise ValueError('bounded stopped-guest probe without ownership/provisioning claim required')
        root_fd = os.open(mounted['data']['mount'], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for row in q['data_probe']['writes']:
                fd = directory(root_fd, row['directory'])
                try:
                    try:
                        os.stat(p['probe']['probe_name'], dir_fd=fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise ValueError('probe entry remains after successful child')
                finally:
                    os.close(fd)
        finally:
            os.close(root_fd)
        q['probe_removal_independently_verified'] = True
        frozen()
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if guard is not None:
            cleanup_metadata(guard.release, q)
        if attachment_attempted:
            try:
                attached, cleanup_inventory = own_attachment('cleanup-image', cleanup=True)
                if attached is not None:
                    own_devices = {row.get('dev-entry') for image_row in cleanup_inventory['matching_images']
                        for row in image_row['system-entities']}
                    for role, record in reversed(list(mounted.items())):
                        try:
                            if attached != q['layout']['whole'] or record['member']['device'] not in own_devices:
                                raise ValueError('cleanup selected volume is no longer in the exact own image')
                            info = plistlib.loads(run('cleanup-' + role + '-info', ['/usr/sbin/diskutil', 'info', '-plist', record['member']['device']], cleanup=True))
                            if (info['DeviceNode'] != record['member']['device'] or info['VolumeUUID'] != record['member']['uuid']
                                    or info['APFSPhysicalStores'] != [{'APFSPhysicalStore': pair['physical_store'][5:]}]):
                                raise ValueError('cleanup volume no longer binds exact task image')
                            if info.get('MountPoint'):
                                if info['MountPoint'] != str(record['mount']):
                                    raise ValueError('cleanup mountpoint changed')
                                run('unmount-' + role, ['/usr/sbin/diskutil', 'unmount', record['member']['device']], cleanup=True)
                            after_info = plistlib.loads(run('unmounted-' + role + '-info', ['/usr/sbin/diskutil', 'info', '-plist', record['member']['device']], cleanup=True))
                            if after_info.get('MountPoint') or record['mount'].is_mount():
                                raise ValueError('task volume remains mounted')
                        except BaseException as error:
                            q.setdefault('unmount_errors', []).append(repr(error))
                    # Detach the current exact own-image device even if a mount query failed.
                    attached, _ = own_attachment('before-detach-image', cleanup=True)
                    if attached is not None:
                        run('detach-image', ['/usr/bin/hdiutil', 'detach', attached], cleanup=True)
                after, _ = own_attachment('after-image', cleanup=True)
                if after is not None or any(record['mount'].is_mount() for record in mounted.values()):
                    raise ValueError('owned image or mount remains attached')
                q['owned_image_detached'] = True
            except BaseException as error:
                q['cleanup_error'] = repr(error)
        # Source checks occur after teardown; no admission error may veto cleanup.
        if source_lock is not None:
            try:
                if original_seed() != original_identity:
                    raise ValueError('original booted seed changed during access/cleanup')
                q['original_seed_unchanged_after_cleanup'] = True
            except BaseException as error:
                q['source_preservation_error'] = repr(error)
        if (q.get('probe_removal_independently_verified') is True and q['owned_image_detached']
                and not any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors', 'unmount_errors'])):
            try:
                accessed = installed_seed_metrics(output / 'vm-seed', pq['booted_seed'], uid=os.getuid())
                for name in accessed:
                    if name != 'disk.img' and any(accessed[name][key] != pq['booted_seed'][name][key]
                            for key in ['bytes', 'sha256', 'mode', 'uid', 'links']):
                        raise ValueError('paired platform/auxiliary metadata changed without a VM: ' + name)
                accessed['disk.img']['content_evidence'] = 'OS clone of qualified stopped booted seed, then readwrite filesystem access and confirmed unmount/detach; no installer or VM in this attempt, no whole-disk hash.'
                q['accessed_seed'] = accessed
            except BaseException as error:
                q['post_detach_seed_error'] = repr(error)
        if lock is not None:
            os.close(lock)
            q['exclusive_lock_released'] = True
        if source_lock is not None:
            os.close(source_lock)
            q['source_exclusive_lock_released'] = True
        if guard is not None:
            q['storage'] = guard.record
        q['data_access_passed'] = (q.get('probe_removal_independently_verified') is True
            and q['system_data_group_verified'] and q['owned_image_detached']
            and 'accessed_seed' in q
            and q.get('original_seed_unchanged_after_cleanup') is True
            and not any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors', 'unmount_errors', 'source_preservation_error', 'post_detach_seed_error']))
        q['clone_quarantined'] = not q['data_access_passed']
        persist()
        signal.signal(signal.SIGTERM, previous_handler)
        signal.signal(signal.SIGINT, previous_interrupt)
    if not q['data_access_passed']:
        raise ValueError('Data access or cleanup failed; see qualification record')


if __name__ == '__main__':
    main()
