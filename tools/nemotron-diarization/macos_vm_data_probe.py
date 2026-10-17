"""Bounded stopped-guest access checks using anchored no-follow descriptors.

Only task-named probe files are written, then removed; no provisioning or stored
owner claim. Private host keys, account data and guest credentials are never read.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import stat


def directory(root, relative):
    current = os.dup(root)
    try:
        for component in relative.split('/'):
            if component in {'', '.', '..'}:
                raise ValueError('literal guest-relative components required')
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            if os.fstat(following).st_dev != os.fstat(root).st_dev:
                os.close(following)
                raise ValueError('guest path crosses mounted filesystem')
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def metadata(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size,
        'mode': stat.S_IMODE(info.st_mode), 'observed_uid': info.st_uid, 'observed_gid': info.st_gid,
        'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def bounded_file(root, relative, maximum):
    parent, name = relative.rsplit('/', 1)
    parent_fd = directory(root, parent)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_dev != os.fstat(root).st_dev or not 0 < info.st_size <= maximum:
                raise ValueError('bounded regular guest file required')
            data = os.read(fd, maximum + 1)
            if len(data) != info.st_size or len(data) > maximum:
                raise ValueError('guest file changed or exceeded bound')
            return data
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def write_probe(root, relative, name):
    parent = directory(root, relative)
    fd = None
    created = None
    try:
        before = metadata(os.fstat(parent))
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        created = os.fstat(fd)
        payload = b'Gen2 disposable guest-volume access control; no account or app.\n'
        if os.write(fd, payload) != len(payload):
            raise OSError('partial probe write')
        os.fsync(fd)
        os.fchmod(fd, 0o600)
        os.lseek(fd, 0, os.SEEK_SET)
        if os.read(fd, len(payload) + 1) != payload or stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
            raise ValueError('guest write/read/mode control failed')
        return {'directory': relative, 'parent': before, 'probe': metadata(os.fstat(fd)),
            'payload': {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}, 'probe_removed': True}
    finally:
        if fd is not None:
            os.close(fd)
        try:
            if created is not None:
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if current.st_dev != created.st_dev or current.st_ino != created.st_ino or not stat.S_ISREG(current.st_mode):
                    raise ValueError('probe entry replaced; refusing removal')
                os.unlink(name, dir_fd=parent)
                os.fsync(parent)
        finally:
            os.close(parent)


def existing_write_open(root, relative):
    parent, name = relative.rsplit('/', 1)
    parent_fd = directory(root, parent)
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_dev != os.fstat(root).st_dev:
            raise ValueError('same-mount regular guest file required')
        fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            if metadata(os.fstat(fd)) != metadata(before):
                raise ValueError('existing guest file changed before open')
        finally:
            os.close(fd)
        if metadata(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != metadata(before):
            raise ValueError('existing guest file changed during non-writing open')
        return {'path': relative, 'metadata': metadata(before), 'write_open_without_write': True}
    finally:
        os.close(parent_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not re.fullmatch(r'gen2-data-probe-[0-9a-f]{32}', config['probe_name']):
        raise ValueError('task-only exclusive probe basename required')
    roots = {}
    report = {'schema_version': 1, 'stored_guest_ownership_verified': False,
        'provisioning_performed': False, 'access_passed': False, 'writes': [],
        'existing_write_opens': [], 'directory_metadata': {}, 'public_host_key': {'present': False}}
    try:
        for role in ['system', 'data']:
            mount = Path(config[role])
            fd = os.open(mount, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            roots[role] = fd
            readonly = bool(os.fstatvfs(fd).f_flag & os.ST_RDONLY)
            if (not mount.is_mount() or os.fstat(fd).st_dev != config[role + '_device']
                    or mount.stat().st_dev != os.fstat(fd).st_dev
                    or readonly is not (role == 'system') or os.fstat(fd).st_dev == mount.parent.stat().st_dev):
                raise ValueError('exact separately mounted SystemRO/DataRW required')
        if os.fstat(roots['system']).st_dev == os.fstat(roots['data']).st_dev:
            raise ValueError('distinct System and Data filesystems required')
        report['step'] = 'SystemVersion'
        version_data = bounded_file(roots['system'], 'System/Library/CoreServices/SystemVersion.plist', 1024**2)
        with Path(config['version_output']).open('xb') as stream:
            stream.write(version_data)
            stream.flush()
            os.fsync(stream.fileno())
        report['system_version_file'] = {'bytes': len(version_data), 'sha256': hashlib.sha256(version_data).hexdigest()}
        version = plistlib.loads(version_data)
        if not isinstance(version, dict):
            raise ValueError('guest SystemVersion dictionary required')
        for source_key, report_key in [('ProductVersion', 'product_version'), ('ProductBuildVersion', 'product_build')]:
            value = version.get(source_key)
            report[report_key] = value if isinstance(value, str) and len(value) <= 128 else repr(value)[:128]
        raw_version = version['ProductVersion']
        if not isinstance(raw_version, str) or not re.fullmatch(r'[0-9]+\.[0-9]+(?:\.[0-9]+)?', raw_version):
            raise ValueError('actual numeric guest product version required')
        components = [int(part) for part in raw_version.split('.')]
        components += [0] * (3 - len(components))
        if components != config['product_version_components'] or version['ProductBuildVersion'] != config['product_build']:
            raise ValueError('actual guest version/build differs from pinned restore')
        directories = ['private/var/db', 'private/var/db/dslocal/nodes/Default/users',
            'private/var/db/dslocal/nodes/Default/groups', 'private/var/db/com.apple.xpc.launchd',
            'Library/Preferences', 'Users', 'private/etc']
        for relative in directories:
            report['step'] = 'directory metadata: ' + relative
            fd = directory(roots['data'], relative)
            try:
                report['directory_metadata'][relative] = metadata(os.fstat(fd))
            finally:
                os.close(fd)
        for relative in directories:
            report['step'] = 'write probe: ' + relative
            report['writes'].append(write_probe(roots['data'], relative, config['probe_name']))
        for relative in ['private/var/db/dslocal/nodes/Default/users/root.plist', 'private/var/db/dslocal/nodes/Default/groups/admin.plist']:
            report['step'] = 'existing non-writing open: ' + relative
            report['existing_write_opens'].append(existing_write_open(roots['data'], relative))
        report['step'] = 'optional public host key'
        try:
            key = bounded_file(roots['data'], 'private/etc/ssh/ssh_host_ed25519_key.pub', 8192)
        except FileNotFoundError:
            pass
        else:
            fields = key.decode('ascii').strip().split()
            if (len(key.strip().splitlines()) != 1 or len(fields) < 2 or fields[0] != 'ssh-ed25519'
                    or len(base64.b64decode(fields[1], validate=True)) != 51
                    or not base64.b64decode(fields[1], validate=True).startswith(b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20')):
                raise ValueError('actual observed public host-key format required')
            report['public_host_key'] = {'present': True, 'bytes': len(key), 'sha256': hashlib.sha256(key).hexdigest(),
                'public_key': ' '.join(fields[:2])}
        report['access_passed'] = True
        report['step'] = 'completed'
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        report['access_passed'] = False
        report['error'] = repr(error)
        print(json.dumps(report, sort_keys=True), flush=True)
        raise
    finally:
        for fd in roots.values():
            os.close(fd)


if __name__ == '__main__':
    main()
