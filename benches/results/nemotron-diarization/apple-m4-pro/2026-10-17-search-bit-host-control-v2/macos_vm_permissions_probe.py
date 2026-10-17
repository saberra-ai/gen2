"""Read-only, parent-relative inspection of known stopped-guest directories.

No account content, directory enumeration, host keys, permission repair or
provisioning. Returned ACL metadata is evidence, never an access-policy verdict.
"""
import argparse
import base64
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct


CAPACITY = 65536
TARGETS = ['private/var/db/dslocal/nodes/Default/users', 'private/var/db/dslocal/nodes/Default/groups',
    'private/var/db/com.apple.xpc.launchd', 'Library/Preferences', 'Users', 'private/etc']


class AttrList(ctypes.Structure):
    _fields_ = [('bitmapcount', ctypes.c_uint16), ('reserved', ctypes.c_uint16),
        ('commonattr', ctypes.c_uint32), ('volattr', ctypes.c_uint32), ('dirattr', ctypes.c_uint32),
        ('fileattr', ctypes.c_uint32), ('forkattr', ctypes.c_uint32)]


def packed_security(raw):
    """Check the one-variable-attribute ABI, including silent-truncation hazards."""
    if not 12 <= len(raw) <= CAPACITY:
        raise ValueError('bounded attribute buffer required')
    size, offset, length = struct.unpack_from('<IiI', raw)
    start = 4 + offset  # Relative to attrreference at byte4, not the buffer base.
    null_reference = offset == 0 and length == 0
    if (size != len(raw) or (not null_reference and (offset < 8 or offset % 4 or start > size or length > size - start))):
        raise ValueError('untruncated aligned attribute reference required')
    payload = b'' if null_reference else raw[start:start + length]
    result = {'bytes': size, 'sha256': hashlib.sha256(raw).hexdigest(),
        'raw_base64': base64.b64encode(raw).decode('ascii'), 'attribute_bytes': length,
        'attribute_offset': None if null_reference else start, 'null_reference': null_reference,
        'acl_policy_qualified': False, 'recognized_filesec': False}
    known_magic = len(payload) >= 4 and struct.unpack_from('<I', payload)[0] == 0x012CC16D
    if known_magic and len(payload) < 44:
        raise ValueError('truncated kauth_filesec header')
    if known_magic:
        count, flags = struct.unpack_from('<II', payload, 36)
        entries = 0 if count == 0xFFFFFFFF else count
        if entries > 128 or len(payload) != 44 + 24 * entries:
            raise ValueError('complete bounded kauth_filesec entry layout required')
        result.update(recognized_filesec=True, acl_entrycount=count, acl_flags=flags)
    result['canonical_empty'] = size == 12 and offset == 8 and length == 0
    result['abi_shape_qualified'] = result['canonical_empty'] or result['recognized_filesec']
    return result


def security_at(parent, name):
    if name in {'', '.', '..'} or not re.fullmatch(r'[A-Za-z0-9._-]+', name):
        raise ValueError('one literal anchored leaf required')
    if ctypes.sizeof(AttrList) != 24 or ctypes.sizeof(ctypes.c_ulong) != 8:
        raise ValueError('pinned Darwin arm64 ABI required')
    library = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    function = library.getattrlistat
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(AttrList), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
    function.restype = ctypes.c_int
    attributes = AttrList(5, 0, 0x00400000, 0, 0, 0, 0)  # ATTR_CMN_EXTENDED_SECURITY
    buffer = ctypes.create_string_buffer(CAPACITY)
    ctypes.set_errno(0)
    result = function(parent, os.fsencode(name), ctypes.byref(attributes), buffer, CAPACITY, 1)  # FSOPT_NOFOLLOW
    if result != 0:
        error = ctypes.get_errno()
        return {'status': 'query_error', 'returncode': result, 'errno': error, 'message': os.strerror(error),
            'acl_policy_qualified': False}
    size = struct.unpack_from('<I', buffer.raw)[0]
    retained = buffer.raw[:min(size, CAPACITY)]
    try:
        decoded = packed_security(retained)
    except ValueError as error:
        return {'status': 'unparsed', 'returncode': 0, 'reported_bytes': size,
            'bytes': len(retained), 'sha256': hashlib.sha256(retained).hexdigest(),
            'raw_base64': base64.b64encode(retained).decode('ascii'), 'error': str(error), 'acl_policy_qualified': False}
    return {'status': 'ok' if decoded['abi_shape_qualified'] else 'unparsed', 'returncode': 0, **decoded}


def metadata(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'full_mode': info.st_mode,
        'permissions': stat.S_IMODE(info.st_mode), 'file_type': stat.S_IFMT(info.st_mode), 'links': info.st_nlink,
        'observed_uid': info.st_uid, 'observed_gid': info.st_gid, 'flags': getattr(info, 'st_flags', None),
        'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def inspect_paths(root, query=security_at, opener=os.open, rows=None):
    paths = set()
    for target in TARGETS:
        parts = target.split('/')
        paths.update('/'.join(parts[:index]) for index in range(1, len(parts) + 1))
    descriptors = {'': os.dup(root)}
    rows = {} if rows is None else rows
    root_device = os.fstat(root).st_dev
    try:
        for path in sorted(paths, key=lambda item: (item.count('/'), item)):
            parent, separator, name = path.rpartition('/')
            if not separator:
                parent, name = '', path
            if parent not in descriptors:
                rows[path] = {'status': 'parent_not_opened', 'parent': parent}
                continue
            parent_fd = descriptors[parent]
            try:
                before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.ENOENT}:
                    raise
                rows[path] = {'status': 'stat_error', 'errno': error.errno, 'message': str(error)}
                continue
            row = rows[path] = {'status': 'metadata_observed', 'stat': metadata(before)}
            if not stat.S_ISDIR(before.st_mode) or before.st_dev != root_device:
                raise ValueError('only same-mount literal guest directories permitted: ' + path)
            row['security'] = query(parent_fd, name)
            if metadata(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != metadata(before):
                raise ValueError('entry changed during security metadata query: ' + path)
            try:
                fd = opener(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.ENOENT}:
                    raise
                row['open'] = {'success': False, 'errno': error.errno, 'message': str(error)}
                if metadata(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != metadata(before):
                    raise ValueError('denied entry changed: ' + path)
                continue
            descriptors[path] = fd
            if metadata(os.fstat(fd)) != metadata(before):
                raise ValueError('opened node differs from inspected node: ' + path)
            row['open'] = {'success': True}
        return rows
    finally:
        for fd in descriptors.values():
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    if (args.config is None) == (args.control is None):
        parser.error('exactly one control or guest config required')
    if args.control is not None:
        args.control.mkdir(mode=0o700)
        fd = os.open(args.control.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            result = security_at(fd, args.control.name)
        finally:
            os.close(fd)
        print(json.dumps({'attrlist_bytes': ctypes.sizeof(AttrList), 'capacity': CAPACITY, 'result': result}, sort_keys=True))
        if result['status'] != 'ok':
            raise ValueError('bounded actual getattrlistat ABI control required')
        return
    config = json.loads(args.config.read_text())
    mount = Path(config['data_mount'])
    fd = os.open(mount, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    report = {'schema_version': 1, 'inspection_passed': False, 'permission_changes': 0,
        'directory_contents_read': False, 'account_content_read': False, 'data_access_passed': False,
        'provisioning_performed': False, 'stored_guest_ownership_verified': False}
    try:
        if (not mount.is_mount() or os.fstat(fd).st_dev != config['data_device']
                or mount.stat().st_dev != os.fstat(fd).st_dev
                or os.fstat(fd).st_dev == mount.parent.stat().st_dev
                or not os.fstatvfs(fd).f_flag & os.ST_RDONLY):
            raise ValueError('independently validated actual DataRO mount required')
        report['nodes'] = {}
        inspect_paths(fd, rows=report['nodes'])
        users = report['nodes']['private/var/db/dslocal/nodes/Default/users']
        if users.get('status') != 'metadata_observed':
            raise ValueError('parent-relative users metadata not reached; retain earlier denial')
        report['inspection_passed'] = True
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        report['error'] = repr(error)
        print(json.dumps(report, sort_keys=True), flush=True)
        raise
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
