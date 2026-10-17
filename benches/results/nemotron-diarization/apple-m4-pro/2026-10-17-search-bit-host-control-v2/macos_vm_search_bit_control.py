"""Host-only ordinary-descriptor search-bit control on a fresh owned fixture.

The frozen command selects a fresh host fixture; existing paths are rejected.
No directory contents or account records are read. No guest operation is admitted.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import platform
import signal
import stat

from macos_vm_permissions_probe import metadata, security_at


CHILDREN = ('users', 'groups')


def stable_identity(info):
    """Mode and ctime necessarily change; all other retained fields must agree."""
    return {key: value for key, value in metadata(info).items()
        if key not in {'permissions', 'full_mode', 'ctime_ns'}}


def control(root, report, *, query=security_at, changer=os.fchmod, observe=None):
    root = Path(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    # Exclusive creation makes every changed inode a newly created host fixture.
    root.mkdir(mode=0o700)
    parent = target = None
    attempted = False
    original = security = handlers = None
    deferred_signals = []
    report.update(control_passed=False, restoration_confirmed=False,
        guest_access_qualified=False, acl_policy_qualified=False,
        directory_contents_read=False, account_content_read=False,
        permission_operation='ordinary O_RDONLY|O_DIRECTORY|O_NOFOLLOW descriptor; fchmod')
    try:
        parent = os.open(root, flags)
        if (metadata(root.lstat()) != metadata(os.fstat(parent))
                or os.fstat(parent).st_uid != os.geteuid()
                or stat.S_IMODE(os.fstat(parent).st_mode) != 0o700
                or os.fstatvfs(parent).f_flag & os.ST_RDONLY):
            raise ValueError('fresh owned writable host fixture required')
        os.mkdir('Default', mode=0o700, dir_fd=parent)
        target = os.open('Default', flags, dir_fd=parent)
        for name in CHILDREN:
            os.mkdir(name, mode=0o600, dir_fd=target)
        os.fchmod(target, 0o600)
        os.close(target)
        target = None
        root_before = metadata(os.fstat(parent))
        if metadata(root.lstat()) != root_before:
            raise ValueError('fresh fixture root path changed')
        report['root_before'] = root_before
        before = os.stat('Default', dir_fd=parent, follow_symlinks=False)
        original = metadata(before)
        report['before'] = original
        if (not stat.S_ISDIR(before.st_mode) or original['permissions'] != 0o600
                or original['observed_uid'] != os.geteuid() or original['flags'] not in {0, None}):
            raise ValueError('exact owned mode0600 directory without flags required')
        security = query(parent, 'Default')
        report['security_before'] = security
        if security.get('status') != 'ok':
            raise ValueError('retained actual security response required')
        target = os.open('Default', flags, dir_fd=parent)
        if metadata(os.fstat(target)) != original:
            raise ValueError('ordinary descriptor must identify the exact inspected directory')
        report['denials_before'] = {}
        for name in CHILDREN:
            try:
                os.stat(name, dir_fd=target, follow_symlinks=False)
            except PermissionError as error:
                if error.errno != errno.EACCES:
                    raise
                report['denials_before'][name] = error.errno
            else:
                raise ValueError('mode0600 must deny child lookup in selected runtime')
        if metadata(os.stat('Default', dir_fd=parent, follow_symlinks=False)) != original:
            raise ValueError('fixture changed before descriptor-bound operation')
        # OS masks affect only this thread. Python can dispatch a handler for
        # a signal delivered to a different thread, so defer in the handlers.
        handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        def defer(number, frame):
            deferred_signals.append(number)
        for number in handlers:
            signal.signal(number, defer)
        attempted = True  # Restore even if the mutating call raises after changing mode.
        changer(target, 0o700)
        after = os.fstat(target)
        report['during'] = metadata(after)
        if stable_identity(after) != stable_identity(before) or stat.S_IMODE(after.st_mode) != 0o700:
            raise ValueError('only owner search bit may change')
        report['security_during'] = query(parent, 'Default')
        if report['security_during'] != security:
            raise ValueError('search-bit operation changed retained security metadata')
        report['children'] = {}
        for name in CHILDREN:
            info = os.stat(name, dir_fd=target, follow_symlinks=False)
            report['children'][name] = metadata(info)
            if not stat.S_ISDIR(info.st_mode) or info.st_dev != before.st_dev or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('same-device unchanged child directory metadata required')
        if observe is not None:
            observe(target)
    finally:
        try:
            if attempted:
                try:
                    current = os.fstat(target)
                    if (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode)) != (
                            before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode)):
                        raise ValueError('changed descriptor identity; restoration unconfirmed')
                    changer(target, original['permissions'])
                    restored = os.fstat(target)
                    report['restored'] = metadata(restored)
                    if metadata(os.stat('Default', dir_fd=parent, follow_symlinks=False)) != metadata(restored):
                        raise ValueError('restored inode no longer occupies inspected fixture path')
                    report['root_restored'] = metadata(os.fstat(parent))
                    if report['root_restored'] != root_before or metadata(root.lstat()) != root_before:
                        raise ValueError('fixture root no longer occupies original path')
                    report['security_restored'] = query(parent, 'Default')
                    if (stable_identity(restored) != stable_identity(before)
                            or stat.S_IMODE(restored.st_mode) != original['permissions']
                            or report['security_restored'] != security):
                        raise ValueError('exact mode, identity and retained security restoration required')
                    report['ctime_changed'] = restored.st_ctime_ns != before.st_ctime_ns
                    report['restoration_confirmed'] = True
                except BaseException as error:
                    report['restoration_error'] = repr(error)
                    raise
        finally:
            try:
                for fd in (target, parent):
                    if fd is not None:
                        os.close(fd)
            finally:
                if handlers is not None:
                    try:
                        signal.signal(signal.SIGINT, handlers[signal.SIGINT])
                    finally:
                        signal.signal(signal.SIGTERM, handlers[signal.SIGTERM])
                    for number in deferred_signals:
                        signal.raise_signal(number)
    report['control_passed'] = report['restoration_confirmed']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or os.geteuid() == 0:
        parser.error('non-root Darwin arm64 runtime required')
    report = {'schema_version': 1, 'host_only': True, 'runtime': os.sys.executable,
        'python_version': platform.python_version()}
    def interrupted(number, frame):
        raise InterruptedError('host control signal ' + str(number))
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        control(args.control, report)
    except BaseException as error:
        report['control_passed'] = False
        report['error'] = repr(error)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
