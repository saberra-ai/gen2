"""Build/run a host-only Apple VZ probe and mount one fresh blank APFS image.

No restore download, VM creation/boot, guest provisioning or app/model execution.
"""
import argparse
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
        while raw := stream.read(1024 * 1024):
            digest.update(raw)
            size += len(raw)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def policy(denied):
    quote = lambda path: json.dumps(str(path), ensure_ascii=False)
    return ('(version 1)\n(allow default)\n(deny network*)\n'
        + '(deny file-read* ' + ' '.join('(subpath ' + quote(p) + ')' for p in denied) + ')\n'
        + '(deny file-read* (regex #".*\\.(nemo|gguf|safetensors|onnx|npy|f32)$"))\n'
        + '(deny file-write* (subpath ' + quote(Path(__file__).resolve().parents[2]) + '))\n'
        + '(deny file-write* ' + ' '.join('(subpath ' + quote(p) + ')' for p in denied) + ')\n')


def execute_child(argv, *, cwd, environment, record, persist, timeout, guard=None, stdout=None, stderr=None):
    """Cover spawn/PID persistence/wait with one kill-and-reap boundary.

    Cleanup uses capture in memory, independent of storage and metadata writes.
    """
    process = None
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment,
            stdout=subprocess.PIPE if guard is None else stdout,
            stderr=subprocess.PIPE if guard is None else stderr, start_new_session=True)
        record['pid'] = process.pid
        if guard is None:
            captured = process.communicate(timeout=timeout)
            record['returncode'] = process.returncode
            return captured
        persist()
        record['returncode'] = guard.wait(process, record, timeout)
    except BaseException as error:
        if process is None:
            record['startup_error'] = repr(error)
        else:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    record['forced_process_group_termination'] = True
                except ProcessLookupError:
                    pass
            record['returncode'] = process.wait()
        if guard is not None:
            guard.release()
        raise


def cleanup_metadata(action, qualification):
    """Diagnostic writes must not prevent exact-image cleanup."""
    try:
        action()
    except BaseException as error:
        qualification.setdefault('cleanup_metadata_errors', []).append(repr(error))


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
            raise ValueError('host admission protocol changed')
        for name, expected in p['tools'].items():
            if descriptor(tools / name) != expected:
                raise ValueError('host admission source changed: ' + name)
        for name, expected in p['external_files'].items():
            if descriptor(name) != expected:
                raise ValueError('host admission tool/SDK/source metadata changed: ' + name)

    frozen()
    from frozen_git_sources import verify_git_sources
    from qualification_storage import StorageGuard, atomic_json
    from macos_vm_host_metrics import attached_image_device, apfs_probe_volume, probe_metrics
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    verify_git_sources(repo, source, {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()})
    if raw != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-vm-host.protocol.json'], cwd=repo):
        raise ValueError('committed host admission protocol required')
    parent = Path(p['output_parent']).resolve(strict=True)
    if Path(p['output_name']).name != p['output_name']:
        raise ValueError('single output directory name required')
    output = parent / p['output_name']
    if output.exists() or output.is_symlink():
        raise ValueError('fresh host admission output required')
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw)
    q = {'schema_version': 1, 'source_commit': source, 'protocol': descriptor(args.protocol),
         'host_prerequisites_passed': False, 'fresh_machine_qualified': False, 'release_qualified': False,
         'restore_image_downloaded': False, 'virtual_machine_created': False, 'guest_booted': False,
         'models_or_applications_executed': False, 'commands': [], 'scope': __doc__}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    guard = None
    image = output / 'blank-apfs.dmg'
    mount = output / 'mount'
    sandbox = output / 'offline.sb'
    home, temporary = output / 'home', output / 'tmp'
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}

    def run(name, arguments, *, sandboxed=True, cleanup=False):
        if not cleanup:
            frozen()
            guard.before_model()
        argv = list(map(str, arguments))
        if sandboxed:
            argv = ['/usr/bin/sandbox-exec', '-f', str(sandbox), *argv]
        row = {'name': name, 'argv': argv, 'cwd': str(output), 'environment': env}
        q['commands'].append(row)
        started = time.monotonic()
        if cleanup:
            # No source/storage/persistence/log-file admission before detach.
            captured = execute_child(argv, cwd=output, environment=env, record=row,
                persist=persist, timeout=p['command_timeout_seconds'])
            for label, data in zip(['stdout', 'stderr'], captured):
                cleanup_metadata(lambda: (output / (name + '.' + label + '.log')).write_bytes(data), q)
            result = captured[0]
        else:
            persist()
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                execute_child(argv, cwd=output, environment=env, record=row, persist=persist,
                    timeout=p['command_timeout_seconds'], guard=guard, stdout=stdout, stderr=stderr)
            result = (output / (name + '.stdout.log')).read_bytes()
        row['wall_seconds'] = time.monotonic() - started
        if cleanup:
            cleanup_metadata(persist, q)
        else:
            persist()
        if row['returncode'] != 0:
            raise ValueError('host admission child failed: ' + name)
        return result

    def own_attachment(name, *, cleanup=False):
        value = plistlib.loads(run(name, ['/usr/bin/hdiutil', 'info', '-plist'], cleanup=cleanup))
        device = attached_image_device(value, image)
        write = lambda: atomic_json(output / (name + '.own-image.json'), {'image': str(image), 'device': device,
            'matching_images': [v for v in value.get('images', []) if v.get('image-path') == str(image)]})
        if cleanup:
            cleanup_metadata(write, q)
        else:
            write()
        return device

    try:
        guard = StorageGuard(output, **p['storage'])
        home.mkdir()
        temporary.mkdir()
        mount.mkdir()
        sandbox.write_text(policy(p['denied_paths']))
        q['host'] = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                     'platform': platform.platform(), 'uid': os.getuid()}
        if q['host']['cpu'] != p['expected_cpu']:
            raise ValueError('named physical host required')
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for path in probes:
            path.write_bytes(b'not a model')
        control = ('import pathlib,socket,sys\n'
            'for name in sys.argv[1:]:\n'
            ' try:pathlib.Path(name).read_bytes()\n'
            ' except PermissionError:pass\n'
            ' else:raise RuntimeError("model read allowed")\n'
            's=socket.socket()\n'
            'try:s.connect(("127.0.0.1",9))\n'
            'except PermissionError:pass\n'
            'else:raise RuntimeError("network allowed")\n')
        run('offline-controls', ['/usr/bin/python3', '-c', control, *probes])
        executable = output / 'vm-host-probe'
        run('build', [p['swiftc'], '-sdk', p['sdk'], '-target', 'arm64-apple-macosx26.0', '-warnings-as-errors',
                      '-j', '2', '-module-cache-path', output / 'module-cache', tools / 'macos_vm_host_probe.swift', '-o', executable])
        run('sign', ['/usr/bin/codesign', '--force', '--sign', '-', '--timestamp=none', '--options', 'runtime',
                     '--entitlements', tools / 'macos_vm_host.entitlements', executable])
        run('signature-verify', ['/usr/bin/codesign', '--verify', '--strict', '--deep', executable])
        run('signature-metadata', ['/usr/bin/codesign', '--display', '--verbose=4', executable])
        entitlements = plistlib.loads(run('entitlements', ['/usr/bin/codesign', '--display', '--entitlements', '-', executable]))
        if entitlements != {'com.apple.security.virtualization': True}:
            raise ValueError('only the unrestricted virtualization entitlement allowed')
        signature = (output / 'signature-metadata.stderr.log').read_text()
        if 'Signature=adhoc' not in signature or not re.search(r'flags=0x[0-9a-f]+\([^)]*runtime', signature):
            raise ValueError('actual ad-hoc hardened signature required')
        if run('architecture', ['/usr/bin/lipo', '-archs', executable]).strip() != b'arm64':
            raise ValueError('actual arm64 probe required')
        load = run('deployment', ['/usr/bin/otool', '-l', executable]).decode()
        if not re.search(r'\bminos 26\.0\s', load):
            raise ValueError('fixed macOS26.0 deployment required')
        run('dependencies', ['/usr/bin/otool', '-L', executable])
        q['probe'] = probe_metrics(json.loads(run('host-probe', [executable, 'probe'])))
        q['probe_executable'] = descriptor(executable)
        q['entitlements'] = entitlements
        if own_attachment('before-image') is not None:
            raise ValueError('fresh image unexpectedly attached')
        run('create-image', ['/usr/bin/hdiutil', 'create', '-size', '256m', '-layout', 'GPTSPUD', '-fs', 'APFS',
                             '-volname', 'Gen2VMHostProbe', '-type', 'UDIF', '-nospotlight', image])
        if image.is_symlink() or not image.is_file() or image.stat().st_size > 300 * 1024**2:
            raise ValueError('bounded fresh regular image required')
        run('attach-image', ['/usr/bin/hdiutil', 'attach', '-nomount', '-readwrite', '-nobrowse', '-plist', image])
        whole = own_attachment('attached-image')
        if whole is None:
            raise ValueError('exact new image attachment missing')
        partition = plistlib.loads(run('partition-info', ['/usr/sbin/diskutil', 'info', '-plist', whole + 's2']))
        container = partition['APFSContainerReference']
        listing = plistlib.loads(run('apfs-list', ['/usr/sbin/diskutil', 'apfs', 'list', '-plist', container]))
        volume = apfs_probe_volume(listing, whole, container)
        if own_attachment('before-mount') != whole:
            raise ValueError('owned image changed before mount')
        run('mount-volume', ['/usr/sbin/diskutil', 'mount', 'nobrowse', '-mountPoint', mount, volume])
        info = plistlib.loads(run('mounted-info', ['/usr/sbin/diskutil', 'info', '-plist', volume]))
        if info['MountPoint'] != str(mount) or info['Mounted'] is not True or not mount.is_mount():
            raise ValueError('actual exact task mount required before write')
        probe = mount / 'gen2-vm-host-write-control'
        probe.mkdir(mode=0o700)
        written = probe / 'control.txt'
        payload = b'Task-owned blank APFS control; no OS, app or model.\n'
        with written.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        written.chmod(0o600)
        if written.read_bytes() != payload or written.stat().st_mode & 0o777 != 0o600:
            raise ValueError('mounted image write/read/mode control failed')
        q['blank_volume'] = {'whole_device': whole, 'container': container, 'volume': volume,
            'mountpoint': str(mount), 'write_control': descriptor(written), 'mode': 0o600,
            'owners_enabled': info.get('OwnersEnabled'), 'filesystem': info.get('FilesystemType')}
        persist()
        frozen()
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        try:
            if guard is not None and sandbox.exists():
                # Free the physical reserve before cleanup writes, including ENOSPC failures.
                cleanup_metadata(guard.release, q)
                whole = own_attachment('cleanup-image', cleanup=True)
                if whole is not None:
                    run('detach-image', ['/usr/bin/hdiutil', 'detach', whole], cleanup=True)
                if own_attachment('after-image', cleanup=True) is not None or mount.is_mount():
                    raise ValueError('task image or mount remains attached')
                q['owned_image_detached'] = True
                if image.is_file():
                    q['detached_image'] = descriptor(image)
                q['host_prerequisites_passed'] = ('error' not in q and 'cleanup_metadata_errors' not in q
                    and 'blank_volume' in q)
        except BaseException as error:
            q['cleanup_error'] = repr(error)
            q['host_prerequisites_passed'] = False
            raise
        finally:
            if guard is not None:
                guard.release()
                q['storage'] = guard.record
            persist()
    if q.get('cleanup_metadata_errors'):
        raise ValueError('cleanup diagnostic write failed; see qualification record')


if __name__ == '__main__':
    main()
