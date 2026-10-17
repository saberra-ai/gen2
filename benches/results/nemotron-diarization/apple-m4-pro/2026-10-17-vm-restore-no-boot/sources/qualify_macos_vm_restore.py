"""Acquire a pinned public IPSW and validate a fresh sparse VM seed without a VM.

No installer, guest, mount, application/model invocation or custom listener.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
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
            raise ValueError('restore protocol changed')
        for name, want in p['tools'].items():
            if descriptor(tools / name) != want:
                raise ValueError('restore source changed: ' + name)
        for name, want in p['external_files'].items():
            if descriptor(name) != want:
                raise ValueError('restore prerequisite/tool/SDK metadata changed: ' + name)

    frozen()
    from frozen_git_sources import verify_git_sources
    from qualification_storage import StorageGuard, atomic_json
    from qualify_macos_vm_host_v4 import execute_child, policy
    from macos_vm_restore_metrics import image_metrics, seed_metrics
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    verify_git_sources(repo, source, {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()})
    if raw != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-vm-restore.protocol.json'], cwd=repo):
        raise ValueError('committed restore protocol required')
    prior = Path(p['host_prerequisite']['root'])
    manifest = json.loads((prior / 'collection.json').read_text())
    for name, want in manifest['files'].items():
        if descriptor(prior / name) != want:
            raise ValueError('prior host receipt member changed: ' + name)
    pq = json.loads((prior / 'qualification.json').read_text())
    summary = json.loads((prior / 'collection-qualification.json').read_text())
    if (pq['source_commit'] != p['host_prerequisite']['source_commit']
            or pq['host_prerequisites_passed'] is not True or pq['owned_image_detached'] is not True
            or pq['storage']['released'] is not True or summary['host_prerequisites_passed'] is not True):
        raise ValueError('passing actual host prerequisite required')
    parent = Path(p['output_parent']).resolve(strict=True)
    if Path(p['output_name']).name != p['output_name']:
        raise ValueError('single fresh output name required')
    output = parent / p['output_name']
    if output.exists() or output.is_symlink():
        raise ValueError('fresh restore output required; no automatic retry')
    output.mkdir(mode=0o700)
    (output / 'protocol.json').write_bytes(raw)
    q = {'schema_version': 1, 'source_commit': source, 'protocol': descriptor(args.protocol),
        'no_boot_restore_passed': False, 'fresh_machine_qualified': False, 'release_qualified': False,
        'virtual_machine_created': False, 'installation_started': False, 'guest_booted': False,
        'models_or_applications_executed': False, 'commands': [], 'download': {}, 'scope': __doc__,
        'prior_host_receipt_members_verified': len(manifest['files'])}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    guard = None
    sandbox = output / 'offline.sb'
    home, temporary = output / 'home', output / 'tmp'
    environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home),
        'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}

    def run(name, arguments, *, network_download=False):
        frozen()
        if guard.check() < guard.running_minimum:
            raise OSError('restore child running storage floor exceeded')
        argv = list(map(str, arguments))
        if not network_download:
            argv = ['/usr/bin/sandbox-exec', '-f', str(sandbox), *argv]
        row = {'name': name, 'argv': argv,
            'cwd': str(output), 'environment': environment}
        q['commands'].append(row)
        persist()
        started = time.monotonic()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_child(row['argv'], cwd=output, environment=environment, record=row, persist=persist,
                timeout=(p['download_limits']['deadline_seconds'] + 30 if network_download
                    else p['command_timeout_seconds']), guard=guard, stdout=stdout, stderr=stderr)
        row['wall_seconds'] = time.monotonic() - started
        persist()
        if row['returncode'] != 0:
            raise ValueError('restore child failed: ' + name)
        return (output / (name + '.stdout.log')).read_bytes()

    try:
        guard = StorageGuard(output, **p['storage'])
        q['storage'] = guard.record
        home.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
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
        executable = output / 'vm-restore-probe'
        run('build', [p['swiftc'], '-sdk', p['sdk'], '-target', 'arm64-apple-macosx26.0',
            '-swift-version', '5', '-warnings-as-errors', '-j', '2', '-module-cache-path', output / 'module-cache',
            tools / 'macos_vm_restore_probe.swift', '-o', executable])
        run('sign', ['/usr/bin/codesign', '--force', '--sign', '-', '--timestamp=none', '--options', 'runtime',
            '--entitlements', tools / 'macos_vm_host.entitlements', executable])
        run('signature-verify', ['/usr/bin/codesign', '--verify', '--strict', '--deep', executable])
        run('signature-metadata', ['/usr/bin/codesign', '--display', '--verbose=4', executable])
        q['entitlements'] = plistlib.loads(run('entitlements', ['/usr/bin/codesign', '--display', '--xml', '--entitlements', '-', executable]))
        if q['entitlements'] != {'com.apple.security.virtualization': True}:
            raise ValueError('only unrestricted virtualization entitlement allowed')
        signature = (output / 'signature-metadata.stderr.log').read_text()
        if 'Signature=adhoc' not in signature or not re.search(r'flags=0x[0-9a-f]+\([^)]*runtime', signature):
            raise ValueError('actual ad-hoc hardened signature required')
        if run('architecture', ['/usr/bin/lipo', '-archs', executable]).strip() != b'arm64':
            raise ValueError('arm64 probe required')
        if not re.search(r'\bminos 26\.0\s', run('deployment', ['/usr/bin/otool', '-l', executable]).decode()):
            raise ValueError('macOS26.0 deployment required')
        run('dependencies', ['/usr/bin/otool', '-L', executable])
        q['probe_executable'] = descriptor(executable)
        # Only this separately bounded acquisition worker has network access.
        try:
            run('download', [p['python'], '-B', tools / 'macos_restore_download.py',
                '--protocol', args.protocol.resolve(), '--output', output], network_download=True)
        finally:
            if (output / 'download.json').is_file():
                q['download'] = json.loads((output / 'download.json').read_text())
        if q['download'].get('complete') is not True or q['download'].get('actual') != {
                key: p['restore_image'][key] for key in ['bytes', 'sha256']}:
            raise ValueError('complete hash-verified download required')
        image = output / p['restore_image']['name']
        frozen()
        q['image'] = image_metrics(json.loads(run('inspect', [executable, 'inspect', image])), operation='inspect')
        prepared = json.loads(run('prepare', [executable, 'prepare', image, output / 'vm-seed']))
        q['prepared'] = prepared
        q['seed'] = seed_metrics(output / 'vm-seed', prepared, q['image'], uid=os.getuid())
        frozen()
        q['no_boot_restore_passed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        q['no_boot_restore_passed'] = False
        raise
    finally:
        if guard is not None:
            guard.release()
            q['storage'] = guard.record
        persist()


if __name__ == '__main__':
    main()
