"""Freeze metadata only for pinned IPSW acquisition and no-boot configuration."""
from pathlib import Path
import hashlib
import json
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')

def descriptor(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}

prior = json.loads((tools / 'macos-vm-host-v4.protocol.json').read_text())
for name, want in prior['tools'].items():
    assert descriptor(tools / name) == want
for name, want in prior['external_files'].items():
    assert descriptor(name) == want
receipt = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-prerequisites'
manifest = json.loads((receipt / 'collection.json').read_text())
for name, want in manifest['files'].items():
    assert descriptor(receipt / name) == want
assert json.loads((receipt / 'collection-qualification.json').read_text())['host_prerequisites_passed'] is True
names = ['qualify_macos_vm_restore.py', 'macos_restore_download.py', 'macos_vm_restore_metrics.py',
    'macos_vm_restore_probe.swift', 'test_macos_restore_download.py', 'test_macos_vm_restore_metrics.py',
    'qualify_macos_vm_host_v4.py', 'qualification_storage.py', 'frozen_git_sources.py',
    'macos_vm_host.entitlements', 'macos_vm_apple_LICENSE.txt', 'test_macos_vm_host_cleanup.py']
external = dict(prior['external_files'])
sdk = Path(prior['sdk'])
headers = ['VZMacOSRestoreImage', 'VZMacOSConfigurationRequirements', 'VZMacHardwareModel',
    'VZMacMachineIdentifier', 'VZMacAuxiliaryStorage', 'VZMacPlatformConfiguration', 'VZMacOSBootLoader',
    'VZMacGraphicsDeviceConfiguration', 'VZMacGraphicsDisplayConfiguration', 'VZUSBKeyboardConfiguration',
    'VZUSBScreenCoordinatePointingDeviceConfiguration', 'VZVirtioNetworkDeviceConfiguration',
    'VZMACAddress', 'VZNATNetworkDeviceAttachment', 'VZVirtioBlockDeviceConfiguration', 'VZDiskImageStorageDeviceAttachment']
records = [sdk / ('System/Library/Frameworks/Virtualization.framework/Headers/' + name + '.h') for name in headers]
records += [Path(__file__), Path(sys.executable), Path('/usr/bin/python3'), Path('/usr/sbin/sysctl'),
    receipt / 'collection.json', receipt / 'collection-qualification.json', receipt / 'qualification.json',
    cache / 'macos-vm-host-admission-local-v4/qualification.json',
    cache / 'research/local-vm-receiver/lifecycle/manifest.json',
    cache / 'research/local-vm-receiver/lifecycle/sdk-headers.json']
external.update({str(path): descriptor(path) for path in records})
protocol = {'schema_version': 1,
    'scope': 'Exact public Apple IPSW acquisition and offline no-boot configuration validation only. No VM instance, installation, guest or app/model.',
    'output_parent': str(cache), 'output_name': 'macos-vm-restore-local-v1', 'expected_cpu': 'Apple M4 Pro',
    'swiftc': prior['swiftc'], 'sdk': str(sdk), 'python': sys.executable,
    'tools': {name: descriptor(tools / name) for name in names}, 'external_files': external,
    'host_prerequisite': {'root': str(receipt), 'source_commit': '5a9dcbd97bfc29ffe4e6b2113d3b13b24cb6f2be'},
    'denied_paths': sorted(set(prior['denied_paths']) | {str(path) for path in cache.iterdir() if path.is_dir()}),
    'command_timeout_seconds': 300,
    'storage': {'start_minimum': 140 * 1024**3, 'running_minimum': 120 * 1024**3, 'reserve_bytes': 256 * 1024**2},
    'download_limits': {'deadline_seconds': 7200, 'read_timeout_seconds': 30},
    'restore_image': {'name': 'UniversalMac_26.3_25D125_Restore.ipsw',
        'url': 'https://updates.cdn-apple.com/2026WinterFCS/fullrestores/047-60229/6D5DBEA5-75A0-4BEF-ACC9-5ACF9B8DF6B7/UniversalMac_26.3_25D125_Restore.ipsw',
        'bytes': 18877622182, 'sha256': '7d22296ca6ab46f255bed8f5fc80845edab14fb781f845d7cae9427a22970b53'},
    'limits': [
        'Reviewed committed source/protocol required before invocation. Bind actual passing host receipt and all its members. Named physical M4 Pro only; retain every failure, no automatic retries/overwrites.',
        'Before the large acquisition build the new helper offline with Swift language mode5, SDK26.5, arm64/min26.0, warnings-as-errors and two compiler jobs. New local ad-hoc hardened signature, exact one unrestricted virtualization entitlement. No signing identity, certificate/profile/Keychain, host sudo/security/settings or private checkout changes.',
        'All build/sign/inspection/helper children deny network and model/private reads. Sole exception: one acquisition worker with isolated HOME/TMPDIR/environment, proxy/cookies/redirects disabled, exact public Apple HTTPS URL, size and SHA256. Partial file retained, never admitted. Worker soft deadline7200s/read30s; parent hard timeout7230s covers DNS/TLS stalls, storage floor checked throughout. Reserve owned only by parent.',
        'Initial free140GiB, running120GiB, physical256MiB diagnostic reserve. After the large download, child admission checks running floor rather than reapplying the initial floor. Sparse logical100GiB is not a physical quota.',
        'Require actual local image support, OS26.3.0/build25D125, supported model and image minima within4CPU/8GiB. Prepare a fresh private four-file seed; exactly model/machine-ID/auxiliary/rawdisk. NAT random locally administered unicast MAC, one framebuffer, no audio/share/window/custom listener. NAT is configuration only; no running network or offline-installation claim.',
        'Only configuration.validate(), never construct VZVirtualMachine or VZMacOSInstaller. No mount or attach, guest, private payload, SDK/model/app call. Disk is fresh O_EXCL/ftruncate100GiB with zero allocated blocks; never hash/read the full sparse disk. Hash only bounded small seed metadata. All four files600, directory700, owned regular files/single links.',
        'Immutable local seed preserved for separately reviewed APFS-clone/install lifecycle phase. Keep OS body, auxiliary/model/ID/disk, helper and module cache local; collect hashes/stat/JSON/source/logs only. Selected compiler/tools/SDK headers pinned; no claim of complete SDK closure or module-cache hashing.',
        'Passing this stage does not close installation/boot/provisioning/fresh-machine complete-app, GUI/microphone/playback/IPC, performance/quality/human gold, platform/public-signing/release gates. Future installer cancellation uses progress.cancel after start and terminal callback; no VM stop/pause during install.']}
assert not (cache / protocol['output_name']).exists()
target = tools / 'macos-vm-restore.protocol.json'
target.write_text(json.dumps(protocol, indent=2) + '\n')
print(descriptor(target), 'tools', len(names), 'external', len(external))
