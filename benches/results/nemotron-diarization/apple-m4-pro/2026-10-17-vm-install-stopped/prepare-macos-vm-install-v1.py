"""Freeze metadata only for enforced paired clone and install-only owner."""
from pathlib import Path
import hashlib
import json

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

prior = json.loads((tools / 'macos-vm-restore.protocol.json').read_text())
for name, want in prior['tools'].items():
    assert descriptor(tools / name) == want
for name, want in prior['external_files'].items():
    assert descriptor(name) == want
receipt = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-restore-no-boot'
manifest = json.loads((receipt / 'collection.json').read_text())
for name, want in manifest['files'].items():
    assert descriptor(receipt / name) == want
assert json.loads((receipt / 'collection-qualification.json').read_text())['no_boot_restore_passed'] is True
origin = cache / 'macos-vm-restore-local-v1'
assert (origin / 'qualification.json').read_bytes() == (receipt / 'qualification.json').read_bytes()
external = dict(prior['external_files'])
sdk = Path(prior['sdk'])
records = [Path(__file__), receipt / 'collection.json', receipt / 'collection-qualification.json', receipt / 'qualification.json',
    origin / 'qualification.json', origin / 'protocol.json',
    sdk / 'usr/include/sys/clonefile.h',
    sdk / 'System/Library/Frameworks/Foundation.framework/Headers/NSError.h',
    cache / 'research/local-vm-receiver/installer-seed/manifest.json',
    sdk / 'System/Library/Frameworks/Virtualization.framework/Modules/Virtualization.swiftmodule/arm64e-apple-macos.swiftinterface']
records += [sdk / ('System/Library/Frameworks/Virtualization.framework/Headers/' + name + '.h') for name in
    ['VZMacOSInstaller', 'VZVirtualMachine', 'VZVirtualMachineDelegate', 'VZError']]
external.update({str(path): descriptor(path) for path in records})
names = ['qualify_macos_vm_install.py', 'macos_vm_install_owner.swift', 'macos_vm_seed_clone.py',
    'macos_vm_install_supervision.py', 'macos_vm_install_metrics.py',
    'test_macos_vm_install_controls.py', 'test_macos_vm_install_metrics.py']
sources = dict(prior['tools'])
sources.update({name: descriptor(tools / name) for name in names})
denied = set(prior['denied_paths']) | {str(path) for path in cache.iterdir() if path.is_dir()}
denied.discard(str(origin))
protocol = {'schema_version': 1, 'scope': 'Enforced paired seed clone and installer-only VM owner; setup networking permitted. No normal boot, mount/provisioning, private app/model or release qualification.',
    'output_parent': str(cache), 'output_name': 'macos-vm-install-local-v1', 'expected_cpu': 'Apple M4 Pro',
    'swiftc': prior['swiftc'], 'sdk': prior['sdk'], 'tools': sources, 'external_files': external,
    'restore_prerequisite': {'receipt': str(receipt), 'local_root': str(origin),
        'source_commit': '4c9eb3612defd9c1ab6a8dfc982d489d23bb084b'},
    'restore_image': prior['restore_image'], 'denied_paths': sorted(denied),
    'storage': {'start_minimum': 140 * 1024**3, 'running_minimum': 80 * 1024**3, 'reserve_bytes': 256 * 1024**2},
    'command_timeout_seconds': 300,
    'owner_limits': {'operation_seconds': 1800.0, 'cleanup_seconds': 180.0},
    'supervision': {'operation_seconds': 2010, 'cleanup_seconds': 180},
    'limits': [
        'Reviewed committed source/protocol and actual passing no-boot47-member receipt required. Preserve every prior result. Named physical M4 Pro only, distinct fresh output/no retry, original pinned image and paired seed unchanged before/after.',
        'Build new local qualification owner offline first: exact SDK26.5/Swift mode5/arm64/min26.0/warnings-as-errors/twojobs, new ad-hoc hardened signature with exactly one unrestricted virtualization entitlement. No host sudo/credential/certificate/profile/Keychain/security/settings/private-checkout changes.',
        'Hold new task-owned exclusive owner.lock across clone/config/VM lifetime and terminal owner wait. Clone exactly four declared paired files through descriptor-anchored same-volume fclonefileat(CLONE_NOFOLLOW), return0/no copy fallback/freshdest/private ownership/regular/single-link/distinct inode. Record before/after stat; fully hash only small metadata, never read/hash100GiB sparse disk. Clone provenance is OS CoW equivalence, not full-disk SHA or physical-share measurement.',
        'Initial free140GiB/running80GiB/physical256MiB reserve. Sparse100GiB is no physical quota. Supervise storage throughout, free reserve before cancellation diagnostics, never let source/storage/persist failures bypass owner cancellation/reaping. Preserve original seed, image, prior apps/assets; no mount or deletion.',
        'Offline build/sign/inspection policy denies network/model/private reads and original-restore-root writes. Setup-controls verify model denial/original write-open denial before VM. Installer policy explicitly permits setup networking with the same file denials; Apple-managed networking/personalization and temporary files are not endpoint/zero-egress/whole-system-sandbox guarantees. No private app/model enters guest. No implicit network retry/fallback.',
        'Create VM on main serial queue from exact cloned hardware-model/ID/aux/disk and persisted MAC,4CPU/8GiB,NAT,oneframebuffer,keyboard/pointing,zeroaudio/shares/window/customlistener. Validate configuration, require initial stopped state, retain VM/installer/delegate/timer/signals strongly. Installer executes guest code; no subsequent normal start.',
        'Owner operation1800s from owner creation. Timer cancellation or SIGTERM/INT before start never cancels progress; while installing only progress.cancel after start, then await callback. Never VM pause/stop/start during install. Installer callback success alone is insufficient: require observed stopped state and successful normal owner exit with no cancellation/error/logfailure. Only after callback, Running/Paused+canStop permits destructive stop and successful completion+stopped observation; transient states get bounded cleanup wait.',
        'Synchronously reject diagnostic failure before VM creation and before actual installer invocation. Offline Foundation-only error-control before cloning verifies both NSUnderlyingErrorKey and NSMultipleUnderlyingErrorsKey branches with shared-reference/cycle-safe graph capture; preserve all standard underlying errors.',
        'Owner cleanup180s, including post-callback stop. Parent waits2010s (1800+180+30 allowance); on storage/deadline/error requests SIGTERM then180s grace, finally kill/reap only if needed. Maximum parent owner wait2190s plus bounded process-reap scheduling. Owner exit/kill is not public proof of framework teardown: missing/failed/cancelled/unconfirmed report quarantines disk; no mount/clone/reuse/replacement VM. No stop/pause during active installer even after timeout.',
        'Require exact ordered JSONL lifecycle, callback success+stopped final0+uncancelled terminal+exit0, unchanged original source/image/seed and owned paired installed files before admission. Installed disk records logical/allocated stat only, platform IDs remain byte-exact, auxiliary metadata fully hashed. New owner.lock must be reused by cooperating later phases; releasing it is not teardown proof.',
        'Keep image, paired seed binaries/disk, executable/modulecache local. No full SDK/OS/runtime closure claim. This may establish installed stopped seed only, not normal boot, account/SSH/provisioning, fresh-machine complete-app, GUI/microphone/IPC/playback, quality/human annotation, other platforms or public-signing/release. Installation10006/10007 and underlying errors remain failures; no host update/security or credential fallback.']}
assert not (cache / protocol['output_name']).exists()
target = tools / 'macos-vm-install.protocol.json'
target.write_text(json.dumps(protocol, indent=2) + '\n')
print(descriptor(target), 'tools', len(sources), 'external', len(external))
