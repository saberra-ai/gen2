"""Metadata-only freeze for one normal boot and stopped readonly disk discovery."""
from pathlib import Path
import hashlib
import json
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(tools))
from macos_vm_seed_clone import file_identity
from macos_vm_install_metrics import installed_seed_metrics


def descriptor(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


prior = json.loads((tools / 'macos-vm-install.protocol.json').read_text())
for name, want in prior['tools'].items():
    assert descriptor(tools / name) == want
for name, want in prior['external_files'].items():
    assert descriptor(name) == want
receipt = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-install-stopped'
manifest = json.loads((receipt / 'collection.json').read_text())
for name, want in manifest['files'].items():
    assert descriptor(receipt / name) == want
assert json.loads((receipt / 'collection-qualification.json').read_text())['installation_passed'] is True
origin = cache / 'macos-vm-install-local-v1'
assert (origin / 'qualification.json').read_bytes() == (receipt / 'qualification.json').read_bytes()
pq = json.loads((origin / 'qualification.json').read_text())
assert pq['owner_result']['cleanup_confirmed'] is True and pq['owner_result']['final_state'] == 0
assert pq['installed_disk_quarantined'] is False and pq['exclusive_lock_released'] is True
assert installed_seed_metrics(origin / 'vm-seed', pq['installed_seed'], uid=502) == pq['installed_seed']
seed_identity = {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()}
external = dict(prior['external_files'])
records = [Path(__file__), receipt / 'collection.json', receipt / 'collection-qualification.json', receipt / 'qualification.json',
    origin / 'qualification.json', origin / 'protocol.json', origin / 'owner.lock',
    Path('/usr/share/man/man1/hdiutil.1'), Path('/usr/share/man/man8/diskutil.8')]
external.update({str(path): descriptor(path) for path in records})
names = ['qualify_macos_vm_boot.py', 'macos_vm_boot_owner.swift', 'macos_vm_boot_metrics.py',
    'test_macos_vm_boot_metrics.py', 'macos_vm_host_metrics.py']
sources = dict(prior['tools'])
sources.update({name: descriptor(tools / name) for name in names})
denied = set(prior['denied_paths']) | {str(path) for path in cache.iterdir() if path.is_dir()}
denied.discard(str(origin))
protocol = {'schema_version': 1,
    'scope': 'One fresh installed paired clone, normal headless boot, supported stop and readonly metadata discovery. Setup networking permitted; no guest readiness/provisioning, mount, SSH, private app/model or release qualification.',
    'output_parent': str(cache), 'output_name': 'macos-vm-boot-local-v1', 'expected_cpu': 'Apple M4 Pro',
    'swiftc': prior['swiftc'], 'sdk': prior['sdk'], 'tools': sources, 'external_files': external,
    'installed_prerequisite': {'receipt': str(receipt), 'local_root': str(origin),
        'source_commit': '550905e469c95e1373f75f330f68d32acfebf2bc', 'seed_identity': seed_identity},
    'mac_address': json.loads((receipt / 'owner-config.json').read_text())['mac_address'],
    'denied_paths': sorted(denied),
    'storage': {'start_minimum': 120 * 1024**3, 'running_minimum': 80 * 1024**3, 'reserve_bytes': 256 * 1024**2},
    'command_timeout_seconds': 300,
    'owner_limits': {'run_seconds': 120.0, 'operation_seconds': 300.0, 'graceful_seconds': 30.0, 'cleanup_seconds': 90.0},
    'supervision': {'operation_seconds': 330, 'cleanup_seconds': 90},
    'limits': [
        'Reviewed committed source/protocol and actual passing installed-stopped56-member receipt required. Named physical M4 Pro only, fresh output/no retry, preserve all earlier receipts and original paired seed. Reacquire source owner.lock and hold it plus new owner.lock across clone, normal VM lifetime, terminal owner wait and readonly attach/detach.',
        'Exactly four descriptor-anchored same-volume fclonefileat(CLONE_NOFOLLOW) clones, return0/no copy fallback/fresh owned600 single-link regular distinct-inode destination under700. Compare exact bounded metadata and stat identities, never read/hash100GiB disk. Freeze original installed source identities and preserve before/after. No clone or reuse following an unconfirmed owner.',
        'Build new qualification owner offline, pinned SDK26.5/Swift mode5/arm64/min26.0/twojobs/warnings-as-errors. New local ad-hoc hardened signature/exact unrestricted virtualization entitlement. No host sudo, credentials, certificate/profile/Keychain/security/settings/private-checkout changes or custom host listener. Foundation error-graph control before VM; synchronous diagnostic checks before VM creation and normal start.',
        'Initial free120GiB/running80GiB/physically allocated256MiB diagnostic reserve. Offline controls deny model/private reads, networking and original installed-root writes. Normal-boot owner explicitly permits NAT/setup networking only with same file denials. No private app/model or guest account/SSH/patching. Apple-managed traffic/temp files are not a whole-system confinement or zero-egress guarantee.',
        'One VM on main serial queue from exact cloned hardware/ID/aux/disk/persistedMAC,4CPU/8GiB,NAT,oneframebuffer,keyboard/pointing,noaudio/shares/window. Validate and require stopped before start. Start callback plus120s Running observation qualifies public lifecycle only, never OS ready, Setup Assistant readiness or app execution.',
        'Owner operation300s,120s run,30s graceful wait,90s cleanup. After run or cancellation, await active start callback before stop; requestStop only when supported. After30s allow declared destructive stop only Running/Paused+canStop; require completion+observed stopped and normal exit. Guest-stop delegate is required if no forced stop. No pause/resume/replacement, no install call.',
        'Parent owner wait330s; on storage/deadline/diagnostic error release reserve, SIGTERM then90s grace before kill/reap failure. Owner kill/exit alone is not teardown proof. Failed/cancelled/unconfirmed owner quarantines clone; no disk attach/mount/clone/replacement. Callback, full run interval, stopped event, zero exit/no errors/cancellation required before disk discovery.',
        'After qualified stopped owner exit only: exact new RAW image attach via hdiutil CRawDiskImage/read-only/nomount/nobrowse/plist. Discover actual own GUID whole and all observed physical child partitions, not s2. Query diskutil info/list on exact observed device only; APFSContainerReference and sole physical-store membership must bind each scoped container. Query actual observed volumes/roles and require no MountPoint.',
        'Acquire exact scoped diskutil apfs listVolumeGroups -plist container bytes without guessing group dictionary schema or choosing first Data/name. System/Data roles must be observed but system_data_group_verified remainsfalse. No filesystem mount, root writes, ownership/settings changes or group-pair admission until actual metadata review in a later stage.',
        'Always attempt exact own-image detach and verify absence independent of source/storage/persistence failure after attach begins. Free reserve before cleanup diagnostics; raw global image inventory stays local with only exact own-image filtered rows public. Hold both locks until cleanup; detach/absence plus all diagnostics required for pass.',
        'Keep paired disk/metadata binaries/helper/modulecache local. Preserve raw failure diagnostics, no silent retry or changed threshold. Passing result establishes normal-start/stopped lifecycle and scoped readonly metadata acquisition only; OS readiness/account/SSH/provisioning, fresh-machine complete app, GUI/microphone/playback/IPC, quality/human gold, platform performance and public signing/release/main promotion remain open.']}
assert not (cache / protocol['output_name']).exists()
target = tools / 'macos-vm-boot.protocol.json'
target.write_text(json.dumps(protocol, indent=2) + '\n')
print(descriptor(target), 'tools', len(sources), 'external', len(external))
