"""Freeze local no-boot VZ and blank-image admission; no build/sign/mount/VM calls."""
from pathlib import Path
import hashlib,json,subprocess
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb') as stream:
  while raw:=stream.read(1024**2): d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
research=cache/'research/local-vm-receiver/alternative';manifest=json.loads((research/'manifest.json').read_text())
for name,want in manifest['files'].items():assert descriptor(research/name)==want
assert descriptor(research/'manifest.json')=={'bytes':6444,'sha256':'0bab2f296d4d0d8b8360c10500c29f2a36e41f4130a8cf9b91849b7e653c8c3f'}
assert (tools/'macos_vm_apple_LICENSE.txt').read_bytes()==(research/'sample_LICENSE.txt').read_bytes()
assert (tools/'macos_vm_host.entitlements').read_bytes()==(research/'sample_InstallationTool.entitlements').read_bytes()
swift=Path(subprocess.check_output(['/usr/bin/xcrun','--find','swiftc'],text=True).strip())
sdk=Path(subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip()).resolve(strict=True)
headers=sdk/'System/Library/Frameworks/Virtualization.framework/Headers'
external=[swift,sdk/'SDKSettings.json',headers/'VZVirtualMachine.h',headers/'VZVirtualMachineConfiguration.h',
 research/'manifest.json',research/'sample_LICENSE.txt',research/'sample_InstallationTool.entitlements',
 research/'sample_Swift_Common_MacOSVirtualMachineConfigurationHelper.swift',
 research/'sample_Swift_InstallationTool_MacOSVirtualMachineInstaller.swift',Path(__file__)]
external.extend(Path(n) for n in ['/usr/bin/codesign','/usr/bin/hdiutil','/usr/sbin/diskutil','/usr/bin/otool','/usr/bin/lipo','/usr/bin/sandbox-exec'])
output=cache/'macos-vm-host-admission-local-v1';assert not output.exists()
names=['qualify_macos_vm_host.py','macos_vm_host_probe.swift','macos_vm_host.entitlements','macos_vm_apple_LICENSE.txt',
'macos_vm_host_metrics.py','test_macos_vm_host_metrics.py','test_macos_vm_host_cleanup.py','qualification_storage.py','frozen_git_sources.py']
denied=[str(p) for p in cache.iterdir() if p.is_dir()]
denied += [str(repo/'benches/results'),'/Users/victor/workspace/pnola','/Users/victor/workspace/gen2','/Users/victor/.ssh','/Users/victor/Library/Keychains']
p={'schema_version':1,'scope':'Actual local helper build/ad-hoc sign/API host probe plus one task-owned blank256MiB APFS create/attach/mount/write/detach. No OS/VM/app/model/GUI/microphone/fresh-machine qualification.',
'output_parent':str(cache),'output_name':output.name,'expected_cpu':'Apple M4 Pro','swiftc':str(swift),'sdk':str(sdk),
'tools':{n:descriptor(tools/n) for n in names},'external_files':{str(n):descriptor(n) for n in external},'denied_paths':sorted(set(denied)),
'storage':{'start_minimum':5*1024**3,'running_minimum':1024**3,'reserve_bytes':256*1024**2},'command_timeout_seconds':300,
'limits':['Review and commit sources/protocol before actual build/sign/probe/mount. No inference or compilation concurrent with other model work.',
'Only a new local helper is signed ad-hoc with exactly com.apple.security.virtualization and runtime hardening; no host identity/certificate/Keychain/settings/profile changes. No restricted networking entitlement. Public VZ support/geometry API only; no VM instance, boot, OS image or network attachment.',
'Host macOS26.3; local Apple SDK26.5/Swift6.3.3 measured with explicit arm64/macOS26.0 target and only older available APIs. Freeze actual compiler/tool and selected SDK/header/source bytes; this is not a complete SDK source-reproducibility claim.',
'Offline sandbox/weight-format/network-denial controls precede compilation. Original cache/private app/Keychain/SSH roots denied and checkout read-only for children. Task-private HOME/TMPDIR/module-cache and minimal environment; no inherited Apple credentials.',
'Create one fresh256MiB GUID/APFS image without auto-attach/Spotlight; attach nomount then explicitly mount its single volume under task output. Select whole device only from exact-image hdiutil identity; require exact APFS physical-store/container/volume bridge before mount. Never use a guessed/prior/first disk number.',
'Write/read/fsync/chmod600 only a new dummy file on the actual new volume. Record ownership policy without overriding it. Blank user-owned-volume access does not establish root-owned restored Data-volume patch permission.',
'Always re-query exact image before detach; do not detach other images. Cleanup captures in memory with bounded process-group deadlines and bypasses storage floors, changed-input admission and diagnostic-write gates; free the reserve first. Any metadata failure prevents a passing gate. Spawn/PID persistence/wait share a kill-and-reap boundary. Require actual detach and absent task mount. Retain failures; no privilege or GUI fallback or silent rerun.',
'5GiB start/1GiB running/256MiB physicalreserve and300s per-child deadlines. OS-managed disk services are expected; no custom listener, LaunchAgent, VNC, HTTP server, clipboard, guest or OS restore. NAT/offline guest support and locked-session VM boot remain untested.',
'Helper/blank-disk prerequisites only: human word review remains deferred and all20%DER, GUI/microphone, fresh-machine complete-app, platform/performance and public signing/release gates stay open. No main/default promotion.']}
path=tools/'macos-vm-host.protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2)+'\n');print(descriptor(path),len(p['tools']),len(p['external_files']),str(sdk))
