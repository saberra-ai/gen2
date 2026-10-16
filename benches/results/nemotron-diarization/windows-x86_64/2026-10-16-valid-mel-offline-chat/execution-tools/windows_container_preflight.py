"""Read-only discovery of hosted Windows process-container prerequisites.

No image pulls, daemon changes, containers, installs or model execution. A
candidate result establishes only that a subsequent isolated test is feasible.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def candidate(version, info, host, context_overrides):
    return (not context_overrides and host.get('build') == '20348'
            and version.get('Server', {}).get('Os') == 'windows'
            and info.get('OSType') == 'windows'
            and info.get('Architecture') in {'x86_64', 'amd64'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        parser.error('requires the actual Windows x64 host')
    args.output.mkdir(parents=True, exist_ok=False)
    commands = []

    def observe(name, argv):
        started = time.monotonic()
        record = {'name': name, 'argv': argv}
        commands.append(record)
        try:
            result = subprocess.run(argv, capture_output=True, encoding='utf-8-sig', errors='replace', timeout=30)
            record.update(returncode=result.returncode, seconds=time.monotonic() - started)
            if result.returncode:
                record['error'] = result.stderr[-4000:]
                return None
            return result.stdout
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(error=str(error), seconds=time.monotonic() - started)
            return None

    def decode(name, value, lines=False):
        try:
            result = [json.loads(line) for line in value.splitlines() if line.strip()] if lines else json.loads(value)
            if not all(isinstance(item, dict) for item in (result if lines else [result])):
                raise ValueError('expected JSON objects')
            return result
        except (ValueError, TypeError) as error:
            commands.append({'name': name + '-parse', 'error': str(error)})
            return [] if lines else {}

    host_text = observe('host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', '''
$ErrorActionPreference='Stop'
$os=Get-CimInstance Win32_OperatingSystem
$cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,VMMonitorModeExtensions,SecondLevelAddressTranslationExtensions,VirtualizationFirmwareEnabled)
$services=@(Get-Service docker,vmcompute,hns -ErrorAction SilentlyContinue | Select-Object Name,Status)
[ordered]@{caption=$os.Caption;version=$os.Version;build=$os.BuildNumber;architecture=$os.OSArchitecture;cpu=$cpu;services=$services} | ConvertTo-Json -Depth 5
'''])
    host = decode('host', host_text) if host_text else {}
    # Never contact a configured remote/context override or print its value.
    overrides = sorted(name for name in ['DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'] if os.environ.get(name))
    version, info, images = {}, {}, []
    docker = shutil.which('docker')
    if docker and not overrides:
        local = [docker, '--context', 'default']
        version_text = observe('docker-version', [*local, 'version', '--format', '{{json .}}'])
        info_text = observe('docker-info', [*local, 'info', '--format', '{{json .}}'])
        images_text = observe('local-images', [*local, 'image', 'ls', '--digests', '--no-trunc', '--format', '{{json .}}'])
        version = decode('docker-version', version_text) if version_text else {}
        info = decode('docker-info', info_text) if info_text else {}
        images = decode('local-images', images_text, lines=True) if images_text else []
    keep = lambda value, keys: {key: value[key] for key in keys if key in value}
    record = {'schema_version': 1, 'runtime_qualified': False,
              'scope': 'Read-only prerequisites on the named hosted build VM; no isolated container or model execution.',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'host': host,
              'docker_environment_override_names': overrides,
              'docker_executable': {'path': docker, 'sha256': hashlib.sha256(Path(docker).read_bytes()).hexdigest()} if docker else None,
              'docker_version': {key: keep(version.get(key, {}), ['Version', 'ApiVersion', 'MinAPIVersion', 'Os', 'Arch', 'KernelVersion', 'GitCommit']) for key in ['Client', 'Server']},
              'docker_info': keep(info, ['OSType', 'Architecture', 'OperatingSystem', 'OSVersion', 'KernelVersion', 'ServerVersion', 'Isolation', 'Driver', 'SecurityOptions']),
              'local_images': [keep(value, ['Repository', 'Tag', 'Digest', 'ID', 'Size']) for value in images],
              'server2022_process_container_candidate': candidate(version, info, host, overrides),
              'limits': ['Fresh build VM contains Visual Studio/vcpkg; it is not a clean runtime machine.',
                         'No image identity, isolated operation, offline controls, actual DLL paths or desktop compatibility is qualified.',
                         'Reported CPU virtualization flags do not establish parent-hypervisor configuration or usable nested Hyper-V.'],
              'commands': commands}
    (args.output / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'server2022_process_container_candidate': record['server2022_process_container_candidate'], 'runtime_qualified': False}))


if __name__ == '__main__':
    main()
