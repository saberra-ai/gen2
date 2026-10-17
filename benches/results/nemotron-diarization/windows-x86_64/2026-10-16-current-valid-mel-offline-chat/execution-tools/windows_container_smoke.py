"""Exercise immutable Server Core container controls without any model payload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import uuid

from evidence_commands import CommandRunner
from sdk_archive import inventory


class Smoke(CommandRunner):
    def __init__(self, output):
        self.output, self.repo = output.resolve(), Path(__file__).resolve().parents[2]
        self.tools = Path(__file__).resolve().parent
        self.environment, self.commands = dict(os.environ), []
        self.docker = [shutil.which('docker'), '--context', 'default']
        if not self.docker[0] or any(self.environment.get(name) for name in
                                    ['DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH']):
            raise ValueError('requires existing local default-context Windows Docker')
        self.created = None

    def execute(self, prerequisites):
        preflight = json.loads(prerequisites.read_text(encoding='utf-8-sig'))
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        if (not preflight['server2022_process_container_candidate'] or preflight['source_commit'] != source
                or preflight['tool_sha256'] != hashlib.sha256((self.tools / 'windows_container_preflight.py').read_bytes()).hexdigest()):
            raise ValueError('same-source successful Windows prerequisites required')
        lock = json.loads((self.tools / 'windows-container.lock.json').read_text())
        image = lock['repository'] + '@' + lock['digest']
        self.run('image-inspect', [*self.docker, 'image', 'inspect', image])
        description, = json.loads((self.output / 'image-inspect.stdout.log').read_text(encoding='utf-8-sig'))
        if (description['Id'] != lock['image_id'] or image not in description['RepoDigests']
                or description['Architecture'] != lock['architecture'] or description['Os'] != 'windows'
                or description['OsVersion'].split('.')[:3] != ['10', '0', lock['host_kernel_build']]):
            raise ValueError('local Windows runtime image differs from frozen fixture')
        inputs, results = self.output / 'inputs ü', self.output / 'results'
        inputs.mkdir()
        results.mkdir()
        (inputs / 'control.txt').write_text('Gen2 offline read-only input control', encoding='utf-8')
        shutil.copy2(self.tools / 'windows_smoke_controls.ps1', inputs / 'controls.ps1')
        before = inventory(inputs)
        frozen = {'source_commit': source, 'lock': lock, 'image': description,
                  'prerequisites_sha256': hashlib.sha256(prerequisites.read_bytes()).hexdigest(),
                  'inputs': before, 'tools': {name: hashlib.sha256((self.tools / name).read_bytes()).hexdigest()
                    for name in ['windows_container_smoke.py', 'windows_smoke_controls.ps1', 'evidence_commands.py', 'sdk_archive.py']}}
        (self.output / 'frozen-inputs.json').write_text(json.dumps(frozen, indent=2) + '\n')
        for path in [inputs, results]:
            if ',' in str(path):
                raise ValueError('bind mount paths cannot contain commas')
        name = 'gen2-nemo-controls-' + uuid.uuid4().hex
        self.run('create', [*self.docker, 'create', '--name', name, '--pull', 'never', '--isolation', 'process',
                 '--network', 'none', '--mount', f'type=bind,source={inputs},target=C:\\inputs,readonly',
                 '--mount', f'type=bind,source={results},target=C:\\results',
                 '--env', 'PATH=C:\\Windows\\System32;C:\\Windows', description['Id'],
                 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
                 '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', 'C:\\inputs\\controls.ps1'])
        self.created = (self.output / 'create.stdout.log').read_text(encoding='utf-8-sig').strip()
        if len(self.created) != 64 or any(char not in '0123456789abcdef' for char in self.created):
            self.created = None
            raise ValueError('Docker did not return a canonical managed container ID')
        self.run('start', [*self.docker, 'start', '--attach', self.created])
        self.run('container-inspect', [*self.docker, 'inspect', self.created])
        actual, = json.loads((self.output / 'container-inspect.stdout.log').read_text(encoding='utf-8-sig'))
        mounts = {row['Destination'].lower(): row for row in actual['Mounts']}
        if (actual['Image'] != description['Id'] or actual['HostConfig']['Isolation'] != 'process'
                or actual['HostConfig']['NetworkMode'] != 'none' or actual['State']['ExitCode'] != 0 or actual['State']['Running']
                or set(mounts) != {'c:\\inputs', 'c:\\results'} or mounts['c:\\inputs']['RW']
                or not mounts['c:\\results']['RW']
                or Path(mounts['c:\\inputs']['Source']).resolve(strict=True) != inputs
                or Path(mounts['c:\\results']['Source']).resolve(strict=True) != results):
            raise ValueError('actual container controls differ from frozen command')
        control = json.loads((results / 'controls.json').read_text(encoding='utf-8-sig'))
        if (control['input_sha256'] != before['control.txt']['sha256'] or not control['input_read']
                or control['input_write_win32_error'] != 5 or not control['input_write_denied']
                or control['operational_non_loopback_interfaces'] or control['process_architecture'] != 'AMD64'
                or not control['development_tools_absent_from_path'] or not control['visual_studio_directory_absent']
                or inventory(inputs) != before):
            raise ValueError('actual read/write/tool/network controls failed')
        return {'source_commit': source, 'image_id': description['Id'], 'controls': control,
                'frozen_inputs_sha256': hashlib.sha256((self.output / 'frozen-inputs.json').read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prerequisites', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows':
        parser.error('requires actual Windows Docker host')
    args.output.mkdir(parents=True, exist_ok=False)
    runner = None
    receipt = {'schema_version': 1, 'runtime_qualified': False, 'container_controls_passed': False,
               'scope': 'Actual Server Core process-container controls only; no Gen2/model/package payload, desktop VM or nested Hyper-V claim.'}
    try:
        runner = Smoke(args.output)
        receipt.update(runner.execute(args.prerequisites))
        receipt['container_controls_passed'] = True
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        if runner and runner.created:
            try:
                runner.run('remove-managed-container', [*runner.docker, 'rm', '--force', runner.created])
            except Exception as error:
                receipt.update(container_controls_passed=False, cleanup_error=str(error))
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['container_controls_passed']:
        raise RuntimeError('container cleanup failed; see retained receipt')


if __name__ == '__main__':
    main()
