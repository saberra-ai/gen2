"""Provision and re-observe a VC prerequisite in a disposable local Windows image.

No host installation or image publication. This is installer/state qualification,
not a substitute for real Gen2 native/public/lifecycle/coexistence execution.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import uuid

from artifacts import acquire
from windows_archive import byte_inventory, digest
from windows_central_state import audit_state, require_setup_success, require_signature
from windows_container_smoke import Smoke

MINIMUM = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-static/minimum-x64-cab-inspection.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


class Setup(Smoke):
    def __init__(self, args):
        super().__init__(args.output)
        self.args = args
        self.receipt = {'schema_version': 1, 'setup_and_fresh_image_state_passed': False,
                        'central_runtime_qualified': False, 'release_qualified': False,
                        'redistribution_qualified': False,
                        'limits': ['Disposable CI image only; no host installation, image publication or Microsoft bytes in Gen2 runtime archive.',
                                   'Installer/setup/registry/file/signature observations only; actual Gen2/native/public/lifecycle/chat loading remains separate.',
                                   'Exit 3010 or any observed pending restart stays unqualified; container recreation is not an actual Windows reboot.',
                                   'Observed publisher-selected terms are not a distribution grant or evidence of individual assent.']}

    def prepare(self):
        self.source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        prerequisite = read(self.args.prerequisites)
        if (not prerequisite['server2022_process_container_candidate'] or prerequisite['source_commit'] != self.source
                or prerequisite['tool_sha256'] != digest(self.tools / 'windows_container_preflight.py')):
            raise ValueError('same-source successful Windows preflight required')
        self.lock = read(self.tools / 'windows-vc-runtime.lock.json')
        image_lock = read(self.tools / 'windows-container.lock.json')
        self.run('base-image-inspect', [*self.docker, 'image', 'inspect', image_lock['repository'] + '@' + image_lock['digest']])
        self.base_image, = read(self.output / 'base-image-inspect.stdout.log')
        if (self.base_image['Id'] != image_lock['image_id'] or self.base_image['Architecture'] != 'amd64'
                or self.base_image['Os'] != 'windows' or self.base_image['OsVersion'].split('.')[:3] != ['10', '0', image_lock['host_kernel_build']]
                or image_lock['repository'] + '@' + image_lock['digest'] not in self.base_image['RepoDigests']):
            raise ValueError('actual base image differs from frozen Windows fixture')
        self.inputs = self.output / 'setup inputs ü'
        self.inputs.mkdir()
        acquire(self.lock['installer'], self.inputs)
        installer = self.inputs / self.lock['installer']['name']
        self.run('installer-host-signature', ['pwsh', '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
                 self.tools / 'windows_vc_signature.ps1', '-InputFile', installer,
                 '-Output', self.output / 'installer-host-signature.json'])
        signature = read(self.output / 'installer-host-signature.json')
        require_signature(signature)
        if {key: signature[key] for key in ['bytes', 'sha256']} != {'bytes': self.lock['installer']['size'], 'sha256': self.lock['installer']['sha256']}:
            raise ValueError('signature observation not bound to exact publisher installer')
        minimum = read(self.repo / MINIMUM)
        if minimum['installer_sha256'] != self.lock['installer']['sha256']:
            raise ValueError('static DLL payload identity differs from prerequisite')
        self.expected = {row['runtime_name']: {key: row[key] for key in ['bytes', 'sha256']} for row in minimum['selected_dlls']}
        for row in minimum['selected_dlls']:
            if row['machine'] != 'AMD64' or not row['is_dll'] or row['file_version_static_string'] != self.lock['version']:
                raise ValueError('unsupported expected central payload')
        expected = {'files': self.expected, 'installer': self.lock['installer'], 'version': self.lock['version']}
        (self.inputs / 'expected.json').write_text(json.dumps(expected, indent=2) + '\n', encoding='utf-8')
        (self.inputs / 'control.txt').write_text('Gen2 offline read-only input control', encoding='utf-8')
        shutil.copy2(self.tools / 'windows_smoke_controls.ps1', self.inputs / 'controls.ps1')
        shutil.copy2(self.tools / 'windows_central_setup.ps1', self.inputs / 'setup.ps1')
        self.inventory = byte_inventory(self.inputs)
        frozen = {'source_commit': self.source, 'base_image': self.base_image,
                  'installer_host_signature_sha256': digest(self.output / 'installer-host-signature.json'),
                  'minimum_static_receipt_sha256': digest(self.repo / MINIMUM), 'inputs': self.inventory,
                  'tools': {name: digest(self.tools / name) for name in ['windows_central_setup.py', 'windows_central_setup.ps1',
                            'windows_central_state.py', 'windows_vc_signature.ps1', 'windows_smoke_controls.ps1',
                            'windows_archive.py', 'windows_container_smoke.py', 'windows_container_preflight.py',
                            'evidence_commands.py', 'artifacts.py', 'package_windows.py', 'windows_notices.py',
                            'windows_pe.py', 'windows_naming.py', 'windows-vc-runtime.lock.json', 'windows-container.lock.json']}}
        (self.output / 'frozen-inputs.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
        self.receipt.update(source_commit=self.source, frozen_inputs_sha256=digest(self.output / 'frozen-inputs.json'),
                            base_image_id=self.base_image['Id'], installer_sha256=self.lock['installer']['sha256'])

    def container(self, name, image, mode, commit=False):
        results = self.output / name
        results.mkdir()
        if any(',' in str(path) for path in [self.inputs, results]):
            raise ValueError('comma in Windows bind mount source')
        try:
            self.run(name + '-create', [*self.docker, 'create', '--name', 'gen2-vc-' + uuid.uuid4().hex,
                     '--pull', 'never', '--isolation', 'process', '--network', 'none',
                     '--mount', f'type=bind,source={self.inputs},target=C:\\inputs,readonly',
                     '--mount', f'type=bind,source={results},target=C:\\results', '--env', 'PATH=C:\\Windows\\System32;C:\\Windows',
                     image, 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
                     '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                     '-File', 'C:\\inputs\\setup.ps1', '-Mode', mode])
            created = (self.output / (name + '-create.stdout.log')).read_text(encoding='utf-8-sig').strip()
            if len(created) != 64 or any(char not in '0123456789abcdef' for char in created):
                raise ValueError('canonical managed setup container ID required')
            self.created = created
            self.run(name + '-start', [*self.docker, 'start', '--attach', self.created])
            self.run(name + '-inspect', [*self.docker, 'inspect', self.created])
            actual, = read(self.output / (name + '-inspect.stdout.log'))
            mounts = {row['Destination'].lower(): row for row in actual['Mounts']}
            if (actual['Image'] != image or actual['HostConfig']['Isolation'] != 'process' or actual['HostConfig']['NetworkMode'] != 'none'
                    or actual['State']['Running'] or actual['State']['ExitCode'] != 0
                    or set(mounts) != {'c:\\inputs', 'c:\\results'} or mounts['c:\\inputs']['RW'] or not mounts['c:\\results']['RW']
                    or Path(mounts['c:\\inputs']['Source']).resolve(strict=True) != self.inputs
                    or Path(mounts['c:\\results']['Source']).resolve(strict=True) != results):
                raise ValueError('actual setup container controls differ')
            control = read(results / 'controls.json')
            if (not control['input_read'] or not control['input_write_denied'] or control['input_write_win32_error'] != 5
                    or control['input_sha256'] != self.inventory['control.txt']['sha256']
                    or control['operational_non_loopback_interfaces'] or control['process_architecture'] != 'AMD64'
                    or not control['development_tools_absent_from_path'] or not control['visual_studio_directory_absent']
                    or byte_inventory(self.inputs) != self.inventory):
                raise ValueError('actual setup read/network/tool controls failed')
            observation = read(results / 'state.json')
            if not observation['completed'] or observation.get('error') or observation['mode'] != mode:
                raise ValueError('incomplete setup/state observation')
            if mode == 'Install':
                if observation['installer_executed'] is not True:
                    raise ValueError('installer was not executed')
                require_setup_success(observation, self.expected, self.lock['version'])
            else:
                if observation['installer_executed'] is not False:
                    raise ValueError('fresh state observation unexpectedly executed installer')
                audit_state(observation['after'], self.expected, self.lock['version'])
            if commit:
                self.run('freeze-provisioned-image', [*self.docker, 'commit', self.created])
                self.provisioned = (self.output / 'freeze-provisioned-image.stdout.log').read_text(encoding='utf-8-sig').strip()
                if len(self.provisioned) != 71 or not self.provisioned.startswith('sha256:') or any(char not in '0123456789abcdef' for char in self.provisioned[7:]):
                    raise ValueError('canonical provisioned image ID required')
            return observation
        finally:
            if self.created:
                self.run(name + '-remove', [*self.docker, 'rm', '--force', self.created])
                self.created = None

    def execute(self):
        try:
            self.prepare()
            installed = self.container('install', self.base_image['Id'], 'Install', commit=True)
            self.run('provisioned-image-inspect', [*self.docker, 'image', 'inspect', self.provisioned])
            image, = read(self.output / 'provisioned-image-inspect.stdout.log')
            if (image['Id'] != self.provisioned or image['Os'] != 'windows' or image['Architecture'] != 'amd64'
                    or image['RootFS']['Layers'][:-1] != self.base_image['RootFS']['Layers']):
                raise ValueError('provisioned image does not extend exact base image')
            observed = self.container('fresh-state', self.provisioned, 'Observe')
            # Signature timestamps can differ; the admitted bytes/version/registry
            # contract is independently checked in each fresh container above.
            self.receipt.update(setup_and_fresh_image_state_passed=True, provisioned_image_id=self.provisioned,
                                installed_state_sha256=digest(self.output / 'install/state.json'),
                                fresh_state_sha256=digest(self.output / 'fresh-state/state.json'),
                                actual_installer_exit_code=installed['installer_exit_code'],
                                actual_central_files=observed['after']['files'])
        except Exception as error:
            self.receipt['error'] = str(error)
            raise
        finally:
            (self.output / 'receipt.json').write_text(json.dumps(self.receipt, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prerequisites', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows':
        parser.error('requires actual Windows Docker host')
    args.output.mkdir(parents=True, exist_ok=False)
    Setup(args).execute()


if __name__ == '__main__':
    main()
