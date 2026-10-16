"""Exercise a delivered Windows CPU runtime in fresh offline Server Core containers.

The native SDK is an earlier qualified, byte-pinned input. Gen2 helpers are built
from the current checkout. No SDK, checkout or build tools are mounted inside.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import uuid

from artifacts import acquire, matches
from platform_valid_mel_native import compare_complete_60s as compare
from runtime_profiles import load_profile, profile_provenance, validate_provenance
from package_windows import CRT, qualified_sdk, verify
from qualify_linux_package import LIFECYCLE_GATES, require_lifecycle
from windows_archive import byte_inventory, digest, restore_archive, write_archive
from windows_container_smoke import Smoke
from windows_pe import inspect_pe
from windows_runtime_audit import loaded_modules
from windows_central_state import audit_state
from windows_central_setup import MINIMUM


class Qualification(Smoke):
    def __init__(self, args):
        super().__init__(args.output)
        self.args = args
        self.central_crt = None
        self.work = self.output / 'base inputs ü'
        lock_path, lock = load_profile()
        self.receipt = {**profile_provenance(lock_path, lock), 'schema_version': 2, 'release_qualified': False, 'redistribution_qualified': False,
                        'offline_container_gates_passed': False,
                        'scope': 'Delivered Windows Server 2022 AMD64 AVX2/FMA/F16C runtime, fresh process containers with network none, native/public parity, lifecycle and observed modules.',
                        'limits': ['No Windows desktop/physical-machine, chat/Bark/Laya coexistence or universal namespace guarantee.',
                                   'No held-out quality, live/uncontended-performance or Microsoft redistribution qualification.',
                                   'Test executables carry exact app-local CRT copies to control their process-start imports; these are separate from the runtime archive.',
                                   'The independent native probe alone uses the relocated runtime directory on PATH for its static private C ABI import. Gen2 uses its explicit library loader with system-only PATH.']}

    def prepare(self):
        self.source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        prerequisite = json.loads(self.args.prerequisites.read_text(encoding='utf-8-sig'))
        if (not prerequisite['server2022_process_container_candidate'] or prerequisite['source_commit'] != self.source
                or prerequisite['tool_sha256'] != digest(self.tools / 'windows_container_preflight.py')):
            raise ValueError('same-source successful Windows prerequisites required')
        lock = json.loads((self.tools / 'windows-container.lock.json').read_text())
        self.run('image-inspect', [*self.docker, 'image', 'inspect', lock['repository'] + '@' + lock['digest']])
        self.image, = json.loads((self.output / 'image-inspect.stdout.log').read_text(encoding='utf-8-sig'))
        if (self.image['Id'] != lock['image_id'] or lock['repository'] + '@' + lock['digest'] not in self.image['RepoDigests']
                or self.image['Architecture'] != lock['architecture'] or self.image['Os'] != 'windows'
                or self.image['OsVersion'].split('.')[:3] != ['10', '0', lock['host_kernel_build']]):
            raise ValueError('actual Windows runtime image differs from frozen fixture')
        evidence = self.args.package_evidence.resolve(strict=True)
        construction = json.loads((evidence / 'receipt.json').read_text(encoding='utf-8-sig'))
        validate_provenance(construction)
        if construction['package_construction_passed'] is not True:
            raise ValueError('passing Windows runtime construction required')
        archive = json.loads((evidence / 'development-windows-runtime-files.json').read_text(encoding='utf-8-sig'))
        if any(archive[key] != construction['archive'][key] for key in ['archive', 'archive_bytes', 'archive_sha256']):
            raise ValueError('delivered Windows archive identity differs')
        self.work.mkdir()
        self.runtime = restore_archive(evidence / archive['archive'], archive, self.work / 'relocated runtime ü')
        self.package = verify(self.runtime)
        if self.package['format'] == 'gen2-nemo-windows-cpu-central-v2':
            self.use_central_image()
        elif getattr(self.args, 'central_setup', None) is not None:
            raise ValueError('central setup evidence requires the five-DLL central runtime format')
        if digest(self.runtime / 'package.json') != construction['package_manifest_sha256']:
            raise ValueError('delivered Windows package manifest differs')
        native = self.args.native_evidence.resolve(strict=True)
        if digest(native / 'receipt.json') != construction['native_receipt_sha256']:
            raise ValueError('package/native receipt differs')
        _, runtime_lock = load_profile()
        build, _, _ = qualified_sdk(self.args.sdk_prefix.resolve(strict=True), native, runtime_lock)
        if build['source_commit'] != self.package['native_source_commit']:
            raise ValueError('qualified native source differs from runtime')
        for name in ['windows_package_commands.ps1', 'windows_smoke_controls.ps1']:
            shutil.copy2(self.tools / name, self.work / ('launch.ps1' if name == 'windows_package_commands.ps1' else 'controls.ps1'))
        (self.work / 'control.txt').write_text('Gen2 offline read-only input control', encoding='utf-8')
        assets = self.work / 'assets'
        assets.mkdir()
        for entry in runtime_lock['artifacts']:
            if entry['role'] != 'reference':
                acquire(entry, assets)
                if not matches(assets / entry['name'], entry):
                    raise ValueError('locked Windows qualification asset differs')
        self.run('build-public-helper', ['cargo', 'build', '--locked', '--release', '--no-default-features',
                 '--features', 'backend-diar-nemo', '--example', 'nemotron_valid_mel_stream'])
        metadata = json.loads(self.run('cargo-metadata', ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1']))
        test = self.run('build-lifecycle-helper', ['cargo', 'test', '--locked', '--release', '--no-default-features',
                        '--features', 'backend-diar-nemo', '--test', 'diarization_native', '--no-run', '--message-format=json'])
        executables = [row['executable'] for line in test.splitlines()
                       if (row := json.loads(line)).get('reason') == 'compiler-artifact'
                       and row['target']['name'] == 'diarization_native' and row.get('executable')]
        if len(executables) != 1:
            raise ValueError('expected exactly one compiled Windows lifecycle executable')
        binaries = self.work / 'bin'
        binaries.mkdir()
        helpers = {'native-probe.exe': self.args.sdk_prefix.resolve() / 'bin/gen2_native_stream_probe_valid_mel.exe',
                   'public-stream.exe': Path(metadata['target_directory']) / 'release/examples/nemotron_valid_mel_stream.exe',
                   'native-lifecycle.exe': Path(executables[0])}
        installed = {name.replace('\\', '/'): row for name, row in build['installed_files'].items()}
        probe = helpers['native-probe.exe']
        if {'bytes': probe.stat().st_size, 'sha256': digest(probe)} != installed['bin/gen2_native_stream_probe_valid_mel.exe']:
            raise ValueError('independent native probe differs from qualified SDK')
        self.helpers = {}
        for name, path in helpers.items():
            shutil.copy2(path, binaries / name)
            pe = inspect_pe(binaries / name)
            if pe['machine'] != 'AMD64' or pe['is_dll'] or pe['bound_imports_present']:
                raise ValueError('unsupported qualification executable')
            self.helpers[name] = {'bytes': pe['bytes'], 'sha256': pe['sha256']}
            (self.output / (name + '-pe.json')).write_text(json.dumps(pe, indent=2) + '\n', encoding='utf-8')
        for name in (CRT if self.central_crt is None else set()):
            shutil.copy2(self.runtime / 'lib' / name, binaries / name)
        helper_archive = write_archive(binaries, self.output, 'development-windows-test-helpers')
        self.base_inventory = byte_inventory(self.work)
        frozen = {'source_commit': self.source, 'native_source_commit': build['source_commit'],
                  'native_receipt_sha256': construction['native_receipt_sha256'],
                  'construction_receipt_sha256': digest(evidence / 'receipt.json'),
                  'package_manifest_sha256': construction['package_manifest_sha256'], 'archive': construction['archive'],
                  'image': self.image, 'helpers': self.helpers, 'files': self.base_inventory,
                  'test_helper_archive': {key: value for key, value in helper_archive.items() if key != 'files'},
                  'tools': {name: digest(self.tools / name) for name in ['qualify_windows_package.py', 'windows_package_commands.ps1',
                    'windows_runtime_audit.py', 'windows_smoke_controls.ps1', 'windows_container_smoke.py',
                    'windows_archive.py', 'package_windows.py', 'windows_pe.py', 'windows_naming.py',
                    'runtime.lock.json', 'windows-container.lock.json', 'evidence_commands.py', 'compare_stream_valid_mel.py', 'platform_valid_mel_native.py', 'platform_valid_mel_sources.py', 'platform_native.py', 'platform_sdk_receipt.py', 'runtime_profiles.py', 'build_valid_mel_candidate.py', 'qualification_storage.py', 'runtime-valid-mel.lock.json', 'valid-mel-platform-native.protocol.json', 'qualify_linux_package.py']}}
        if self.central_crt is not None:
            frozen['central_setup'] = {'receipt_sha256': digest(self.args.central_setup / 'receipt.json'),
                'frozen_inputs_sha256': digest(self.args.central_setup / 'frozen-inputs.json'),
                'base_image': self.base_image, 'central_crt': self.central_crt}
            frozen['tools'].update({name: digest(self.tools / name) for name in
                ['windows_central_state.py', 'windows_central_setup.py', 'windows-vc-runtime.lock.json']})
        (self.output / 'frozen-inputs.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
        self.receipt.update(source_commit=self.source, native_source_commit=build['source_commit'], image_id=self.image['Id'],
                            frozen_inputs_sha256=digest(self.output / 'frozen-inputs.json'),
                            package_manifest_sha256=construction['package_manifest_sha256'])

    def use_central_image(self):
        directory = getattr(self.args, 'central_setup', None)
        if directory is None:
            raise ValueError('five-DLL runtime requires same-source successful central setup evidence')
        read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
        setup, frozen = read(directory / 'receipt.json'), read(directory / 'frozen-inputs.json')
        if (setup['setup_and_fresh_image_state_passed'] is not True or setup['source_commit'] != self.source
                or frozen['source_commit'] != self.source or setup['base_image_id'] != self.image['Id']
                or digest(directory / 'frozen-inputs.json') != setup['frozen_inputs_sha256']
                or digest(directory / 'fresh-state/state.json') != setup['fresh_state_sha256']
                or any(digest(self.tools / name) != sha256 for name, sha256 in frozen['tools'].items())):
            raise ValueError('central setup source/image/provenance differs')
        minimum = read(self.repo / MINIMUM)
        lock = read(self.tools / 'windows-vc-runtime.lock.json')
        if (digest(self.repo / MINIMUM) != frozen['minimum_static_receipt_sha256']
                or minimum['installer_sha256'] != setup['installer_sha256'] or setup['installer_sha256'] != lock['installer']['sha256']):
            raise ValueError('central prerequisite differs from pinned publisher bytes')
        expected = {row['runtime_name']: {key: row[key] for key in ['bytes', 'sha256']} for row in minimum['selected_dlls']}
        fresh = read(directory / 'fresh-state/state.json')
        if fresh['mode'] != 'Observe' or fresh['installer_executed'] is not False or not fresh['completed'] or fresh.get('error'):
            raise ValueError('fresh central state was not observed independently of installation')
        audit_state(fresh['after'], expected, lock['version'])
        self.run('central-image-inspect', [*self.docker, 'image', 'inspect', setup['provisioned_image_id']])
        image, = read(self.output / 'central-image-inspect.stdout.log')
        if (image['Id'] != setup['provisioned_image_id'] or image['Architecture'] != 'amd64' or image['Os'] != 'windows'
                or image['RootFS']['Layers'][:-1] != self.image['RootFS']['Layers']):
            raise ValueError('actual central image does not extend frozen base')
        self.base_image, self.image, self.central_crt = self.image, image, expected
        self.central_state = fresh['after']
        self.receipt['limits'][2] = 'No CRT/OpenMP copies beside test executables or in the five-DLL runtime; exact central System32 bytes come from the separately qualified publisher installer.'

    def audit_modules(self, observation, executable):
        return loaded_modules(observation, self.package['libraries'], executable, self.helpers[executable], central_crt=self.central_crt)

    def container(self, name, executable, arguments, capture=True, negative=False, readonly_mounts=(), remove_private=True):
        run = self.output / name
        run.mkdir()
        inputs, results = run / 'inputs ü', run / 'results'
        shutil.copytree(self.work, inputs)
        results.mkdir()
        (results / 'tmp').mkdir()
        (results / 'local').mkdir()
        if negative and remove_private:
            (inputs / 'relocated runtime ü/lib/gen2_nemo_windows_ggml-base.dll').unlink()
        environment = {'SystemRoot': 'C:\\Windows', 'WINDIR': 'C:\\Windows',
                       'PATH': 'C:\\Windows\\System32;C:\\Windows',
                       'TEMP': 'C:\\results\\tmp', 'TMP': 'C:\\results\\tmp',
                       'USERPROFILE': 'C:\\results', 'LOCALAPPDATA': 'C:\\results\\local',
                       'GEN2_NEMOTRON_ASSETS': 'C:\\inputs\\assets',
                       'GEN2_NEMO_LIBRARY': 'C:\\inputs\\relocated runtime ü\\lib\\' + self.package['entry_library'],
                       'GEN2_NEMOTRON_REFERENCE': 'C:\\inputs\\reference', 'GEN2_NEMO_EXECUTION': 'cpu'}
        if executable == 'native-probe.exe':
            environment['PATH'] = 'C:\\inputs\\relocated runtime ü\\lib;' + environment['PATH']
        spec = {'executable': 'C:\\inputs\\bin\\' + executable, 'argv': arguments,
                'command_line': subprocess.list2cmdline(arguments), 'capture_modules': capture,
                'expected_failure': negative, 'environment': environment}
        (inputs / 'command.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
        before = byte_inventory(inputs)
        (run / 'frozen-inputs.json').write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
        readonly_mounts = [(Path(source).resolve(strict=True), target) for source, target in readonly_mounts]
        if any(',' in str(path) for path in [inputs, results, *[source for source, _ in readonly_mounts]]):
            raise ValueError('Windows bind mount paths cannot contain commas')
        expected_mounts = {'c:\\inputs': (inputs, False), 'c:\\results': (results, True)}
        extra = []
        for source, target in readonly_mounts:
            if target.lower() in expected_mounts or target != 'C:\\chat':
                raise ValueError('unsupported additional Windows qualification mount')
            expected_mounts[target.lower()] = (source, False)
            extra += ['--mount', f'type=bind,source={source},target={target},readonly']
        try:
            self.run(name + '-create', [*self.docker, 'create', '--name', 'gen2-nemo-runtime-' + uuid.uuid4().hex,
                     '--pull', 'never', '--isolation', 'process', '--network', 'none',
                     '--mount', f'type=bind,source={inputs},target=C:\\inputs,readonly',
                     '--mount', f'type=bind,source={results},target=C:\\results',
                     *extra,
                     '--env', 'PATH=C:\\Windows\\System32;C:\\Windows', self.image['Id'],
                     'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
                     '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', 'C:\\inputs\\launch.ps1'])
            created = (self.output / (name + '-create.stdout.log')).read_text(encoding='utf-8-sig').strip()
            if len(created) != 64 or any(char not in '0123456789abcdef' for char in created):
                raise ValueError('Docker did not return canonical managed runtime container ID')
            self.created = created
            self.run(name + '-start', [*self.docker, 'start', '--attach', self.created])
            self.run(name + '-inspect', [*self.docker, 'inspect', self.created])
            actual, = json.loads((self.output / (name + '-inspect.stdout.log')).read_text(encoding='utf-8-sig'))
            mounts = {row['Destination'].lower(): row for row in actual['Mounts']}
            if (actual['Image'] != self.image['Id'] or actual['HostConfig']['Isolation'] != 'process'
                    or actual['HostConfig']['NetworkMode'] != 'none' or actual['State']['ExitCode'] != 0 or actual['State']['Running']
                    or set(mounts) != set(expected_mounts)
                    or any(row['RW'] != expected_mounts[target][1]
                           or Path(row['Source']).resolve(strict=True) != expected_mounts[target][0]
                           for target, row in mounts.items())):
                raise ValueError('actual Windows runtime container controls differ')
            control = json.loads((results / 'controls.json').read_text(encoding='utf-8-sig'))
            if (not control['input_read'] or control['input_sha256'] != before['control.txt']['sha256']
                    or not control['input_write_denied'] or control['input_write_win32_error'] != 5
                    or control['operational_non_loopback_interfaces'] or control['process_architecture'] != 'AMD64'
                    or not control['development_tools_absent_from_path'] or not control['visual_studio_directory_absent']
                    or byte_inventory(inputs) != before):
                raise ValueError('actual Windows runtime read/network/tool/input controls failed')
            child = json.loads((results / 'child.json').read_text(encoding='utf-8-sig'))
            if child['completed'] is not True or child.get('error') or ((child['returncode'] == 0) == negative):
                raise ValueError('actual Windows helper result differs from contract')
            if capture:
                audited = self.audit_modules(child, executable)
                (run / 'loaded-modules.json').write_text(json.dumps(audited, indent=2) + '\n', encoding='utf-8')
            return results
        finally:
            if self.created:
                self.run(name + '-remove', [*self.docker, 'rm', '--force', self.created])
                self.created = None

    def exercise(self):
        assets, library = 'C:\\inputs\\assets', 'C:\\inputs\\relocated runtime ü\\lib\\' + self.package['entry_library']
        wav = assets + '\\ami_en2002d_2132.wav'
        native_dir = self.container('native', 'native-probe.exe', [assets + '\\Nemotron-3-Diarization.q8_0.gguf', wav, '1', '-1', 'C:\\results\\native.json'])
        native = json.loads((native_dir / 'native.json').read_text(encoding='utf-8-sig'))
        public_dir = self.container('public', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\public.f32'])
        public = json.loads((public_dir / 'child.stdout.log').read_text(encoding='utf-8-sig'))
        (public_dir / 'public.json').write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8')
        comparison = compare(public, native, (native_dir / 'native.json.f32').read_bytes(), (public_dir / 'public.f32').read_bytes())
        expected_pcm = json.loads((self.tools / 'valid-mel-platform-native.protocol.json').read_text())['cases']['actual_pcm_sha256']
        if public['input_pcm_f32le_sha256'] != expected_pcm or native['input_pcm_f32le_sha256'] != expected_pcm or native['input_repeats'] != 1 or public['input_repeats'] != 1:
            raise ValueError('actual Windows native/public PCM identity differs')
        reference = self.work / 'reference'
        reference.mkdir()
        (reference / 'ami.json').write_text(json.dumps({'segments': [{'start': turn['start_seconds'], 'end': turn['end_seconds'], 'speaker': turn['speaker']} for turn in native['turns']]}) + '\n', encoding='utf-8')
        shutil.copy2(native_dir / 'native.json.f32', reference / 'ami.json.f32')
        self.with_reference_inventory = byte_inventory(self.work)
        (self.output / 'lifecycle-inputs.json').write_text(json.dumps(self.with_reference_inventory, indent=2) + '\n', encoding='utf-8')
        listed = self.container('lifecycle-list', 'native-lifecycle.exe', ['--ignored', '--list'], capture=False)
        require_lifecycle((listed / 'child.stdout.log').read_text(encoding='utf-8-sig'))
        completed = self.container('lifecycle', 'native-lifecycle.exe', ['--ignored', '--test-threads=1'])
        require_lifecycle((completed / 'child.stdout.log').read_text(encoding='utf-8-sig'), completed=True)
        negative = self.container('missing-private-library', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\negative.f32'], capture=False, negative=True)
        error = (negative / 'child.stderr.log').read_text(encoding='utf-8-sig')
        if 'os error 126' not in error or (negative / 'negative.f32').exists():
            raise ValueError('missing private Windows library did not fail closed with native missing-module cause')
        if self.central_crt is not None:
            installed_image = self.image
            try:
                self.image = self.base_image
                absent = self.container('missing-central-runtime', 'public-stream.exe',
                    [assets, library, wav, '1', 'C:\\results\\absent.f32'], capture=False, negative=True, remove_private=False)
            finally:
                self.image = installed_image
            child = json.loads((absent / 'child.json').read_text(encoding='utf-8-sig'))
            if child['returncode'] != -1073741515 or (absent / 'absent.f32').exists():
                raise ValueError('missing central runtime did not fail with Windows STATUS_DLL_NOT_FOUND and no output')
            self.receipt['missing_central_runtime_control_passed'] = True
        if byte_inventory(self.work) != self.with_reference_inventory or verify(self.runtime) != self.package:
            raise ValueError('base Windows runtime inputs changed during qualification')
        self.receipt.update(offline_container_gates_passed=True, native_public_comparison=comparison,
                            lifecycle_tests=sorted(LIFECYCLE_GATES), negative_control_passed=True)

    def main(self):
        try:
            self.prepare()
            self.exercise()
        except Exception as error:
            self.receipt['error'] = str(error)
            raise
        finally:
            (self.output / 'receipt.json').write_text(json.dumps(self.receipt, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sdk-prefix', 'native-evidence', 'package-evidence', 'prerequisites', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--central-setup', type=Path, help='Successful same-source setup evidence for a five-DLL central runtime only.')
    args = parser.parse_args()
    if platform.system() != 'Windows':
        parser.error('requires actual Windows Docker host')
    args.output.mkdir(parents=True, exist_ok=False)
    Qualification(args).main()


if __name__ == '__main__':
    main()
