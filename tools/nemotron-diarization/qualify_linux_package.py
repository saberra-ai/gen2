"""Exercise a delivered Ubuntu 24.04 runtime in fresh network-disabled containers.

Qualification helpers are prebuilt on the CI host. The runtime image is pulled
and frozen by content identity before testing; no package installation or model
acquisition occurs inside it. This does not qualify chat symbol isolation,
other Linux baselines, physical machines or held-out accuracy.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess

from artifacts import acquire, matches
from evidence_commands import CommandRunner
from platform_valid_mel_native import compare_complete_60s as compare
from platform_sdk_receipt import validate_native_receipt
from runtime_profiles import load_profile, profile_provenance, validate_provenance
from package_linux import PLATFORM_LIBRARIES, digest, inspect_elf, verify
from sdk_archive import inventory, restore_archive

LIFECYCLE_GATES = {
    'dropping_a_diarization_child_keeps_the_parent_pipeline_active',
    'silence_short_tail_and_idle_deadline_have_explicit_terminal_results',
    'repeated_reload_full_queue_cancel_and_dropped_stream_release_resources',
    'streamed_uneven_pcm_matches_recording_and_finish_is_idempotent',
    'native_admission_cancellation_and_failed_load_release_reservations',
    'verified_provisioned_cache_is_reused_offline',
    'real_diarizer_matches_native_reference_and_resets_recordings',
}
RUNTIME_PATH = '/work/relocated runtime ü'


def loader_paths(text, libraries, extra_paths=()):
    """Require actual initialization of the complete bundled closure, including GCC."""
    raw = {path.strip() for path in re.findall(r'calling init:\s*([^\r\n]+)', text)}
    if any(not path.startswith('/') for path in raw):
        raise ValueError('loader trace contains a non-absolute library path')
    # The loader can retain $ORIGIN/../ in l_name. Both mounted helper/package
    # directories are real directories with verified inventories, not aliases.
    paths = {posixpath.normpath(path) for path in raw}
    expected = {RUNTIME_PATH + '/lib/' + name for name in libraries} | set(extra_paths)
    if not expected <= paths:
        raise ValueError('bundled libraries were not initialized: ' + str(sorted(expected - paths)))
    platform_paths = set()
    for path in paths - expected:
        name = PurePosixPath(path).name
        if name not in PLATFORM_LIBRARIES or not path.startswith(('/lib/', '/lib64/', '/usr/lib/')):
            raise ValueError('unexpected loaded library outside the package: ' + path)
        platform_paths.add(path)
    return {'bundled': sorted(expected), 'platform': sorted(platform_paths),
            'raw_initialization_paths': sorted(raw)}


def require_lifecycle(text, completed=False):
    if completed:
        passed = set(re.findall(r'^test ([A-Za-z0-9_]+) \.{3} ok$', text, re.MULTILINE))
    else:
        passed = {line.removesuffix(': test') for line in text.splitlines() if line.endswith(': test')}
    if passed != LIFECYCLE_GATES:
        raise ValueError('compiled/executed real-model gates differ: ' + str(sorted(passed ^ LIFECYCLE_GATES)))



def require_constructor_sources(build, tools):
    # SDK, package and runtime jobs may have distinct commits. Require the
    # constructor bytes used by that package; actual receipts stay bound.
    for field, name in [('packager_sha256', 'package_linux.py'),
                        ('archive_tool_sha256', 'sdk_archive.py'),
                        ('runner_sha256', 'platform_linux_package.py')]:
        if build.get(field) != digest(tools / name):
            raise ValueError('qualified package constructor source differs: ' + name)


def container_command(image, mounts, results, command, environment=None):
    args = ['docker', 'run', '--rm', '--pull', 'never', '--platform', 'linux/amd64',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m',
            '--workdir', '/tmp']
    for source, destination in [*mounts, (results, '/results')]:
        if ',' in str(source) or ',' in destination:
            raise ValueError('Docker mount path cannot contain a comma')
        mode = '' if destination == '/results' else ',readonly'
        args += ['--mount', f'type=bind,src={Path(source).resolve()},dst={destination}{mode}']
    variables = {'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'LC_ALL': 'C.UTF-8', **(environment or {})}
    if any(name in variables for name in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT']):
        raise ValueError('qualification cannot override dependency resolution')
    return [*args, image, '/usr/bin/env', '-i', *[name + '=' + value for name, value in variables.items()], *command]


class Qualification(CommandRunner):
    def __init__(self, args):
        self.args = args
        self.tools = Path(__file__).resolve().parent
        self.repo = self.tools.parents[1]
        self.output = args.output.resolve()
        self.work = self.output / 'inputs'
        self.results = self.output / 'results'
        super().__init__(self.output, self.repo)
        lock_path, lock = load_profile()
        self.receipt = {**profile_provenance(lock_path, lock), 'schema_version': 2, 'release_qualified': False,
                        'offline_container_gates_passed': False,
                        'scope': 'Delivered Ubuntu 24.04 x86_64 CPU runtime relocation, native/public parity and lifecycle in fresh offline containers.',
                        'limits': ['No chat/Bark/Laya namespace coexistence qualification.',
                                   'No other OS/glibc/CPU baseline or separate physical machine claim.',
                                   'No held-out accuracy or uncontended performance claim.']}

    def container(self, name, command, environment=None, runtime=None, expected_failure=False):
        mounts = [(self.work / 'bin', '/work/bin'), (self.work / 'assets', '/work/assets'),
                  (self.work / 'reference', '/work/reference'),
                  (runtime or self.runtime, RUNTIME_PATH)]
        return self.run(name, container_command(self.image, mounts, self.results, command, environment), expected_failure)

    def prepare(self):
        self.work.mkdir()
        self.results.mkdir()
        evidence = self.args.package_evidence.resolve()
        build = json.loads((evidence / 'receipt.json').read_text())
        validate_provenance(build)
        if build.get('package_construction_passed') is not True:
            raise ValueError('runtime construction prerequisite has not passed')
        archive = json.loads((evidence / 'development-runtime-files.json').read_text())
        for key in ['archive', 'archive_bytes', 'archive_sha256']:
            if archive[key] != build['archive'][key]:
                raise ValueError('archive identity differs from construction receipt')
        self.runtime = restore_archive(evidence / 'development-runtime.tar.gz', archive, self.work / 'relocated runtime ü')
        if digest(self.runtime / 'package.json') != build['package_manifest_sha256']:
            raise ValueError('delivered package manifest differs from construction receipt')
        self.package = verify(self.runtime)
        native = self.args.native_evidence.resolve()
        native_receipt = json.loads((native / 'receipt.json').read_text())
        validate_native_receipt(native_receipt)
        if digest(native / 'receipt.json') != build['native_receipt_sha256'] or native_receipt.get('development_native_gates_passed') is not True:
            raise ValueError('package does not bind a successful native prerequisite')
        current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        require_constructor_sources(build, self.tools)
        _, lock = load_profile()
        assets = self.work / 'assets'
        assets.mkdir()
        for entry in lock['artifacts']:
            if entry['role'] == 'reference':
                continue
            source = native / 'assets' / entry['name']
            if source.exists():
                if not matches(source, entry):
                    raise ValueError('locked asset differs: ' + entry['name'])
                shutil.copy2(source, assets / entry['name'])
            else:
                # CI artifacts omit models; acquisition is host-side only.
                acquire(entry, assets)
            if not matches(assets / entry['name'], entry):
                raise ValueError('staged locked asset differs: ' + entry['name'])
        shutil.copytree(native / 'reference', self.work / 'reference')
        # Bind the expected lifecycle reference to the same-host native result.
        source_native = json.loads((native / 'native-ami.json').read_text())
        reference = json.loads((self.work / 'reference/ami.json').read_text())
        if reference != {'segments': [{'start': t['start_seconds'], 'end': t['end_seconds'], 'speaker': t['speaker']} for t in source_native['turns']]}:
            raise ValueError('lifecycle turn reference differs from native prerequisite')
        if digest(self.work / 'reference/ami.json.f32') != digest(native / 'native-ami.json.f32'):
            raise ValueError('lifecycle probability reference differs from native prerequisite')
        binaries = self.work / 'bin'
        binaries.mkdir()
        target = Path(os.environ.get('CARGO_TARGET_DIR', self.repo / 'target')).resolve()
        self.run('build-public-helper', ['cargo', 'build', '--locked', '--release',
                 '--no-default-features', '--features', 'backend-diar-nemo',
                 '--example', 'nemotron_valid_mel_stream'])
        metadata = self.run('discover-lifecycle-binary', ['cargo', 'test', '--locked', '--release',
                            '--no-default-features', '--features', 'backend-diar-nemo', '--test',
                            'diarization_native', '--no-run', '--message-format=json'])
        executables = [item['executable'] for line in metadata.splitlines() if (item := json.loads(line)).get('reason') == 'compiler-artifact'
                       and item['target']['name'] == 'diarization_native' and item.get('executable')]
        if len(executables) != 1:
            raise ValueError('expected one compiled native lifecycle executable')
        probe = self.args.sdk_prefix.resolve() / 'bin/gen2_native_stream_probe_valid_mel'
        if {'sha256': digest(probe), 'bytes': probe.stat().st_size} != native_receipt['installed_files']['bin/gen2_native_stream_probe_valid_mel']:
            raise ValueError('independent probe differs from qualified SDK')
        helper_sources = {'native-probe': probe, 'public-stream': target / 'release/examples/nemotron_valid_mel_stream',
                          'native-lifecycle': Path(executables[0])}
        helpers = {}
        for name, source in helper_sources.items():
            destination = binaries / name
            helpers[name] = {'source_sha256': digest(source), 'source': str(source)}
            shutil.copy2(source, destination)
            if name == 'native-probe':
                self.run('probe-private-dependency', ['patchelf', '--replace-needed', 'libnemo_speech_asr_c.so.1', self.package['entry_library'], destination])
            self.run(name + '-rpath', ['patchelf', '--force-rpath', '--set-rpath', '$ORIGIN/../relocated runtime ü/lib', destination])
            elf = inspect_elf(destination, require_soname=False)
            if elf['rpath'] != ['$ORIGIN/../relocated runtime ü/lib'] or elf['runpath']:
                raise ValueError('qualification helper has an unapproved search path')
            if any(item not in PLATFORM_LIBRARIES | set(self.package['libraries']) for item in elf['needed']):
                raise ValueError('qualification helper has an unbundled dependency')
            helpers[name].update(sha256=digest(destination), elf=elf)
        self.run('pull-runtime-image', ['docker', 'pull', '--platform', 'linux/amd64', 'ubuntu:24.04'])
        image = json.loads(self.run('inspect-runtime-image', ['docker', 'image', 'inspect', 'ubuntu:24.04']))
        if len(image) != 1 or image[0]['Architecture'] != 'amd64' or image[0]['Os'] != 'linux' or not image[0]['RepoDigests']:
            raise ValueError('runtime image lacks content-bound amd64 Linux provenance')
        self.image = image[0]['Id']
        inputs = {'source_commit': current, 'native_source_commit': native_receipt['source_commit'],
                  'construction_source_commit': build['source_commit'], 'construction_receipt_sha256': digest(evidence / 'receipt.json'),
                  'native_receipt_sha256': digest(native / 'receipt.json'), 'archive': build['archive'],
                  'package_manifest_sha256': build['package_manifest_sha256'], 'image': image[0],
                  'helpers': helpers, 'inputs': inventory(self.work),
                  'tools': {name: digest(self.tools / name) for name in ['qualify_linux_package.py', 'package_linux.py', 'sdk_archive.py', 'compare_stream_valid_mel.py', 'platform_valid_mel_native.py', 'platform_valid_mel_sources.py', 'platform_native.py', 'platform_sdk_receipt.py', 'runtime_profiles.py', 'build_valid_mel_candidate.py', 'qualification_storage.py', 'runtime-valid-mel.lock.json', 'runtime.lock.json', 'valid-mel-platform-native.protocol.json', 'evidence_commands.py']},
                  'geometry': {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222},
                  'negative_control': 'Remove private ggml-base; mount that copy at the identical runtime path so helper RPATH cannot see the good package.'}
        (self.output / 'frozen-inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
        self.receipt.update(source_commit=current, image_id=self.image, image_repo_digests=image[0]['RepoDigests'],
                            frozen_inputs_sha256=digest(self.output / 'frozen-inputs.json'), package_manifest_sha256=build['package_manifest_sha256'])

    def exercise(self):
        # The base runtime is deliberately tool-free; all build inputs remain unmounted.
        control = '''set -eu
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_ID" = 24.04
id
test -r /work/assets/ami_en2002d_2132.wav
test -r /work/assets/Nemotron-3-Diarization.q8_0.gguf
test -w /results
for tool in cc c++ gcc g++ clang cmake cargo; do
    if command -v "$tool"; then exit 10; fi
done
test ! -e /home/runner/work && test ! -e /opt/hostedtoolcache
test ! -e /usr/lib/x86_64-linux-gnu/libsentencepiece.so.0
set -- /sys/class/net/*
test "$#" -eq 1 && test "$1" = /sys/class/net/lo
if touch /work/assets/forbidden-write 2>/dev/null; then exit 11; fi
if touch /work/relocated\\ runtime\\ ü/forbidden-write 2>/dev/null; then exit 12; fi
cat /etc/os-release
cat /proc/cpuinfo
cat /proc/net/route
printenv
'''
        self.container('container-controls', ['/bin/sh', '-c', control])
        library = RUNTIME_PATH + '/lib/' + self.package['entry_library']
        environment = {'LD_DEBUG': 'libs'}
        self.container('packaged-native', ['/work/bin/native-probe', '/work/assets/Nemotron-3-Diarization.q8_0.gguf',
                       '/work/assets/ami_en2002d_2132.wav', '1', '-1', '/results/native.json'], environment)
        public_text = self.container('packaged-public', ['/work/bin/public-stream', '/work/assets', library,
                                     '/work/assets/ami_en2002d_2132.wav', '1', '/results/public.f32'], environment)
        public, native = json.loads(public_text), json.loads((self.results / 'native.json').read_text())
        (self.results / 'public.json').write_text(json.dumps(public, indent=2) + '\n')
        comparison = compare(public, native, (self.results / 'native.json.f32').read_bytes(), (self.results / 'public.f32').read_bytes())
        expected_pcm = json.loads((self.tools / 'valid-mel-platform-native.protocol.json').read_text())['cases']['actual_pcm_sha256']
        if public['input_pcm_f32le_sha256'] != expected_pcm or native['input_pcm_f32le_sha256'] != expected_pcm or public['input_repeats'] != native['input_repeats'] or native['input_repeats'] != 1:
            raise ValueError('packaged native/public actual PCM identity differs')
        (self.output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        lifecycle_env = {**environment, 'GEN2_NEMOTRON_ASSETS': '/work/assets', 'GEN2_NEMO_LIBRARY': library,
                         'GEN2_NEMOTRON_REFERENCE': '/work/reference', 'GEN2_NEMO_EXECUTION': 'cpu'}
        listed = self.container('packaged-lifecycle-list', ['/work/bin/native-lifecycle', '--ignored', '--list'], lifecycle_env)
        require_lifecycle(listed)
        completed = self.container('packaged-lifecycle', ['/work/bin/native-lifecycle', '--ignored', '--test-threads=1'], lifecycle_env)
        require_lifecycle(completed, completed=True)
        loaded = {name: loader_paths((self.output / (name + '.stderr.log')).read_text(), self.package['libraries'])
                  for name in ['packaged-native', 'packaged-public', 'packaged-lifecycle']}
        (self.output / 'loaded-libraries.json').write_text(json.dumps(loaded, indent=2) + '\n')
        platform_paths = sorted({path for item in loaded.values() for path in item['platform']})
        self.container('platform-library-hashes', ['sha256sum', '--', *platform_paths])
        negative = self.output / 'missing-library-control'
        shutil.copytree(self.runtime, negative, symlinks=True)
        missing = 'libgen2_nemo_linux_ggml-base.so.0.25.1'
        (negative / 'lib' / missing).unlink()
        self.container('missing-library-control', ['/work/bin/public-stream', '/work/assets', library,
                       '/work/assets/ami_en2002d_2132.wav', '1', '/results/negative.f32'], environment,
                       runtime=negative, expected_failure=True)
        error = (self.output / 'missing-library-control.stderr.log').read_text()
        if missing + ': cannot open shared object file' not in error or (self.results / 'negative.f32').exists():
            raise ValueError('missing-library control did not establish fail-closed loading')
        frozen = json.loads((self.output / 'frozen-inputs.json').read_text())
        if inventory(self.work) != frozen['inputs'] or verify(self.runtime) != self.package:
            raise ValueError('qualified input files changed during execution')
        self.receipt.update(offline_container_gates_passed=True, lifecycle_tests=sorted(LIFECYCLE_GATES),
                            native_public_comparison=comparison, negative_control_passed=True)

    def main(self):
        try:
            self.prepare()
            self.exercise()
        except Exception as error:
            self.receipt['error'] = str(error)
            raise
        finally:
            (self.output / 'receipt.json').write_text(json.dumps(self.receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sdk-prefix', 'native-evidence', 'package-evidence', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    Qualification(args).main()


if __name__ == '__main__':
    main()
