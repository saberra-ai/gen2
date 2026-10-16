"""Fresh tagged Linux/Windows CPU SDK and same-host qualification; no release claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from artifacts import acquire, digest, matches
from build_valid_mel_candidate import descriptor, source_inventory
from platform_native import (Qualification as BaseQualification, capture, msvc_compiler_definitions,
                             msvc_probe_command, msvc_tool, powershell_literal)
from platform_valid_mel_sources import audit_cpu_tree, verify_completed_source
from qualification_storage import atomic_json
from compare_stream_valid_mel import compare
from runtime_profiles import load_profile, profile_provenance, verify_source as verify_profile_source
from sdk_archive import write_archive
from windows_compat import apply_windows_lf_compat
from windows_naming import prepare_private_names, verify_private_outputs


def compare_complete_60s(public, native, raw, actual_raw):
    # This platform runner stays stdlib-only; historical macOS scripts remain frozen.
    if (public['accepted_samples'] != 960000 or native['audio_samples'] != 960000
            or public['retained_probabilities'] != {'first_frame': 0, 'rows': 6000}
            or native['retained_first_frame'] != 0 or native['retained_rows'] != 6000
            or len(raw) != 192000 or len(actual_raw) != 192000):
        raise ValueError('platform qualification requires complete6000row histories')
    return compare(public, native, raw, actual_raw)


class Qualification(BaseQualification):
    def __init__(self, source, prefix, output, jobs, private_windows_names=False):
        super().__init__(source, prefix, output, jobs, private_windows_names)
        self.lock_path, self.lock = load_profile()
        self.source_commit = capture(['git', 'rev-parse', 'HEAD'], cwd=self.repo)
        self.reported_version = None
        self.compiler_audit = None
        self.completed_lifecycle_tests = []
        self.protocol_path = self.tools / 'valid-mel-platform-native.protocol.json'
        self.protocol_bytes = self.protocol_path.read_bytes()
        self.protocol = json.loads(self.protocol_bytes)
        self.verify_protocol()
        (self.output / 'protocol.json').write_bytes(self.protocol_bytes)

    def verify_protocol(self):
        if self.protocol_path.read_bytes() != self.protocol_bytes:
            raise ValueError('platform protocol changed during qualification')
        for key, root in [('tools', self.tools), ('repository_files', self.repo)]:
            for name, expected in self.protocol[key].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen platform qualification input differs: ' + name)
        if not re.fullmatch('[0-9a-f]{64}', self.protocol['cases']['actual_pcm_sha256']):
            raise ValueError('platform protocol PCM digest is invalid')
        if self.protocol['runtime_provenance'] != profile_provenance(self.lock_path, self.lock):
            raise ValueError('platform protocol runtime profile differs')


    def run(self, name, command, cwd=None, environment=None):
        command = list(map(str, command))
        env = environment if environment is not None else self.environment
        record = {'name': name, 'command': command, 'cwd': str(cwd or self.repo),
                  'environment_overrides': {k: v for k, v in env.items() if self.environment.get(k) != v}}
        self.commands.append(record)
        atomic_json(self.output / 'commands.json', self.commands)
        print('START ' + name, flush=True)
        started = time.monotonic()
        try:
            with (self.output / (name + '.stdout.log')).open('xb') as out, (self.output / (name + '.log')).open('xb') as log:
                result = subprocess.run(command, cwd=cwd or self.repo, env=env, stdout=out, stderr=log)
            record['returncode'] = result.returncode
        except OSError as error:
            record['startup_error'] = str(error)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(self.output / 'commands.json', self.commands)
        result.check_returncode()
        print('PASS ' + name, flush=True)

    def verify_source(self):
        # The original runner's pristine-source checks remain historical controls.
        current = self.lock
        self.lock = json.loads((self.tools / 'runtime.lock.json').read_text())
        try:
            super().verify_source()
        finally:
            self.lock = current
        patch = self.tools / self.lock['runtime']['compatibility_patches'][0]['path']
        self.run('valid-mel-patch-check', ['git', 'apply', '--check', patch], self.source)
        self.run('valid-mel-patch', ['git', 'apply', patch], self.source)
        verify_profile_source(self.source, self.lock)
        shutil.copy2(patch, self.output / 'valid-mel-inference.patch')
        shutil.copy2(self.lock_path, self.output / 'runtime-profile.json')

    def build(self):
        flags = [f'-D{k}={v}' for k, v in self.lock['runtime']['cmake_definitions'].items()]
        flags += ['-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_AVX512=OFF',
                  '-DGGML_AVX_VNNI=OFF', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
                  '-DNEMO_SPEECH_WITH_NORM=OFF', '-DCMAKE_INSTALL_LIBDIR=lib',
                  '-DNEMO_SPEECH_BUILD_TESTS=ON', '-DBUILD_TESTING=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                  '-DNEMO_SPEECH_GGML_PATCHED=ON', '-DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR:PATH=',
                  '-DGGML_BACKEND_DL=OFF', '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
                  f'-DCMAKE_INSTALL_PREFIX={self.prefix}']
        if self.windows:
            flags += msvc_compiler_definitions(self.environment)
            if self.private_windows_names:
                hook, self.naming = prepare_private_names(self.source, self.tools, self.output)
                flags.append(f'-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES:FILEPATH={hook}')
            self.compatibility = apply_windows_lf_compat(self.source, self.output)
            self.run('cmake-version', ['cmake', '--version'])
            build = self.source / 'build/gen2-cpu-diar'
            flags += ['-DNEMO_SPEECH_BUILD_ASR=OFF', '-DNEMO_SPEECH_BUILD_TTS=OFF']
            script = self.source / 'scripts/windows/build.ps1'
            command = f'& {powershell_literal(script)} -Backend cpu -Profile asr -Jobs {self.jobs} -BuildDir {powershell_literal(build)} -CMakeArgs @({",".join(map(powershell_literal, flags))})'
            self.run('git-line-ending-config', ['git', '-C', self.source / 'llama.cpp', 'config', '--show-origin', '--get-regexp', r'^core\.(autocrlf|eol)$'])
            try:
                self.run('sdk-build', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', command])
            finally:
                cache = build / 'CMakeCache.txt'
                if cache.is_file():
                    shutil.copy2(cache, self.output / 'cmake-cache.log')
                paths = build / 'gen2-private-paths-Release.txt'
                if paths.is_file():
                    shutil.copy2(paths, self.output / 'windows-private-paths.log')
                # Preserve the actual patch and exported context when upstream
                # configuration fails, rather than guessing from Git globals.
                staging = build / '_deps/llama.cpp.staging'
                diagnostic = {}
                for relative in ['ggml/include/ggml.h', 'ggml/src/ggml.c']:
                    path = staging / relative
                    if path.is_file():
                        raw = path.read_bytes()
                        diagnostic[relative] = {'bytes': len(raw), 'sha256': digest(path),
                                                'crlf_count': raw.count(b'\r\n'),
                                                'head': raw[:256].decode('utf-8', errors='replace')}
                patch = build / '_deps/llama.cpp.staging.patch'
                if patch.is_file():
                    shutil.copy2(patch, self.output / 'failed-upstream.patch.log')
                (self.output / 'upstream-patch-diagnostic.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
        else:
            build = self.source / 'build/cpu-diar'
            self.run('sdk-configure', ['cmake', '--preset', 'cpu-diar', *flags], self.source)
            self.run('sdk-build', ['cmake', '--build', str(build), '--parallel', self.jobs])
        self.run('sdk-install', ['cmake', '--install', str(build), '--config', 'Release'])
        self.build_directory = build
        patch = self.tools / self.lock['runtime']['compatibility_patches'][0]['path']
        verify_completed_source(self.source, build, self.lock, patch, self.windows)
        def replay_run(name, argv, cwd=None, overrides=None):
            self.run(name, argv, cwd, {**self.environment, **(overrides or {})})
        self.compiler_audit = audit_cpu_tree(self.source, build, self.output,
                                           self.lock['runtime']['llama_revision'], replay_run)
        for name in ['CMakeCache.txt', 'compile_commands.json']:
            shutil.copy2(build / name, self.output / name)
        self.run('model-free-state-test', ['ctest', '--test-dir', build, '--output-on-failure', '-R', '^diar_state$'])
        state = (self.output / 'model-free-state-test.stdout.log').read_text()
        if not re.search(r'1/1 Test #\d+: diar_state\s+.*Passed', state) or '100% tests passed, 0 tests failed out of 1' not in state:
            raise ValueError('actual model-free platform state test did not execute and pass')
        stem = 'gen2_nemo_windows_nemo_speech_asr_c' if self.private_windows_names else 'nemo_speech_asr_c'
        self.library = self.prefix / (f'bin/{stem}.dll' if self.windows else 'lib/libnemo_speech_asr_c.so')
        if not self.library.is_file():
            raise ValueError('installed C ABI library missing')
        self.probe = self.prefix / ('bin/gen2_native_stream_probe_valid_mel.exe' if self.windows else 'bin/gen2_native_stream_probe_valid_mel')
        hashing = self.source / 'llama.cpp/vendor/hash'
        sources = [self.tools / 'native_stream_probe_valid_mel.cpp', hashing / 'sha256/sha256.c']
        includes = [self.prefix / 'include', self.source / 'examples', hashing]
        if self.windows:
            command = msvc_probe_command(self.environment, sources, includes,
                                         self.prefix / f'lib/{stem}.lib', self.probe)
        else:
            command = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-x', 'c++', *sources,
                       *['-I' + str(p) for p in includes], '-L' + str(self.prefix / 'lib'), '-lnemo_speech_asr_c',
                       '-Wl,-rpath,' + str(self.prefix / 'lib'), '-o', self.probe]
        self.run('native-probe-build', command, self.output)
        if self.private_windows_names:
            if digest(self.tools / 'windows_private_dlls.cmake') != self.naming['hook_sha256']:
                raise ValueError('Windows naming hook changed during build')
            self.naming.update(verify_private_outputs(self.prefix, self.output))
        if self.windows:
            self.run('windows-provenance', ['pwsh', '-NoProfile', '-NonInteractive', '-File',
                     self.tools / 'windows_provenance.ps1', '-SdkRoot', self.prefix,
                     '-CMakeCache', build / 'CMakeCache.txt', '-Compiler', msvc_tool(self.environment, 'cl.exe'),
                     '-Output', self.output])
        self.run('gen2-build', ['cargo', 'build', '--locked', '--release', '--no-default-features',
                              '--features', 'backend-diar-nemo', '--example', 'nemotron_valid_mel_stream', '--test', 'diarization_native'])
        self.run('loader-dependencies', self.dependency_command(self.library))
        query = ('import ctypes,os,sys; directory=os.add_dll_directory(sys.argv[2]) if os.name=="nt" else None; '
                 'library=ctypes.CDLL(sys.argv[1]); function=library.nemo_speech_asr_version; '
                 'function.restype=ctypes.c_char_p; print(function().decode("utf-8"))')
        self.run('version-only', [sys.executable, '-c', query, self.library, self.library.parent])
        self.reported_version = (self.output / 'version-only.stdout.log').read_text().strip()
        if self.reported_version != self.lock['runtime']['reported_version']:
            raise ValueError('actual platform runtime tag differs before model admission')
        verify_completed_source(self.source, build, self.lock, patch, self.windows)

    def dependency_command(self, library):
        if not self.windows:
            return ['ldd', library]
        return [msvc_tool(self.environment, 'dumpbin.exe'), '/nologo', '/dependents', library]

    def infer(self):
        assets = self.output / 'assets'
        assets.mkdir()
        entries = [item for item in self.lock['artifacts'] if item['role'] != 'reference']
        for item in entries:
            acquire(item, assets)
            if not matches(assets / item['name'], item):
                raise ValueError('real-model prerequisite corrupt: ' + item['name'])
        model = assets / 'Nemotron-3-Diarization.q8_0.gguf'
        wav = assets / 'ami_en2002d_2132.wav'
        native = self.output / 'native-ami.json'
        self.run('native-inference', [self.probe, model, wav, '1', '-1', native])
        if self.windows:
            # CPython's CDLL constructor replaces ERROR_MOD_NOT_FOUND with a
            # FileNotFoundError lacking winerror. Read the native code directly.
            self.run('windows-default-loader-control', [sys.executable,
                     self.tools / 'windows_loader_control.py', self.library])
        target = Path(self.environment.get('CARGO_TARGET_DIR', self.repo / 'target'))
        if not target.is_absolute():
            target = self.repo / target
        executable = target / ('release/examples/nemotron_valid_mel_stream.exe' if self.windows else 'release/examples/nemotron_valid_mel_stream')
        # Deliberately do not prepend the SDK to PATH/LD_LIBRARY_PATH: an
        # absolute host library path must load its installed siblings itself.
        self.run('public-inference', [executable, assets, self.library, wav, '1', self.output / 'public.f32'])
        public = json.loads((self.output / 'public-inference.stdout.log').read_text())
        (self.output / 'public.json').write_text(json.dumps(public, indent=2) + '\n')
        native_output = json.loads(native.read_text())
        comparison = compare_complete_60s(public, native_output, Path(str(native) + '.f32').read_bytes(), (self.output / 'public.f32').read_bytes())
        expected_pcm = self.protocol['cases']['actual_pcm_sha256']
        if (public['input_pcm_f32le_sha256'] != expected_pcm
                or native_output['input_pcm_f32le_sha256'] != expected_pcm
                or public['input_repeats'] != 1 or native_output['input_repeats'] != 1):
            raise ValueError('public/native actual PCM identity differs')
        (self.output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        reference = self.output / 'reference'
        reference.mkdir()
        (reference / 'ami.json').write_text(json.dumps({'segments': [
            {'start': s['start_seconds'], 'end': s['end_seconds'], 'speaker': s['speaker']} for s in native_output['turns']]}) + '\n')
        shutil.copy2(str(native) + '.f32', reference / 'ami.json.f32')
        env = {**self.environment, 'GEN2_NEMOTRON_ASSETS': str(assets), 'GEN2_NEMO_LIBRARY': str(self.library),
               'GEN2_NEMOTRON_REFERENCE': str(reference), 'GEN2_NEMO_EXECUTION': 'cpu'}
        command = ['cargo', 'test', '--locked', '--release', '--no-default-features', '--features',
                   'backend-diar-nemo', '--test', 'diarization_native', '--']
        self.run('native-lifecycle-list', [*command, '--ignored', '--list'], environment=env)
        expected = {'dropping_a_diarization_child_keeps_the_parent_pipeline_active',
                    'silence_short_tail_and_idle_deadline_have_explicit_terminal_results',
                    'repeated_reload_full_queue_cancel_and_dropped_stream_release_resources',
                    'streamed_uneven_pcm_matches_recording_and_finish_is_idempotent',
                    'native_admission_cancellation_and_failed_load_release_reservations',
                    'verified_provisioned_cache_is_reused_offline',
                    'real_diarizer_matches_native_reference_and_resets_recordings'}
        if sorted(expected) != self.protocol['required_lifecycle_tests']:
            raise ValueError('platform lifecycle selection differs from frozen protocol')
        listed = {line.removesuffix(': test') for line in (self.output / 'native-lifecycle-list.stdout.log').read_text().splitlines() if line.endswith(': test')}
        if not expected <= listed:
            raise ValueError('required real-model lifecycle gates were not compiled: ' + str(sorted(expected - listed)))
        for index, test in enumerate(sorted(expected)):
            phase = 'native-lifecycle-' + str(index)
            self.run(phase, [*command, '--ignored', '--exact', test, '--test-threads=1'], environment=env)
            stdout = (self.output / (phase + '.stdout.log')).read_text()
            if not re.search(r'^test ' + re.escape(test) + r' \.\.\. ok$', stdout, re.MULTILINE) or '1 passed; 0 failed; 0 ignored;' not in stdout:
                raise ValueError('required platform lifecycle test did not execute: ' + test)
            self.completed_lifecycle_tests.append(test)
        self.verify_protocol()
        patch = self.tools / self.lock['runtime']['compatibility_patches'][0]['path']
        verify_completed_source(self.source, self.build_directory, self.lock, patch, self.windows)
        if source_inventory(self.build_directory / '_deps/llama.cpp') != self.compiler_audit['files']:
            raise ValueError('platform compiler tree changed during neural qualification')

    def receipt(self, passed, error=None):
        files = {}
        for p in sorted(self.prefix.rglob('*')):
            if p.is_file() and not p.is_symlink():
                files[str(p.relative_to(self.prefix))] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        return {**profile_provenance(self.lock_path, self.lock), 'schema_version': 2, 'backend': 'cpu', 'release_qualified': False, 'development_native_gates_passed': passed,
                'error': error, 'scope': 'Linux/Windows x86_64 AVX2 CPU source SDK and same-host actual native/public lifecycle; not offline packaging',
                'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
                'source_commit': self.source_commit,
                'protocol_sha256': hashlib.sha256(self.protocol_bytes).hexdigest(),
                'actual_reported_version': self.reported_version,
                'model_free_state_test_completed': any(c['name'] == 'model-free-state-test' and c.get('returncode') == 0 for c in self.commands),
                'completed_lifecycle_tests': self.completed_lifecycle_tests,
                'compiler_audit': descriptor(self.output / 'platform-llama-materialized-audit.json') if self.compiler_audit else None,
                'compiler_input_audit_timing': 'after SDK build and before model admission',
                'runtime_revision': self.lock['runtime']['revision'], 'llama_revision': self.lock['runtime']['llama_revision'],
                'lock_sha256': digest(self.lock_path), 'runner_sha256': digest(Path(__file__)), 'commands': self.commands,
                'windows_loader_control_sha256': digest(self.tools / 'windows_loader_control.py') if self.windows else None,
                'windows_provenance_script_sha256': digest(self.tools / 'windows_provenance.ps1') if self.windows else None,
                'installed_files': files, 'build_compatibility': self.compatibility,
                'windows_private_names': self.naming,
                'development_sdk_archive': self.archive,
                'limits': ['Pinned valid-Mel inference patch and explicit Windows LF-only build overlay; independent GGML source replay and actual compiler-path audit occur after construction, before model admission.', 'No offline/fresh-machine relocation claim.',
                'No chat/Bark/Laya coexistence or heldout quality claim.', 'CPU baseline explicitly requires AVX2/FMA/F16C.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--windows-private-dlls', action='store_true',
                        help='Relink five Windows native DLLs under guarded private output names.')
    args = parser.parse_args()
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('this development run requires Linux/Windows x86_64')
    if args.jobs < 1 or args.output.exists() or args.prefix.exists() or (args.source / 'build').exists():
        parser.error('positive jobs, a fresh source build and new output/SDK directories required')
    roots = [p.resolve() for p in [args.source, args.prefix, args.output]]
    if len(set(roots)) != 3 or any(a != b and a.is_relative_to(b) for a in roots for b in roots):
        parser.error('isolated source, SDK and evidence roots must be disjoint')
    if args.windows_private_dlls and platform.system() != 'Windows':
        parser.error('private DLL names require Windows')
    args.output.mkdir(parents=True)
    runner = Qualification(args.source, args.prefix, args.output, args.jobs, args.windows_private_dlls)
    try:
        runner.verify_source()
        runner.build()
        runner.verify_protocol()
        runner.infer()
        if not runner.windows:
            runner.archive = write_archive(runner.prefix, runner.output)
    except Exception as error:
        (args.output / 'receipt.json').write_text(json.dumps(runner.receipt(False, str(error)), indent=2) + '\n')
        raise
    (args.output / 'receipt.json').write_text(json.dumps(runner.receipt(True), indent=2) + '\n')


if __name__ == '__main__':
    main()
