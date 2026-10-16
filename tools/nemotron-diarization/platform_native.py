"""Linux/Windows CPU development qualification; not a portable release package.

Build the pinned upstream SDK (with a recorded Windows CMake-only LF overlay),
compare real public/native output on the same host, then run Gen2 lifecycle gates.
Installed dependency paths and all commands
are retained. Offline relocation and mixed-model packaging are separate gates.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from artifacts import acquire, digest, matches
from sdk_archive import write_archive
from windows_compat import apply_windows_lf_compat


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, **kwargs).strip()


def msvc_environment(environment):
    """Import the compiler environment only into this process's child mapping."""
    env = environment.copy()
    variables = {name.upper(): value for name, value in env.items()}
    vswhere = Path(variables['PROGRAMFILES(X86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    install = capture([str(vswhere), '-latest', '-products', '*', '-requires',
                       'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'])
    vcvars = Path(install) / 'VC/Auxiliary/Build/vcvars64.bat'
    if not install or not vcvars.is_file() or '"' in str(vcvars):
        raise ValueError('MSVC x64 compiler environment unavailable')
    result = subprocess.check_output(f'cmd.exe /d /u /s /c ""{vcvars}" >NUL 2>&1 && set"', env=env)
    for line in result.decode('utf-16-le').splitlines():
        name, separator, value = line.partition('=')
        if name and separator:
            # Windows environment keys are case-insensitive; avoid duplicate
            # Path/PATH entries when passing a Python mapping to CreateProcess.
            for old in list(env):
                if old.lower() == name.lower():
                    del env[old]
            env[name] = value
    return env


def powershell_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def msvc_tool(environment, name):
    variables = {name.upper(): value for name, value in environment.items()}
    executable = shutil.which(name, path=variables.get('PATH', ''))
    if not executable:
        raise ValueError(f'MSVC compiler environment does not provide {name}')
    return Path(executable).resolve(strict=True)


def msvc_compiler_definitions(environment):
    compiler = msvc_tool(environment, 'cl.exe')
    # Ninja otherwise prefers an ambient cc/c++ (MinGW on hosted Windows),
    # even after vcvars has provided a valid MSVC include/library environment.
    return [f'-DCMAKE_C_COMPILER={compiler}', f'-DCMAKE_CXX_COMPILER={compiler}']


def msvc_probe_command(environment, sources, includes, import_library, output):
    return [msvc_tool(environment, 'cl.exe'), '/nologo', '/std:c++17', '/O2', '/EHsc', '/TP',
            *sources, *['/I' + str(path) for path in includes],
            '/link', import_library, '/OUT:' + str(output)]


def pinned_git_environment(environment):
    """Keep archive/apply byte semantics consistent in all build descendants."""
    env = environment.copy()
    count = int(env.get('GIT_CONFIG_COUNT', '0'))
    if count < 0:
        raise ValueError('invalid Git environment configuration count')
    for name, value in [('core.autocrlf', 'false'), ('core.eol', 'lf')]:
        env[f'GIT_CONFIG_KEY_{count}'] = name
        env[f'GIT_CONFIG_VALUE_{count}'] = value
        count += 1
    env['GIT_CONFIG_COUNT'] = str(count)
    return env


class Qualification:
    def __init__(self, source, prefix, output, jobs):
        self.repo = Path(__file__).resolve().parents[2]
        self.tools = Path(__file__).resolve().parent
        self.source, self.prefix, self.output = source.resolve(), prefix.resolve(), output.resolve()
        self.jobs = jobs
        self.windows = platform.system() == 'Windows'
        environment = msvc_environment(dict(os.environ)) if self.windows else dict(os.environ)
        self.environment = pinned_git_environment(environment)
        self.commands = []
        self.compatibility = None
        self.archive = None
        self.lock_path = self.tools / 'runtime.lock.json'
        self.lock = json.loads(self.lock_path.read_text())

    def run(self, name, command, cwd=None, environment=None):
        command = list(map(str, command))
        self.commands.append({'name': name, 'command': command, 'cwd': str(cwd or self.repo)})
        (self.output / 'commands.json').write_text(json.dumps(self.commands, indent=2) + '\n')
        print('START ' + name, flush=True)
        with (self.output / (name + '.stdout.log')).open('x', encoding='utf-8') as out, (self.output / (name + '.log')).open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=cwd or self.repo, env=environment or self.environment,
                           stdout=out, stderr=log, check=True)
        print('PASS ' + name, flush=True)

    def verify_source(self):
        runtime = self.lock['runtime']
        if capture(['git', 'rev-parse', 'HEAD'], cwd=self.source) != runtime['revision']:
            raise ValueError('native source revision differs')
        if capture(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=self.source):
            raise ValueError('native source has tracked changes')
        if capture(['git', 'rev-parse', 'HEAD'], cwd=self.source / 'llama.cpp') != runtime['llama_revision']:
            raise ValueError('native llama revision differs')
        for name, expected in runtime['source_sha256'].items():
            if digest(self.source / name) != expected:
                raise ValueError('native source hash differs: ' + name)
        # The independent probe's hashing code must also be pristine.
        for name in ['sha256/sha256.c', 'sha256/sha256.h', 'rotate-bits/rotate-bits.h']:
            path = self.source / 'llama.cpp/vendor/hash' / name
            pinned = subprocess.check_output(['git', 'show', 'HEAD:vendor/hash/' + name], cwd=self.source / 'llama.cpp')
            if path.read_bytes() != pinned:
                raise ValueError('probe SHA256 source changed: ' + name)

    def build(self):
        flags = [f'-D{k}={v}' for k, v in self.lock['runtime']['cmake_definitions'].items()]
        flags += ['-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_AVX512=OFF',
                  '-DGGML_AVX_VNNI=OFF', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
                  '-DNEMO_SPEECH_WITH_NORM=OFF', '-DCMAKE_INSTALL_LIBDIR=lib',
                  f'-DCMAKE_INSTALL_PREFIX={self.prefix}']
        if self.windows:
            flags += msvc_compiler_definitions(self.environment)
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
        self.library = self.prefix / ('bin/nemo_speech_asr_c.dll' if self.windows else 'lib/libnemo_speech_asr_c.so')
        if not self.library.is_file():
            raise ValueError('installed C ABI library missing')
        self.probe = self.prefix / ('bin/gen2_native_stream_probe.exe' if self.windows else 'bin/gen2_native_stream_probe')
        hashing = self.source / 'llama.cpp/vendor/hash'
        sources = [self.tools / 'native_stream_probe.cpp', hashing / 'sha256/sha256.c']
        includes = [self.prefix / 'include', self.source / 'examples', hashing]
        if self.windows:
            command = msvc_probe_command(self.environment, sources, includes,
                                         self.prefix / 'lib/nemo_speech_asr_c.lib', self.probe)
        else:
            command = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-x', 'c++', *sources,
                       *['-I' + str(p) for p in includes], '-L' + str(self.prefix / 'lib'), '-lnemo_speech_asr_c',
                       '-Wl,-rpath,' + str(self.prefix / 'lib'), '-o', self.probe]
        self.run('native-probe-build', command, self.output)
        self.run('gen2-build', ['cargo', 'build', '--locked', '--release', '--no-default-features',
                              '--features', 'backend-diar-nemo', '--example', 'nemotron_stream', '--test', 'diarization_native'])
        self.run('loader-dependencies', self.dependency_command(self.library))

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
            control = '''import ctypes, sys
try:
    ctypes.CDLL(sys.argv[1], winmode=0)
except OSError as error:
    if error.winerror != 126:
        raise
    print('Confirmed default Windows loader cannot resolve sibling SDK DLLs:', error)
else:
    raise RuntimeError('Default Windows loader unexpectedly resolved SDK dependencies; control invalid')
'''
            self.run('windows-default-loader-control', [sys.executable, '-c', control, self.library])
        target = Path(self.environment.get('CARGO_TARGET_DIR', self.repo / 'target'))
        if not target.is_absolute():
            target = self.repo / target
        executable = target / ('release/examples/nemotron_stream.exe' if self.windows else 'release/examples/nemotron_stream')
        # Deliberately do not prepend the SDK to PATH/LD_LIBRARY_PATH: an
        # absolute host library path must load its installed siblings itself.
        self.run('public-inference', [executable, assets, self.library, wav, '1', self.output / 'public.f32'])
        public = json.loads((self.output / 'public-inference.stdout.log').read_text())
        (self.output / 'public.json').write_text(json.dumps(public, indent=2) + '\n')
        from compare_stream import compare
        native_output = json.loads(native.read_text())
        comparison = compare(public, native_output, Path(str(native) + '.f32').read_bytes(), (self.output / 'public.f32').read_bytes())
        if public['input_pcm_f32le_sha256'] != native_output['input_pcm_f32le_sha256'] or public['input_repeats'] != 1 or native_output['input_repeats'] != 1:
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
        listed = {line.removesuffix(': test') for line in (self.output / 'native-lifecycle-list.stdout.log').read_text().splitlines() if line.endswith(': test')}
        if not expected <= listed:
            raise ValueError('required real-model lifecycle gates were not compiled: ' + str(sorted(expected - listed)))
        self.run('native-lifecycle', [*command, '--ignored', '--test-threads=1'], environment=env)

    def receipt(self, passed, error=None):
        files = {}
        for p in sorted(self.prefix.rglob('*')):
            if p.is_file() and not p.is_symlink():
                files[str(p.relative_to(self.prefix))] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        return {'schema_version': 1, 'release_qualified': False, 'development_native_gates_passed': passed,
                'error': error, 'scope': 'Linux/Windows x86_64 AVX2 CPU source SDK and same-host actual native/public lifecycle; not offline packaging',
                'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
                'source_commit': capture(['git', 'rev-parse', 'HEAD'], cwd=self.repo),
                'runtime_revision': self.lock['runtime']['revision'], 'llama_revision': self.lock['runtime']['llama_revision'],
                'lock_sha256': digest(self.lock_path), 'runner_sha256': digest(Path(__file__)), 'commands': self.commands,
                'installed_files': files, 'build_compatibility': self.compatibility,
                'development_sdk_archive': self.archive,
                'limits': ['No offline/fresh-machine relocation claim.',
                'No chat/Bark/Laya coexistence or heldout quality claim.', 'CPU baseline explicitly requires AVX2/FMA/F16C.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('this development run requires Linux/Windows x86_64')
    if args.jobs < 1 or args.output.exists() or args.prefix.exists():
        parser.error('positive jobs and new output/SDK directories required')
    args.output.mkdir(parents=True)
    runner = Qualification(args.source, args.prefix, args.output, args.jobs)
    try:
        runner.verify_source()
        runner.build()
        runner.infer()
        if not runner.windows:
            runner.archive = write_archive(runner.prefix, runner.output)
    except Exception as error:
        (args.output / 'receipt.json').write_text(json.dumps(runner.receipt(False, str(error)), indent=2) + '\n')
        raise
    (args.output / 'receipt.json').write_text(json.dumps(runner.receipt(True), indent=2) + '\n')


if __name__ == '__main__':
    main()
