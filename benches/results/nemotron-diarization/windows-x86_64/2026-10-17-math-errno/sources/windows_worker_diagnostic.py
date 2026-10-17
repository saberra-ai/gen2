"""Windows native worker matrix; diagnostic builds cannot qualify packages."""
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
from ctest_output import require_single_test_pass
from frozen_git_sources import verify_git_sources
from platform_native import (Qualification as BaseQualification, capture, msvc_compiler_definitions,
                             msvc_probe_command, msvc_tool, powershell_literal)
from platform_valid_mel_sources import audit_cpu_tree
from platform_bounded_sources import verify_completed_source as verify_bounded_source
from pooled_runtime import BOUNDED_PROFILE, load_profile, compatibility_patch_bytes
from qualification_storage import atomic_json
from windows_compat import apply_windows_lf_compat
from windows_naming import prepare_private_names, verify_private_outputs
from windows_worker_metrics import ORDER, PCM_SHA256, DIARIZER, BACKEND, metadata, version, transform, derive_probe, telemetry, compare_native
from windows_worker_sources import verify_completed_source


class Diagnostic(BaseQualification):
    # Frozen Windows build mechanics reused by value in this separately versioned
    # runner; the historical producer and production runtime lock stay unchanged.
    def __init__(self, source, prefix, output, jobs, cap, protocol, frozen):
        super().__init__(source, prefix, output, jobs, True)
        self.lock_path, self.lock = load_profile(BOUNDED_PROFILE)
        self.cap, self.protocol = cap, protocol
        self.frozen = frozen
        self.native_probe_source = None
        self.compiler_audit = None
        self.lock['runtime']['cmake_definitions']['NEMO_SPEECH_VERSION_METADATA'] = metadata(cap)

    def verify_sources(self, build=None):
        verify_completed_source(self.source, build, self.lock, self.patch_path, True,
                                self.protocol['variants'][str(self.cap)]['source_sha256'])
        if self.native_probe_source is not None:
            if (self.output / 'native_stream_probe_bounded.cpp').read_bytes() != derive_probe(
                    (self.tools / 'native_stream_probe_valid_mel.cpp').read_bytes(), self.cap):
                raise ValueError('diagnostic probe changed')

    def verify_source(self):
        current = self.lock
        self.lock = json.loads((self.tools / 'runtime.lock.json').read_text())
        try:
            super().verify_source()
        finally:
            self.lock = current
        for index, item in enumerate(self.lock['runtime']['compatibility_patches']):
            patch = self.tools / item['path']
            self.run('inference-patch-check-' + str(index), ['git', 'apply', '--check', patch], self.source)
            self.run('inference-patch-' + str(index), ['git', 'apply', patch], self.source)
            shutil.copy2(patch, self.output / patch.name)
        self.patch_path = self.output / 'bounded-inference.patch'
        self.patch_path.write_bytes(compatibility_patch_bytes(self.lock))
        verify_bounded_source(self.source, None, self.lock, self.patch_path, True)
        for name in (DIARIZER, BACKEND):
            path = self.source / name
            path.write_bytes(transform(name, path.read_bytes(), self.cap))
        self.verify_sources()
        (self.output / 'diagnostic-inference.diff').write_bytes(subprocess.check_output(
            ['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'], cwd=self.source,
            env=self.environment))

    def run(self, name, command, cwd=None, environment=None):
        self.frozen()
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

    def build(self):
        flags = [f'-D{k}={v}' for k, v in self.lock['runtime']['cmake_definitions'].items()]
        flags += ['-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_AVX512=OFF',
                  '-DGGML_AVX_VNNI=OFF', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
                  '-DNEMO_SPEECH_WITH_NORM=OFF', '-DCMAKE_INSTALL_LIBDIR=lib',
                  '-DNEMO_SPEECH_BUILD_TESTS=ON', '-DBUILD_TESTING=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                  '-DNEMO_SPEECH_GGML_PATCHED=ON', '-DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR:PATH=',
                  '-DGGML_BACKEND_DL=OFF', '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
                  f'-DCMAKE_INSTALL_PREFIX={self.prefix}']
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
        self.run('sdk-install', ['cmake', '--install', str(build), '--config', 'Release'])
        self.build_directory = build
        self.verify_sources(build)
        def replay_run(name, argv, cwd=None, overrides=None):
            self.run(name, argv, cwd, {**self.environment, **(overrides or {})})
        self.compiler_audit = audit_cpu_tree(self.source, build, self.output,
                                           self.lock['runtime']['llama_revision'], replay_run)
        for name in ['CMakeCache.txt', 'compile_commands.json']:
            shutil.copy2(build / name, self.output / name)
        self.run('model-free-state-test', ['ctest', '--test-dir', build, '--output-on-failure', '-R', '^diar_state$'])
        state = (self.output / 'model-free-state-test.stdout.log').read_text()
        require_single_test_pass(state, 'diar_state')
        stem = 'gen2_nemo_windows_nemo_speech_asr_c' if self.private_windows_names else 'nemo_speech_asr_c'
        self.library = self.prefix / (f'bin/{stem}.dll' if self.windows else 'lib/libnemo_speech_asr_c.so')
        if not self.library.is_file():
            raise ValueError('installed C ABI library missing')
        self.probe = self.prefix / ('bin/gen2_native_stream_probe_valid_mel.exe' if self.windows else 'bin/gen2_native_stream_probe_valid_mel')
        hashing = self.source / 'llama.cpp/vendor/hash'
        original = self.tools / 'native_stream_probe_valid_mel.cpp'
        derived = self.output / 'native_stream_probe_bounded.cpp'
        derived.write_bytes(derive_probe(original.read_bytes(), self.cap))
        self.native_probe_source = {'original': descriptor(original), 'derived': descriptor(derived),
            'derivation': 'One exact historical quoted version literal replaced with diagnostic-only cap identity.'}
        sources = [derived, hashing / 'sha256/sha256.c']
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
        self.run('loader-dependencies', [msvc_tool(self.environment, 'dumpbin.exe'), '/nologo', '/dependents', self.library])
        query = ('import ctypes,os,sys; directory=os.add_dll_directory(sys.argv[2]) if os.name=="nt" else None; '
                 'library=ctypes.CDLL(sys.argv[1]); function=library.nemo_speech_asr_version; '
                 'function.restype=ctypes.c_char_p; print(function().decode("utf-8"))')
        self.run('version-only', [sys.executable, '-c', query, self.library, self.library.parent])
        self.reported_version = (self.output / 'version-only.stdout.log').read_text().strip()
        if self.reported_version != version(self.cap):
            raise ValueError('actual platform runtime tag differs before model admission')
        self.verify_sources(build)


def installed_files(root):
    return {p.relative_to(root).as_posix(): descriptor(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='pristine pinned native checkout with llama submodule')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 host required')
    tools = Path(__file__).resolve().parent
    repo, source, output = tools.parents[1], args.source.resolve(strict=True), args.output.resolve()
    if (args.jobs < 1 or output.exists() or output.is_symlink() or output.is_relative_to(repo)
            or output.is_relative_to(source) or source.is_relative_to(output)):
        parser.error('fresh disjoint output outside repository/native source and positive jobs required')
    protocol_path = tools / 'windows-worker-diagnostic.protocol.json'
    raw_protocol = protocol_path.read_bytes()
    p = json.loads(raw_protocol)
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)

    def frozen():
        if protocol_path.read_bytes() != raw_protocol or p['order'] != ORDER or p['actual_pcm_sha256'] != PCM_SHA256:
            raise ValueError('worker diagnostic protocol/order/PCM changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('diagnostic frozen input differs: ' + name)

    frozen()
    expected = {**{'tools/nemotron-diarization/' + k: v for k, v in p['tools'].items()}, **p['repository_files']}
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + protocol_path.name],
            cwd=repo, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}) != raw_protocol:
        raise ValueError('committed diagnostic protocol required before execution')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(raw_protocol)
    r = {'schema_version': 1, 'source_commit': commit, 'protocol': descriptor(protocol_path),
         'diagnostic_completed': False, 'release_qualified': False, 'package_qualified': False,
         'development_native_gates_passed': False, 'order': ORDER, 'builds': {}, 'cases': [],
         'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    persist()
    runners = {}
    try:
        for cap in (4, 2, 1):
            evidence = output / ('evidence-' + str(cap))
            evidence.mkdir()
            runner = Diagnostic(output / ('source-' + str(cap)), output / ('sdk-' + str(cap)),
                                evidence, args.jobs, cap, p, frozen)
            runners[cap] = runner
            for root, origin, revision in [(runner.source, source, runner.lock['runtime']['revision']),
                    (runner.source / 'llama.cpp', source / 'llama.cpp', runner.lock['runtime']['llama_revision'])]:
                name = 'native' if root == runner.source else 'llama'
                runner.run('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, root])
                runner.run('checkout-' + name, ['git', '-C', root, 'checkout', '--detach', revision])
            try:
                runner.verify_source()
                runner.build()
            finally:
                r['builds'][str(cap)] = {'reported_version': getattr(runner, 'reported_version', None),
                    'commands': runner.commands, 'native_probe_source': runner.native_probe_source,
                    'installed_files': installed_files(runner.prefix),
                    'compiler_audit': descriptor(evidence / 'platform-llama-materialized-audit.json') if runner.compiler_audit else None,
                    'windows_private_names': runner.naming, 'build_compatibility': runner.compatibility}
                persist()
        # No model is acquired until all three source/compiler/state/version
        # admissions succeed. Each inference uses a new OS process.
        assets = output / 'assets'
        assets.mkdir()
        for item in runners[4].lock['artifacts']:
            if item['role'] != 'reference':
                frozen()
                acquire(item, assets)
                if not matches(assets / item['name'], item):
                    raise ValueError('diagnostic asset differs: ' + item['name'])
        asset_inventory = installed_files(assets)
        baseline = baseline_raw = None
        for index, cap in enumerate(ORDER):
            frozen()
            for runner in runners.values():
                runner.verify_sources(runner.build_directory)
                if (installed_files(runner.prefix) != r['builds'][str(runner.cap)]['installed_files']
                        or source_inventory(runner.build_directory / '_deps/llama.cpp') != runner.compiler_audit['files']):
                    raise ValueError('diagnostic SDK/compiler tree changed')
            if installed_files(assets) != asset_inventory:
                raise ValueError('diagnostic assets changed')
            runner = runners[cap]
            phase = 'case-' + str(index) + '-cap-' + str(cap)
            target = runner.output / (phase + '.json')
            case = {'index': index, 'cap': cap, 'command_started': True}
            r['cases'].append(case)
            persist()
            runner.run(phase, [runner.probe, assets / 'Nemotron-3-Diarization.q8_0.gguf',
                       assets / 'ami_en2002d_2132.wav', '1', '-1', target])
            actual, raw = json.loads(target.read_text()), Path(str(target) + '.f32').read_bytes()
            case['workers'] = telemetry((runner.output / (phase + '.log')).read_text(), cap)
            if case['workers']['effective_threads'] != cap:
                raise ValueError('host does not support the requested fixed worker matrix')
            if baseline is None:
                baseline, baseline_raw = actual, raw
            case.update(result=actual, result_file=descriptor(target), probabilities=descriptor(Path(str(target) + '.f32')),
                        comparison=compare_native(baseline, actual, baseline_raw, raw, 4, cap))
            # Record all original push median/p95/max/finish/load timings above;
            # no timing averaging or automatic performance promotion occurs.
            persist()
        frozen()
        if installed_files(assets) != asset_inventory:
            raise ValueError('diagnostic assets changed after inference')
        for runner in runners.values():
            runner.verify_sources(runner.build_directory)
            if (installed_files(runner.prefix) != r['builds'][str(runner.cap)]['installed_files']
                    or source_inventory(runner.build_directory / '_deps/llama.cpp') != runner.compiler_audit['files']):
                raise ValueError('diagnostic SDK/compiler tree changed after inference')
        r['assets'] = asset_inventory
        r['diagnostic_completed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        for cap, runner in runners.items():
            r.setdefault('commands', {})[str(cap)] = runner.commands
        persist()


if __name__ == '__main__':
    main()
