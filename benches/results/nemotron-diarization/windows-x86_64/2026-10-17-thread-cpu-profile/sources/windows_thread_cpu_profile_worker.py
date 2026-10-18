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
from windows_thread_cpu_profile_metrics import ORDER, PCM_SHA256, DIARIZER, BACKEND, metadata, version, transform, derive_probe, telemetry, compare_native
from windows_thread_cpu_profile_verify import checkout_environment, verify_completed_source
from windows_thread_cpu_profile_sources import install_profile


class Diagnostic(BaseQualification):
    # Frozen Windows build mechanics reused by value in this separately versioned
    # runner; the historical producer and production runtime lock stay unchanged.
    def __init__(self, source, prefix, output, jobs, cap, protocol, frozen):
        super().__init__(source, prefix, output, jobs, True)
        self.environment = checkout_environment(self.environment)
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
        install_profile(self.source, self.output)
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
