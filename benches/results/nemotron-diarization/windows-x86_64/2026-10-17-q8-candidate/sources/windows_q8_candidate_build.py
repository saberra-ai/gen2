"""Exact bounded SDK build mechanics; candidate source admission only."""
import json
import shutil
import sys

from platform_bounded_windows_native import (Qualification as BoundedQualification,
    msvc_compiler_definitions, msvc_probe_command, msvc_tool, powershell_literal,
    prepare_private_names, apply_windows_lf_compat, digest, audit_cpu_tree,
    require_single_test_pass, derive_native_probe, descriptor, verify_private_outputs)
from windows_q8_candidate_verify import verify_completed_source


class Qualification(BoundedQualification):
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
        patch = self.patch_path
        verify_completed_source(self.source, build, self.lock, patch, self.windows)
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
        derived.write_bytes(derive_native_probe(original.read_bytes()))
        self.native_probe_source = {'original': descriptor(original), 'derived': descriptor(derived),
            'derivation': 'Replace exactly one quoted historical version literal with exact bounded CPU-v2 identity; all other bytes unchanged.'}
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
