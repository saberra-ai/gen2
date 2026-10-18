"""Fresh MSVC/clang-cl native comparison with identical admitted inference source."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from artifacts import acquire, matches
from build_valid_mel_candidate import descriptor, source_inventory
from frozen_git_sources import verify_git_sources
from platform_native import capture, msvc_tool
from profile_windows_sessions import HOST_SCRIPT
from qualification_storage import atomic_json
from windows_job import run_in_job
from windows_compiler_metrics import (ORDER, admit_compiler_paths, build_command, command_compiler,
                                      compiler_environment, compiler_evidence, compiler_kind, timing_evidence)
from windows_session_timing import file_inventory, profiling_environment, summarize
from windows_worker_diagnostic_v2 import Diagnostic as WorkerDiagnostic
from windows_worker_metrics import PCM_SHA256, compare_native, telemetry


REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'


class Diagnostic(WorkerDiagnostic):
    def __init__(self, source, prefix, output, jobs, kind, protocol, frozen):
        super().__init__(source, prefix, output, jobs, 4, protocol, frozen)
        self.environment = compiler_environment(profiling_environment(self.environment, 'control'))
        self.kind = compiler_kind(kind)
        self.msvc = msvc_tool(self.environment, 'cl.exe')
        self.probe_compiler_identity = descriptor(self.msvc)
        self.compiler = msvc_tool(self.environment, 'cl.exe' if kind == 'msvc' else 'clang-cl.exe')
        self.compiler_identity = descriptor(self.compiler)

    def run(self, name, command, cwd=None, environment=None):
        self.frozen()
        command = list(map(str, command))
        if name == 'sdk-build':
            command = build_command(command, self.msvc, self.compiler, self.kind)
        if name == 'windows-provenance':
            index = command.index('-Compiler') + 1
            if command[index] != str(self.msvc):
                raise ValueError('original provenance compiler differs')
            command[index] = str(self.compiler)
        if name == 'native-probe-build' and (Path(command[0]).resolve(strict=True) != self.msvc
                or descriptor(self.msvc) != self.probe_compiler_identity):
            raise ValueError('fixed MSVC probe compiler required')
        record = {'name': name, 'command': command, 'cwd': str(cwd or self.repo),
                  'process_supervision': 'Windows kill-on-close Job with gated worker', 'process_tree_reaped': False}
        self.commands.append(record)
        atomic_json(self.output / 'commands.json', self.commands)
        print('START ' + self.kind + ' ' + name, flush=True)
        started = time.monotonic()
        try:
            with (self.output / (name + '.stdout.log')).open('xb') as out, (self.output / (name + '.log')).open('xb') as err:
                result = run_in_job(command, cwd=cwd or self.repo,
                    env=environment if environment is not None else self.environment,
                    stdout=out, stderr=err, timeout=1800 if name == 'sdk-build' else 600)
            record['returncode'] = result.returncode
            record['process_tree_reaped'] = result.process_tree_reaped
            result.check_returncode()
        except BaseException as error:
            record['error'] = repr(error)
            record['process_tree_reaped'] = getattr(error, 'process_tree_reaped', record['process_tree_reaped'])
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(self.output / 'commands.json', self.commands)
        print('PASS ' + self.kind + ' ' + name, flush=True)

    def build(self):
        if self.kind == 'clang-cl':
            self.run('compiler-version', [self.compiler, '--version'])
        super().build()
        compiler_files = {}
        for language in ('C', 'CXX'):
            path, = self.build_directory.glob('CMakeFiles/*/CMake' + language + 'Compiler.cmake')
            target = self.output / ('compiler-' + language + '.cmake')
            shutil.copy2(path, target)
            compiler_files[language] = target.read_text()
        database = json.loads((self.output / 'compile_commands.json').read_text())
        self.compiler_aliases = admit_compiler_paths(self.compiler,
            {command_compiler(row['command']) for row in database})
        evidence = compiler_evidence((self.output / 'CMakeCache.txt').read_text(),
            database, compiler_files, self.compiler, self.kind, self.compiler_aliases)
        evidence['compiler_binary'] = self.compiler_identity
        evidence['compiler_path_aliases'] = self.compiler_aliases
        self.linker = Path(evidence['linker']).resolve(strict=True)
        self.linker_identity = descriptor(self.linker)
        evidence['linker_binary'] = self.linker_identity
        if descriptor(self.compiler) != self.compiler_identity:
            raise ValueError('compiler binary changed during construction')
        atomic_json(self.output / 'compiler-admission.json', evidence)
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 compiler diagnostic required')
    tools = Path(__file__).resolve().parent
    repo, source, output = tools.parents[1], args.source.resolve(strict=True), args.output.resolve()
    if (args.jobs < 1 or output.exists() or output.is_symlink() or output.is_relative_to(repo)
            or output.is_relative_to(source) or source.is_relative_to(output)):
        parser.error('fresh disjoint output outside repository/native source and positive jobs required')
    protocol_path = tools / 'windows-compiler-diagnostic.protocol.json'
    raw_protocol = protocol_path.read_bytes()
    p = json.loads(raw_protocol)
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)

    def frozen():
        if protocol_path.read_bytes() != raw_protocol or p['order'] != ORDER or p['actual_pcm_sha256'] != PCM_SHA256:
            raise ValueError('compiler diagnostic frozen protocol/order/PCM changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('compiler diagnostic frozen input differs: ' + name)

    frozen()
    expected = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()}, **p['repository_files']}
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + protocol_path.name],
            cwd=repo, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}) != raw_protocol:
        raise ValueError('committed compiler diagnostic protocol required')
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    q = {'schema_version': 1, 'source_commit': commit, 'protocol': descriptor(protocol_path),
        'diagnostic_completed': False, 'release_qualified': False, 'package_qualified': False,
        'public_or_lifecycle_qualified': False, 'order': ORDER, 'builds': {}, 'cases': [], 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    persist()
    runners = {}
    try:
        for kind in ('msvc', 'clang-cl'):
            evidence = output / ('evidence-' + kind)
            evidence.mkdir()
            runner = Diagnostic(output / ('source-' + kind), output / ('sdk-' + kind), evidence, args.jobs, kind, p, frozen)
            runners[kind] = runner
            q['builds'][kind] = {'construction_completed': False, 'compiler': str(runner.compiler),
                'compiler_binary': runner.compiler_identity,
                'probe_compiler': {'path': str(runner.msvc), **runner.probe_compiler_identity}}
            try:
                for root, origin, revision in [(runner.source, source, runner.lock['runtime']['revision']),
                        (runner.source / 'llama.cpp', source / 'llama.cpp', runner.lock['runtime']['llama_revision'])]:
                    name = 'native' if root == runner.source else 'llama'
                    runner.run('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, root])
                    runner.run('checkout-' + name, ['git', '-C', root, 'checkout', '--detach', revision])
                runner.verify_source()
                admission = runner.build()
                q['builds'][kind].update(construction_completed=True, reported_version=runner.reported_version,
                    compiler_admission=admission,
                    native_probe_source=runner.native_probe_source, installed_files=file_inventory(runner.prefix),
                    compiler_audit=descriptor(evidence / 'platform-llama-materialized-audit.json'),
                    windows_private_names=runner.naming, build_compatibility=runner.compatibility)
                q['builds'][kind]['compiler_evidence_files'] = {name: descriptor(evidence / name) for name in
                    ['compiler-C.cmake', 'compiler-CXX.cmake', 'compiler-admission.json',
                     'CMakeCache.txt', 'compile_commands.json', 'platform-llama-materialized-audit.json']}
            finally:
                q.setdefault('commands', {})[kind] = runner.commands
                persist()
        if (q['builds']['msvc']['probe_compiler'] != q['builds']['clang-cl']['probe_compiler']
                or runners['msvc'].native_probe_source != runners['clang-cl'].native_probe_source):
            raise ValueError('identical MSVC probe compiler/source required for both SDKs')
        host_runner = runners['msvc']
        host_runner.run('receiver-host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST_SCRIPT])
        q['host'] = json.loads((host_runner.output / 'receiver-host.stdout.log').read_text(encoding='utf-8-sig'))
        if q['host']['avx2'] is not True or q['host']['fma'] is not True:
            raise ValueError('actual AVX2/FMA receiver required before model admission')
        assets = output / 'assets'
        assets.mkdir()
        for item in runners['msvc'].lock['artifacts']:
            if item['role'] != 'reference':
                frozen()
                acquire(item, assets)
                if not matches(assets / item['name'], item):
                    raise ValueError('compiler diagnostic asset differs: ' + item['name'])
        q['assets'] = file_inventory(assets)
        reference_path = repo / REFERENCE
        reference, reference_raw = json.loads(reference_path.read_text()), Path(str(reference_path) + '.f32').read_bytes()

        def unchanged():
            frozen()
            if file_inventory(assets) != q['assets']:
                raise ValueError('compiler diagnostic assets changed')
            for kind, runner in runners.items():
                runner.verify_sources(runner.build_directory)
                if any(descriptor(runner.output / name) != expected
                        for name, expected in q['builds'][kind]['compiler_evidence_files'].items()):
                    raise ValueError('compiler diagnostic provenance changed')
                if (file_inventory(runner.prefix) != q['builds'][kind]['installed_files']
                        or descriptor(runner.compiler) != runner.compiler_identity
                        or descriptor(runner.msvc) != runner.probe_compiler_identity
                        or descriptor(runner.linker) != runner.linker_identity
                        or admit_compiler_paths(runner.compiler, runner.compiler_aliases) != runner.compiler_aliases
                        or source_inventory(runner.build_directory / '_deps/llama.cpp') != runner.compiler_audit['files']):
                    raise ValueError('compiler diagnostic SDK/compiler/source changed')

        for index, kind in enumerate(ORDER):
            unchanged()
            runner = runners[kind]
            phase = 'case-' + str(index) + '-' + kind
            target = runner.output / (phase + '.json')
            case = {'index': index, 'compiler': kind, 'command_started': True}
            q['cases'].append(case)
            persist()
            runner.run(phase, [runner.probe, assets / 'Nemotron-3-Diarization.q8_0.gguf',
                assets / 'ami_en2002d_2132.wav', '1', '-1', target],
                environment=profiling_environment(runner.environment, 'control'))
            actual, raw = json.loads(target.read_text()), Path(str(target) + '.f32').read_bytes()
            log = (runner.output / (phase + '.log')).read_text()
            case.update(result=actual, result_file=descriptor(target), probabilities=descriptor(Path(str(target) + '.f32')),
                workers=telemetry(log, 4), timing_control=summarize(log, 'control', actual['elapsed_micros']))
            persist()
            if case['workers']['effective_threads'] != 4 or actual.get('push_count') != 58 or type(actual.get('push_count')) is not int:
                raise ValueError('fixed four-worker/58-push compiler diagnostic required')
            case['comparison'] = compare_native(reference, actual, reference_raw, raw, 4, 4)
            persist()
        unchanged()
        q['timing_comparison'] = timing_evidence(q['cases'])
        q['diagnostic_completed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        for kind, runner in runners.items():
            q.setdefault('commands', {})[kind] = runner.commands
        persist()


if __name__ == '__main__':
    main()
